import math
from dataclasses import replace
from pathlib import Path

from depth_hybrid_slam.command_arbiter_core import (
    CommandCandidate, arbitrate)
from depth_hybrid_slam.csv_road_validator_core import (
    DEGRADED_LANE_UNCERTAIN, INVALID_OUTSIDE_ROAD, ValidationResult)
from depth_hybrid_slam.lidar_local_planner import (
    path_collision_free, plan_detour, plan_parking)
from depth_hybrid_slam.lidar_mission_core import (
    Mode5Avoidance, Mode5ObstacleLatch, Mode9Emergency,
    Mode9EmergencyLatch, Mode11ExitGate, ParkingManeuver,
    StationaryConfirmation, SteeringSlowdownLatch,
    mode5_planning_distance_ready,
    RejoinCandidate,
    bounded_rejoin, parking_decision, route_rejoin_candidates)
from depth_hybrid_slam.lidar_path_tracker import LocalPathTracker
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.route_follower_core import RouteFollower
from depth_hybrid_slam.route_io import forward_tangent_yaw
from depth_hybrid_slam.lidar_scan_core import ScanSafety
from depth_hybrid_slam.mission_core import MissionMachine
from depth_hybrid_slam.models import MissionInputs
from depth_hybrid_slam.prehardware_core import BranchSelector
from depth_hybrid_slam.traffic_gate import (
    IntersectionProgress, IntersectionTrafficGate)
from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.vehicle_kinematics import AckermannPathEvaluator


ROOT = Path(__file__).resolve().parents[3]


def _route_case(case):
    return load_csv_only_route_case(
        ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv",
        ROOT/"routes/network/route_network_segmented_stop_edited_vforward.metadata.yaml",
        case)


def test_camera_advisory_three_state_contract():
    states = {
        ValidationResult(True, "VALID_ROAD_AND_LANE",
                         lane_geometry_confident=True).advisory_state,
        ValidationResult(True, "VALID_ROAD_ONLY").advisory_state,
        ValidationResult(True, DEGRADED_LANE_UNCERTAIN).advisory_state,
        ValidationResult(False, INVALID_OUTSIDE_ROAD,
                         lane_geometry_confident=True,
                         wheel_lane_collision_ratio=.2).advisory_state,
        ValidationResult(False, "STALE_INPUT").advisory_state,
    }
    assert states == {"TRUE", "FAIL", "UNKNOWN"}
    near = ValidationResult(True, DEGRADED_LANE_UNCERTAIN)
    assert near.camera_diagnostics(True)["state"] == "UNKNOWN"
    assert "wheel_lane_collision_ratio" in near.camera_diagnostics(True)


def test_intersection_red_green_and_three_second_unknown_release():
    gate = IntersectionTrafficGate(3.0, 0.5)
    progress = IntersectionProgress(
        True, 4, "COMMON_1", 1, 100, 10.0,
        "COMMON_1", 1, 100, 25, 10.0, 130, 20.0)
    assert gate.evaluate(progress, True, "R", 0.0, 0.0).state == \
        "MINIMUM_3S_HOLD"
    assert gate.evaluate(progress, True, "R", 0.0, 9.0).stop
    gate.reset()
    assert gate.evaluate(progress, True, "G", 0.0, 0.0).stop
    assert not gate.evaluate(progress, True, "G", 0.0, 3.0).stop
    gate.reset()
    assert gate.evaluate(progress, True, "UNKNOWN", 0.0, 0.0).stop
    assert not gate.evaluate(progress, True, "UNKNOWN", 0.0, 3.0).stop
    gate.reset()
    assert gate.evaluate(progress, True, "G", 0.51, 0.0).stop
    assert gate.evaluate(progress, True, "G", 0.51, 3.0).stop


def test_low_confidence_traffic_is_unknown_even_when_fresh():
    machine = MissionMachine(traffic_timeout_s=0.5,
                             min_traffic_confidence=0.65)
    value = MissionInputs(
        now=0.0, csv_stop_line_active=True, traffic_age=0.0,
        traffic_confidence=0.2, traffic_aspect="RED_X",
        intersection_progress=IntersectionProgress(
            True, 4, "COMMON_1", 1, 100, 10.0,
            "COMMON_1", 1, 100, 25, 10.0, 130, 20.0))
    decision = machine.update(value)
    assert decision.state == "MINIMUM_3S_HOLD"
    assert decision.stop_required


def test_mode5_complete_sequence_and_failure_policy():
    core = Mode5Avoidance()
    assert core.update().state == "CSV_TRACKING"
    assert core.update(avoidance_required=True).state == "STOP_FOR_PLANNING"
    assert core.update(avoidance_required=True, path_valid=True).state == \
        "LIDAR_PATH_TRACKING"
    assert core.update(path_complete=True).state == "CSV_REJOIN"
    assert core.update(rejoin_valid=True).state == "CSV_TRACKING"
    assert core.update(True, True, "PATH_ABORT").stop
    assert core.update(True, False, "NO_VALID_DETOUR").owner == "SAFETY"
    failed = core.update(True, True, "NO_FEASIBLE_DETOUR")
    assert failed.stop and failed.owner == "SAFETY"
    cleared = core.update(False, False, "NO_FEASIBLE_DETOUR")
    assert not cleared.stop and cleared.owner == "CSV"
    failed = core.update(True, True, "NO_FEASIBLE_DETOUR")
    assert core.update(True, False).state == "STOP_FOR_PLANNING"
    assert core.update(False, False).state == "CSV_TRACKING"


def test_mode5_obstacle_must_be_visible_two_seconds_then_stays_latched():
    latch = Mode5ObstacleLatch(confirmation_s=2.0)
    assert not latch.update(True, mode=5, now=0.0)
    assert not latch.update(True, mode=5, now=1.99)
    assert latch.update(True, mode=5, now=2.0)
    # A qualified obstacle remains remembered if a later scan loses it.
    assert latch.update(False, mode=5, now=2.1)
    latch.reset()
    assert not latch.update(True, mode=5, now=3.0)
    assert not latch.update(False, mode=5, now=4.9)
    assert not latch.update(True, mode=5, now=5.0)
    assert not latch.update(True, mode=4, now=8.0)


def test_mode5_planning_waits_for_measured_stationary_confirmation():
    gate = StationaryConfirmation(
        duration_s=0.30, linear_limit_mps=0.03,
        angular_limit_rps=0.03)
    assert not gate.update(0.04, 0.0, True, 0.0)
    assert not gate.update(0.0, 0.0, True, 1.0)
    assert not gate.update(0.0, 0.0, True, 1.29)
    assert gate.update(0.0, 0.0, True, 1.30)
    assert not gate.update(0.0, 0.0, False, 2.0)


def test_mode5_planning_is_generated_before_one_metre_only():
    assert not mode5_planning_distance_ready(None)
    assert not mode5_planning_distance_ready(float("nan"))
    assert not mode5_planning_distance_ready(0.999)
    assert mode5_planning_distance_ready(1.0)
    assert mode5_planning_distance_ready(1.5)


def test_bounded_rejoin_rejects_wrong_segment_backtrack_direction_and_limit():
    values = (
        RejoinCandidate("OTHER", 105, 0.1, 0.0, 0.0, 1),
        RejoinCandidate("M5", 99, 0.1, 0.0, 0.0, 1),
        RejoinCandidate("M5", 105, 0.1, 0.0, 0.0, -1),
        RejoinCandidate("M5", 105, 0.1, 0.0, 22.1, 1),
        RejoinCandidate("M5", 106, 0.4, 5.0, 22.0, 1),
    )
    selected = bounded_rejoin(values, "M5", 100, 1)
    assert selected == values[-1]
    assert selected.required_steering_deg <= 22.0


def test_route_rejoin_candidates_use_only_forward_active_segment():
    class Point:
        def __init__(self, x, segment="M5", direction=1):
            self.x, self.y, self.yaw = x, 0.0, 0.0
            self.segment_id, self.direction = segment, direction

    route = tuple(Point(index*0.1) for index in range(20))
    values = route_rejoin_candidates(route, "M5", 5, (0.5, 0.1, 0.0), 5)
    assert values and min(value.index for value in values) == 5
    assert max(value.index for value in values) == 10
    assert all(value.segment == "M5" and value.direction == 1
               for value in values)


def test_local_detour_uses_vehicle_contract():
    plan = plan_detour(
        0.3, wheelbase_m=0.73, planner_max_steering_deg=21.0)
    assert plan.valid
    assert plan.max_steering_deg < 22.0
    assert plan.points[0][1] == 0.0 and abs(plan.points[-1][1]) < 1e-9
    assert path_collision_free(plan.points, ((3.0, 1.5),))
    assert not plan_detour(0.3, obstacles=((0.1, 0.0),)).valid


def test_local_path_tracker_feedback_and_completion():
    plan = plan_detour(0.3)
    tracker = LocalPathTracker()
    assert tracker.set_plan(plan.points, (10.0, 5.0, 0.0), 1.0)
    first = tracker.update((10.0, 5.0, 0.0))
    assert first.valid and not first.complete
    assert abs(first.wheel) <= 22
    final = plan.points[-1]
    before_final = plan.points[-2]
    assert not tracker.update(
        (10.0+before_final[0], 5.0+before_final[1], 0.0)).complete
    assert tracker.update((10.0+final[0], 5.0+final[1], 0.0)).complete
    assert not tracker.update((100.0, 100.0, 0.0)).valid


def test_modes_7_and_10_parking_selection_and_fallback():
    assert parking_decision(7, True, False, False).branch == "A"
    assert parking_decision(7, False, True, False).branch == "B"
    assert parking_decision(7, False, True, True).owner == "LIDAR"
    assert parking_decision(7, False, True, False,
                            "PLANNER_GIVE_UP").owner == "CSV"
    missing = parking_decision(7, False, False, False)
    assert missing.stop and missing.state == "T_WAIT_LIDAR_SLOT"
    assert parking_decision(10, True, False, True).state == "V_LIDAR_PATH"
    assert parking_decision(10, False, True, False,
                            "PATH_ABORT").branch == "B"
    parking = ParkingManeuver(7)
    prepare = parking.update("B", path_valid=True, now=6.0)
    assert prepare.owner == "LIDAR" and prepare.stop
    assert prepare.state == "T_DIRECTION_CHANGE_HOLD"
    tracking = parking.update(
        "B", path_valid=True, drive=-1.0, wheel=22, now=9.0)
    assert tracking.owner == "LIDAR" and tracking.branch == "B"
    rejoin = parking.update("B", path_complete=True, now=10.0)
    assert rejoin.owner == "LIDAR" and rejoin.stop
    assert rejoin.state == "T_CSV_REJOIN"
    assert parking.update("B", rejoin_valid=True, now=12.99).owner == "LIDAR"
    complete = parking.update("B", rejoin_valid=True, now=13.0)
    assert complete.owner == "CSV" and complete.state == "T_COMPLETE"
    assert parking_decision(10, True, False, True,
                            hard_obstacle=True).stop
    for mode in (7, 10):
        for branch in ("A", "B"):
            plan = plan_parking(mode, branch)
            assert plan.valid and plan.max_steering_deg < 22.0


def _assert_parking_path_has_strict_handoff(
        mode, route_case, segment, branch):
    route = _route_case(route_case)
    activation_direction = -1 if mode == 7 else 1
    start = next(point for point in route if point.segment_id == segment and
                 point.direction == activation_direction)
    plan = plan_parking(mode, branch)
    local_x, local_y, local_yaw = plan.points[-1]
    pose = (
        start.x+math.cos(start.yaw)*local_x-math.sin(start.yaw)*local_y,
        start.y+math.sin(start.yaw)*local_x+math.cos(start.yaw)*local_y,
        start.yaw+local_yaw,
    )
    current = min(
        (point for point in route if point.segment_id == segment and
         point.index >= start.index),
        key=lambda point: math.hypot(point.x-pose[0], point.y-pose[1]))
    candidates = route_rejoin_candidates(
        route, segment, current.index, pose, forward_window=120)
    selected = bounded_rejoin(
        candidates, segment, current.index, current.direction,
        forward_window=120, max_distance_m=2.5)
    assert selected is not None
    assert selected.index >= current.index
    assert selected.direction == current.direction
    assert selected.distance_m <= 2.5
    assert abs(selected.required_steering_deg) <= 22.0


def test_parking_local_paths_end_inside_strict_csv_handoff_corridors():
    """Regression: canned parking paths must end at a bounded CSV rejoin."""
    _assert_parking_path_has_strict_handoff(7, "AAAA", "T_A", "A")
    _assert_parking_path_has_strict_handoff(7, "ABAA", "T_B", "B")
    _assert_parking_path_has_strict_handoff(10, "AAAA", "V_A", "A")
    _assert_parking_path_has_strict_handoff(10, "AABA", "V_B", "B")


def test_parking_plan_activation_uses_the_recorded_csv_direction_lineage():
    route = _route_case("AAAA")
    t_points = [point for point in route if point.segment_id == "T_A"]
    v_points = [point for point in route if point.segment_id == "V_A"]
    assert t_points[0].direction == 1 and t_points[5].direction == -1
    assert v_points[0].direction == 1


def _points(distance, count=2):
    return tuple((distance, 0.05*index) for index in range(count))


def test_mode5_direct_start_tangent_sign_and_real_time_tracking():
    points = [point for point in _route_case("AAAA")
              if int(point.mode) == 5]
    points = [replace(point, index=index)
              for index, point in enumerate(points)]
    initial_yaw = forward_tangent_yaw(points)
    points[0] = replace(points[0], yaw=initial_yaw)
    assert (points[0].x, points[0].y) == (
        21.748382018960537, 65.39207863429603)
    assert initial_yaw == math.atan2(
        points[3].y-points[0].y, points[3].x-points[0].x)

    follower = RouteFollower(corridor_m=1.0, max_index_backtrack=0)
    initial = follower.compute(
        Pose2D(points[0].x, points[0].y, initial_yaw, 0.0), points,
        allow_motion=True, global_search=True)
    assert initial.nearest_index == 0 and initial.target_index > 0
    # This opening section bends right from the stabilized tangent.
    assert initial.steering_deg < 0.0

    vehicle = AckermannPathEvaluator(
        {1: 0.527, 2: 0.791, 3: 1.055}, 0.527,
        wheelbase_m=0.730, counts_per_meter=797.0,
        max_steering_deg=22.0, direction_change_hold_s=3.0,
        encoder_signed=False)
    vehicle.x, vehicle.y, vehicle.yaw = (
        points[0].x, points[0].y, initial_yaw)
    maximum_cte = 0.0
    # Ten seconds at the actual 250 Hz integration rate; no speedup.
    for step in range(2500):
        result = follower.compute(
            Pose2D(vehicle.x, vehicle.y, vehicle.yaw, step/250.0), points,
            allow_motion=True)
        assert not result.stop_required
        maximum_cte = max(maximum_cte, abs(result.cross_track_error))
        vehicle.step(result.drive, result.steering_deg, False, 1.0/250.0)
    assert follower.last_index > 30
    assert maximum_cte < 0.15


def test_mode9_is_fixed_stage_three_except_hard_emergency():
    safety = ScanSafety(minimum_points=2, clear_scans=2)
    assert not safety.assess(_points(0.51)).hard_obstacle
    assert safety.assess(_points(0.50)).hard_obstacle
    assert safety.assess(_points(0.49)).hard_obstacle
    assert safety.assess(_points(0.61)).hard_obstacle
    assert not safety.assess(_points(0.61)).hard_obstacle
    mode9 = Mode9Emergency()
    assert mode9.update(True).state == "EMERGENCY_STOP"
    resumed = mode9.update(False)
    assert resumed.state == "ACCEL_TRACKING"
    assert not resumed.stop and resumed.drive == 3.0
    csv = CommandCandidate(2.0, 8, True, True)
    assert arbitrate(
        csv, CommandCandidate(), mode=9).drive == 3.0
    assert arbitrate(
        csv, CommandCandidate(), lidar_slowdown=True, mode=9).drive == 1.0
    assert arbitrate(
        csv, CommandCandidate(), steering_slowdown=True, mode=9).drive == 3.0
    assert arbitrate(
        csv, CommandCandidate(), hard_emergency=True,
        lidar_slowdown=True, mode=9).drive == 0.0


def test_all_mode_steering_slowdown_one_second_enter_and_exit():
    latch = SteeringSlowdownLatch(
        threshold_deg=10.0, enter_duration_s=1.0, exit_duration_s=1.0)

    assert not latch.update(10.0, now=0.0)
    assert not latch.update(10.0, now=0.99)
    assert latch.update(10.0, now=1.0)

    # A short below-threshold interval cannot clear the slowdown.
    assert latch.update(0.0, now=1.1)
    assert latch.update(10.0, now=1.5)
    assert latch.below_since is None

    assert latch.update(0.0, now=1.6)
    assert latch.update(0.0, now=2.59)
    assert not latch.update(0.0, now=2.6)

    # Stale measured steering cannot fabricate a debounce interval.
    assert not latch.update(-10.01, now=4.0)
    assert not latch.update(None, now=5.0)
    assert not latch.update(-10.01, now=5.1)
    assert latch.update(-10.01, now=6.1)

    # Mode 9 obeys the same measured-steering slowdown.
    assert latch.update(-22.0, now=6.2)


def test_mode9_emergency_latch_holds_dropouts_and_near_obstacle_until_clear():
    latch = Mode9EmergencyLatch(clear_distance_m=1.5,
                                clear_duration_s=1.0)
    mode9 = Mode9Emergency()

    # A: a confirmed 0.4 m obstacle remains stopped for a simulated 10 s.
    for index in range(200):
        assert latch.update(True, 0.4, True, True, index*0.05)
        assert mode9.update(latch.latched).stop

    # B: one to three empty scan frames cannot satisfy a one-second clear.
    for index in range(3):
        assert latch.update(False, None, True, True, 10.0+index*0.05)
        assert mode9.update(latch.latched).state == "EMERGENCY_STOP"

    # No scan and stale scan periods are not clear confirmations or timeouts.
    for index in range(200):
        assert latch.update(False, None, False, False, 11.0+index*0.05)
    assert latch.clear_since is None

    # C: any obstacle at or inside 1.5 m resets clear confirmation.
    assert latch.update(False, 1.49, True, True, 22.0)
    assert latch.clear_since is None
    assert mode9.update(latch.latched).state == "EMERGENCY_STOP"

    # D: release happens only after one continuous second beyond 1.5 m.
    assert latch.update(False, 1.51, True, True, 23.0)
    assert latch.update(False, 1.51, True, False, 24.5)
    assert latch.update(False, 1.49, True, True, 24.6)
    assert latch.update(False, 1.51, True, True, 25.0)
    assert latch.update(False, 1.51, True, True, 25.99)
    assert not latch.update(False, 1.51, True, True, 26.0)

    # E: the qualified clear immediately restores fixed stage 3.
    resumed = mode9.update(False)
    assert resumed.state == "ACCEL_TRACKING" and resumed.drive == 3.0


def test_front_rplidar_mount_contract_maps_vehicle_front_to_base_plus_x():
    config = (ROOT/"src/depth_hybrid_slam/config/rplidar_dual.yaml").read_text()
    launch = (ROOT/"src/depth_hybrid_slam/launch/dual_rplidar.launch.py").read_text()
    assert "inverted: false" in config
    assert "angle_compensate: true" in config
    assert "flip_x_axis: true" in config
    assert '"--yaw", "0.0"' in launch

    # A2 +X follows the lead.  With the front unit's lead mounted rearward,
    # an object at vehicle front starts at hardware x=-1. flip_x_axis rotates
    # it to logical front_laser x=+1, and yaw=0 keeps it at base_link +X.
    laser_x, laser_y, yaw = 1.0, 0.0, 0.0
    base_x = math.cos(yaw)*laser_x-math.sin(yaw)*laser_y
    base_y = math.sin(yaw)*laser_x+math.cos(yaw)*laser_y
    assert base_x > 0.99 and abs(base_y) < 1.0e-9


def test_front_only_launch_disables_rear_parking_without_camera_nodes():
    launch = (ROOT/"src/depth_hybrid_slam/launch/" /
              "prehardware_csv_front_lidar.launch.py").read_text()
    assert '"launch_rear_lidar_driver": "false"' in launch
    assert '"rear_lidar_enabled": False' in launch
    route_loop = (ROOT/"src/depth_hybrid_slam/launch"/
                  "prehardware_csv_only_closed_loop.launch.py").read_text()
    assert 'executable="test_odom_publisher"' in route_loop
    assert '"start_rviz"' in launch
    assert "camera_bringup" not in launch
    assert "csv_road_validator" not in launch


def test_front_only_parking_uses_explicit_csv_fallback():
    source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/" /
              "maneuver_manager_node.py").read_text()
    assert 'self.declare_parameter("rear_lidar_enabled", True)' in source
    assert "ManeuverDecision, Mode11ExitGate" in source
    assert 'if not bool(self.get_parameter("rear_lidar_enabled").value):' in source
    assert 'prefix+"_CSV_FALLBACK"' in source


def test_lidar_markers_remove_vehicle_sweep_and_keep_front_sensor_zones():
    source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/" /
              "lidar_perception_node.py").read_text()
    assert 'self.front_frame = "front_laser"' in source
    assert 'line(marker_id, "zones"' in source
    assert 'line(5, "csv_sweep",' not in source
    assert 'line(4, "mode5_broad"' not in source
    assert '"static_obstacles_blue"' in source
    assert '"dynamic_obstacles_green"' in source
    assert '"roi_obstacles_red"' in source


def test_mode11_five_second_default_and_commit_is_immutable():
    gate = Mode11ExitGate(confirmations=2)
    gate.enter(0.0)
    gate.observe("2", 0.1)
    gate.observe("2", 0.2)
    assert gate.evaluate(4.99).stop
    # Stale B evidence is not an explicit fresh B and therefore defaults A.
    assert gate.evaluate(5.0).branch == "A"
    gate.observe("2", 5.1)
    assert gate.evaluate(5.2).branch == "A"
    fresh = Mode11ExitGate(confirmations=2)
    fresh.enter(0.0)
    fresh.observe("2", 4.8)
    fresh.observe("2", 4.9)
    assert fresh.evaluate(5.0).branch == "B"
    fresh.observe("1", 5.1)
    assert fresh.evaluate(6.0).branch == "B"
    first = Mode11ExitGate(confirmations=2)
    first.enter(0.0)
    first.observe("1", 4.8)
    first.observe("1", 4.9)
    assert first.evaluate(5.0).branch == "A"
    unknown = Mode11ExitGate(confirmations=2)
    assert unknown.evaluate(5.0).stop
    assert unknown.evaluate(10.0).branch == "A"


def test_mode11_csv_branch_mapping_is_a_end_aa_b_end_ab():
    root = Path(__file__).resolve().parents[3]
    route = root/"routes/network/route_network_segmented_stop_edited_vforward.csv"
    segments_a = {point.segment_id for point in load_csv_only_route_case(
        route, route.with_suffix(".metadata.yaml"), "AAAA")}
    segments_b = {point.segment_id for point in load_csv_only_route_case(
        route, route.with_suffix(".metadata.yaml"), "AAAB")}
    assert "END_AA" in segments_a and "END_AB" not in segments_a
    assert "END_AB" in segments_b and "END_AA" not in segments_b


def test_mode11_five_second_gate_beats_selector_transport_grace():
    camera = Mode11ExitGate(confirmations=2)
    selector = BranchSelector(command_timeout_s=3.0, mode_11_wait_s=5.5)
    camera.enter(0.0)
    selector.set_mode(11, 0.0)
    camera.observe("2", 4.8)
    camera.observe("2", 4.9)
    assert selector.evaluate(4.99).stop
    decision = camera.evaluate(5.0)
    assert decision.branch == "B" and not decision.stop
    selector.set_command(decision.branch, 5.0)
    selected = selector.evaluate(5.05)
    assert selected.branch == "B" and not selected.stop


def test_command_arbiter_priority_and_contract():
    csv = CommandCandidate(3.0, 22, True, True)
    lidar = CommandCandidate(1.0, -18, True, True)
    assert arbitrate(csv, lidar, True, False).owner == "SAFETY"
    assert arbitrate(csv, lidar, False, True).owner == "MISSION"
    assert arbitrate(csv, lidar).owner == "LIDAR"
    assert arbitrate(csv, CommandCandidate()).owner == "CSV"
    assert arbitrate(CommandCandidate(), CommandCandidate()).drive == 0.0
    assert arbitrate(CommandCandidate(2.0, 23, True, True),
                     CommandCandidate()).owner == "NONE"
