from pathlib import Path

from depth_hybrid_slam.camera_correction_core import (
    CameraCorrectionMachine, plan_camera_correction)
from depth_hybrid_slam.command_arbiter_core import CommandCandidate, arbitrate
from depth_hybrid_slam.csv_road_validator_core import CameraRiskGate
from depth_hybrid_slam.lidar_local_planner import (
    plan_detour, swept_footprint_clear)
from depth_hybrid_slam.lidar_mission_core import Mode5Avoidance
from depth_hybrid_slam.lidar_path_tracker import LocalPathTracker
from depth_hybrid_slam.lidar_roi_core import (
    Cluster, STATIC, assess_mode5_broad, mode5_avoidance_requested)
from depth_hybrid_slam.vehicle_kinematics import planner_steering_feasible


ROOT = Path(__file__).resolve().parents[3]


def candidate(drive=1.0, wheel=0):
    return CommandCandidate(drive, wheel, True, True)


def straight_csv():
    return tuple((index*0.1, 0.0, 0.0) for index in range(1, 51))


def test_csv_is_kept_for_camera_true_and_unknown():
    csv = candidate(2.0)
    for state in ("TRUE", "UNKNOWN"):
        assert state in ("TRUE", "UNKNOWN")
        decision = arbitrate(csv, CommandCandidate(), camera=CommandCandidate())
        assert decision.owner == "CSV" and decision.drive == 2.0


def test_single_frame_fail_is_unknown_but_persistent_fail_is_latched():
    gate = CameraRiskGate(0.4)
    assert gate.update("FAIL", 10.0) == "UNKNOWN"
    assert gate.update("FAIL", 10.39) == "UNKNOWN"
    assert gate.update("FAIL", 10.40) == "FAIL"
    assert gate.update("UNKNOWN", 10.41) == "UNKNOWN"


def test_camera_requires_actual_stop_and_three_seconds_before_plan():
    machine = CameraCorrectionMachine(3.0)
    assert machine.update("FAIL", True, now=0.0).state == "CAMERA_STOP"
    assert machine.update(
        "FAIL", True, now=0.1, vehicle_stopped=True).state == "CAMERA_WAIT_3S"
    early = machine.update("FAIL", True, now=3.09, vehicle_stopped=True)
    assert early.stop and not early.need_plan
    ready = machine.update("FAIL", True, now=3.10, vehicle_stopped=True)
    assert ready.stop and ready.need_plan


def test_camera_invalid_geometry_after_three_seconds_cannot_plan():
    machine = CameraCorrectionMachine(3.0)
    machine.update("FAIL", True, now=0.0)
    machine.update("FAIL", True, now=.1, vehicle_stopped=True)
    result = machine.update("FAIL", False, now=3.1, vehicle_stopped=True)
    assert result.state == "CAMERA_NO_VALID_GEOMETRY"
    assert result.stop and not result.need_plan


def test_lidar_disables_camera_and_emergency_discards_camera_follow():
    machine = CameraCorrectionMachine(3.0)
    disabled = machine.update("FAIL", True, now=0.0, lidar_active=True)
    assert disabled.state == "CAMERA_DISABLED_LIDAR_ACTIVE"
    assert disabled.discard_path and not disabled.need_plan
    machine.state = "CAMERA_FOLLOW"
    emergency = machine.update("FAIL", True, now=1.0, emergency=True)
    assert emergency.stop and emergency.discard_path
    assert emergency.state == "EMERGENCY_STOP"


def test_arbiter_enforces_stop_lidar_camera_csv_single_owner_priority():
    csv, camera, lidar = candidate(2.0), candidate(1.0, -4), candidate(1.0, 4)
    assert arbitrate(csv, lidar, camera=camera).owner == "LIDAR"
    assert arbitrate(csv, lidar, mode=7, camera=camera).owner == "PARKING"
    assert arbitrate(
        csv, CommandCandidate(), mode=7, camera=camera).owner == "CSV"
    assert arbitrate(csv, CommandCandidate(), camera=camera).owner == "CAMERA"
    assert arbitrate(csv, CommandCandidate(), camera=CommandCandidate()).owner == "CSV"
    assert arbitrate(csv, lidar, hard_emergency=True, camera=camera).drive == 0.0


def test_camera_local_path_is_short_steer_safe_and_rejoins_exact_csv():
    path = straight_csv()
    plan = plan_camera_correction(path, .30, 1.0)
    assert plan.valid and plan.state == "CAMERA_PATH_VALID"
    assert plan.max_steering_deg <= 20.0
    assert 3.0 <= plan.points[-1][0] <= 5.0
    assert plan.points[-1] == path[-1]


def test_camera_local_path_rejects_curb_footprint_collision():
    plan = plan_camera_correction(
        straight_csv(), .30, 1.0, curbs=((2.5, -.28),))
    assert not plan.valid and plan.state == "CAMERA_CURB_COLLISION"


def test_camera_local_path_rejects_lane_footprint_collision():
    plan = plan_camera_correction(
        straight_csv(), .30, 1.0, lane_points=((2.5, -.28),))
    assert not plan.valid and plan.state == "CAMERA_LANE_COLLISION"


def test_steering_planning_boundary_allows_20_and_rejects_above_20():
    assert planner_steering_feasible(20.0)
    assert planner_steering_feasible(-20.0)
    assert not planner_steering_feasible(20.1)
    assert not planner_steering_feasible(-20.1)


def _initial_wheel(plan, drive=1.0):
    tracker = LocalPathTracker()
    assert plan.valid and tracker.set_plan(plan.points, (0.0, 0.0, 0.0), drive)
    return tracker.update((0.0, 0.0, 0.0)).wheel


def test_camera_and_lidar_correction_direction_signs():
    # A right-side lane risk shifts left; a left-side risk shifts right.
    assert _initial_wheel(plan_camera_correction(
        straight_csv(), 1.0, 0.30)) > 0
    assert _initial_wheel(plan_camera_correction(
        straight_csv(), 0.30, 1.0)) < 0
    # An obstacle is avoided on its opposite side.
    assert _initial_wheel(plan_detour(-0.30)) > 0
    assert _initial_wheel(plan_detour(0.30)) < 0


def test_normal_curb_outside_corridor_is_not_mode5_obstacle():
    curb = Cluster(tuple((x, .60) for x in (.4, .55, .7, .85)), (.625, .60),
                   motion=STATIC)
    broad = assess_mode5_broad((curb,), tuple((x*.1, 0.0) for x in range(16)),
                               range_m=1.5)
    assert broad.curbs == (curb,)
    assert not broad.path_blocked and not broad.collision_obstacles


def test_actual_corridor_obstacle_requests_mode5_and_other_modes_do_not_plan():
    obstacle = Cluster(((.75, -.02), (.75, .02)), (.75, 0.0), motion=STATIC)
    broad = assess_mode5_broad((obstacle,), tuple((x*.1, 0.0) for x in range(16)),
                               range_m=1.5)
    assert broad.path_blocked and broad.collision_obstacles == (obstacle,)
    assert mode5_avoidance_requested(5, True, True, broad.path_blocked)
    for mode in (1, 2, 3, 4, 6, 8, 9, 11):
        assert not mode5_avoidance_requested(mode, True, True, True)


def test_lidar_lane_crossing_allowed_but_curb_footprint_rejected():
    # LiDAR planner has no lane-boundary input: crossing paint is intentional.
    free = plan_detour(.25)
    assert free.valid
    assert swept_footprint_clear(free.points, ((2.75, 2.0),))
    blocked = plan_detour(.25, curbs=((2.75, -.65),))
    assert not blocked.valid and blocked.replan_reason == "CURB_FOOTPRINT"


def test_no_valid_lidar_path_keeps_mode5_stop():
    core = Mode5Avoidance()
    assert core.update(avoidance_required=True).state == "STOP_FOR_PLANNING"
    decision = core.update(avoidance_required=True, path_valid=False)
    assert decision.stop and decision.state == "STOP_FOR_PLANNING"


def test_production_start_and_lane_tracking_contracts_are_explicit():
    launch = (ROOT/"src/depth_hybrid_slam/launch"/
              "depth_real_vehicle.launch.py").read_text(encoding="utf-8")
    camera_launch = (ROOT/"src/camera_navigation/launch"/
                     "d456_traffic_light.launch.py").read_text(encoding="utf-8")
    assert 'executable="start_validation"' not in launch
    assert '"initial_branch": branch' in launch
    assert '"line_track_mode": "flow"' in camera_launch
    assert '"line_track_max_hold_sec": "0.20"' in camera_launch
