import math
from pathlib import Path

import pytest

from depth_hybrid_slam.lidar_local_planner import (
    HARD_STEERING_LIMIT_DEG, audit_path, path_collision_free, plan_detour,
    plan_parking, plan_route_detour, steering_geometry)
from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.lidar_mission_core import (
    bounded_rejoin, route_rejoin_candidates)
from depth_hybrid_slam.lidar_mission_core import ParkingManeuver
from depth_hybrid_slam.lidar_path_tracker import LocalPathTracker
from depth_hybrid_slam.vehicle_kinematics import AckermannPathEvaluator


ROOT = Path(__file__).resolve().parents[3]


def _assert_generated_feasible(plan, planner_limit=20.0):
    assert plan.valid
    assert plan.path_point_count == len(plan.points) >= 2
    assert plan.max_steering_deg <= planner_limit+1.0e-6
    assert plan.max_steering_deg < HARD_STEERING_LIMIT_DEG
    assert plan.max_curvature <= math.tan(math.radians(planner_limit))/0.73 + 1e-8
    assert plan.min_turn_radius_m >= 0.73/math.tan(
        math.radians(planner_limit))-1e-8
    final = audit_path(plan.points)
    assert final.feasible
    assert final.max_required_steering_deg == pytest.approx(
        plan.max_steering_deg)


def test_canonical_planner_steering_derives_curvature_and_radius():
    curvature, radius = steering_geometry(0.73, 20.0)
    assert curvature == pytest.approx(math.tan(math.radians(20.0))/0.73)
    assert radius == pytest.approx(0.73/math.tan(math.radians(20.0)))
    with pytest.raises(ValueError):
        steering_geometry(0.73, 22.0)


@pytest.mark.parametrize("obstacle_y", (0.3, -0.3))
@pytest.mark.parametrize("length,lateral,expected", (
    (7.0, 0.20, (0.0, 15.0)),
    (6.2, 0.65, (14.0, 16.0)),
    (5.2, 0.65, (19.0, 21.0)),
))
def test_mode5_easy_15_and_20_degree_left_right_paths(
        obstacle_y, length, lateral, expected):
    plan = plan_detour(obstacle_y, length_m=length, lateral_m=lateral)
    _assert_generated_feasible(plan)
    assert plan.max_steering_deg <= 20.0+1.0e-6
    peak_y = max(plan.points, key=lambda point: abs(point[1]))[1]
    assert math.copysign(1.0, peak_y) == -math.copysign(1.0, obstacle_y)


@pytest.mark.parametrize("obstacle_y", (0.3, -0.3))
def test_mode5_29_degree_candidate_is_lengthened_not_clamped(obstacle_y):
    plan = plan_detour(obstacle_y, length_m=4.2, lateral_m=0.65)
    _assert_generated_feasible(plan)
    assert 23.0 <= plan.initial_steering_deg <= 30.0
    assert plan.replans >= 1
    assert "STEERING_LIMIT" in plan.replan_reason
    assert plan.state == "READY_REPLANNED"
    assert plan.points[-1][0] > 4.2
    assert abs(plan.points[0][2]) < 1.0e-12
    assert abs(plan.points[-1][1]) < 1.0e-12
    assert abs(plan.points[-1][2]) < 1.0e-12


def test_mode5_widens_for_vehicle_clearance_without_reducing_clearance():
    obstacle = ((2.0, 0.0),)
    plan = plan_detour(0.1, obstacles=obstacle)
    _assert_generated_feasible(plan)
    assert plan.replans >= 1
    assert path_collision_free(plan.points, obstacle, clearance_m=0.55)


def test_clearance_audit_checks_between_sampled_path_points():
    sparse = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert not path_collision_free(
        sparse, ((1.0, 0.1),), clearance_m=0.55)


def test_mode5_reports_no_feasible_detour_instead_of_clamping():
    plan = plan_detour(0.0, obstacles=((0.1, 0.0),))
    assert not plan.valid
    assert plan.state == "NO_FEASIBLE_DETOUR"


@pytest.mark.parametrize("mode", (7, 10))
@pytest.mark.parametrize("branch", ("A", "B"))
@pytest.mark.parametrize("requested", (15.0, 20.0, 24.0, 30.0))
def test_parking_regenerates_requested_steering_inside_limit(
        mode, branch, requested):
    plan = plan_parking(
        mode, branch, requested_steering_deg=requested)
    _assert_generated_feasible(plan)
    if requested > 20.0:
        assert plan.replans == 1
        assert plan.state == "READY_REPLANNED"
        assert plan.initial_steering_deg == requested
    else:
        assert plan.replans == 0


@pytest.mark.parametrize("branch", ("A", "B"))
def test_mode10_shipped_22_degree_schedule_is_radius_expanded(branch):
    plan = plan_parking(10, branch)
    _assert_generated_feasible(plan)
    assert plan.initial_steering_deg == 22.0
    assert plan.replans == 1
    assert plan.max_steering_deg == pytest.approx(20.0)


def test_parking_reverse_to_forward_handoff_keeps_existing_stop_contract():
    parking = ParkingManeuver(10)
    prepare = parking.update("A", path_valid=True, now=0.0)
    assert prepare.stop and prepare.state == "V_DIRECTION_CHANGE_HOLD"
    assert parking.update("A", path_valid=True, now=2.99).stop
    reverse = parking.update("A", path_valid=True, drive=-1.0, now=3.0)
    assert not reverse.stop and reverse.drive == -1.0
    hold = parking.update("A", path_complete=True, now=5.0)
    assert hold.stop and hold.state == "V_CSV_REJOIN"
    assert parking.update("A", rejoin_valid=True, now=7.99).stop
    forward = parking.update("A", rejoin_valid=True, now=8.0)
    assert not forward.stop and forward.owner == "CSV"


@pytest.mark.parametrize("mode", (7, 10))
@pytest.mark.parametrize("branch", ("A", "B"))
def test_reverse_tracker_follows_every_generated_parking_segment_without_clamp(
        mode, branch):
    plan = plan_parking(mode, branch)
    tracker = LocalPathTracker()
    assert tracker.set_plan(plan.points, (0.0, 0.0, 0.0), -1.0)
    commands = [tracker.update(point) for point in plan.points[:-1]]
    assert all(command.valid for command in commands)
    assert max(abs(command.wheel) for command in commands) <= 21
    assert min(command.wheel for command in commands) < 0
    assert max(command.wheel for command in commands) > 0


@pytest.mark.parametrize("mode", (7, 10))
@pytest.mark.parametrize("branch", ("A", "B"))
def test_parking_path_survives_coarse_reverse_closed_loop_tracking(mode, branch):
    plan = plan_parking(mode, branch)
    tracker = LocalPathTracker()
    tracker.set_plan(plan.points, (0.0, 0.0, 0.0), -1.0)
    vehicle = AckermannPathEvaluator(
        {1: .527, 2: .791, 3: 1.055}, .527)
    vehicle.last_motion_direction = -1
    for _ in range(50):
        command = tracker.update((vehicle.x, vehicle.y, vehicle.yaw))
        assert command.valid
        assert abs(command.wheel) <= 21
        if command.complete:
            break
        vehicle.step(command.drive, command.wheel, False, .5)
    assert command.complete


@pytest.mark.parametrize("mode", (7, 10))
@pytest.mark.parametrize("branch", ("A", "B"))
def test_parking_path_survives_scaled_loop_with_pipeline_delay(mode, branch):
    """10x prehardware loop remains feasible with three command ticks delay."""
    plan = plan_parking(mode, branch)
    tracker = LocalPathTracker()
    tracker.set_plan(plan.points, (0.0, 0.0, 0.0), -1.0)
    vehicle = AckermannPathEvaluator(
        {1: .527, 2: .791, 3: 1.055}, .527)
    vehicle.last_motion_direction = -1
    commands = []
    for _ in range(500):
        command = tracker.update((vehicle.x, vehicle.y, vehicle.yaw))
        assert command.valid
        commands.append(command)
        if command.complete:
            break
        delayed = commands[max(0, len(commands)-4)]
        # 250 Hz wall-clock at 10x simulation speed is 0.04 s vehicle time.
        vehicle.step(delayed.drive, delayed.wheel, False, .04)
    assert command.complete


def test_final_audit_rejects_a_30_degree_path():
    curvature = math.tan(math.radians(30.0))/0.73
    distance = 0.1
    invalid = ((0.0, 0.0, 0.0),
               (distance, 0.0, distance*curvature))
    result = audit_path(invalid)
    assert not result.feasible
    assert result.max_required_steering_deg >= 22.0


def test_tracker_aborts_instead_of_clamping_an_over_limit_recovery():
    tracker = LocalPathTracker()
    assert tracker.set_plan(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        (0.0, 0.0, 0.0), 1.0)
    command = tracker.update((0.0, 0.5, 0.0))
    assert not command.valid
    assert command.drive == 0.0 and command.wheel == 0


def test_tracker_keeps_lidar_ownership_until_final_heading_matches_csv():
    tracker = LocalPathTracker()
    assert tracker.set_plan(
        ((0.0, 0.0, 0.0), (1.0, 0.0, math.radians(8.0))),
        (0.0, 0.0, math.radians(20.0)), 1.0)
    endpoint = (math.cos(math.radians(20.0)),
                math.sin(math.radians(20.0)))
    assert not tracker.update(
        (*endpoint, math.radians(39.0))).complete
    assert tracker.update(
        (*endpoint, math.radians(28.0))).complete


def test_curved_csv_detour_uses_forward_index_and_exact_pose_heading_rejoin():
    route_path = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
    route = load_csv_only_route_case(
        route_path, route_path.with_suffix(".metadata.yaml"), "AAAA")
    start_index = next(
        index for index, point in enumerate(route)
        if point.mode == 5 and point.point_index == 237)
    start = route[start_index]
    plan = plan_route_detour(
        (start.x, start.y, start.yaw), route, start_index, 0.2,
        obstacles=((2.5, 0.0),), maximum_replans=120)
    assert plan.valid and plan.route_index > start_index
    assert route[plan.route_index].segment_id == start.segment_id
    assert plan.max_steering_deg < HARD_STEERING_LIMIT_DEG
    assert plan.replan_reason == "FRENET_OFFSET_SHORTEST"
    assert math.hypot(plan.points[-1][0], plan.points[-1][1]) < 8.0
    local_x, local_y, local_yaw = plan.points[-1]
    cosine, sine = math.cos(start.yaw), math.sin(start.yaw)
    pose = (
        start.x+cosine*local_x-sine*local_y,
        start.y+sine*local_x+cosine*local_y,
        start.yaw+local_yaw,
    )
    goal = route[plan.route_index]
    assert math.hypot(pose[0]-goal.x, pose[1]-goal.y) <= 1.0e-9
    assert abs(math.atan2(
        math.sin(pose[2]-goal.yaw), math.cos(pose[2]-goal.yaw))) <= 1.0e-9
    selected = bounded_rejoin(
        route_rejoin_candidates(
            route, start.segment_id, start_index, pose, forward_window=120),
        start.segment_id, start_index, start.direction,
        forward_window=120, max_distance_m=0.25, max_heading_deg=10.0)
    assert selected is not None and selected.index == plan.route_index


def test_route_detour_never_selects_past_or_other_segment_rejoin():
    route_path = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
    route = load_csv_only_route_case(
        route_path, route_path.with_suffix(".metadata.yaml"), "AAAA")
    start_index = next(
        index for index, point in enumerate(route)
        if point.mode == 5 and point.point_index == 337)
    start = route[start_index]
    plan = plan_route_detour(
        (start.x, start.y, start.yaw), route, start_index, -0.2,
        obstacles=((1.8, 0.0),), maximum_replans=120)
    assert plan.valid
    assert plan.route_index > start_index
    assert route[plan.route_index].segment_id == start.segment_id


def test_route_detour_is_minimal_lateral_offset_and_rejects_curb_overlap():
    class Point:
        def __init__(self, x):
            self.x, self.y, self.yaw = x, 0.0, 0.0
            self.segment_id, self.direction = "M5", 1

    route = tuple(Point(index/10.0) for index in range(121))
    obstacle = ((3.0, -0.02), (3.0, 0.02))
    open_plan = plan_route_detour(
        (0.0, 0.0, 0.0), route, 0, 0.0, obstacles=obstacle,
        maximum_replans=200)
    assert open_plan.valid
    assert open_plan.points[-1][0] < 8.0
    assert min(point[1] for point in open_plan.points) < -0.55
    assert open_plan.replan_reason == "FRENET_OFFSET_SHORTEST"

    narrow = plan_route_detour(
        (0.0, 0.0, 0.0), route, 0, 0.0, obstacles=obstacle,
        left_boundary_m=1.0, right_boundary_m=-1.0,
        maximum_replans=200)
    assert not narrow.valid
    assert "CURB_BOUNDARY" in narrow.replan_reason
