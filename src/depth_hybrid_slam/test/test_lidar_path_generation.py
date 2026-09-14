import math

import pytest

from depth_hybrid_slam.lidar_local_planner import (
    HARD_STEERING_LIMIT_DEG, audit_path, path_collision_free, plan_detour,
    plan_parking, steering_geometry)
from depth_hybrid_slam.lidar_mission_core import ParkingManeuver
from depth_hybrid_slam.lidar_path_tracker import LocalPathTracker
from depth_hybrid_slam.vehicle_kinematics import AckermannPathEvaluator


def _assert_generated_feasible(plan, planner_limit=21.0):
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
    curvature, radius = steering_geometry(0.73, 21.0)
    assert curvature == pytest.approx(math.tan(math.radians(21.0))/0.73)
    assert radius == pytest.approx(1.9017150172)
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
    assert expected[0] <= plan.max_steering_deg <= expected[1]
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
    if requested > 21.0:
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
    assert plan.max_steering_deg == pytest.approx(21.0)


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
