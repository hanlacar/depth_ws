import math

from depth_hybrid_slam.models import Pose2D, RoutePoint
from depth_hybrid_slam.rejoin_core import (
    dubins_path, GridMap, RejoinPlanner, RejoinStateMachine)
from depth_hybrid_slam.route_follower_core import RouteFollower
import pytest


def route(offset=0.0, direction=1):
    return [RoutePoint(index, index*0.5, offset, 0.0, direction)
            for index in range(41)]


def grid(value=0):
    return GridMap(500, 500, 0.1, -15.0, -15.0, 0.0,
                   tuple([value]*(500*500)))


def test_commissioned_geometry_and_minimum_radius():
    follower = RouteFollower()
    assert follower.wheelbase == pytest.approx(0.73)
    assert follower.max_steering == pytest.approx(22.0)
    assert follower.minimum_turning_radius == pytest.approx(1.8068134, abs=0.001)


def test_segment_projection_returns_lateral_error_not_point_distance():
    follower = RouteFollower()
    result = follower.compute(Pose2D(0.5, 0.2, 0.0, 0.0), route(),
                              allow_motion=True)
    assert result.cross_track_error == pytest.approx(0.2)


def test_crossing_and_parallel_return_route_selects_heading_compatible_branch():
    points = [RoutePoint(0, -2, 0, 0), RoutePoint(1, 2, 0, 0),
              RoutePoint(2, 2, .15, math.pi), RoutePoint(3, -2, .15, math.pi)]
    east = RouteFollower().compute(Pose2D(0, .1, 0, 0), points,
                                   allow_motion=True, global_search=True)
    west = RouteFollower().compute(Pose2D(0, .1, math.pi, 0), points,
                                   allow_motion=True, global_search=True)
    assert east.nearest_index == 0
    assert west.nearest_index == 2


def test_progress_index_cannot_jump_backwards():
    follower = RouteFollower(max_index_backtrack=2, search_behind_points=100)
    follower.last_index = 30
    follower.compute(Pose2D(2.0, 0.0, 0.0, 0.0), route(), allow_motion=True)
    assert follower.last_index >= 28


@pytest.mark.parametrize("offset", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("heading_deg", [0.0, 30.0, 60.0, 90.0])
def test_free_space_lateral_rejoin(offset, heading_deg):
    planner = RejoinPlanner(candidate_heading_deg=100.0)
    result = planner.plan(Pose2D(0, offset, math.radians(heading_deg), 0),
                          route(), 2, grid())
    assert result.feasible and result.collision_free
    assert result.maximum_curvature <= 1.0/1.8068134+1.0e-6
    assert all(point.direction == 1 for point in result.path)


def test_forward_only_impossible_but_explicit_reverse_section_is_possible():
    reverse_route = route(direction=-1)
    pose = Pose2D(0, 1.0, math.pi, 0)
    planner = RejoinPlanner(candidate_heading_deg=100.0)
    assert not planner.plan(pose, reverse_route, 1, grid(), False).feasible
    result = planner.plan(pose, reverse_route, 1, grid(), True)
    assert result.feasible and result.direction == -1
    assert all(point.direction == -1 for point in result.path)


def test_obstacle_and_unknown_are_fail_closed():
    planner = RejoinPlanner(candidate_heading_deg=100.0)
    pose = Pose2D(0, 2.0, 0.0, 0)
    assert not planner.plan(pose, route(), 1, grid(100)).feasible
    assert not planner.plan(pose, route(), 1, grid(-1)).feasible
    permissive = RejoinPlanner(candidate_heading_deg=100.0, allow_unknown=True)
    unknown_grid = GridMap(500, 500, .1, -15, -15, 0,
                           tuple([-1]*(500*500)), allow_unknown=True)
    assert permissive.plan(pose, route(), 1, unknown_grid).feasible


def test_dubins_end_pose_and_steering_bound_for_opposite_heading():
    start = Pose2D(0, 0, math.pi, 0)
    goal = RoutePoint(0, 5, 0, 0)
    path = dubins_path(start, goal, 1.8068134, 0.05)
    assert path[-1].x == goal.x and path[-1].y == goal.y
    for first, second in zip(path, path[1:]):
        distance = math.hypot(second.x-first.x, second.y-first.y)
        if distance > 1.0e-6:
            curvature = abs(math.atan2(math.sin(second.yaw-first.yaw),
                                       math.cos(second.yaw-first.yaw)))/distance
            assert curvature <= 1.0/1.8068134+0.01


def test_required_rejoin_state_sequence_and_verify_hold():
    machine = RejoinStateMachine(1.0, 0.25, 10.0, 1.0)
    assert machine.observe_route(0.9) == "FOLLOW_ROUTE"
    assert machine.observe_route(1.01) == "ROUTE_DEVIATION_STOP"
    assert machine.vehicle_stopped() == "PLAN_REJOIN"
    assert machine.plan_completed(True) == "FOLLOW_REJOIN_PATH"
    assert machine.path_completed() == "VERIFY_REJOIN"
    assert machine.verify(.2, math.radians(9), 10.0) == "VERIFY_REJOIN"
    assert machine.verify(.2, math.radians(9), 11.01) == "FOLLOW_ROUTE"


def test_plan_failure_holds_no_feasible_state():
    machine = RejoinStateMachine()
    machine.observe_route(2.0)
    machine.vehicle_stopped()
    assert machine.plan_completed(False) == "REJOIN_NO_FEASIBLE_PATH"


def test_dry_run_launch_cannot_activate_real_command_channels():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    launch = (root/"launch"/"route_follower_dry_run.launch.py").read_text()
    config = (root/"config"/"vehicle_navigation.yaml").read_text()
    assert '"enable_control": False' in launch
    assert '"user_approved": False' in launch
    assert "enable_control: false" in config
    assert "dry_run: true" in config
    assert "user_approved: false" in config
