from pathlib import Path

import pytest

from depth_hybrid_slam.mcu_source_adapter_core import adapt_slam_command
from depth_hybrid_slam.prehardware_core import (
    BranchSelector,
    LidarPolicy,
    RouteLocalConnector,
    StopWaypointMachine,
    select_behavior,
)
from depth_hybrid_slam.route_io import (
    A_EXCLUSIVE_SEGMENTS,
    B_EXCLUSIVE_SEGMENTS,
    load_segmented_route,
)


ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT/"routes/network/route_network_segmented.csv"
METADATA = ROOT/"routes/network/route_network_segmented.metadata.yaml"


def behavior(validator="VALID_ROAD_ONLY", **overrides):
    values = dict(
        candidate_drive=2.0, candidate_wheel=10, candidate_stop=False,
        candidate_fresh=True, localization_state="TRACKING",
        localization_fresh=True, connector_state="READY",
        connector_fresh=True, validator_state=validator,
        validator_fresh=True)
    values.update(overrides)
    return select_behavior(**values)


def test_branch_default_b_timeout_and_mode11_wait():
    selector = BranchSelector(command_timeout_s=1.0, mode_11_wait_s=3.0)
    assert selector.evaluate(0.0).branch == "A"
    selector.set_command("B", 0.1)
    assert selector.evaluate(0.2).branch == "B"
    assert selector.evaluate(1.2).branch == "A"
    selector.set_mode(11, 2.0)
    assert selector.evaluate(4.9).stop
    assert selector.evaluate(5.0).state == "MODE11_TIMEOUT_DEFAULT_A"
    selector.set_mode(1, 5.1)
    selector.set_mode(11, 6.0)
    selector.set_command("B", 6.1)
    selected = selector.evaluate(6.2)
    assert selected.branch == "B" and not selected.stop


def test_actual_a_and_b_routes_change_exclusive_segments():
    route_a = load_segmented_route(CSV, METADATA, branch="A")
    route_b = load_segmented_route(CSV, METADATA, branch="B")
    names_a = {point.segment_id for point in route_a.points}
    names_b = {point.segment_id for point in route_b.points}
    assert set(A_EXCLUSIVE_SEGMENTS) <= names_a
    assert not names_a.intersection(B_EXCLUSIVE_SEGMENTS)
    assert set(B_EXCLUSIVE_SEGMENTS) <= names_b
    assert not names_b.intersection(A_EXCLUSIVE_SEGMENTS)
    assert route_a.points != route_b.points


def test_local_connector_uses_progress_and_rejects_backtrack():
    route = tuple((index*0.1, 0.0, 0.0) for index in range(80))
    connector = RouteLocalConnector(0.5, 4.0)
    first = connector.extract(route, (1.0, 0.0, 0.0), 10)
    assert first.valid and first.state == "READY"
    assert first.points[0][0] == pytest.approx(0.5)
    assert first.points[-1][0] == pytest.approx(4.0)
    assert connector.extract(route, (1.2, 0.0, 0.0), 12).valid
    assert connector.extract(route, (1.1, 0.0, 0.0), 11).state == \
        "INDEX_BACKTRACK"


def test_local_connector_transforms_map_points_to_base_link():
    route = tuple((0.0, index*0.1, 1.5707963267948966)
                  for index in range(60))
    result = RouteLocalConnector().extract(
        route, (0.0, 0.0, 1.5707963267948966), 0)
    assert result.valid
    assert result.points[0][0] == pytest.approx(0.5)
    assert result.points[0][1] == pytest.approx(0.0, abs=1e-9)


def test_behavior_road_only_and_lane_valid_allow_csv():
    assert not behavior("VALID_ROAD_ONLY").stop
    assert not behavior("VALID_ROAD_AND_LANE").stop


def test_behavior_degraded_lane_caps_stage_at_one():
    decision = behavior("DEGRADED_LANE_UNCERTAIN", candidate_drive=3.0)
    assert not decision.stop and decision.drive == 1.0


def test_behavior_camera_unavailable_or_geometry_invalid_falls_back_to_csv():
    for state in ("CAMERA_UNAVAILABLE", "INVALID_GEOMETRY"):
        decision = behavior(state)
        assert not decision.stop and not decision.road_verified
    assert not behavior(validator_fresh=False).stop


def test_behavior_outside_road_and_stale_path_stop():
    assert behavior("INVALID_OUTSIDE_ROAD").stop
    assert behavior("INVALID_INSUFFICIENT_ROAD").stop
    assert behavior(connector_state="STALE_LOCALIZATION").stop
    assert behavior(localization_state="LOST").stop


def test_behavior_stop_sources_and_mcu_adapter_end_to_end():
    for name in ("mission_stop", "camera_stop", "branch_stop", "lidar_stop"):
        assert behavior(**{name: True}).stop
    allowed = behavior(candidate_drive=2.0, candidate_wheel=12)
    adapted = adapt_slam_command(
        allowed.drive, allowed.wheel, allowed.stop, (0.0, 0.0, 0.0), 0.5)
    assert adapted.drive == 2.0 and adapted.wheel == -12 and not adapted.stop


def test_stop_waypoint_holds_three_seconds_and_waits_for_red():
    machine = StopWaypointMachine(3.0)
    assert machine.update("COMMON:7", True, False, 10.0).stop
    assert machine.update("", False, False, 12.99).stop
    assert machine.update("", False, True, 13.0).state == \
        "WAIT_TRAFFIC_RELEASE"
    released = machine.update("", False, False, 14.0)
    assert not released.stop and released.state == "RELEASED"
    assert not machine.update("COMMON:7", True, False, 15.0).stop


def test_mode5_left_right_both_and_rejoin():
    left_policy = LidarPolicy(clear_samples=2)
    left = left_policy.evaluate(5, ((1.0, 0.3),), (), True, False)
    assert left.avoidance_active and left.wheel_internal < 0 and not left.stop
    right = LidarPolicy().evaluate(5, ((1.0, -0.3),), (), True, False)
    assert right.wheel_internal > 0 and not right.stop
    both = LidarPolicy().evaluate(
        5, ((1.0, 0.3), (1.0, -0.3)), (), True, False)
    assert both.stop and both.state == "MODE5_BOTH_BLOCKED"
    assert left_policy.evaluate(5, (), (), True, False).avoidance_active
    clear = left_policy.evaluate(5, (), (), True, False)
    assert not clear.avoidance_active and clear.state == "MODE5_CSV_REJOIN"


@pytest.mark.parametrize("mode", (7, 10))
def test_parking_modes_reverse_then_rear_stop(mode):
    policy = LidarPolicy(rear_stop_distance_m=0.5)
    clear = policy.evaluate(mode, (), ((1.0, 0.0),), True, True)
    assert clear.drive == -1.0 and not clear.stop and clear.parking_active
    blocked = policy.evaluate(mode, (), ((0.4, 0.0),), True, True)
    assert blocked.stop and blocked.state == "PARKING_REAR_STOP"


def test_mode9_three_meter_sudden_obstacle_hard_stop():
    policy = LidarPolicy()
    assert not policy.evaluate(9, (), (), True, False).stop
    stopped = policy.evaluate(9, ((2.5, 0.1),), (), True, False)
    assert stopped.stop and stopped.state == "MODE9_SUDDEN_OBSTACLE"


def test_required_production_topic_owners_are_explicit():
    follower = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
                "route_follower_node.py").read_text()
    behavior_source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
                       "behavior_selector_node.py").read_text()
    lidar = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
             "lidar_source_node.py").read_text()
    camera = (ROOT/"src/camera_navigation/camera_navigation"/
              "camera_command_selector_node.py").read_text()
    assert '"/slam_drive"' not in follower
    assert 'Float32, "/slam_drive"' in behavior_source
    for token in ('"/lidar_drive"', '"/lidar_wheel"', '"/lidar_stop"'):
        assert token in lidar
    for token in ('"/camera_drive"', '"/camera_wheel"', '"/camera_stop"'):
        assert token in camera
    combined = follower+behavior_source+lidar+camera
    assert '"/mcu/cmd_drive"' not in combined
    assert '"/mcu/cmd_wheel"' not in combined
