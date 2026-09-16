from pathlib import Path

import pytest
import yaml

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
CSV = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
METADATA = CSV.with_suffix(".metadata.yaml")


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
    assert selector.evaluate(1.2).branch == "B"
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


def test_behavior_stop_sources():
    for name in ("mission_stop", "camera_stop", "branch_stop", "lidar_stop"):
        assert behavior(**{name: True}).stop
    allowed = behavior(candidate_drive=2.0, candidate_wheel=12)
    assert allowed.drive == 2.0 and allowed.wheel == 12 and not allowed.stop


def test_csv_camera_lidar_arbiter_uses_csv_only_branch_gate():
    launch = (ROOT/"src/depth_hybrid_slam/launch/" /
              "prehardware_csv_camera_lidar.launch.py").read_text()
    assert '("/depth_slam/route/branch_stop",' in launch
    assert '"/depth_slam/csv_only/branch_stop")' in launch


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
    arbiter = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
               "command_arbiter_node.py").read_text()
    lidar = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
             "lidar_perception_node.py").read_text()
    rejoin = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
              "lidar_rejoin_validator_node.py").read_text()
    assert '"/cmd_drive"' not in follower
    assert '"/depth_slam/lidar/candidate_valid"' in follower
    assert 'if not self.external_maneuver_active' in follower
    assert '"EXTERNAL_MANEUVER_ACTIVE"' in follower
    assert 'Float32, "/cmd_drive"' in arbiter
    assert 'Int32, "/cmd_wheel"' in arbiter
    assert '"/depth_slam/camera/candidate_valid"' in arbiter
    assert '"/depth_slam/camera/hold"' in arbiter
    assert '"/depth_slam/path_owner"' in arbiter
    assert 'create_publisher(Float32, "/cmd_drive"' not in lidar
    assert 'create_publisher(Int32, "/cmd_wheel"' not in lidar
    assert '"prehardware_test_only"' not in lidar
    assert 'Int32, "/cmd_wheel", self._fallback_steering' not in lidar
    assert '"/depth_slam/lidar/rear_hard_emergency"' in lidar
    manager = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
               "maneuver_manager_node.py").read_text()
    assert 'front_hard or (self.rear_available and self.rear_hard)' in manager
    assert '"/depth_slam/lidar/rear_scan_available"' in manager
    assert 'parking_reverse_phase(' in manager
    assert 'valid = decision.owner == "LIDAR"' in manager
    assert 'command = f"{choice}:{branch}"' in (
        ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
        "maneuver_manager_node.py").read_text()
    assert 'self.case_choices[prefix]' in (
        ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
        "maneuver_manager_node.py").read_text()
    assert 'ComputePathThroughPoses' in (
        ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
        "maneuver_manager_node.py").read_text()
    manager = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
               "maneuver_manager_node.py").read_text()
    assert 'self.parking_plans = {7: None, 10: None}' in manager
    assert 'self.parking_plans[self.mode] = self.nav2_plan' in manager
    assert 'self.nav2_pending' in manager
    assert 'prefix+"_"+runtime.state, "CSV", False' in manager
    assert 'prefix+"_SLAM_FREEZE"' in manager
    assert '"/depth_slam/lidar/csv_rejoin_valid"' in rejoin
    assert '"/depth_slam/lidar/planned_rejoin_index"' in rejoin
    assert '"PLANNED_EXACT_ENDPOINT"' in rejoin
    assert '"/depth_slam/lidar/planned_rejoin_index"' in manager
    assert not (ROOT/"src/camera_navigation/camera_navigation"/
                "camera_command_selector_node.py").exists()


def test_traffic_gate_is_driven_by_csv_stop_waypoint_state():
    mission = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
               "mission_node.py").read_text()
    assert '"/depth_slam/route/stop_waypoint_state"' in mission
    assert '"MINIMUM_3S_HOLD", "WAIT_TRAFFIC_RELEASE"' in mission
    assert '"/camera_traffic_light"' in mission
    assert '"/camera/traffic_light_rgb/state"' in mission
    assert 'self.value.traffic_red_override = "R" in direct' in mission


def test_hil_uses_real_sensors_and_test_only_odom():
    launch = (ROOT/"src/depth_hybrid_slam/launch"/
              "prehardware_csv_camera_lidar.launch.py").read_text()
    assert '"d456_production.launch.py"' in launch
    assert '"dual_rplidar.launch.py"' in launch
    route_loop = (ROOT/"src/depth_hybrid_slam/launch"/
                  "prehardware_csv_only_closed_loop.launch.py").read_text()
    assert 'executable="test_odom_publisher"' in route_loop
    assert '"test_only_acknowledged": True' in route_loop
    for forbidden in ("synthetic_scan", "virtual_vehicle", "virtual_mcu"):
        assert forbidden not in launch+route_loop


def test_production_lidar_drivers_have_one_explicit_owner_per_scan_topic():
    launch = (ROOT/"src/depth_hybrid_slam/launch"/
              "dual_rplidar.launch.py").read_text()
    production = (ROOT/"src/depth_hybrid_slam/launch"/
                  "competition_csv_camera_lidar.launch.py").read_text()
    assert 'remappings=[("scan", "/front/scan")]' in launch
    assert 'remappings=[("scan", "/rear/scan")]' in launch
    assert 'default_value="false"' in launch
    assert '"dual_rplidar.launch.py"' in production


def test_d456_production_tf_reads_the_single_commissioned_mount_source():
    mount_path = ROOT/"src/camera_bringup/config/camera_mount.yaml"
    mount = yaml.safe_load(mount_path.read_text())["/**"]["ros__parameters"][
        "camera_mount"]
    assert mount == {
        "configured": True,
        "position_x_m": 0.32,
        "position_y_m": 0.0,
        "height_z_m": 0.85,
        "reference_roll_deg": 0.0,
        "reference_pitch_deg": -5.0,
        "reference_yaw_deg": 0.0,
    }
    launch = (ROOT/"src/camera_bringup/launch"/
              "d456_production.launch.py").read_text()
    assert '(bringup/"config"/"camera_mount.yaml")' in launch
    assert "_mount_tf(bringup)" in launch


def test_production_mode11_selector_has_post_window_transport_grace():
    launch = (ROOT/"src/depth_hybrid_slam/launch"/
              "competition_csv_camera_lidar.launch.py").read_text()
    assert '"mode_11_wait_s": 5.5' in launch
