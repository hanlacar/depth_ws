import csv
import hashlib
from pathlib import Path

import yaml

from depth_hybrid_slam.command_arbiter_core import (
    CommandCandidate, arbitrate)
from depth_hybrid_slam.localization_core import align_odom_pose_to_route_entry
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.route_follower_core import RouteFollower, select_mode_range
from depth_hybrid_slam.route_io import load_segmented_route
from depth_hybrid_slam.traffic_gate import (
    IntersectionProgress, IntersectionTrafficGate)


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT/"src/depth_hybrid_slam"
LAUNCH = PACKAGE/"launch/depth_real_vehicle.launch.py"
SPLIT_LAUNCH = PACKAGE/"launch/depth_csv_camera_lidar.launch.py"
CAMERA_LAUNCH = ROOT/"src/camera_navigation/launch/d456_traffic_light.launch.py"
ROUTE = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
METADATA = ROUTE.with_suffix(".metadata.yaml")


def test_real_launch_contains_only_measured_odom_and_one_front_lidar():
    source = LAUNCH.read_text(encoding="utf-8").lower()
    for forbidden in (
            "controlled_synthetic_odom", "synthetic_odom", "fake_odom",
            "virtual_odom", "virtual_vehicle", "mock_odom", "gazebo",
            "test_odom_publisher", "rear_rplidar_node"):
        assert forbidden not in source
    assert 'package="t870_mcu_simple", executable="bridge"' not in source
    assert '"launch_rear_lidar_driver": launchconfiguration(' in source
    assert 'declarelaunchargument("use_rear_lidar", default_value="false")' in source
    assert '"--frame-id", "map", "--child-frame-id", "odom"' not in source
    assert '"hybrid_localization.launch.py"' in source
    assert '"use_vehicle_odom": "true"' in source
    assert '"cuvslam_only.launch.py"' not in source
    assert 'executable="start_validation"' not in source
    assert '"initial_branch": branch' in source


def test_split_sensor_route_launch_waits_for_separate_real_mcu_odom():
    source = SPLIT_LAUNCH.read_text(encoding="utf-8")
    assert '"launch_mcu_odom": "false"' not in source
    assert '"use_lidar": "true"' in source
    assert '"use_camera": "true"' in source
    assert "test_odom_publisher" not in source
    assert "synthetic_odom" not in source
    real = LAUNCH.read_text(encoding="utf-8")
    assert 'LaunchConfiguration("launch_mcu_odom")' not in real
    assert 'executable="csv_only_network_visualizer"' in real
    assert 'executable="route_local_path"' in real


def test_production_rear_lidar_is_optional_and_explicitly_enabled():
    wrapper = SPLIT_LAUNCH.read_text(encoding="utf-8")
    real = LAUNCH.read_text(encoding="utf-8")
    for source in (wrapper, real):
        assert 'DeclareLaunchArgument("use_rear_lidar", default_value="false")' in source
        assert 'DeclareLaunchArgument("rear_serial_port", default_value="/dev/ttyUSB1")' in source
    assert '"use_rear_lidar": LaunchConfiguration("use_rear_lidar")' in wrapper
    assert '"rear_serial_port": LaunchConfiguration("rear_serial_port")' in wrapper
    assert '"rear_lidar_enabled": rear_lidar_enabled' in real


def test_production_launch_exposes_vslam_and_mode_range_controls():
    real = LAUNCH.read_text(encoding="utf-8")
    wrapper = SPLIT_LAUNCH.read_text(encoding="utf-8")
    for source in (real, wrapper):
        assert 'DeclareLaunchArgument("enable_vslam", default_value="true")' in source
        assert 'DeclareLaunchArgument("start_mode", default_value="1")' in source
        assert 'DeclareLaunchArgument("end_mode", default_value="11")' in source
    assert '"enable_vslam": LaunchConfiguration("enable_vslam")' in wrapper
    assert '"start_mode": LaunchConfiguration("start_mode")' in wrapper
    assert '"end_mode": LaunchConfiguration("end_mode")' in wrapper
    assert 'validate_mode_range(' in real
    assert 'if vslam_enabled:' in real
    assert real.count('share/"launch"/"hybrid_localization.launch.py"') == 1
    assert '"enable_vslam": vslam_enabled' in real
    assert '"start_mode": start_mode' in real
    assert '"end_mode": end_mode' in real
    assert '"initial_branch": branch' in real
    assert '"VSLAM" if vslam_enabled else "ODOM_ONLY"' in real
    assert 'if not vslam_enabled:' in real
    assert ('follower_overrides["localization_stability_s"] = 0.0'
            in real)


def test_odom_only_skips_only_vslam_stability_delay():
    real = LAUNCH.read_text(encoding="utf-8")
    config = yaml.safe_load(
        (PACKAGE/"config/vehicle_navigation.yaml").read_text(
            encoding="utf-8"))
    assert config["route_follower"]["ros__parameters"][
        "localization_stability_s"] == 2.0
    override = real.index(
        'follower_overrides["localization_stability_s"] = 0.0')
    odom_only_branch = real.rindex("if not vslam_enabled:", 0, override)
    follower_build = real.index("follower =", override)
    assert odom_only_branch < override < follower_build


def test_odom_only_disables_vslam_gate_subscription_and_runtime_watchdog():
    localization = (PACKAGE/"depth_hybrid_slam"/
                    "odom_localization_node.py").read_text(encoding="utf-8")
    monitor = (PACKAGE/"depth_hybrid_slam"/
               "runtime_monitor_node.py").read_text(encoding="utf-8")
    assert 'self.declare_parameter("enable_vslam", True)' in localization
    assert 'if self.vslam_enabled else None' in localization
    assert 'if self.vslam_enabled:' in localization
    assert '[LOCALIZATION] ODOM_ONLY - VSLAM DISABLED' in localization
    assert '[LOCALIZATION] ODOM+VSLAM' in localization
    assert 'decision.use_vslam else 0.6' in localization
    assert 'if not source or child != expected_child:' in localization
    assert 'if not self.vslam_enabled or self.centralize_vslam_map_tf else' in \
        localization
    assert 'map_edge = odom_only_map_edge(' in localization
    assert 'align_odom_pose_to_route_entry(' in localization
    assert ('transform.header.frame_id, transform.child_frame_id = map_edge'
            in localization)
    assert 'transform.transform.translation.x = tx' in localization
    assert 'transform.transform.rotation = quaternion_from_yaw(tf_yaw)' in \
        localization
    assert 'self.odom_map_transform.sendTransform(transform)' in localization
    assert '"vslam": ("DISABLED" if not self.vslam_enabled else' in localization
    assert 'if key == "vslam" and not self.vslam_enabled:' in monitor
    assert 'return "DISABLED"' in monitor


def test_real_camera_launch_keeps_mode2_imu_contract_alive():
    source = CAMERA_LAUNCH.read_text(encoding="utf-8")
    assert '"enable_depth": LaunchConfiguration("enable_vslam")' in source
    assert '"enable_vslam": LaunchConfiguration("enable_vslam")' in source
    assert 'imu/"launch"/"imu_manager.launch.py"' in source


def test_mode5_roi_and_csv_collision_planning_distance_contract():
    config = yaml.safe_load(
        (PACKAGE/"config/lidar_autonomy.yaml").read_text(encoding="utf-8"))
    perception = config["depth_lidar_perception"]["ros__parameters"]
    maneuver = config["depth_maneuver_manager"]["ros__parameters"]
    assert perception["mode5_range_m"] == 1.5
    assert perception["csv_path_timeout_s"] == 0.5
    assert maneuver["obstacle_confirmation_s"] == 2.0
    assert maneuver["minimum_planning_lidar_distance_m"] == 1.0


def test_rviz_and_markers_make_a_b_and_active_route_unambiguous():
    rviz = (PACKAGE/"config/prehardware_csv_front_lidar.rviz").read_text()
    visualizer = (PACKAGE/"depth_hybrid_slam"/
                  "csv_only_network_visualizer_node.py").read_text()
    assert "ACTIVE CSV ROUTE (YELLOW)" in rviz
    assert "A=CYAN SOLID / B=MAGENTA DASHED" in rviz
    assert 'suffix = " [ACTIVE]"' in visualizer
    assert 'suffix = " [ACTIVE]" if selected == branch else " [AVAILABLE]"' in visualizer
    assert 'name.startswith("START_")' in visualizer


def test_only_arbiter_publishes_final_drive_and_wheel():
    arbiter = (PACKAGE/"depth_hybrid_slam/command_arbiter_node.py").read_text()
    bridge = (ROOT/"src/t870_mcu_simple/t870_mcu_simple/bridge_node.py").read_text()
    assert 'create_publisher(Float32, "/cmd_drive"' in arbiter
    assert 'create_publisher(Int32, "/cmd_wheel"' in arbiter
    assert 'create_subscription(Float32, "/cmd_drive"' in bridge
    assert 'create_subscription(Int32, "/cmd_wheel"' in bridge
    assert 'create_publisher(Float32, "/cmd_drive"' not in bridge
    assert 'create_publisher(Int32, "/cmd_wheel"' not in bridge


def test_stop_reason_topics_and_transition_only_terminal_logs_remain_visible():
    arbiter = (PACKAGE/"depth_hybrid_slam"/
               "command_arbiter_node.py").read_text(encoding="utf-8")
    safety = (PACKAGE/"depth_hybrid_slam"/
              "safety_node.py").read_text(encoding="utf-8")
    assert '"/depth_slam/safety/state"' in arbiter
    assert '"/depth_slam/safety/stop_reason"' in arbiter
    assert '"stop_reason": stop_reason' in arbiter
    assert '"/depth_slam/path_owner"' in arbiter
    assert 'self.last_stop_reason' in arbiter
    assert 'self.get_logger().warning("[STOP] "+stop_reason)' in arbiter
    assert '"/depth_slam/safety/state"' in safety
    assert '"/depth_slam/safety/stop_reason"' in safety
    assert 'self.last_stop_reason' in safety


def test_mcu_bridge_is_the_odom_and_odom_base_tf_owner():
    bridge = (ROOT/"src/t870_mcu_simple/t870_mcu_simple/bridge_node.py").read_text()
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'self.odom_pub = self.create_publisher(Odometry, "/odom"' in bridge
    assert "TransformBroadcaster(self)" in bridge
    localization = (PACKAGE/"depth_hybrid_slam"/
                    "odom_localization_node.py").read_text(encoding="utf-8")
    assert "if map_edge is not None:" in localization
    assert "owns only map->odom" in localization
    assert '"odom_localization"' in launch
    assert 'executable="test_odom_publisher"' not in launch


def test_odom_only_route_origin_uses_selected_branch_and_mode_entry():
    launch = LAUNCH.read_text(encoding="utf-8")
    assert "load_segmented_route(" in launch
    assert "branch=branch" in launch
    assert "select_mode_range(" in launch
    for parameter in (
            "odom_route_entry_x_m", "odom_route_entry_y_m",
            "odom_route_entry_yaw_rad"):
        assert f'"{parameter}"' in launch


def test_real_odom_origin_produces_mode1_csv_drive_without_deviation():
    route = select_mode_range(
        load_segmented_route(ROUTE, METADATA, branch="A").points, 1, 2)
    entry = route[0]
    x, y, yaw = align_odom_pose_to_route_entry(
        0.0, 0.0, 0.0, entry.x, entry.y, entry.yaw)
    result = RouteFollower(corridor_m=1.0).compute(
        Pose2D(x, y, yaw, 1.0), route,
        allow_motion=True, global_search=True)
    assert result.cross_track_error == 0.0
    assert result.drive == 2.0
    assert not result.stop_required
    assert result.reason == "OK"


def test_active_start_a_mode2_stop_line_and_metadata_hash():
    with ROUTE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    stops = [row for row in rows if row["segment_id"] == "START_A" and
             row["mode"] == "2" and row["event"] == "STOP_LINE"]
    assert [(row["point_index"], row["event"]) for row in stops] == [
        ("173", "STOP_LINE")]
    metadata = yaml.safe_load(METADATA.read_text(encoding="utf-8"))
    assert metadata["route_sha256"] == hashlib.sha256(ROUTE.read_bytes()).hexdigest()


def test_real_rejoin_limits_are_position_025m_heading_10deg():
    config = yaml.safe_load(
        (PACKAGE/"config/lidar_autonomy.yaml").read_text(encoding="utf-8"))
    values = config["depth_lidar_rejoin_validator"]["ros__parameters"]
    assert values["max_distance_m"] == 0.25
    assert values["max_heading_deg"] == 10.0


def test_traffic_is_ignored_outside_modes_and_unknown_releases_after_3s():
    gate = IntersectionTrafficGate()
    outside = IntersectionProgress(
        True, 5, "COMMON_1", 1, 200, 20.0,
        "COMMON_1", 1, 200, 1, 20.0, 220, 25.0)
    decision = gate.evaluate(outside, True, "R", 0.0, 0.0)
    assert not decision.active and not decision.stop
    line = IntersectionProgress(
        True, 4, "COMMON_1", 1, 100, 10.0,
        "COMMON_1", 1, 100, 25, 10.0, 130, 20.0)
    assert gate.evaluate(line, True, "UNKNOWN", 0.0, 1.0).stop
    assert gate.evaluate(line, True, "UNKNOWN", 0.0, 3.999).stop
    assert not gate.evaluate(line, True, "UNKNOWN", 0.0, 4.0).stop
    config = yaml.safe_load(
        (PACKAGE/"config/mission.yaml").read_text(encoding="utf-8"))
    assert config["/**"]["ros__parameters"]["unknown_hold_s"] == 3.0


def test_emergency_wins_and_final_wheel_never_exceeds_22():
    csv_candidate = CommandCandidate(3.0, 22, True, True)
    lidar_candidate = CommandCandidate(1.0, -22, True, True)
    emergency = arbitrate(
        csv_candidate, lidar_candidate, hard_emergency=True)
    assert emergency.drive == 0.0 and emergency.wheel == 0
    assert emergency.state == "HARD_EMERGENCY_STOP"
    rejected = arbitrate(CommandCandidate(2.0, 23, True, True),
                         CommandCandidate())
    assert rejected.drive == 0.0 and rejected.wheel == 0
