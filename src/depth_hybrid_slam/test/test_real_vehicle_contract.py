import csv
import hashlib
from pathlib import Path

import yaml

from depth_hybrid_slam.command_arbiter_core import (
    CommandCandidate, arbitrate)
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
    assert '"launch_rear_lidar_driver": "false"' in source
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


def test_mcu_bridge_is_the_odom_and_odom_base_tf_owner():
    bridge = (ROOT/"src/t870_mcu_simple/t870_mcu_simple/bridge_node.py").read_text()
    launch = LAUNCH.read_text(encoding="utf-8")
    assert 'self.odom_pub = self.create_publisher(Odometry, "/odom"' in bridge
    assert "TransformBroadcaster(self)" in bridge
    assert '"odom_localization"' in launch
    assert 'executable="test_odom_publisher"' not in launch


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
