"""Pure synthetic tests for the low-speed rosbag analyzer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


TOOL = Path(__file__).parents[1]/"tools"/"analyze_a6_low_speed_bag.py"
SPEC = importlib.util.spec_from_file_location("a6_bag_analysis", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def empty_series():
    return {topic: [] for topic in MODULE.TOPICS.values()}


def test_marker_segments_require_ordered_start_and_end():
    markers = [(0.0, "V1_RUN1_START"), (5.0, "V1_RUN1_END"),
               (6.0, "V2_RUN1_END")]
    assert MODULE.marker_segments(markers) == [{
        "name": "V1_RUN1", "stage": "V1", "start": 0.0, "end": 5.0}]


def test_causal_path_error_is_zero_for_exact_straight_tracking():
    data = empty_series()
    data[MODULE.TOPICS["odom"]] = [(float(i), {
        "x": float(i), "y": 0.0, "yaw_deg": 0.0,
        "speed": 1.0, "yaw_rate": 0.0}) for i in range(7)]
    path = np.array([[0.2, 0.0], [1.0, 0.0], [2.0, 0.0]])
    data[MODULE.TOPICS["path"]] = [(float(i), path) for i in range(5)]
    data[MODULE.TOPICS["controller"]] = [(float(i), {
        "target_point": [1.0, 0.0]}) for i in range(5)]
    errors = MODULE.causal_path_errors(data)
    assert errors
    assert max(abs(row["cross_track_error_m"]) for row in errors) == 0.0
    assert max(abs(row["heading_error_deg"]) for row in errors) == 0.0


def test_analyzer_checks_owner_sign_limits_and_invalid_stop():
    data = empty_series()
    for topic, values in {
        "camera_wheel": [(0.0, -2), (0.1, -2), (0.2, 0)],
        "mcu_wheel": [(0.0, -2), (0.1, -2), (0.2, 0)],
        "camera_drive": [(0.0, 1.0), (0.2, 0.0)],
        "mcu_drive": [(0.0, 1.0), (0.2, 0.0)],
        "steer": [(0.0, 0.0), (0.1, -2.0), (0.2, 0.0)],
        "wheel_owner": [(0.0, "camera"), (0.1, "camera")],
        "drive_owner": [(0.0, "camera")],
        "safety": [(0.0, "OK")],
        "controller": [(0.0, {"target_point": [2.0, 0.2]})],
        "state": [(0.0, {"state": "VALID", "reasons": []}),
                  (0.1, {"state": "INVALID", "reasons": ["INPUT_TIMEOUT"]})],
    }.items():
        data[MODULE.TOPICS[topic]] = values
    result, _, _, _ = MODULE.analyze(data, {}, 0.52)
    assert result["wheel_owner_camera_percent"] == 100.0
    assert result["camera_to_mcu_wheel_error_deg"]["max_abs"] == 0.0
    assert result["path_to_camera_wheel_sign_errors"] == 0
    assert result["camera_wheel_over_22"] == 0
    assert result["invalid_transitions"] == 1
    assert result["timeout_samples"] == 1
    assert result["invalid_stop_responses"][0]["drive_zero_delay_sec"] == 0.1
    assert result["invalid_stop_responses"][0]["wheel_zero_delay_sec"] == 0.1


def test_steering_response_delay_uses_ninety_percent_threshold():
    command = [(0.0, 0), (1.0, 10)]
    feedback = [(0.9, 0.0), (1.0, 0.0), (1.1, 4.0),
                (1.2, 9.1), (1.3, 10.0)]
    assert MODULE.steering_response_delays(command, feedback) == pytest.approx([0.2])


def test_rosbag_reader_deserializes_real_ros_messages(tmp_path):
    import rosbag2_py
    from nav_msgs.msg import Odometry, Path as PathMessage
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Int32, String

    bag = tmp_path/"synthetic_bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    definitions = (
        (MODULE.TOPICS["odom"], "nav_msgs/msg/Odometry"),
        (MODULE.TOPICS["path"], "nav_msgs/msg/Path"),
        (MODULE.TOPICS["controller"], "std_msgs/msg/String"),
        (MODULE.TOPICS["camera_wheel"], "std_msgs/msg/Int32"),
    )
    for name, message_type in definitions:
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=0, name=name, type=message_type, serialization_format="cdr"))
    writer.write(MODULE.TOPICS["odom"], serialize_message(Odometry()), 1_000_000_000)
    writer.write(MODULE.TOPICS["path"], serialize_message(PathMessage()), 1_000_000_001)
    writer.write(MODULE.TOPICS["controller"], serialize_message(String(
        data='{"target_point":[1.0,0.0]}')), 1_000_000_002)
    writer.write(MODULE.TOPICS["camera_wheel"], serialize_message(Int32(data=0)),
                 1_000_000_003)
    del writer
    raw, types, origin = MODULE.read_bag(bag)
    assert types[MODULE.TOPICS["odom"]] == "nav_msgs/msg/Odometry"
    assert origin == pytest.approx(1.0)
    assert raw[MODULE.TOPICS["controller"]][0][1]["target_point"] == [1.0, 0.0]
    assert raw[MODULE.TOPICS["camera_wheel"]][0][1] == 0
