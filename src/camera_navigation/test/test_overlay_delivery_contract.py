"""Regression tests for RQT delivery, synchronization and subscriptions."""

import ast
from pathlib import Path

import numpy as np
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image

from camera_navigation.direct_bev_planner_node import render_diagnostic_overlay
from camera_navigation.timestamp_sync import (
    TimestampedMessageCache, subscription_transition,
)


WORKSPACE_SRC = Path(__file__).resolve().parents[2]


def image(stamp_ns, value=0):
    message = Image()
    message.header.stamp.sec = stamp_ns//1_000_000_000
    message.header.stamp.nanosec = stamp_ns%1_000_000_000
    message.header.frame_id = "camera_color_optical_frame"
    message.height, message.width = 8, 12
    message.encoding = "bgr8"
    message.step = 36
    message.data = np.full((8, 12, 3), value, np.uint8).tobytes()
    return message


def test_all_create_subscription_calls_have_four_required_arguments():
    failures = []
    for path in WORKSPACE_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr == "create_subscription"):
                keywords = {item.arg for item in node.keywords}
                complete_keywords = {
                    "msg_type", "topic", "callback", "qos_profile"
                }.issubset(keywords)
                if len(node.args) < 4 and not complete_keywords:
                    failures.append(f"{path}:{node.lineno}")
    assert failures == []


def test_direct_bev_semantic_subscription_has_callback_and_qos():
    path = WORKSPACE_SRC/"camera_navigation"/"camera_navigation"/\
        "direct_bev_planner_node.py"
    source = path.read_text(encoding="utf-8")
    assert 'self.get_parameter("semantic_topic").value,\n            self._on_semantic, latest_sensor_qos)' in source


def test_rqt_image_publishers_are_reliable_depth_one():
    paths = [
        WORKSPACE_SRC/"camera_navigation"/"camera_navigation"/
        "camera_image_path_node.py",
        WORKSPACE_SRC/"camera_navigation"/"camera_navigation"/
        "direct_bev_planner_node.py",
        WORKSPACE_SRC/"camera_yolo_inference"/"camera_yolo_inference"/
        "camera_yolo_inference_node.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "depth=1" in source
        assert "reliability=QoSReliabilityPolicy.RELIABLE" in source
    assert QoSReliabilityPolicy.RELIABLE.value == 1


def test_exact_nearest_outside_and_stale_timestamp_matching():
    cache = TimestampedMessageCache(4)
    exact = image(1_000_000_000, 1)
    near = image(1_040_000_000, 2)
    cache.add(exact, received_wall=10.0)
    cache.add(near, received_wall=10.0)
    match = cache.nearest_match(1_000_000_000, .05, .5, now_wall=10.1)
    assert match.message is exact and match.exact and match.delta_ns == 0
    match = cache.nearest_match(1_050_000_000, .05, .5, now_wall=10.1)
    assert match.message is near and not match.exact
    assert cache.nearest_match(1_100_000_000, .05, .5, now_wall=10.1) is None
    assert cache.nearest_match(1_040_000_000, .05, .5, now_wall=10.6) is None


def test_semantic_before_rgb_can_match_after_arrival():
    cache = TimestampedMessageCache(2)
    assert cache.nearest_match(123, .08) is None
    rgb = image(123)
    cache.add(rgb)
    assert cache.nearest_match(123, .08).message is rgb


def test_late_subscriber_disconnect_and_reconnect_transitions():
    assert subscription_transition(1, False) == "CREATE"
    assert subscription_transition(1, True) == "NONE"
    assert subscription_transition(0, True) == "DESTROY"
    assert subscription_transition(1, False) == "CREATE"


def test_bev_waiting_diagnostic_is_visible_on_black_and_rgb():
    black = render_diagnostic_overlay(None, (80, 120), "WAITING_CAMERA_INFO")
    rgb = render_diagnostic_overlay(image(1, 200), (80, 120), "WAITING_IMU")
    assert black.shape == (80, 120, 3) and np.any(black)
    assert rgb.shape == (80, 120, 3) and np.any(rgb)
