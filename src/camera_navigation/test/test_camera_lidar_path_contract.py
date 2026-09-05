from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from builtin_interfaces.msg import Time
from nav_msgs.msg import Path as PathMessage

from camera_navigation.camera_reference_path_adapter_node import (
    CameraReferencePathAdapterNode)


ROOT = Path(__file__).parents[1]


def test_adapter_uses_exact_source_stamp_without_latest_fallback():
    source = (ROOT / "camera_navigation" /
              "camera_reference_path_adapter_node.py").read_text()
    assert "Time.from_msg(message.header.stamp)" in source
    assert "lookup_transform(" in source
    assert "metric_stamp_regression" in source
    assert "Time()" not in source


def test_validation_launch_connects_direct_bev_path_only():
    source = (ROOT / "launch" /
              "camera_lidar_path_validation.launch.py").read_text()
    assert '"metric_path_topic": "/camera/bev/path"' in source
    assert '"reference_path_topic": "/avoidance/route/reference_path"' in source
    assert '"mode_topic": "/mcu/current_mode"' in source
    assert "avoidance_coordinator" not in source
    assert "lidar_safety" not in source
    assert "static_transform_publisher" not in source
    assert "fake_odom" not in source


def test_production_planner_default_remains_production():
    source = (ROOT / "launch" / "camera_bev_standalone.launch.py").read_text()
    assert 'DeclareLaunchArgument("planner_variant", default_value="production")' in source


def test_adapter_defaults_are_mode_5_avoidance_reference_only():
    adapter_launch = (ROOT / "launch" /
                      "camera_reference_path_adapter.launch.py").read_text()
    config = (ROOT / "config" /
              "camera_reference_path_adapter.yaml").read_text()
    production = (ROOT / "launch" /
                  "camera_bev_standalone.launch.py").read_text()
    for source in (adapter_launch, config):
        assert "/camera/bev/path" in source
        assert "/avoidance/route/reference_path" in source
        assert "/mcu/current_mode" in source
    assert 'allowed_modes: ["5"]' in config
    assert "camera_reference_path_adapter_node" not in production


def _callback_harness():
    return SimpleNamespace(
        base_frame="base_link",
        current_mode="5",
        mode_allowed=True,
        adapter_state="initial",
        last_reject_reason="none",
        pending_path="sentinel",
        pending_received_time="sentinel",
        last_metric_stamp_ns=None,
        stamp_ns=CameraReferencePathAdapterNode.stamp_ns,
        publish_diagnostics=Mock(),
        process_pending=Mock(),
        get_clock=Mock(),
    )


def _path(frame="base_link", sec=1, nanosec=0):
    message = PathMessage()
    message.header.frame_id = frame
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec
    return message


def test_source_frame_and_nonzero_stamp_are_mandatory():
    for message, reason in (
            (_path(frame="map"), "metric_frame_mismatch"),
            (_path(sec=0, nanosec=0), "metric_stamp_missing")):
        node = _callback_harness()
        CameraReferencePathAdapterNode.on_metric_path(node, message)
        assert node.last_reject_reason == reason
        assert node.pending_path is None
        assert node.pending_received_time is None
        node.process_pending.assert_not_called()


def test_timestamp_regression_is_rejected_before_tf_lookup():
    node = _callback_harness()
    node.last_metric_stamp_ns = 2_000_000_000
    CameraReferencePathAdapterNode.on_metric_path(node, _path(sec=1))
    assert node.last_reject_reason == "metric_stamp_regression"
    assert node.pending_path is None
    node.process_pending.assert_not_called()


def test_published_reference_preserves_source_stamp_frame_and_orientations():
    node = SimpleNamespace(
        current_mode="5", mode_allowed=True,
        odom_frame="odom", reference_pub=Mock(),
        last_published_stamp_ns=None,
        stamp_ns=CameraReferencePathAdapterNode.stamp_ns)
    stamp = Time(sec=123, nanosec=456)
    CameraReferencePathAdapterNode.publish_reference(
        node, [[1.0, 2.0], [2.0, 3.0]], [0.25, -0.5], stamp)
    message = node.reference_pub.publish.call_args.args[0]
    assert message.header.frame_id == "odom"
    assert message.header.stamp == stamp
    assert all(pose.header == message.header for pose in message.poses)
    assert message.poses[0].pose.orientation.w != 1.0
    assert message.poses[1].pose.orientation.z < 0.0


def _mode_harness(current_mode=None, mode_allowed=False):
    core = Mock()
    node = SimpleNamespace(
        current_mode=current_mode,
        mode_allowed=mode_allowed,
        allowed_modes=("5",),
        core=core,
        pending_path="old_pending",
        pending_received_time="old_time",
        tf_available=True,
        last_metric="old_metric",
        last_stitch="old_stitch",
        last_metric_stamp_ns=123,
        last_published_stamp_ns=123,
        adapter_state="active",
        last_reject_reason="none",
        publish_diagnostics=Mock(),
    )
    node.clear_mode_scoped_state = (
        lambda: CameraReferencePathAdapterNode.clear_mode_scoped_state(node))
    return node


def _set_mode(node, value):
    CameraReferencePathAdapterNode.on_mode(
        node, SimpleNamespace(data=value))


def test_no_mode_received_rejects_metric_path_fail_closed():
    node = _callback_harness()
    node.current_mode = None
    node.mode_allowed = False
    CameraReferencePathAdapterNode.on_metric_path(node, _path())
    assert node.adapter_state == "inactive_mode"
    assert node.last_reject_reason == "mode_unavailable"
    assert node.pending_path is None
    node.process_pending.assert_not_called()


def test_mode_4_does_not_process_or_publish_reference_path():
    node = _callback_harness()
    node.current_mode = "4"
    node.mode_allowed = False
    CameraReferencePathAdapterNode.on_metric_path(node, _path())
    assert node.last_reject_reason == "mode_not_allowed"
    node.process_pending.assert_not_called()

    publisher = Mock()
    publish_node = SimpleNamespace(
        current_mode="4", mode_allowed=False, adapter_state="active",
        last_reject_reason="none", odom_frame="odom",
        reference_pub=publisher, last_published_stamp_ns=None,
        stamp_ns=CameraReferencePathAdapterNode.stamp_ns)
    published = CameraReferencePathAdapterNode.publish_reference(
        publish_node, [[1.0, 0.0]], [0.0], Time(sec=1))
    assert published is False
    publisher.publish.assert_not_called()


def test_mode_5_enables_path_processing_and_reference_publication():
    node = _mode_harness()
    _set_mode(node, "5")
    assert node.mode_allowed is True
    assert node.adapter_state == "waiting_for_metric_path"
    assert node.core.reset.call_count == 1

    callback = _callback_harness()
    CameraReferencePathAdapterNode.on_metric_path(callback, _path())
    callback.process_pending.assert_called_once()


def test_mode_4_to_5_clears_pending_and_stitched_state():
    node = _mode_harness(current_mode="4", mode_allowed=False)
    _set_mode(node, "5")
    assert node.mode_allowed is True
    assert node.pending_path is None
    assert node.last_metric_stamp_ns is None
    node.core.reset.assert_called_once()


def test_mode_5_to_4_stops_output_and_clears_state_immediately():
    node = _mode_harness(current_mode="5", mode_allowed=True)
    _set_mode(node, "4")
    assert node.mode_allowed is False
    assert node.adapter_state == "inactive_mode"
    assert node.last_reject_reason == "mode_not_allowed"
    assert node.pending_path is None
    assert node.last_stitch is None
    node.core.reset.assert_called_once()


def test_mode_5_reentry_starts_a_fresh_segment():
    node = _mode_harness(current_mode="5", mode_allowed=True)
    _set_mode(node, "4")
    _set_mode(node, "5")
    assert node.core.reset.call_count == 2
    assert node.last_published_stamp_ns is None
    assert node.adapter_state == "waiting_for_metric_path"


def test_continuous_mode_5_does_not_reset_stitched_path():
    node = _mode_harness(current_mode="5", mode_allowed=True)
    node.pending_path = None
    node.pending_received_time = None
    _set_mode(node, "5")
    assert node.core.reset.call_count == 0
    assert node.adapter_state == "active"


def test_adapter_has_no_control_or_lidar_command_side_effects():
    source = (ROOT / "camera_navigation" /
              "camera_reference_path_adapter_node.py").read_text()
    for forbidden in (
            "/camera_drive", "/camera_wheel", "/avoidance/active",
            "/mcu_drive", "/mcu_wheel", "LaserScan"):
        assert forbidden not in source
