#!/usr/bin/env python3
"""Transform valid base_link camera paths into a stitched odom reference."""

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .metric_path_quality import MetricPathQualityConfig
from .reference_path_stitching import (
    PlanarTransform, ReferencePathAdapterCore, StitchConfig)


def yaw_from_quaternion(x, y, z, w):
    values = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm <= 1.0e-9:
        raise ValueError("quaternion must be finite and nonzero")
    x, y, z, w = values/norm
    return math.atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))


class CameraReferencePathAdapterNode(Node):
    def __init__(self):
        super().__init__("camera_reference_path_adapter_node")
        defaults = {
            "metric_path_topic": "/camera/bev/path",
            "reference_path_topic": "/avoidance/route/reference_path",
            "mode_topic": "/mcu/current_mode",
            "allowed_modes": ["5"],
            "base_frame_id": "base_link",
            "odom_frame_id": "odom",
            "tf_lookup_timeout_sec": 0.02,
            "pending_retry_rate_hz": 10.0,
            "pending_path_timeout_sec": 0.50,
            "metric_minimum_points": 3,
            "metric_minimum_spacing_m": 0.05,
            "metric_maximum_point_jump_m": 4.0,
            "metric_maximum_reverse_step_m": 0.25,
            "metric_minimum_path_length_m": 1.0,
            "stitch_overlap_distance_m": 0.75,
            "stitch_max_position_error_m": 1.5,
            "stitch_max_heading_error_deg": 30.0,
            "stitch_min_overlap_points": 3,
            "reference_path_keep_behind_m": 5.0,
            "reference_path_max_total_m": 50.0,
            "reference_path_target_forward_m": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        metric_config = MetricPathQualityConfig(
            minimum_points=int(self.get_parameter("metric_minimum_points").value),
            minimum_spacing_m=float(
                self.get_parameter("metric_minimum_spacing_m").value),
            maximum_point_jump_m=float(
                self.get_parameter("metric_maximum_point_jump_m").value),
            maximum_reverse_step_m=float(
                self.get_parameter("metric_maximum_reverse_step_m").value),
            minimum_path_length_m=float(
                self.get_parameter("metric_minimum_path_length_m").value),
        )
        stitch_config = StitchConfig(
            overlap_distance_threshold_m=float(
                self.get_parameter("stitch_overlap_distance_m").value),
            stitch_max_position_error_m=float(
                self.get_parameter("stitch_max_position_error_m").value),
            stitch_max_heading_error_deg=float(
                self.get_parameter("stitch_max_heading_error_deg").value),
            stitch_min_overlap_points=int(
                self.get_parameter("stitch_min_overlap_points").value),
            reference_path_keep_behind_m=float(
                self.get_parameter("reference_path_keep_behind_m").value),
            reference_path_max_total_m=float(
                self.get_parameter("reference_path_max_total_m").value),
            reference_path_target_forward_m=float(
                self.get_parameter("reference_path_target_forward_m").value),
            minimum_spacing_m=metric_config.minimum_spacing_m,
        )
        self.core = ReferencePathAdapterCore(metric_config, stitch_config)
        self.base_frame = str(self.get_parameter("base_frame_id").value)
        self.odom_frame = str(self.get_parameter("odom_frame_id").value)
        self.tf_timeout = float(
            self.get_parameter("tf_lookup_timeout_sec").value)
        self.pending_timeout = float(
            self.get_parameter("pending_path_timeout_sec").value)
        retry_rate = float(self.get_parameter("pending_retry_rate_hz").value)
        if (not self.base_frame or not self.odom_frame or
                not math.isfinite(self.tf_timeout) or self.tf_timeout < 0.0 or
                not math.isfinite(self.pending_timeout) or
                self.pending_timeout <= 0.0 or
                not math.isfinite(retry_rate) or retry_rate <= 0.0):
            raise ValueError("adapter frame/time parameters are invalid")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.metric_topic = str(self.get_parameter("metric_path_topic").value)
        self.reference_topic = str(
            self.get_parameter("reference_path_topic").value)
        self.mode_topic = str(self.get_parameter("mode_topic").value)
        self.allowed_modes = tuple(str(value) for value in
                                   self.get_parameter("allowed_modes").value)
        if (not self.metric_topic or not self.reference_topic or
                not self.mode_topic or not self.allowed_modes or
                any(not value for value in self.allowed_modes)):
            raise ValueError("adapter topic/mode parameters are invalid")
        self.reference_pub = self.create_publisher(
            Path, self.reference_topic, 10)
        self.diagnostics_pub = self.create_publisher(
            String, "/camera/reference_path_adapter_diagnostics", 10)
        self.create_subscription(
            Path, self.metric_topic,
            self.on_metric_path, 10)
        self.create_subscription(
            String, self.mode_topic,
            self.on_mode, 10)
        self.create_timer(1.0/retry_rate, self.retry_pending)

        self.current_mode = None
        self.mode_allowed = False
        self.pending_path = None
        self.pending_received_time = None
        self.adapter_state = "inactive_mode"
        self.tf_available = False
        self.last_reject_reason = "mode_unavailable"
        self.last_metric = None
        self.last_stitch = None
        self.last_metric_stamp_ns = None
        self.last_published_stamp_ns = None
        self.get_logger().info(
            f"reference adapter ready: {self.base_frame} -> {self.odom_frame}; "
            f"mode topic={self.mode_topic}, allowed={list(self.allowed_modes)}; "
            "fail-closed until an allowed mode is received")

    def clear_mode_scoped_state(self):
        """Clear anything that could leak a path across mission transitions."""
        self.core.reset()
        self.pending_path = None
        self.pending_received_time = None
        self.tf_available = False
        self.last_metric = None
        self.last_stitch = None
        self.last_metric_stamp_ns = None
        self.last_published_stamp_ns = None

    def on_mode(self, message):
        new_mode = str(message.data)
        new_allowed = new_mode in self.allowed_modes
        changed = (new_mode != self.current_mode or
                   new_allowed != self.mode_allowed)
        self.current_mode = new_mode
        self.mode_allowed = new_allowed
        if changed:
            self.clear_mode_scoped_state()
        if new_allowed:
            if changed:
                self.adapter_state = "waiting_for_metric_path"
                self.last_reject_reason = "none"
        else:
            # Repeated disallowed messages also enforce fail-closed state.
            self.pending_path = None
            self.pending_received_time = None
            self.adapter_state = "inactive_mode"
            self.last_reject_reason = "mode_not_allowed"
        self.publish_diagnostics()

    @staticmethod
    def stamp_ns(header):
        return (int(header.stamp.sec)*1_000_000_000 +
                int(header.stamp.nanosec))

    @staticmethod
    def points_from_path(message):
        return np.asarray([
            (pose.pose.position.x, pose.pose.position.y)
            for pose in message.poses
        ], dtype=float).reshape((-1, 2))

    @staticmethod
    def headings_from_path(message):
        return np.asarray([
            yaw_from_quaternion(
                pose.pose.orientation.x, pose.pose.orientation.y,
                pose.pose.orientation.z, pose.pose.orientation.w)
            for pose in message.poses
        ], dtype=float)

    def on_metric_path(self, message):
        if not self.mode_allowed:
            self.pending_path = None
            self.pending_received_time = None
            self.adapter_state = "inactive_mode"
            self.last_reject_reason = (
                "mode_unavailable" if self.current_mode is None else
                "mode_not_allowed")
            self.publish_diagnostics()
            return
        if message.header.frame_id != self.base_frame:
            self.adapter_state = "metric_path_invalid"
            self.last_reject_reason = "metric_frame_mismatch"
            self.pending_path = None
            self.pending_received_time = None
            self.publish_diagnostics()
            return
        stamp_ns = self.stamp_ns(message.header)
        if stamp_ns <= 0:
            self.adapter_state = "metric_path_invalid"
            self.last_reject_reason = "metric_stamp_missing"
            self.pending_path = None
            self.pending_received_time = None
            self.publish_diagnostics()
            return
        if (self.last_metric_stamp_ns is not None and
                stamp_ns < self.last_metric_stamp_ns):
            self.adapter_state = "metric_path_invalid"
            self.last_reject_reason = "metric_stamp_regression"
            self.pending_path = None
            self.pending_received_time = None
            self.publish_diagnostics()
            return
        self.last_metric_stamp_ns = stamp_ns
        self.pending_path = message
        self.pending_received_time = self.get_clock().now()
        self.process_pending()

    def retry_pending(self):
        if not self.mode_allowed:
            self.pending_path = None
            self.pending_received_time = None
            self.adapter_state = "inactive_mode"
            self.last_reject_reason = (
                "mode_unavailable" if self.current_mode is None else
                "mode_not_allowed")
            self.publish_diagnostics()
            return
        if self.pending_path is not None:
            age = (self.get_clock().now()-self.pending_received_time).nanoseconds/1e9
            if age > self.pending_timeout:
                self.adapter_state = "waiting_for_tf"
                self.last_reject_reason = "pending_path_timeout"
                self.pending_path = None
                self.pending_received_time = None
                self.publish_diagnostics()
                return
            self.process_pending()
        else:
            self.publish_diagnostics()

    def process_pending(self):
        if not self.mode_allowed:
            self.pending_path = None
            self.pending_received_time = None
            self.adapter_state = "inactive_mode"
            self.last_reject_reason = (
                "mode_unavailable" if self.current_mode is None else
                "mode_not_allowed")
            self.publish_diagnostics()
            return
        message = self.pending_path
        if message is None:
            return
        base_points = self.points_from_path(message)
        try:
            base_headings = self.headings_from_path(message)
        except ValueError:
            self.adapter_state = "metric_path_invalid"
            self.last_reject_reason = "invalid_pose_orientation"
            self.pending_path = None
            self.pending_received_time = None
            self.publish_diagnostics()
            return
        waiting = self.core.process(base_points, None, base_headings)
        self.last_metric = waiting.metric
        if waiting.state == "metric_path_invalid":
            self.adapter_state = waiting.state
            self.last_reject_reason = waiting.reason
            self.pending_path = None
            self.pending_received_time = None
            self.publish_diagnostics()
            return
        try:
            transform_message = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=self.tf_timeout))
            translation = transform_message.transform.translation
            rotation = transform_message.transform.rotation
            transform = PlanarTransform(
                float(translation.x), float(translation.y),
                yaw_from_quaternion(
                    rotation.x, rotation.y, rotation.z, rotation.w))
        except (TransformException, ValueError) as error:
            self.tf_available = False
            self.adapter_state = "waiting_for_tf"
            self.last_reject_reason = f"tf_unavailable:{type(error).__name__}"
            self.publish_diagnostics()
            return

        self.tf_available = True
        result = self.core.process(base_points, transform, base_headings)
        self.last_metric = result.metric
        self.last_stitch = result.stitch
        self.adapter_state = result.state
        self.last_reject_reason = result.reason
        self.pending_path = None
        self.pending_received_time = None
        if result.accepted and result.stitch is not None:
            self.publish_reference(
                result.stitch.points, result.stitch.headings_rad,
                message.header.stamp)
        self.publish_diagnostics()

    def publish_reference(self, points, headings, stamp):
        if not self.mode_allowed:
            self.adapter_state = "inactive_mode"
            self.last_reject_reason = (
                "mode_unavailable" if self.current_mode is None else
                "mode_not_allowed")
            return False
        message = Path()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            yaw = float(headings[index])
            pose.pose.orientation.z = math.sin(0.5*yaw)
            pose.pose.orientation.w = math.cos(0.5*yaw)
            message.poses.append(pose)
        self.reference_pub.publish(message)
        self.last_published_stamp_ns = self.stamp_ns(message.header)
        return True

    def publish_diagnostics(self):
        metric = self.last_metric
        stitch = self.last_stitch
        now = self.get_clock().now()
        pending_age = None
        if self.pending_received_time is not None:
            pending_age = max(
                0.0, (now-self.pending_received_time).nanoseconds/1.0e9)
        source_age = None
        if self.last_metric_stamp_ns is not None:
            source_age = max(
                0.0, (now.nanoseconds-self.last_metric_stamp_ns)/1.0e9)
        data = {
            "current_mode": self.current_mode,
            "allowed_modes": list(self.allowed_modes),
            "mode_allowed": self.mode_allowed,
            "metric_path_valid": bool(metric.valid) if metric else False,
            "metric_path_points": int(metric.point_count) if metric else 0,
            "metric_forward_length_m": (
                metric.forward_usable_length_m if metric else 0.0),
            "metric_path_length_m": metric.path_length_m if metric else 0.0,
            "metric_minimum_x_m": metric.minimum_x_m if metric else None,
            "metric_maximum_x_m": metric.maximum_x_m if metric else None,
            "metric_curvature_per_m": (
                metric.maximum_curvature_per_m if metric else 0.0),
            "tf_available": self.tf_available,
            "adapter_state": self.adapter_state,
            "stitched_points": int(len(self.core.stitcher.points)),
            "stitched_length_m": stitch.stitched_length_m if stitch else 0.0,
            "forward_usable_length_m": (
                stitch.forward_usable_length_m if stitch else 0.0),
            "target_forward_length_m": (
                self.core.stitcher.config.reference_path_target_forward_m),
            "last_reject_reason": self.last_reject_reason,
            "metric_path_topic": self.metric_topic,
            "reference_path_topic": self.reference_topic,
            "last_source_stamp_ns": self.last_metric_stamp_ns,
            "last_published_stamp_ns": self.last_published_stamp_ns,
            "pending_path_age_sec": pending_age,
            "last_source_age_sec": source_age,
            "mode_topic": self.mode_topic,
        }
        self.diagnostics_pub.publish(String(
            data=json.dumps(data, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = CameraReferencePathAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
