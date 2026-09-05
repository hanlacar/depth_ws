#!/usr/bin/env python3
"""Steering-only ROS controller for /camera/bev/path."""

import json
import math
import time

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Int32, String

from .direct_bev_controller import BevControllerConfig, DirectBevController
from .timestamp_sync import ExactStampPairCache


class DirectBevControllerNode(Node):
    def __init__(self):
        super().__init__("direct_bev_controller_node")
        defaults = {name: field.default for name, field in
                    BevControllerConfig.__dataclass_fields__.items()}
        defaults.update({"control_rate_hz": 20.0,
                         "fixed_output_rate_enabled": False,
                         "pair_cache_size": 32,
                         "pair_cache_timeout_sec": 0.25})
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.controller = DirectBevController(BevControllerConfig(**{
            name: self.get_parameter(name).value
            for name in BevControllerConfig.__dataclass_fields__}))
        self.fixed_output_rate_enabled = bool(
            self.get_parameter("fixed_output_rate_enabled").value)
        self.pairs = ExactStampPairCache(
            self.get_parameter("pair_cache_size").value,
            self.get_parameter("pair_cache_timeout_sec").value)
        self.last_context = {}
        self.wheel_pub = self.create_publisher(Int32, "/camera/bev/wheel", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/bev/controller_diagnostics", 10)
        self.create_subscription(Path, "/camera/bev/path", self._on_path, 10)
        self.create_subscription(String, "/camera/bev/state", self._on_state, 10)
        rate = float(self.get_parameter("control_rate_hz").value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("control rate must be positive")
        self.create_timer(1.0/rate, self._expire_unmatched)

    @staticmethod
    def _stamp(header):
        return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nanosec)

    def _on_path(self, message):
        stamp = self._stamp(message.header)
        pair = self.pairs.add_path(
            stamp, [(pose.pose.position.x, pose.pose.position.y)
                    for pose in message.poses])
        if pair is not None:
            self._control_pair(pair)

    def _on_state(self, message):
        try:
            data = json.loads(message.data)
        except (TypeError, ValueError):
            data = {"state": "INVALID", "reasons": ["STATUS_DECODE_ERROR"]}
        stamp = int(data.get("stamp_ns", -1))
        if data.get("state") == "INVALID":
            self.pairs.discard_through(stamp)
            self._publish(self.controller.neutral(), "PATH_INVALID", {
                "status": data, "state_received_monotonic_ns": time.monotonic_ns(),
                "path_stamp_ns": None, "path_point_count": 0})
            return
        pair = self.pairs.add_state(stamp, data)
        if pair is not None:
            self._control_pair(pair)

    def _control_pair(self, pair):
        status = pair.state
        now_ns = time.monotonic_ns()
        context = {
            "status": status, "path_stamp_ns": pair.stamp_ns,
            "path_point_count": len(pair.path),
            "path_received_monotonic_ns": pair.path_received_ns,
            "state_received_monotonic_ns": pair.state_received_ns,
            "pair_arrival_delta_ms": pair.arrival_delta_ns/1.0e6,
            "pair_exact": True,
        }
        if (now_ns-min(pair.path_received_ns, pair.state_received_ns) >
                int(self.controller.config.path_timeout_sec*1.0e9)):
            return self._publish(self.controller.neutral(), "PAIR_STALE", context)
        if status.get("state") not in ("VALID", "DEGRADED"):
            return self._publish(self.controller.neutral(), "PATH_INVALID", context)
        command = self.controller.command(
            pair.path, float(status.get("confidence", 1.0)),
            status.get("state") == "DEGRADED", now_ns*1.0e-9)
        self._publish(command, context=context)

    def _expire_unmatched(self):
        expired = self.pairs.expire()
        if expired:
            self._publish(self.controller.neutral(), "PAIR_TIMEOUT", {
                "status": {"state": "INVALID", "stamp_ns": max(expired),
                           "source_stamp_ns": max(expired)},
                "path_stamp_ns": None, "path_point_count": 0,
                "pair_exact": False})

    def _publish(self, command, reason=None, context=None):
        command = dict(command)
        if reason is not None:
            command["reason"] = reason
        context = context or {}
        status = context.get("status", {})
        self.last_context = context
        command.update({
            "planner_state": status.get("state"),
            "state_stamp_ns": status.get("stamp_ns"),
            "source_stamp_ns": status.get("source_stamp_ns"),
            "path_stamp_ns": context.get("path_stamp_ns"),
            "path_point_count": context.get("path_point_count", 0),
            "planner_curvature_per_m": status.get("curvature_per_m"),
            "publish_monotonic_ns": time.monotonic_ns(),
            "path_received_monotonic_ns": context.get(
                "path_received_monotonic_ns"),
            "state_received_monotonic_ns": context.get(
                "state_received_monotonic_ns"),
            "pair_arrival_delta_ms": context.get("pair_arrival_delta_ms"),
            "pair_exact": context.get("pair_exact", False),
            "pair_cache": self.pairs.stats(),
            "wheelbase_m": self.controller.config.wheelbase_m,
            "steering_gain": self.controller.config.steering_gain,
            "steering_rate_deg_per_sec":
                self.controller.config.steering_rate_deg_per_sec,
            "maximum_steering_deg":
                self.controller.config.maximum_steering_deg,
            "fractional_deadband_deg":
                self.controller.config.fractional_deadband_deg,
        })
        self.wheel_pub.publish(Int32(data=int(command.get("wheel", 0))))
        self.diag_pub.publish(String(data=json.dumps(
            command, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = DirectBevControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
