#!/usr/bin/env python3
"""Thin internal drive-candidate producer for the direct BEV stack.

Why this node exists: this codebase's only existing /camera_drive publisher
is camera_path_controller_node, which is hard-gated on
/camera/calibration_valid, /camera/image_path_valid, /camera/metric_path_valid
and /camera/metric_path_status (see CameraController._step in
camera_path_controller_node.py) -- four topics only the *legacy*
camera_metric_path_node/camera_image_path_node stack publishes
(camera_bev_control.launch.py). The direct BEV stack
(direct_bev_planner_node/direct_bev_controller_node/bev_wheel_selector_node)
never populates them, so reusing camera_path_controller_node as-is would sit
at permanent STOP with reason=calibration_invalid regardless of the YOLO
model or road conditions. Rather than touch that node or fake its four
legacy inputs (option B), this node derives a drive stage straight from the
direct BEV stack's own path-validity signal (option A).

Contract mirrors direct_bev_controller_node's own /camera/bev/state
subscription exactly: JSON String, "state" in {VALID, DEGRADED, INVALID,
HOLD} (direct_bev_core.py). INVALID also covers calibration failure, since
direct_bev_planner_node emits state=INVALID via _handle_invalid() whenever
_calibration_ready() is false -- so no separate calibration check is needed
here; it is surfaced in the diagnostics topic below for visibility only.

It never owns either final camera command topic.

Publishes on its own fixed-rate timer (drive_rate_hz) regardless of upstream
message timing, so the candidate stays at a steady hz even while stopped: a
gap in publish rate would look identical to a dead node from the outside,
and "publishing STOP steadily" is a very different failure signature from
"not publishing at all" during validation.
"""
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

def drive_command_for_state(state, valid_command, degraded_command,
                            stop_command):
    """Map path validity to the explicit conservative speed policy."""
    if state == "VALID":
        return float(valid_command)
    if state == "DEGRADED":
        return float(degraded_command)
    return float(stop_command)


class DriveSafetyPolicy:
    """State-stamped, fail-closed drive policy with no command memory."""
    def __init__(self, stale_timeout, valid_command, degraded_command,
                 stop_command):
        self.stale_timeout = float(stale_timeout)
        self.valid_command = float(valid_command)
        self.degraded_command = float(degraded_command)
        self.stop_command = float(stop_command)
        self.last_state = None
        self.last_state_wall = None

    def update(self, payload, now):
        self.last_state = dict(payload)
        self.last_state_wall = float(now)
        return self.evaluate(now)

    def evaluate(self, now):
        stale = (self.last_state_wall is None or
                 float(now)-self.last_state_wall > self.stale_timeout)
        state = (None if stale or self.last_state is None else
                 self.last_state.get("state"))
        drive = drive_command_for_state(
            state, self.valid_command, self.degraded_command,
            self.stop_command)
        return {"drive": drive, "state": state, "stale": stale}


class DirectBevDriveNode(Node):
    def __init__(self):
        super().__init__("direct_bev_drive_node")
        self.declare_parameter("drive_rate_hz", 30.0)
        self.declare_parameter("state_stale_timeout_sec", 0.5)
        # Values match camera_path_controller_node.DriveCommand exactly
        # (STOP=0.0, SLOW=1.0, CRUISE=2.0, FAST=3.0) so downstream MCU
        # tooling built against that enum needs no changes.
        self.declare_parameter("go_drive_command", 2.0)
        self.declare_parameter("degraded_drive_command", 1.0)
        self.declare_parameter("stop_drive_command", 0.0)

        rate_hz = float(self.get_parameter("drive_rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("drive_rate_hz must be > 0")
        self.stale_timeout = float(self.get_parameter("state_stale_timeout_sec").value)
        self.go_value = float(self.get_parameter("go_drive_command").value)
        self.degraded_value = float(
            self.get_parameter("degraded_drive_command").value)
        self.stop_value = float(self.get_parameter("stop_drive_command").value)

        self.policy = DriveSafetyPolicy(
            self.stale_timeout, self.go_value, self.degraded_value,
            self.stop_value)
        self.command_sequence = 0

        self.drive_pub = self.create_publisher(
            Float32, "/camera/candidate/bev/drive", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/bev_drive_diagnostics", 10)
        self.create_subscription(String, "/camera/bev/state", self._on_state, 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"direct_bev_drive_node ready at {rate_hz:.1f} Hz -> candidate drive "
            f"(valid={self.go_value}, degraded={self.degraded_value}, "
            f"stop={self.stop_value}); reads /camera/bev/state "
            "only; final command ownership remains in the selector")

    def _on_state(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"state": "INVALID", "reasons": ["STATE_DECODE_ERROR"]}
        decision = self.policy.update(payload, time.monotonic())
        # Do not retain the preceding go command until the next timer tick.
        # INVALID (including malformed state) publishes STOP in this callback.
        if payload.get("state") not in ("VALID", "DEGRADED"):
            self._publish(decision["drive"], decision["state"],
                          decision["stale"],
                          "state_callback")

    def _tick(self):
        decision = self.policy.evaluate(time.monotonic())
        self._publish(decision["drive"], decision["state"],
                      decision["stale"], "timer")

    def _publish(self, drive, state, stale, trigger):
        go = float(drive) != self.stop_value
        self.command_sequence += 1
        self.drive_pub.publish(Float32(data=drive))
        self.diag_pub.publish(String(data=json.dumps({
            "drive": float(drive), "go": go, "state": state, "stale": stale,
            "trigger": trigger, "command_sequence": self.command_sequence,
            "publish_monotonic_ns": time.monotonic_ns(),
            "state_received_monotonic_ns": (
                None if self.policy.last_state_wall is None else
                int(self.policy.last_state_wall*1e9)),
            "state_stamp_ns": (None if self.policy.last_state is None else
                               self.policy.last_state.get("stamp_ns")),
            "source_stamp_ns": (None if self.policy.last_state is None else
                                self.policy.last_state.get("source_stamp_ns")),
            "calibration_state": (
                None if self.policy.last_state is None else
                self.policy.last_state.get("calibration_state")),
            "reasons": (None if self.policy.last_state is None else
                       self.policy.last_state.get("reasons")),
        }, separators=(",", ":"))))


def main():
    rclpy.init()
    node = DirectBevDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
