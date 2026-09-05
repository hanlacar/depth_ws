#!/usr/bin/env python3
"""Fail-safe Pure Pursuit producer for the internal camera candidate contract."""

from dataclasses import dataclass
from enum import Enum
import json
import math
import time

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .pure_pursuit import steering_angle_deg


class DriveCommand(float, Enum):
    """Discrete vehicle drive contract (these values are not m/s)."""

    STOP = 0.0
    SLOW = 1.0
    CRUISE = 2.0
    FAST = 3.0


ALLOWED_DRIVE_COMMANDS = frozenset(command.value for command in DriveCommand)
DRIVE_TOPIC = "/camera/candidate/path/drive"
WHEEL_TOPIC = "/camera/candidate/path/wheel"
DRIVE_MESSAGE_TYPE = Float32
WHEEL_MESSAGE_TYPE = Int32


@dataclass(frozen=True)
class ControllerConfig:
    wheelbase_m: float = 0.73
    lookahead_m: float = 3.0
    maximum_steering_deg: float = 22.0
    # /camera/path is REP-103 (+y left); the vehicle contract is negative left.
    steering_sign: float = -1.0
    normal_drive_command: float = DriveCommand.CRUISE.value
    path_timeout_sec: float = 0.15
    source_stamp_timeout_sec: float = 0.15
    minimum_path_points: int = 3
    expected_frame_id: str = "base_link"
    minimum_confidence: float = 0.0
    # --- Steering-based longitudinal rule (additive; gates above untouched) ---
    # Drive is CRUISE (normal_drive_command) on gentle steering, and drops to
    # SLOW once the commanded |steering| reaches steering_slowdown_threshold_deg.
    # No other speed logic: FAST is never emitted, no curvature planning.
    slow_drive_command: float = DriveCommand.SLOW.value
    steering_slowdown_threshold_deg: float = 10.0

    def validate(self):
        finite = (
            self.wheelbase_m, self.lookahead_m, self.maximum_steering_deg,
            self.steering_sign, self.normal_drive_command,
            self.path_timeout_sec, self.source_stamp_timeout_sec,
            self.minimum_confidence,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("controller parameters must be finite")
        if self.wheelbase_m <= 0.0 or self.lookahead_m <= 0.0:
            raise ValueError("wheelbase and lookahead must be positive")
        if not 0.0 < self.maximum_steering_deg <= 22.0:
            raise ValueError("maximum_steering_deg must be in (0, 22]")
        if self.normal_drive_command not in ALLOWED_DRIVE_COMMANDS:
            raise ValueError("normal_drive_command must be one of 0.0, 1.0, 2.0, 3.0")
        if self.slow_drive_command not in ALLOWED_DRIVE_COMMANDS:
            raise ValueError("slow_drive_command must be one of 0.0, 1.0, 2.0, 3.0")
        if not math.isfinite(self.steering_slowdown_threshold_deg) or \
                self.steering_slowdown_threshold_deg <= 0.0:
            raise ValueError("steering_slowdown_threshold_deg must be positive")
        if self.path_timeout_sec != 0.15 or self.source_stamp_timeout_sec != 0.15:
            raise ValueError("camera safety timeouts are fixed at 0.15 seconds")
        if self.minimum_path_points < 3:
            raise ValueError("minimum_path_points must be at least 3")
        if not self.expected_frame_id:
            raise ValueError("expected_frame_id must not be empty")


@dataclass(frozen=True)
class MetricStatus:
    stamp_ns: int
    path_valid: bool
    calibration_valid: bool
    confidence: float


@dataclass(frozen=True)
class PathSample:
    stamp_ns: int
    frame_id: str
    points: tuple
    received_at: float
    finite: bool


@dataclass(frozen=True)
class CameraCommand:
    drive: float
    wheel: int
    steering_deg: float
    reason: str
    valid: bool


class CameraController:
    """ROS-independent latest-only controller and safety state machine."""

    def __init__(self, config=ControllerConfig()):
        config.validate()
        self.config = config
        self.calibration_valid = False
        self.image_path_valid = False
        self.metric_path_valid = False
        self.metric_status = None
        self.path = None
        self.last_path_stamp_ns = None

    def ingest_path(self, stamp_ns, frame_id, points, received_at):
        stamp_ns = int(stamp_ns)
        if self.last_path_stamp_ns is not None and stamp_ns <= self.last_path_stamp_ns:
            return False
        converted = []
        finite = True
        for point in points:
            try:
                x_m, y_m = float(point[0]), float(point[1])
            except (IndexError, TypeError, ValueError):
                finite = False
                continue
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                finite = False
            converted.append((x_m, y_m))
        self.path = PathSample(
            stamp_ns, str(frame_id), tuple(converted),
            float(received_at), finite)
        self.last_path_stamp_ns = stamp_ns
        return True

    def ingest_metric_status(self, status):
        if not isinstance(status, MetricStatus):
            raise TypeError("status must be MetricStatus")
        if self.metric_status is None or status.stamp_ns >= self.metric_status.stamp_ns:
            self.metric_status = status

    @staticmethod
    def stop(reason):
        return CameraCommand(DriveCommand.STOP.value, 0, 0.0, str(reason), False)

    def step(self, now, ros_now_ns):
        """Decide drive and wheel together for one controller tick."""
        try:
            return self._step(float(now), int(ros_now_ns))
        except Exception:  # No internal control error may preserve propulsion.
            return self.stop("internal_exception")

    def _step(self, now, ros_now_ns):
        if not self.calibration_valid:
            return self.stop("calibration_invalid")
        if not self.image_path_valid:
            return self.stop("image_path_invalid")
        if not self.metric_path_valid:
            return self.stop("metric_path_invalid")
        if self.path is None:
            return self.stop("path_missing")
        if self.metric_status is None:
            return self.stop("metric_status_missing")
        if self.metric_status.stamp_ns != self.path.stamp_ns:
            return self.stop("path_status_stamp_mismatch")
        if not self.metric_status.calibration_valid:
            return self.stop("metric_status_calibration_invalid")
        if not self.metric_status.path_valid:
            return self.stop("metric_status_path_invalid")
        if not math.isfinite(self.metric_status.confidence):
            return self.stop("nonfinite_confidence")
        if self.metric_status.confidence <= self.config.minimum_confidence:
            return self.stop("low_confidence")
        if self.path.frame_id != self.config.expected_frame_id:
            return self.stop("path_frame_mismatch")
        if not self.path.finite:
            return self.stop("nonfinite_path")
        if len(self.path.points) < self.config.minimum_path_points:
            return self.stop("insufficient_path_points")
        forward_points = tuple(point for point in self.path.points if point[0] > 0.0)
        if len(forward_points) < self.config.minimum_path_points:
            return self.stop("insufficient_forward_points")
        if not math.isfinite(now) or not math.isfinite(self.path.received_at):
            return self.stop("nonfinite_time")
        if now - self.path.received_at > self.config.path_timeout_sec:
            return self.stop("path_stale")
        if self.path.stamp_ns <= 0:
            return self.stop("source_stamp_missing")
        source_age = (ros_now_ns - self.path.stamp_ns) * 1.0e-9
        if not math.isfinite(source_age) or source_age < 0.0:
            return self.stop("source_stamp_invalid")
        if source_age > self.config.source_stamp_timeout_sec:
            return self.stop("source_stamp_stale")

        # Pure Pursuit -> degrees -> physical sign -> clamp -> integer.
        steering = steering_angle_deg(
            forward_points, 0.0, self.config.wheelbase_m,
            self.config.lookahead_m, self.config.lookahead_m, 0.0,
            180.0)
        steering *= self.config.steering_sign
        if not math.isfinite(steering):
            return self.stop("steering_nonfinite")
        steering = max(-self.config.maximum_steering_deg,
                       min(self.config.maximum_steering_deg, steering))
        wheel = max(-22, min(22, int(round(steering))))

        # Longitudinal rule: CRUISE on gentle steering, SLOW once the
        # commanded |steering| reaches the threshold. This is the only speed
        # logic; FAST is never produced.
        if abs(steering) >= self.config.steering_slowdown_threshold_deg:
            drive = float(self.config.slow_drive_command)
        else:
            drive = float(self.config.normal_drive_command)
        if drive not in ALLOWED_DRIVE_COMMANDS:
            return self.stop("invalid_drive_policy")
        return CameraCommand(drive, wheel, steering, "ok", True)


class CameraPathController(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__(
            "camera_path_controller_node",
            parameter_overrides=parameter_overrides or [])
        defaults = {
            "wheelbase_m": 0.73, "lookahead_m": 3.0,
            "maximum_steering_deg": 22.0, "steering_sign": -1.0,
            "normal_drive_command": 2.0, "path_timeout_sec": 0.15,
            "source_stamp_timeout_sec": 0.15, "minimum_path_points": 3,
            "expected_frame_id": "base_link", "minimum_confidence": 0.0,
            "control_rate_hz": 20.0,
            "slow_drive_command": 1.0,
            "steering_slowdown_threshold_deg": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        config = ControllerConfig(**{
            name: self.get_parameter(name).value
            for name in ControllerConfig.__dataclass_fields__
        })
        self.controller = CameraController(config)
        rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive")

        self.drive_pub = self.create_publisher(DRIVE_MESSAGE_TYPE, DRIVE_TOPIC, 10)
        self.wheel_pub = self.create_publisher(WHEEL_MESSAGE_TYPE, WHEEL_TOPIC, 10)
        self.diagnostics_pub = self.create_publisher(
            String, "/camera/controller_diagnostics", 10)
        self.create_subscription(Path, "/camera/path", self.on_path, 10)
        self.create_subscription(
            Bool, "/camera/calibration_valid",
            lambda msg: setattr(self.controller, "calibration_valid", bool(msg.data)), 10)
        self.create_subscription(
            Bool, "/camera/image_path_valid",
            lambda msg: setattr(self.controller, "image_path_valid", bool(msg.data)), 10)
        self.create_subscription(
            Bool, "/camera/metric_path_valid",
            lambda msg: setattr(self.controller, "metric_path_valid", bool(msg.data)), 10)
        self.create_subscription(
            String, "/camera/metric_path_status", self.on_metric_status, 10)
        self.last_command = CameraController.stop("controller_starting")
        self.create_timer(1.0 / rate_hz, self.control)
        self.get_logger().info(
            f"camera command controller ready at {rate_hz:.1f} Hz; "
            f"outputs={DRIVE_TOPIC},{WHEEL_TOPIC}; MCU relay disabled")

    @staticmethod
    def stamp_ns(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def on_path(self, msg):
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.controller.ingest_path(
            self.stamp_ns(msg.header), msg.header.frame_id, points, time.monotonic())

    def on_metric_status(self, msg):
        try:
            payload = json.loads(msg.data)
            self.controller.ingest_metric_status(MetricStatus(
                stamp_ns=int(payload["stamp_ns"]),
                path_valid=bool(payload["path_valid"]),
                calibration_valid=bool(payload["calibration_valid"]),
                confidence=float(payload["confidence"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.controller.metric_status = None
            self.get_logger().warning(f"rejected metric path status: {exc}")

    def control(self):
        try:
            command = self.controller.step(
                time.monotonic(), self.get_clock().now().nanoseconds)
        except Exception as exc:  # Defensive node boundary; publish both stops.
            command = CameraController.stop("node_internal_exception")
            self.get_logger().error(f"controller exception: {exc}")
        self.last_command = command
        # Both messages come from the same immutable control result.
        self.drive_pub.publish(Float32(data=float(command.drive)))
        self.wheel_pub.publish(Int32(data=int(command.wheel)))
        self.diagnostics_pub.publish(String(data=json.dumps({
            "valid": command.valid, "reason": command.reason,
            "drive": command.drive, "wheel": command.wheel,
            "steering_deg": command.steering_deg,
        }, separators=(",", ":"))))

    def destroy_node(self):
        if rclpy.ok():
            self.drive_pub.publish(Float32(data=DriveCommand.STOP.value))
            self.wheel_pub.publish(Int32(data=0))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPathController()
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
