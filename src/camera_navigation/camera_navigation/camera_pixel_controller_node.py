#!/usr/bin/env python3
"""Non-BEV (pixel-space) camera driving command controller.

Subscribes to the pixel-space path (race_interfaces/ImagePath on
/camera/image_path_typed) and produces the same discrete vehicle command
contract as the metric controller:

  /camera/candidate/pixel/drive  (std_msgs/Float32)
  /camera/candidate/pixel/wheel  (std_msgs/Int32), clamped to +/-22

Steering is a PD response to the lateral pixel offset of a look-ahead row
(see pixel_lateral_control). This node needs NO camera extrinsics, NO
ground-plane calibration, and NO IMU attitude lock -- it is the geometry-free
alternative to camera_path_controller_node.

Longitudinal rule is CRUISE only for a confident lane-dominant path with a
small final wheel angle. ROAD_CENTER, sustained single-boundary, temporal, or
lower-confidence valid paths are capped at SLOW; the steering threshold can
also cap any valid path at SLOW. FAST and REVERSE are never emitted by this
normal camera-driving policy. The same fail-safe gates apply (validity, stamp
freshness, confidence, timeouts): any failure commands STOP + centered wheel.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import math
import time

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32, String

from race_interfaces.msg import ImagePath, ImagePathPoint, SemanticPathFrame

from .pixel_lateral_control import lookahead_offset_px, steering_from_offset_deg
from .semantic_path_contract import decode_binary_rle
from .stop_line_control import (
    StopLineConfig, StopLinePhase, StopLinePolicy, estimate_stop_line_distance)
from .stop_line_memory import (
    StopLineMemory, StopLineMemoryConfig, StopLineObservation,
    StopLineState, stop_line_pixel_row)
from .uphill_stop_control import UphillStopConfig, UphillStopController


class DriveCommand(float, Enum):
    REVERSE = -1.0
    STOP = 0.0
    SLOW = 1.0
    CRUISE = 2.0
    FAST = 3.0


class SafetyState(str, Enum):
    WAITING_FOR_PATH = "WAITING_FOR_PATH"
    ACTIVE = "ACTIVE"
    PATH_INVALID = "PATH_INVALID"
    PATH_TIMEOUT = "PATH_TIMEOUT"


ALLOWED_DRIVE_COMMANDS = frozenset(command.value for command in DriveCommand)
CAMERA_DRIVE_COMMANDS = frozenset((
    DriveCommand.STOP.value,
    DriveCommand.SLOW.value,
    DriveCommand.CRUISE.value,
))
DRIVE_TOPIC = "/camera/candidate/pixel/drive"
WHEEL_TOPIC = "/camera/candidate/pixel/wheel"
DEFAULT_STEERING_SLOWDOWN_THRESHOLD_DEG = 5.0
IMU_SLOPE_TOPIC = "/imu/slope"
IMU_VALID_TOPIC = "/imu/valid"


@dataclass(frozen=True)
class PixelControllerConfig:
    image_path_topic: str = "/camera/image_path_typed"
    maximum_steering_deg: float = 22.0
    # ImagePath x_px grows to the right; vehicle wheel is negative-left, so a
    # path offset to the right (positive) should steer right. Sign folds any
    # convention flip in one place, matching the metric controller's pattern.
    steering_sign: float = 1.0
    lookahead_y_ratio: float = 0.5      # 0=near/bottom, 1=far/top
    proportional_gain_deg_per_norm: float = 22.0
    derivative_gain_deg_per_norm_per_s: float = 2.0
    normal_drive_command: float = DriveCommand.CRUISE.value
    slow_drive_command: float = DriveCommand.SLOW.value
    steering_slowdown_threshold_deg: float = \
        DEFAULT_STEERING_SLOWDOWN_THRESHOLD_DEG
    minimum_confidence: float = 0.0
    minimum_path_points: int = 3
    path_timeout_sec: float = 0.15
    source_stamp_timeout_sec: float = 0.15
    max_steering_rate_deg_per_sec: float = 180.0
    low_confidence_slow_threshold: float = 0.65
    road_source_slow_ratio: float = 0.40
    single_source_slow_ratio: float = 0.50

    def validate(self):
        finite = (
            self.maximum_steering_deg, self.steering_sign,
            self.lookahead_y_ratio, self.proportional_gain_deg_per_norm,
            self.derivative_gain_deg_per_norm_per_s,
            self.normal_drive_command, self.slow_drive_command,
            self.steering_slowdown_threshold_deg, self.minimum_confidence,
            self.path_timeout_sec, self.source_stamp_timeout_sec,
            self.max_steering_rate_deg_per_sec,
            self.low_confidence_slow_threshold,
            self.road_source_slow_ratio, self.single_source_slow_ratio,
        )
        if not all(math.isfinite(v) for v in finite):
            raise ValueError("controller parameters must be finite")
        if not 0.0 < self.maximum_steering_deg <= 22.0:
            raise ValueError("maximum_steering_deg must be in (0, 22]")
        if not 0.0 <= self.lookahead_y_ratio <= 1.0:
            raise ValueError("lookahead_y_ratio must be in [0, 1]")
        if self.normal_drive_command not in CAMERA_DRIVE_COMMANDS:
            raise ValueError("normal_drive_command must be one of 0, 1, 2")
        if self.slow_drive_command not in CAMERA_DRIVE_COMMANDS:
            raise ValueError("slow_drive_command must be one of 0, 1, 2")
        if self.steering_slowdown_threshold_deg <= 0.0:
            raise ValueError("steering_slowdown_threshold_deg must be positive")
        if self.minimum_path_points < 3:
            raise ValueError("minimum_path_points must be at least 3")
        if self.path_timeout_sec <= 0.0 or self.source_stamp_timeout_sec <= 0.0:
            raise ValueError("camera safety timeouts must be positive")
        if self.max_steering_rate_deg_per_sec <= 0.0:
            raise ValueError("max_steering_rate_deg_per_sec must be positive")
        if not self.minimum_confidence <= self.low_confidence_slow_threshold <= 1.0:
            raise ValueError("low_confidence_slow_threshold must be in [minimum_confidence, 1]")
        if not 0.0 <= self.road_source_slow_ratio <= 1.0:
            raise ValueError("road_source_slow_ratio must be in [0, 1]")
        if not 0.0 <= self.single_source_slow_ratio <= 1.0:
            raise ValueError("single_source_slow_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PixelPathSample:
    stamp_ns: int
    image_width: float
    points: tuple
    path_valid: bool
    confidence: float
    received_at: float
    finite: bool
    sources: tuple = ()
    path_state: int = ImagePath.STATE_VALID


@dataclass(frozen=True)
class PixelCommand:
    drive: float
    wheel: int
    steering_deg: float
    reason: str
    valid: bool


def apply_stop_line_limit(command, decision):
    """Apply only a longitudinal ceiling; preserve valid path steering."""
    if not command.valid or decision.phase == StopLinePhase.NORMAL:
        return command
    drive = min(float(command.drive), float(decision.maximum_drive))
    reason = ("stop_line_stop" if decision.phase == StopLinePhase.STOP
              else "stop_line_slowdown")
    return PixelCommand(
        drive, command.wheel, command.steering_deg, reason, command.valid)


def apply_uphill_stop_limit(command, stop_active):
    """Veto valid-path propulsion without changing steering."""
    if (not command.valid or not stop_active or
            command.drive == DriveCommand.STOP.value):
        return command
    return PixelCommand(
        DriveCommand.STOP.value, command.wheel, command.steering_deg,
        "uphill_stop", command.valid)


class PixelController:
    """ROS-independent latest-only controller + safety state machine."""

    def __init__(self, config=PixelControllerConfig()):
        config.validate()
        self.config = config
        self.path = None
        self.last_path_stamp_ns = None
        self.last_valid_path_received_at = None
        self.previous_offset_px = None
        self.last_step_monotonic = None
        self.last_output_steering_deg = 0.0
        self.last_output_monotonic = None

    def ingest_path(self, stamp_ns, image_width, points, path_valid,
                    confidence, received_at, sources=None,
                    path_state=ImagePath.STATE_VALID):
        stamp_ns = int(stamp_ns)
        if self.last_path_stamp_ns is not None and stamp_ns <= self.last_path_stamp_ns:
            return False
        converted = []
        finite = math.isfinite(image_width)
        for point in points:
            try:
                x_px, y_px = float(point[0]), float(point[1])
            except (IndexError, TypeError, ValueError):
                finite = False
                continue
            if not (math.isfinite(x_px) and math.isfinite(y_px)):
                finite = False
            converted.append((x_px, y_px))
        converted_sources = ()
        if sources is not None:
            try:
                converted_sources = tuple(int(source) for source in sources)
            except (TypeError, ValueError):
                finite = False
                converted_sources = ()
            if len(converted_sources) != len(converted):
                finite = False
            known_sources = {
                ImagePathPoint.BOTH_BOUNDARIES,
                ImagePathPoint.LEFT_BOUNDARY,
                ImagePathPoint.RIGHT_BOUNDARY,
                ImagePathPoint.ROAD_CENTER,
                ImagePathPoint.TEMPORAL_FALLBACK,
            }
            if any(source not in known_sources
                   for source in converted_sources):
                finite = False
        try:
            converted_path_state = int(path_state)
        except (TypeError, ValueError):
            converted_path_state = ImagePath.STATE_INVALID
            finite = False
        if converted_path_state not in {
                ImagePath.STATE_INVALID, ImagePath.STATE_VALID,
                ImagePath.STATE_DEGRADED, ImagePath.STATE_INACTIVE}:
            finite = False
        self.path = PixelPathSample(
            stamp_ns, float(image_width), tuple(converted),
            bool(path_valid), float(confidence), float(received_at), finite,
            converted_sources, converted_path_state)
        self.last_path_stamp_ns = stamp_ns
        if (path_valid and finite and image_width > 0.0 and
                math.isfinite(confidence) and
                confidence > self.config.minimum_confidence and
                len(converted) >= self.config.minimum_path_points):
            self.last_valid_path_received_at = float(received_at)
        return True

    @staticmethod
    def stop(reason):
        return PixelCommand(DriveCommand.STOP.value, 0, 0.0, str(reason), False)

    def drive_for_wheel(self, wheel):
        """Return the longitudinal command for the final integer wheel angle."""
        if abs(int(wheel)) >= self.config.steering_slowdown_threshold_deg:
            return float(self.config.slow_drive_command)
        return float(self.config.normal_drive_command)

    def drive_for_path(self, path, wheel):
        """Apply path-quality ceilings after the steering slowdown rule."""
        drive = self.drive_for_wheel(wheel)
        slow = float(self.config.slow_drive_command)
        if path.confidence < self.config.low_confidence_slow_threshold:
            drive = min(drive, slow)
        if path.path_state != ImagePath.STATE_VALID:
            drive = min(drive, slow)
        if path.sources:
            count = len(path.sources)
            road_ratio = path.sources.count(ImagePathPoint.ROAD_CENTER)/count
            single_ratio = sum(
                source in (ImagePathPoint.LEFT_BOUNDARY,
                           ImagePathPoint.RIGHT_BOUNDARY)
                for source in path.sources)/count
            temporal_ratio = path.sources.count(
                ImagePathPoint.TEMPORAL_FALLBACK)/count
            both_ratio = path.sources.count(
                ImagePathPoint.BOTH_BOUNDARIES)/count
            if (road_ratio >= self.config.road_source_slow_ratio or
                    single_ratio >= self.config.single_source_slow_ratio or
                    temporal_ratio > 0.0 or both_ratio < 0.60):
                drive = min(drive, slow)
        return drive

    def step(self, now, ros_now_ns):
        try:
            return self._step(float(now), int(ros_now_ns))
        except Exception:
            return self.stop("internal_exception")

    def _step(self, now, ros_now_ns):
        if self.path is None:
            return self.stop("path_missing")
        path = self.path
        if not path.path_valid:
            return self.stop("path_invalid")
        if path.path_state in (ImagePath.STATE_INVALID,
                               ImagePath.STATE_INACTIVE):
            return self.stop("path_state_invalid")
        if self.last_valid_path_received_at is None:
            return self.stop("valid_path_missing")
        if now - self.last_valid_path_received_at > self.config.path_timeout_sec:
            return self.stop("path_timeout")
        if not path.finite:
            return self.stop("nonfinite_path")
        if not math.isfinite(path.image_width) or path.image_width <= 0.0:
            return self.stop("bad_image_width")
        if not math.isfinite(path.confidence):
            return self.stop("nonfinite_confidence")
        if path.confidence <= self.config.minimum_confidence:
            return self.stop("low_confidence")
        if len(path.points) < self.config.minimum_path_points:
            return self.stop("insufficient_path_points")
        if not math.isfinite(now) or not math.isfinite(path.received_at):
            return self.stop("nonfinite_time")
        if now - path.received_at > self.config.path_timeout_sec:
            return self.stop("path_stale")
        if path.stamp_ns <= 0:
            return self.stop("source_stamp_missing")
        source_age = (ros_now_ns - path.stamp_ns) * 1.0e-9
        if not math.isfinite(source_age) or source_age < 0.0:
            return self.stop("source_stamp_invalid")
        if source_age > self.config.source_stamp_timeout_sec:
            return self.stop("source_stamp_stale")

        offset = lookahead_offset_px(
            path.points, path.image_width, self.config.lookahead_y_ratio)
        if offset is None or not math.isfinite(offset):
            return self.stop("offset_unavailable")

        dt = None
        if self.last_step_monotonic is not None:
            dt = now - self.last_step_monotonic
        steering = steering_from_offset_deg(
            offset, self.previous_offset_px, dt if dt is not None else 0.0,
            path.image_width,
            self.config.proportional_gain_deg_per_norm,
            self.config.derivative_gain_deg_per_norm_per_s,
            self.config.maximum_steering_deg)
        steering *= self.config.steering_sign
        if not math.isfinite(steering):
            return self.stop("steering_nonfinite")
        steering = max(-self.config.maximum_steering_deg,
                       min(self.config.maximum_steering_deg, steering))
        self.previous_offset_px = offset
        self.last_step_monotonic = now

        wheel = max(-22, min(22, int(round(steering))))
        drive = self.drive_for_path(path, wheel)
        if drive not in CAMERA_DRIVE_COMMANDS:
            return self.stop("invalid_drive_policy")
        return PixelCommand(drive, wheel, steering, "ok", True)

    @staticmethod
    def safety_state(command):
        if command.valid:
            return SafetyState.ACTIVE
        if command.reason == "path_missing":
            return SafetyState.WAITING_FOR_PATH
        if command.reason in ("path_timeout", "path_stale", "source_stamp_stale"):
            return SafetyState.PATH_TIMEOUT
        return SafetyState.PATH_INVALID

    def finalize_output(self, command, now):
        """Apply final actuator safety immediately before ROS publication."""
        now = float(now)
        limit = self.config.maximum_steering_deg
        if not command.valid:
            # Stop propulsion and center steering immediately. Recovery starts
            # from this real output, so it remains slew-limited.
            self.last_output_steering_deg = 0.0
            self.last_output_monotonic = now
            return self.stop(command.reason)

        requested = max(-limit, min(limit, float(command.steering_deg)))
        if self.last_output_monotonic is None or not math.isfinite(now):
            dt = 0.0
        else:
            dt = max(0.0, now-self.last_output_monotonic)
        maximum_delta = self.config.max_steering_rate_deg_per_sec*dt
        delta = max(-maximum_delta, min(maximum_delta,
                                        requested-self.last_output_steering_deg))
        steering = self.last_output_steering_deg+delta
        # Final saturation directly before conversion to the actuator message.
        steering = max(-limit, min(limit, steering))
        wheel_limit = int(math.floor(limit))
        wheel = max(-wheel_limit, min(wheel_limit, int(round(steering))))
        self.last_output_steering_deg = steering
        self.last_output_monotonic = now
        # Preserve source/confidence slowdown selected in step(); finalization
        # may only reduce propulsion further after steering slew/saturation.
        drive = min(float(command.drive), self.drive_for_wheel(wheel))
        return PixelCommand(drive, wheel, steering,
                            command.reason, command.valid)


class CameraPixelController(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__(
            "camera_pixel_controller_node",
            parameter_overrides=parameter_overrides or [])
        defaults = {
            "image_path_topic": "/camera/image_path_typed",
            "maximum_steering_deg": 22.0, "steering_sign": 1.0,
            "lookahead_y_ratio": 0.5,
            "proportional_gain_deg_per_norm": 22.0,
            "derivative_gain_deg_per_norm_per_s": 2.0,
            "normal_drive_command": 2.0, "slow_drive_command": 1.0,
            "steering_slowdown_threshold_deg":
                DEFAULT_STEERING_SLOWDOWN_THRESHOLD_DEG,
            "minimum_confidence": 0.0, "minimum_path_points": 3,
            "path_timeout_sec": 0.15, "source_stamp_timeout_sec": 0.15,
            "max_steering_rate_deg_per_sec": 180.0,
            "low_confidence_slow_threshold": 0.65,
            "road_source_slow_ratio": 0.40,
            "single_source_slow_ratio": 0.50,
            "control_rate_hz": 20.0,
            "semantic_path_frame_topic": "/perception/semantic_path_frame",
            "aligned_depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "depth_unit_scale_m": 0.001,
            "depth_sync_tolerance_sec": 0.05,
            "depth_cache_size": 120,
            "stop_line_slowdown_distance_m": 2.0,
            "stop_line_stop_distance_m": 0.7,
            # base_link camera x=0.015, measured vehicle front x=0.500.
            "camera_to_front_bumper_m": 0.485,
            "stop_line_release_margin_m": 0.2,
            "stop_line_confirmation_frames": 3,
            "stop_line_measurement_timeout_sec": 0.5,
            "stop_line_center_roi_width_ratio": 0.6,
            "stop_line_min_depth_m": 0.1,
            "stop_line_max_depth_m": 20.0,
            "stop_line_min_valid_depth_pixels": 20,
            "stop_line_depth_percentile": 50.0,
            "stop_line_depth_mad_scale": 3.5,
            "imu_slope_topic": IMU_SLOPE_TOPIC,
            "imu_valid_topic": IMU_VALID_TOPIC,
            "imu_state_timeout_sec": 0.2,
            "uphill_stop_duration_sec": 5.0,
            # --- Stop-line memory / front-axle crossing (req 2) -----------
            # Independent of image_path_planner's ego-bumper exclusion: fed
            # directly from the raw stop-line mask/depth, tracks distance
            # through occlusion using the MCU's cumulative wheel distance
            # forward travel, and releases the depth-latch stop above once
            # the front axle has crossed the line.
            "distance_topic": "/mcu/distance_m",
            "front_axle_offset_m": 0.350,
            "camera_to_bumper_m": 0.485,
            "stopline_near_bumper_row_ratio": 0.92,
            "stopline_memory_timeout_sec": 3.0,
            "stopline_depth_min_confidence": 0.35,
            "stopline_passed_confirm_sec": 0.15,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        config = PixelControllerConfig(**{
            name: self.get_parameter(name).value
            for name in PixelControllerConfig.__dataclass_fields__
        })
        self.controller = PixelController(config)
        stop_config = StopLineConfig(**{
            name: self.get_parameter(name).value
            for name in StopLineConfig.__dataclass_fields__
        })
        self.stop_line_policy = StopLinePolicy(stop_config)
        self.stop_line_memory = StopLineMemory(StopLineMemoryConfig(
            camera_to_bumper_m=float(
                self.get_parameter("camera_to_bumper_m").value),
            front_axle_offset_m=float(
                self.get_parameter("front_axle_offset_m").value),
            near_bumper_row_ratio=float(
                self.get_parameter("stopline_near_bumper_row_ratio").value),
            slowdown_distance_m=float(
                self.get_parameter("stop_line_slowdown_distance_m").value),
            stop_distance_m=float(
                self.get_parameter("stop_line_stop_distance_m").value),
            depth_min_confidence=float(
                self.get_parameter("stopline_depth_min_confidence").value),
            memory_timeout_sec=float(
                self.get_parameter("stopline_memory_timeout_sec").value),
            passed_confirm_sec=float(
                self.get_parameter("stopline_passed_confirm_sec").value)))
        self._mcu_distance_m = None
        self.uphill_stop = UphillStopController(UphillStopConfig(
            uphill_stop_duration_sec=float(
                self.get_parameter("uphill_stop_duration_sec").value)))
        self.imu_state_timeout_sec = float(
            self.get_parameter("imu_state_timeout_sec").value)
        if (not math.isfinite(self.imu_state_timeout_sec) or
                self.imu_state_timeout_sec <= 0.0):
            raise ValueError("imu_state_timeout_sec must be finite and positive")
        self.imu_slope = False
        self.imu_valid = False
        self.imu_slope_received_at = None
        self.imu_valid_received_at = None
        self.bridge = CvBridge()
        self.depth_unit_scale_m = float(
            self.get_parameter("depth_unit_scale_m").value)
        self.depth_sync_tolerance_ns = int(float(
            self.get_parameter("depth_sync_tolerance_sec").value)*1.0e9)
        depth_cache_size = int(self.get_parameter("depth_cache_size").value)
        if (not math.isfinite(self.depth_unit_scale_m) or
                self.depth_unit_scale_m <= 0.0):
            raise ValueError("depth_unit_scale_m must be finite and positive")
        if self.depth_sync_tolerance_ns < 0:
            raise ValueError("depth_sync_tolerance_sec must be nonnegative")
        if depth_cache_size < 1:
            raise ValueError("depth_cache_size must be positive")
        self.depth_cache = deque(maxlen=depth_cache_size)
        self.last_stop_semantic_stamp_ns = None
        self.last_stop_measurement = None
        rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive")

        self.drive_pub = self.create_publisher(Float32, DRIVE_TOPIC, 10)
        self.wheel_pub = self.create_publisher(Int32, WHEEL_TOPIC, 10)
        self.diagnostics_pub = self.create_publisher(
            String, "/camera/pixel_controller_diagnostics", 10)
        self.create_subscription(
            ImagePath, config.image_path_topic, self.on_path, 10)
        self.create_subscription(
            Image, self.get_parameter("aligned_depth_topic").value,
            self.on_depth, qos_profile_sensor_data)
        self.create_subscription(
            Float32, self.get_parameter("distance_topic").value,
            self.on_mcu_distance, 10)
        self.create_subscription(
            SemanticPathFrame,
            self.get_parameter("semantic_path_frame_topic").value,
            self.on_semantic_path_frame, qos_profile_sensor_data)
        self.create_subscription(
            Bool, self.get_parameter("imu_slope_topic").value,
            self.on_imu_slope, 10)
        self.create_subscription(
            Bool, self.get_parameter("imu_valid_topic").value,
            self.on_imu_valid, 10)
        self.create_timer(1.0 / rate_hz, self.control)
        self.get_logger().info(
            f"camera PIXEL (non-BEV) controller ready at {rate_hz:.1f} Hz; "
            f"candidate outputs={DRIVE_TOPIC},{WHEEL_TOPIC}; "
            "aligned depth stop-line and one-shot uphill stop enabled")

    @staticmethod
    def stamp_ns(header):
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def on_path(self, msg):
        points = [(p.x_px, p.y_px) for p in msg.points]
        sources = [p.source for p in msg.points]
        now = time.monotonic()
        accepted = self.controller.ingest_path(
            self.stamp_ns(msg.header), float(msg.image_width), points,
            bool(msg.path_valid), float(msg.path_confidence), now, sources,
            int(msg.path_state))
        if accepted and not msg.path_valid:
            self.controller.previous_offset_px = None
            self.controller.last_step_monotonic = None
            self.publish_command(PixelController.stop("path_invalid"), now)

    def on_imu_slope(self, msg):
        self.imu_slope = bool(msg.data)
        self.imu_slope_received_at = time.monotonic()

    def on_imu_valid(self, msg):
        self.imu_valid = bool(msg.data)
        self.imu_valid_received_at = time.monotonic()

    def imu_input_valid(self, now):
        return bool(
            self.imu_valid and
            self.imu_slope_received_at is not None and
            self.imu_valid_received_at is not None and
            0.0 <= now-self.imu_slope_received_at <=
            self.imu_state_timeout_sec and
            0.0 <= now-self.imu_valid_received_at <=
            self.imu_state_timeout_sec)

    def on_depth(self, msg):
        stamp = self.stamp_ns(msg.header)
        if stamp > 0:
            self.depth_cache.append((stamp, msg))

    def on_mcu_distance(self, msg):
        value = float(msg.data)
        if math.isfinite(value) and value >= 0.0:
            self._mcu_distance_m = value

    def depth_to_meters(self, msg):
        image = np.asarray(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        if image.ndim != 2:
            raise ValueError("aligned depth image must be single-channel")
        if image.dtype == np.uint16:
            return image.astype(np.float64)*self.depth_unit_scale_m
        if np.issubdtype(image.dtype, np.floating):
            return image.astype(np.float64)
        raise ValueError(f"unsupported aligned depth dtype: {image.dtype}")

    def on_semantic_path_frame(self, msg):
        stamp = self.stamp_ns(msg.header)
        if (stamp <= 0 or
                (self.last_stop_semantic_stamp_ns is not None and
                 stamp <= self.last_stop_semantic_stamp_ns)):
            return
        self.last_stop_semantic_stamp_ns = stamp
        now = time.monotonic()
        try:
            stop_mask = decode_binary_rle(
                msg.stop_line_rle, int(msg.image_height), int(msg.image_width))
        except Exception as exc:
            self.stop_line_policy.observe_unavailable("stop_line_decode_error")
            self.get_logger().warning(f"stop-line mask decode failed: {exc}")
            return
        measurement = None
        if not self.depth_cache:
            self.stop_line_policy.observe_unavailable("aligned_depth_missing")
        else:
            depth_stamp, depth_msg = min(
                self.depth_cache, key=lambda item: abs(item[0]-stamp))
            if abs(depth_stamp-stamp) > self.depth_sync_tolerance_ns:
                self.stop_line_policy.observe_unavailable("aligned_depth_unsynchronized")
            else:
                try:
                    depth_m = self.depth_to_meters(depth_msg)
                    measurement = estimate_stop_line_distance(
                        stop_mask, depth_m, self.stop_line_policy.config)
                except Exception as exc:
                    self.stop_line_policy.observe_unavailable("stop_line_decode_error")
                    self.get_logger().warning(
                        f"stop-line measurement rejected: {exc}")
        self.last_stop_measurement = measurement
        if measurement is not None and measurement.valid:
            self.stop_line_policy.ingest_camera_distance(
                measurement.camera_distance_m, now)
        elif measurement is not None:
            self.stop_line_policy.observe_unavailable(measurement.reason)

        # req 2: independent stop-line memory. Fed the raw (un-excluded)
        # mask directly, never image_path_planner's ego-excluded geometry.
        detected = bool(np.any(stop_mask))
        observation = StopLineObservation(
            detected=detected,
            pixel_row=stop_line_pixel_row(stop_mask) if detected else None,
            image_height=float(msg.image_height),
            camera_distance_m=(measurement.camera_distance_m
                               if measurement is not None and measurement.valid
                               else None),
            confidence=1.0 if measurement is not None and measurement.valid else 0.0)
        status = self.stop_line_memory.update(
            observation, self._mcu_distance_m, now)
        if status.crossed_front_axle:
            # The front axle has passed the line: release the depth-latch
            # stop instead of waiting on an external arbiter that never
            # calls release_stop() (req 2.5).
            self.stop_line_policy.release_stop()

    def control(self):
        now = time.monotonic()
        try:
            command = self.controller.step(
                now, self.get_clock().now().nanoseconds)
        except Exception as exc:
            command = PixelController.stop("node_internal_exception")
            self.get_logger().error(f"controller exception: {exc}")
        if not command.valid:
            self.controller.previous_offset_px = None
            self.controller.last_step_monotonic = None
        self.publish_command(command, now)

    def publish_command(self, command, now):
        command = self.controller.finalize_output(command, now)
        stop_decision = self.stop_line_policy.decision(now)
        command = apply_stop_line_limit(command, stop_decision)
        uphill_stop_active = self.uphill_stop.update(
            self.imu_slope, self.imu_input_valid(now), now)
        command = apply_uphill_stop_limit(command, uphill_stop_active)
        state = self.controller.safety_state(command)
        self.drive_pub.publish(Float32(data=float(command.drive)))
        self.wheel_pub.publish(Int32(data=int(command.wheel)))
        self.diagnostics_pub.publish(String(data=json.dumps({
            "valid": command.valid, "reason": command.reason,
            "safety_state": state.value,
            "drive": command.drive, "wheel": command.wheel,
            "steering_deg": command.steering_deg,
            "stop_line_phase": stop_decision.phase.value,
            "front_bumper_stop_distance_m":
                stop_decision.front_bumper_distance_m,
            "stop_required": stop_decision.stop_required,
            "stop_latched": self.stop_line_policy.stop_latched,
            "stop_confirmation_count":
                self.stop_line_policy.confirmation_count,
            "stop_measurement_reason":
                self.stop_line_policy.last_measurement_reason,
            "uphill_stop_state": self.uphill_stop.state.value,
            "uphill_stop_active": uphill_stop_active,
            "imu_slope": self.imu_slope,
            "imu_slope_input_valid": self.imu_input_valid(now),
            "stopline_memory_state": self.stop_line_memory.state.value,
            "stopline_memory_camera_distance_m":
                self.stop_line_memory.status().camera_distance_m,
            "stopline_memory_front_axle_distance_m":
                self.stop_line_memory.status().front_axle_distance_m,
            "stopline_memory_crossed_front_axle":
                self.stop_line_memory.status().crossed_front_axle,
            "stopline_memory_reason": self.stop_line_memory.status().reason,
            "mcu_distance_m": self._mcu_distance_m,
            "motion_distance_source": "/mcu/distance_m",
        }, separators=(",", ":"))))
        return command

    def destroy_node(self):
        if rclpy.ok():
            self.controller.finalize_output(
                PixelController.stop("node_shutdown"), time.monotonic())
            self.drive_pub.publish(Float32(data=DriveCommand.STOP.value))
            self.wheel_pub.publish(Int32(data=0))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPixelController()
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
