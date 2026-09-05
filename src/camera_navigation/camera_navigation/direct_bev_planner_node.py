#!/usr/bin/env python3
"""Direct semantic-mask to metric BEV path ROS node."""

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Bool, Float32, Header, Int32, String

from race_interfaces.msg import SemanticPathFrame

from .direct_bev_core import DirectBevConfig, DirectBevPlanner
from .hybrid_bev_candidate import HybridCandidateOptions, HybridDirectBevCandidate
from .direct_bev_projection import (
    CameraModel, build_ground_remap, ground_points_to_pixels,
    project_mask_to_bev, warp_rgb_to_bev,
)
from .ground_plane_calibration import (
    CALIBRATION_VALID, InitConfig, NOT_CONFIGURED, OPTICAL_TO_MECHANICAL,
    StationaryAttitudeEstimator, ValidityConfig, compose_effective_orientation,
    evaluate_calibration_state, load_camera_mount_config,
)
from .metric_path_quality import path_heading
from .overlay_worker import LatestOnlyWorker, OverlayRateLimiter
from .semantic_path_contract import decode_binary_rle
from .timestamp_sync import TimestampedMessageCache, subscription_transition


def image_message(array, header, encoding):
    array = np.ascontiguousarray(array)
    message = Image()
    message.header = header
    message.height, message.width = array.shape[:2]
    message.encoding = encoding
    message.is_bigendian = False
    message.step = int(array.strides[0])
    message.data = array.tobytes()
    return message


def message_stamp_ns(message):
    return (int(message.header.stamp.sec)*1_000_000_000 +
            int(message.header.stamp.nanosec))


def stage_pixel_count(refinement_diagnostics, stage_key):
    """Pull an already-computed road pixel count out of a
    refinement_diagnostics dict (parsed from SemanticPathFrame's
    refinement_diagnostics_json, produced by camera_yolo_inference_node's
    perception_refinement) -- reused, never recomputed here. Returns None
    if the stage/road key is absent or malformed, so callers can tell
    "genuinely zero road pixels" apart from "no data for this stage"."""
    stage = refinement_diagnostics.get(stage_key)
    return stage.get("road") if isinstance(stage, dict) else None


def bev_overlay_requested(enabled, subscriber_count, limiter_ready):
    """Cheap gate kept separate so disabled overlays perform no warp/draw."""
    return bool(enabled and int(subscriber_count) > 0 and limiter_ready)


def wall_input_fresh(last_wall, now_wall, timeout_sec):
    return (last_wall is not None and
            0.0 <= float(now_wall)-float(last_wall) <= float(timeout_sec))


TimestampedImageCache = TimestampedMessageCache


def build_direct_bev_planner(variant, config, road_boundary_fallback="none"):
    """Build an explicitly selected planner; production remains the default."""
    if variant == "production":
        if road_boundary_fallback != "none":
            raise ValueError("road_boundary_fallback is hybrid_a6-only")
        return DirectBevPlanner(config)
    if variant == "hybrid_a6":
        if road_boundary_fallback not in ("none", "basic", "gated"):
            raise ValueError("road_boundary_fallback must be none/basic/gated")
        return HybridDirectBevCandidate(config, HybridCandidateOptions(
            temporal_smoothing=True, curvature_stabilization=True,
            mode_hysteresis_frames=3, fixed_resample_origin=True,
            fail_closed_hold=True,
            road_boundary_fallback=road_boundary_fallback))
    raise ValueError(f"unsupported planner_variant: {variant}")


def image_to_bgr(message):
    """Decode bgr8/rgb8 while respecting per-row sensor_msgs/Image padding."""
    encoding = str(message.encoding).lower()
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"unsupported camera overlay encoding: {encoding}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    row_bytes = width*3
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError("malformed camera image dimensions or step")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height*step:
        raise ValueError("camera image data is shorter than height*step")
    pixels = raw[:height*step].reshape(height, step)[:, :row_bytes]
    image = pixels.reshape(height, width, 3)
    if encoding == "rgb8":
        image = image[:, :, ::-1]
    # sensor_msgs data may be backed by immutable ``bytes``. OpenCV drawing
    # requires a writable destination even when the view is already contiguous.
    return np.ascontiguousarray(image).copy()


def render_camera_path_overlay(message, points, camera, rotation, position,
                               line_thickness, point_radius,
                               anchor_x_ratio=0.5, anchor_y_ratio=1.0,
                               path_valid=True):
    """Render only the final metric path over an original camera frame."""
    overlay = image_to_bgr(message)
    if not path_valid or not len(points):
        return overlay
    pixels, indices = ground_points_to_pixels(
        points, camera, rotation, position)
    rounded = np.rint(pixels).astype(np.int32)
    if len(rounded):
        width, height = int(message.width), int(message.height)
        anchor = np.array([
            np.clip(float(anchor_x_ratio), 0.0, 1.0)*(width-1),
            np.clip(float(anchor_y_ratio), 0.0, 1.0)*(height-1),
        ], float)
        endpoint = pixels[0]
        vertical = max(1.0, anchor[1]-endpoint[1])
        control_one = anchor+np.array([0.0, -0.35*vertical])
        if len(pixels) >= 2 and indices[1] == indices[0]+1:
            tangent = pixels[1]-pixels[0]
            norm = float(np.linalg.norm(tangent))
            tangent = (np.array([0.0, -1.0]) if norm <= 1.0e-6 else
                       tangent/norm)
        else:
            tangent = np.array([0.0, -1.0])
        control_two = endpoint-tangent*(0.25*vertical)
        control_one = np.clip(control_one, [0.0, 0.0],
                              [width-1.0, height-1.0])
        control_two = np.clip(control_two, [0.0, 0.0],
                              [width-1.0, height-1.0])
        samples = []
        for t_value in np.linspace(0.0, 1.0, 32):
            inverse = 1.0-t_value
            samples.append(
                inverse**3*anchor+3.0*inverse**2*t_value*control_one+
                3.0*inverse*t_value**2*control_two+t_value**3*endpoint)
        samples = np.rint(samples).astype(np.int32)
        cv2.polylines(overlay, [samples], False, (255, 0, 255),
                      int(line_thickness), cv2.LINE_AA)
    for offset in range(1, len(rounded)):
        if indices[offset] == indices[offset-1]+1:
            cv2.line(overlay, tuple(rounded[offset-1]), tuple(rounded[offset]),
                     (255, 0, 255), int(line_thickness), cv2.LINE_AA)
    for pixel in rounded:
        cv2.circle(overlay, tuple(pixel), int(point_radius),
                   (255, 0, 255), -1, cv2.LINE_AA)
    return overlay


def safely_render_camera_overlay(render, on_error):
    """Keep optional camera visualization failures out of the path worker."""
    try:
        return render()
    except Exception as error:  # Visualization must never stop path/state output.
        on_error(error)
        return None


def render_diagnostic_overlay(source, output_shape, reason):
    """Create a visible low-cost diagnostic frame from RGB or black."""
    height, width = (int(output_shape[0]), int(output_shape[1]))
    if source is None:
        overlay = np.zeros((height, width, 3), np.uint8)
    else:
        overlay = cv2.resize(image_to_bgr(source), (width, height),
                             interpolation=cv2.INTER_AREA)
        overlay = (overlay.astype(np.float32)*0.30).astype(np.uint8)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, 34), (0, 0, 0), -1)
    cv2.putText(overlay, str(reason), (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 180, 255), 1, cv2.LINE_AA)
    return overlay


class EventRate:
    """Thread-safe observed event-rate meter, separate from processing time."""

    def __init__(self, maximum_samples=600):
        self._times = deque(maxlen=int(maximum_samples))
        self._lock = threading.Lock()

    def tick(self, now=None):
        with self._lock:
            self._times.append(time.monotonic() if now is None else float(now))

    def rate(self):
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1]-self._times[0]
            return 0.0 if elapsed <= 0.0 else (len(self._times)-1)/elapsed


@dataclass(frozen=True)
class CachedPlannerResult:
    generation: int
    points: np.ndarray
    state: str
    valid: bool
    confidence: float
    diagnostics: dict
    source_stamp_ns: int
    calculated_wall: float
    overlay_message: object = None
    overlay_source_stamp_ns: int = 0
    overlay_key: tuple = None


class LatestPlannerResultCache:
    """Atomically replace all fields consumed by the fixed-rate publisher."""

    def __init__(self):
        self._lock = threading.Lock()
        self._result = None
        self._generation = 0

    def replace(self, points, state, valid, confidence, diagnostics,
                source_stamp_ns, calculated_wall=None):
        points = np.asarray(points, float).reshape(-1, 2).copy()
        points.setflags(write=False)
        with self._lock:
            self._generation += 1
            self._result = CachedPlannerResult(
                self._generation, points, str(state), bool(valid),
                float(confidence), dict(diagnostics), int(source_stamp_ns),
                time.monotonic() if calculated_wall is None else
                float(calculated_wall))
            return self._generation

    def set_overlay(self, generation, message, source_stamp_ns, overlay_key):
        with self._lock:
            current = self._result
            if current is None or current.generation != int(generation):
                return False
            self._result = CachedPlannerResult(
                current.generation, current.points, current.state,
                current.valid, current.confidence, current.diagnostics,
                current.source_stamp_ns, current.calculated_wall, message,
                int(source_stamp_ns), tuple(overlay_key))
            return True

    def snapshot(self):
        with self._lock:
            return self._result


def evaluate_fixed_result(snapshot, now_wall, calibration_ready,
                          path_stale_timeout_sec, state_stale_timeout_sec,
                          maximum_steering_deg):
    """Return ``(path_valid, reason, age)`` for a fixed-rate output tick."""
    age = (float("inf") if snapshot is None else
           max(0.0, float(now_wall)-snapshot.calculated_wall))
    if not calibration_ready:
        return False, "CALIBRATION_INVALID", age
    if snapshot is None:
        return False, "NO_RESULT", age
    if (age > float(state_stale_timeout_sec) or
            age > float(path_stale_timeout_sec)):
        return False, "PATH_STALE", age
    if (not snapshot.valid or snapshot.state not in ("VALID", "DEGRADED") or
            len(snapshot.points) < 3 or
            not np.all(np.isfinite(snapshot.points))):
        reason = (snapshot.diagnostics.get("reasons") or ["PATH_INVALID"])[-1]
        return False, reason, age
    steering = snapshot.diagnostics.get("required_steering_deg")
    try:
        steering = float(steering)
    except (TypeError, ValueError):
        steering = float("inf")
    if not math.isfinite(steering) or abs(steering) > maximum_steering_deg:
        return False, "STEERING_LIMIT_EXCEEDED", age
    return True, None, age


class DirectBevPlannerNode(Node):
    def __init__(self):
        super().__init__("direct_bev_planner_node")
        for name, value in {
            "camera_mount.configured": False,
            "camera_mount.position_x_m": 0.0,
            "camera_mount.position_y_m": 0.0,
            "camera_mount.height_z_m": 0.0,
            "camera_mount.reference_roll_deg": 0.0,
            "camera_mount.reference_pitch_deg": 0.0,
            "camera_mount.reference_yaw_deg": 0.0,
            "init_window_sec": 1.0, "init_min_samples": 15,
            "init_max_samples": 400, "init_max_accel_stddev_mps2": 0.25,
            "init_gravity_norm_min_mps2": 8.5,
            "init_gravity_norm_max_mps2": 11.0,
            "init_max_gyro_norm_rad_s": 0.08,
            "init_low_pass_alpha": 0.2, "init_outlier_mad_k": 3.5,
            "max_runtime_pitch_correction_deg": 5.0,
            "max_runtime_roll_correction_deg": 5.0,
            "max_calibration_age_sec": 3600.0,
            "imu_stale_timeout_sec": 2.0,
            "camera_info_stale_timeout_sec": 2.0,
            "semantic_topic": "/perception/semantic_path_frame",
            "camera_info_topic": "/camera/camera_info",
            "accel_topic": "/camera/camera/accel/sample",
            "gyro_topic": "/camera/camera/gyro/sample",
            "input_timeout_sec": 0.25,
            "debug_enabled": False, "debug_publish_rate_hz": 10.0,
            "camera_overlay_enabled": True,
            "camera_overlay_topic": "/camera/bev/camera_overlay",
            "camera_overlay_sync_tolerance_sec": 0.05,
            "camera_overlay_cache_frames": 20,
            "camera_overlay_line_thickness": 4,
            "camera_overlay_point_radius": 3,
            "camera_overlay_publish_rate_hz": 30.0,
            "camera_overlay_anchor_x_ratio": 0.5,
            "camera_overlay_anchor_y_ratio": 1.0,
            "bev_overlay_enabled": True,
            "bev_overlay_topic": "/camera/bev/overlay_image",
            "bev_overlay_max_fps": 55.0,
            "bev_diagnostic_fps": 3.0,
            "rgb_stale_timeout_sec": 0.50,
            "fixed_output_rate_enabled": False,
            "output_rate_hz": 60.0,
            "path_stale_timeout_sec": 0.20,
            "state_stale_timeout_sec": 0.20,
            "camera_overlay_stale_timeout_sec": 0.20,
            "planner_variant": "production",
            "road_boundary_fallback": "none",
        }.items():
            self.declare_parameter(name, value)
        planner_defaults = {name: field.default for name, field in
                            DirectBevConfig.__dataclass_fields__.items()}
        for name, value in planner_defaults.items():
            self.declare_parameter(name, value)
        self.mount = load_camera_mount_config({
            "configured": self.get_parameter("camera_mount.configured").value,
            "position_x_m": self.get_parameter(
                "camera_mount.position_x_m").value,
            "position_y_m": self.get_parameter(
                "camera_mount.position_y_m").value,
            "height_z_m": self.get_parameter("camera_mount.height_z_m").value,
            "reference_roll_deg": self.get_parameter(
                "camera_mount.reference_roll_deg").value,
            "reference_pitch_deg": self.get_parameter(
                "camera_mount.reference_pitch_deg").value,
            "reference_yaw_deg": self.get_parameter(
                "camera_mount.reference_yaw_deg").value,
        })
        self.planner_variant = str(self.get_parameter("planner_variant").value)
        self.planner = build_direct_bev_planner(
            self.planner_variant, DirectBevConfig(**{
                name: self.get_parameter(name).value
                for name in planner_defaults}),
            str(self.get_parameter("road_boundary_fallback").value))
        self.latest_camera_drive = 0.0
        self.latest_camera_wheel = 0
        self.init_config = InitConfig(**{
            "window_sec": self.get_parameter("init_window_sec").value,
            "min_samples": self.get_parameter("init_min_samples").value,
            "max_samples": self.get_parameter("init_max_samples").value,
            "max_accel_stddev_mps2": self.get_parameter(
                "init_max_accel_stddev_mps2").value,
            "gravity_norm_min_mps2": self.get_parameter(
                "init_gravity_norm_min_mps2").value,
            "gravity_norm_max_mps2": self.get_parameter(
                "init_gravity_norm_max_mps2").value,
            "max_gyro_norm_rad_s": self.get_parameter(
                "init_max_gyro_norm_rad_s").value,
            "low_pass_alpha": self.get_parameter("init_low_pass_alpha").value,
            "outlier_mad_k": self.get_parameter("init_outlier_mad_k").value,
        })
        self.validity_config = ValidityConfig(
            max_runtime_pitch_correction_deg=float(self.get_parameter(
                "max_runtime_pitch_correction_deg").value),
            max_runtime_roll_correction_deg=float(self.get_parameter(
                "max_runtime_roll_correction_deg").value),
            max_calibration_age_sec=float(self.get_parameter(
                "max_calibration_age_sec").value))
        self.estimator = StationaryAttitudeEstimator(self.init_config)
        self.init_result = None
        self.locked_wall = None
        self.pending_gyro = None
        self.last_accel_wall = self.last_gyro_wall = None
        self.rotation = self.position = None
        self.camera_model = None
        self.camera_info_frame_id = ""
        self.last_camera_info_wall = None
        self.map_x = self.map_y = None
        self.state, self.state_reasons = NOT_CONFIGURED, []
        self.last_semantic_wall = None
        self.last_processed_stamp = None
        self.processing_times = []
        self.end_to_end_times = deque(maxlen=300)
        self.debug_enabled = bool(self.get_parameter("debug_enabled").value)
        self.debug_limiter = OverlayRateLimiter(float(
            self.get_parameter("debug_publish_rate_hz").value))
        self.camera_overlay_enabled = bool(
            self.get_parameter("camera_overlay_enabled").value)
        self.camera_overlay_cache = TimestampedImageCache(
            self.get_parameter("camera_overlay_cache_frames").value)
        self.camera_overlay_limiter = OverlayRateLimiter(float(
            self.get_parameter("camera_overlay_publish_rate_hz").value))
        self.camera_overlay_last_error_wall = None
        self.camera_overlay_render_lock = threading.Lock()
        self.bev_overlay_enabled = bool(
            self.get_parameter("bev_overlay_enabled").value)
        bev_overlay_max_fps = float(
            self.get_parameter("bev_overlay_max_fps").value)
        if bev_overlay_max_fps <= 0.0:
            raise ValueError("bev_overlay_max_fps must be positive")
        self.bev_overlay_limiter = OverlayRateLimiter(bev_overlay_max_fps)
        self.bev_overlay_rate = EventRate()
        self.bev_overlay_times = deque(maxlen=300)
        self.bev_sync_deltas_ms = deque(maxlen=600)
        self.bev_sync_exact = 0
        self.bev_sync_nearest = 0
        self.bev_sync_miss = 0
        self.bev_overlay_errors = 0
        self.bev_diagnostic_reason = None
        self.last_bev_overlay_wall = None
        self.bev_overlay_worker = LatestOnlyWorker(
            self._publish_bev_overlay, "bev-rgb-overlay")
        self.fixed_output_rate_enabled = bool(
            self.get_parameter("fixed_output_rate_enabled").value)
        output_rate = float(self.get_parameter("output_rate_hz").value)
        if not math.isfinite(output_rate) or output_rate <= 0.0:
            raise ValueError("output_rate_hz must be finite and positive")
        self.result_cache = LatestPlannerResultCache()
        self.semantic_input_rate = EventRate()
        self.planner_processing_rate = EventRate()
        self.path_output_rate = EventRate()
        self.state_output_rate = EventRate()
        self.camera_overlay_output_rate = EventRate()
        self.last_output_generation = None
        self.worker = LatestOnlyWorker(self._process, "direct-bev-planner")

        self.path_pub = self.create_publisher(Path, "/camera/bev/path", 10)
        self.valid_pub = self.create_publisher(Bool, "/camera/bev/valid", 10)
        self.confidence_pub = self.create_publisher(
            Float32, "/camera/bev/confidence", 10)
        self.state_pub = self.create_publisher(String, "/camera/bev/state", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/bev/diagnostics", 10)
        visualization_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.overlay_pub = self.create_publisher(
            Image, "/camera/bev/overlay", visualization_qos)
        self.mask_pub = self.create_publisher(
            Image, "/camera/bev/mask", visualization_qos)
        self.safe_pub = self.create_publisher(
            Image, "/camera/bev/safe_road_mask", visualization_qos)
        self.camera_overlay_pub = self.create_publisher(
            Image, self.get_parameter("camera_overlay_topic").value,
            visualization_qos)
        overlay_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.bev_overlay_pub = self.create_publisher(
            Image, self.get_parameter("bev_overlay_topic").value, overlay_qos)
        latest_sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.rgb_qos = latest_sensor_qos
        self._rgb_subscription = None
        self.create_subscription(
            SemanticPathFrame, self.get_parameter("semantic_topic").value,
            self._on_semantic, latest_sensor_qos)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(
            Float32, "/camera_drive",
            lambda message: setattr(self, "latest_camera_drive",
                                    float(message.data)), 10)
        self.create_subscription(
            Int32, "/camera_wheel",
            lambda message: setattr(self, "latest_camera_wheel",
                                    int(message.data)), 10)
        self.create_subscription(Imu, self.get_parameter("accel_topic").value,
                                 self._on_accel, qos_profile_sensor_data)
        self.create_subscription(Imu, self.get_parameter("gyro_topic").value,
                                 self._on_gyro, qos_profile_sensor_data)
        self.create_timer(0.2, self._update_calibration)
        self.create_timer(0.1, self._check_timeout)
        self.create_timer(0.1, self._update_rgb_subscription)
        diagnostic_fps = float(self.get_parameter("bev_diagnostic_fps").value)
        if diagnostic_fps <= 0.0:
            raise ValueError("bev_diagnostic_fps must be positive")
        self.create_timer(1.0/diagnostic_fps, self._publish_diagnostic_overlay)
        if self.fixed_output_rate_enabled:
            self.create_timer(1.0/output_rate, self._fixed_output_tick)

    @staticmethod
    def _stamp(message):
        return float(message.header.stamp.sec)+message.header.stamp.nanosec*1e-9

    @staticmethod
    def _vector(field):
        return np.array((field.x, field.y, field.z), float)

    def _on_gyro(self, message):
        self.pending_gyro = self._vector(message.angular_velocity)
        self.last_gyro_wall = time.monotonic()

    def _on_accel(self, message):
        self.last_accel_wall = time.monotonic()
        if self.pending_gyro is None or self.estimator.locked:
            return
        self.estimator.add_sample(
            OPTICAL_TO_MECHANICAL@self._vector(message.linear_acceleration),
            OPTICAL_TO_MECHANICAL@self.pending_gyro, self._stamp(message))

    def _on_camera_info(self, message):
        matrix = np.asarray(message.k, float).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            self.camera_model = None
            return
        candidate = CameraModel(
            int(message.width), int(message.height), matrix,
            np.asarray(message.d, float), str(message.distortion_model))
        self.camera_info_frame_id = str(message.header.frame_id)
        self.last_camera_info_wall = time.monotonic()
        if self.camera_model is None or not np.array_equal(
                candidate.matrix, self.camera_model.matrix):
            self.camera_model = candidate
            self.map_x = self.map_y = None

    def _imu_available(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("imu_stale_timeout_sec").value)
        return (self.last_accel_wall is not None and
                self.last_gyro_wall is not None and
                now-self.last_accel_wall <= timeout and
                now-self.last_gyro_wall <= timeout)

    def _update_calibration(self):
        available = self._imu_available()
        if self.mount.is_usable() and available and not self.estimator.locked:
            if self.estimator.ready():
                self.init_result = self.estimator.finalize()
                self.locked_wall = time.monotonic()
                pitch_delta = (self.init_result["measured_pitch_deg"]-
                               self.mount.reference_pitch_deg)
                roll_delta = (self.init_result["measured_roll_deg"]-
                              self.mount.reference_roll_deg)
                _, _, _, self.rotation, _ = compose_effective_orientation(
                    (self.mount.reference_roll_deg,
                     self.mount.reference_pitch_deg,
                     self.mount.reference_yaw_deg), pitch_delta, roll_delta)
                self.position = np.array((self.mount.position_x_m,
                                          self.mount.position_y_m,
                                          self.mount.height_z_m), float)
                self.map_x = self.map_y = None
        age = (None if self.locked_wall is None else
               time.monotonic()-self.locked_wall)
        self.state, self.state_reasons = evaluate_calibration_state(
            mount=self.mount, imu_available=available,
            init_result=self.init_result, validity=self.validity_config,
            calibration_age_sec=age)

    def _on_semantic(self, message):
        stamp = message_stamp_ns(message)
        if self.last_processed_stamp is not None and stamp <= self.last_processed_stamp:
            return
        self.last_processed_stamp = stamp
        self.last_semantic_wall = time.monotonic()
        self.semantic_input_rate.tick(self.last_semantic_wall)
        self.worker.submit(message)

    def _on_camera_image(self, message):
        self.camera_overlay_cache.add(message)

    def _overlay_subscriber_count(self):
        return (self.bev_overlay_pub.get_subscription_count()+
                self.camera_overlay_pub.get_subscription_count())

    def _update_rgb_subscription(self):
        action = subscription_transition(
            self._overlay_subscriber_count(), self._rgb_subscription is not None)
        if action == "CREATE":
            self._rgb_subscription = self.create_subscription(
                Image, "/camera/image_raw", self._on_camera_image, self.rgb_qos)
            self.bev_diagnostic_reason = "WAITING_SEMANTIC"
        elif action == "DESTROY":
            self.destroy_subscription(self._rgb_subscription)
            self._rgb_subscription = None
            self.camera_overlay_cache.clear()
            self.bev_overlay_worker.clear()
            self.bev_overlay_limiter.reset()
            self.camera_overlay_limiter.reset()
            self.bev_diagnostic_reason = None

    def _calibration_ready(self):
        return (self.state == CALIBRATION_VALID and self.rotation is not None and
                self.position is not None and self.camera_model is not None and
                wall_input_fresh(
                    self.last_camera_info_wall, time.monotonic(),
                    self.get_parameter("camera_info_stale_timeout_sec").value))

    def _ensure_remap(self):
        if self.map_x is None:
            self.map_x, self.map_y = build_ground_remap(
                self.planner.config, self.camera_model, self.rotation,
                self.position)

    def _process(self, message):
        started = time.perf_counter()
        stamp_ns = message_stamp_ns(message)
        if not self._calibration_ready():
            self.bev_diagnostic_reason = self._calibration_wait_reason()
            self._handle_invalid(stamp_ns, "CALIBRATION_INVALID")
            return
        if (stamp_ns <= 0 or not message.header.frame_id or
                int(message.image_width) != self.camera_model.width or
                int(message.image_height) != self.camera_model.height or
                (self.camera_info_frame_id and
                 str(message.header.frame_id) != self.camera_info_frame_id)):
            self.bev_diagnostic_reason = "FRAME_CONTRACT_INVALID"
            self._handle_invalid(stamp_ns, "FRAME_CONTRACT_INVALID")
            return
        try:
            self._ensure_remap()
            road_image = decode_binary_rle(
                message.road_rle, message.image_height, message.image_width)
            # Stage-diagnostics: pixel count right after RLE decode, before
            # BEV projection. np.count_nonzero() on an already-decoded mask
            # -- no extra copy/render added to the inference path.
            decoded_road_pixels = int(np.count_nonzero(road_image))
            raw_road_image = decode_binary_rle(
                getattr(message, "raw_road_rle", ()) or message.road_rle,
                message.image_height, message.image_width)
            lane_image = decode_binary_rle(
                message.lane_rle, message.image_height, message.image_width)
            refined_road = project_mask_to_bev(road_image, self.map_x, self.map_y)
            # Stage-diagnostics: pixel count right after BEV projection,
            # before DirectBevPlanner.plan()'s own internal preprocessing
            # (small-component removal / morphology / hole-fill) runs.
            projected_road_pixels = int(np.count_nonzero(refined_road))
            raw_road = project_mask_to_bev(raw_road_image, self.map_x, self.map_y)
            lane = project_mask_to_bev(lane_image, self.map_x, self.map_y)
            result = self.planner.plan(refined_road, lane,
                                       message.header.stamp.sec+
                                       message.header.stamp.nanosec*1e-9)
        except Exception as error:
            self.get_logger().error(f"direct BEV frame failed: {error}")
            self._handle_invalid(
                stamp_ns, f"PROCESSING_ERROR:{type(error).__name__}")
            return
        elapsed = (time.perf_counter()-started)*1000.0
        self.planner_processing_rate.tick()
        try:
            refinement_diagnostics = json.loads(
                getattr(message, "refinement_diagnostics_json", "") or "{}")
        except (TypeError, ValueError):
            refinement_diagnostics = {"error": "INVALID_REFINEMENT_DIAGNOSTICS"}
        # Stage-diagnostics: reuse camera_yolo_inference_node's own
        # raw_pixels.road / refined_pixels.road counts (already computed
        # there, already shipped over the wire in refinement_diagnostics_json
        # -- not recomputed here) so the earliest two stages of the pipeline
        # (pre-refinement, post-refinement, both still on the YOLO side) are
        # visible alongside this node's own decode/projection/plan stages.
        raw_road_pixels = stage_pixel_count(refinement_diagnostics, "raw_pixels")
        refined_road_pixels = stage_pixel_count(refinement_diagnostics, "refined_pixels")
        end_to_end_ms = max(0.0, (
            self.get_clock().now().nanoseconds-stamp_ns)*1.0e-6)
        self.end_to_end_times.append(end_to_end_ms)
        latency_values = np.asarray(self.end_to_end_times, dtype=float)
        overlay_values = np.asarray(self.bev_overlay_times, dtype=float)
        sync_values = np.asarray(self.bev_sync_deltas_ms, dtype=float)
        result.diagnostics.update({
            "stamp_ns": stamp_ns, "source_stamp_ns": stamp_ns,
            "planner_variant": self.planner_variant,
            "calibration_state": self.state,
            "calibration_reasons": self.state_reasons,
            "path_valid": bool(result.valid),
            "confidence": float(result.confidence),
            # Stage-by-stage road pixel counts (per-stamp_ns, same frame):
            # raw/refined come from the YOLO node via refinement_diagnostics
            # (reused, not recomputed); decoded/projected were measured just
            # above; ego_component/safe_road are threaded through from
            # DirectBevPlanner.plan()'s own diagnostics (direct_bev_core.py)
            # since only that function has the component/safe masks in
            # scope. None of these six change existing keys -- see
            # "state"/"path_valid"/"confidence" etc. above, unchanged.
            "raw_road_pixels": raw_road_pixels,
            "refined_road_pixels": refined_road_pixels,
            "decoded_road_pixels": decoded_road_pixels,
            "projected_road_pixels": projected_road_pixels,
            "planner_state": result.diagnostics.get("state"),
            "path_point_count": int(len(result.points)),
            "camera_drive": float(self.latest_camera_drive),
            "camera_wheel": int(self.latest_camera_wheel),
            "processing_ms": elapsed,
            "processing_fps": 1000.0/max(1.0e-6, elapsed),
            "semantic_input_fps": self.semantic_input_rate.rate(),
            "planner_processing_fps": self.planner_processing_rate.rate(),
            "worker": self.worker.snapshot(),
            "bev_overlay_unique_fps": self.bev_overlay_rate.rate(),
            "bev_overlay_worker": self.bev_overlay_worker.snapshot(),
            "rgb_sync_exact": self.bev_sync_exact,
            "rgb_sync_nearest": self.bev_sync_nearest,
            "rgb_sync_miss": self.bev_sync_miss,
            "rgb_sync_delta_ms": ({
                "count": int(sync_values.size),
                "p50": float(np.percentile(sync_values, 50)),
                "p95": float(np.percentile(sync_values, 95)),
                "max": float(sync_values.max())}
                if sync_values.size else
                {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}),
            "bev_overlay_errors": self.bev_overlay_errors,
            "end_to_end_latency_ms": end_to_end_ms,
            "end_to_end_latency": {
                "count": int(latency_values.size),
                "p50_ms": float(np.percentile(latency_values, 50)),
                "p95_ms": float(np.percentile(latency_values, 95)),
                "max_ms": float(latency_values.max())},
            "bev_overlay_render": ({
                "count": int(overlay_values.size),
                "p50_ms": float(np.percentile(overlay_values, 50)),
                "p95_ms": float(np.percentile(overlay_values, 95)),
                "max_ms": float(overlay_values.max())}
                if overlay_values.size else {"count": 0, "p50_ms": 0.0,
                                             "p95_ms": 0.0, "max_ms": 0.0}),
            "refinement": refinement_diagnostics,
            "w_line_pixels": int((refinement_diagnostics.get(
                "raw_pixels", {}) or {}).get("white_line", 0) or 0),
            "y_line_pixels": int((refinement_diagnostics.get(
                "raw_pixels", {}) or {}).get("yellow_line", 0) or 0),
            "stop_line_confidence": float(getattr(
                message, "stop_line_confidence", 0.0)),
            "crosswalk_confidence": float(getattr(
                message, "crosswalk_confidence", 0.0)),
        })
        self.processing_times.append(elapsed)
        self.processing_times = self.processing_times[-120:]
        if (self.debug_enabled and self.debug_limiter.ready(time.monotonic()) and
                (self.overlay_pub.get_subscription_count() or
                 self.mask_pub.get_subscription_count() or
                 self.safe_pub.get_subscription_count())):
            try:
                self._publish_debug(message.header, raw_road, result)
            except Exception as error:
                self.get_logger().error(
                    f"direct BEV debug rendering failed: {error}")
        subscriber_count = self.bev_overlay_pub.get_subscription_count()
        if bev_overlay_requested(
                self.bev_overlay_enabled, subscriber_count,
                subscriber_count > 0 and
                self.bev_overlay_limiter.ready(time.monotonic())):
            match = self.camera_overlay_cache.nearest_match(
                stamp_ns, self.get_parameter(
                    "camera_overlay_sync_tolerance_sec").value,
                self.get_parameter("rgb_stale_timeout_sec").value)
            if match is not None:
                source = match.message
                if (int(source.width) != self.camera_model.width or
                        int(source.height) != self.camera_model.height or
                        (str(source.header.frame_id) and
                         str(source.header.frame_id) !=
                         str(message.header.frame_id))):
                    self.bev_diagnostic_reason = "FRAME_CONTRACT_INVALID"
                    self._handle_invalid(stamp_ns, "FRAME_CONTRACT_INVALID")
                    return
                self.bev_sync_deltas_ms.append(match.delta_ns*1.0e-6)
                if match.exact:
                    self.bev_sync_exact += 1
                else:
                    self.bev_sync_nearest += 1
                self.bev_diagnostic_reason = None
                def decode_optional(field):
                    values = getattr(message, field, ())
                    return (np.zeros_like(road_image) if not values else
                            decode_binary_rle(values, message.image_height,
                                              message.image_width))
                self.bev_overlay_worker.submit({
                    "source": source, "header": message.header,
                    "raw_road": raw_road, "refined_road": refined_road,
                    "white": project_mask_to_bev(
                        decode_optional("white_line_rle"), self.map_x, self.map_y),
                    "yellow": project_mask_to_bev(
                        decode_optional("yellow_line_rle"), self.map_x, self.map_y),
                    "restored": project_mask_to_bev(
                        decode_optional("restored_markings_rle"), self.map_x, self.map_y),
                    "result": result, "map_x": self.map_x, "map_y": self.map_y,
                })
            else:
                self.bev_sync_miss += 1
                self.bev_diagnostic_reason = "RGB_SYNC_MISS"
        if self.fixed_output_rate_enabled:
            generation = self.result_cache.replace(
                result.points, result.state, result.valid, result.confidence,
                result.diagnostics, stamp_ns)
            if result.valid:
                safely_render_camera_overlay(
                    lambda: self._refresh_fixed_overlay(
                        self.result_cache.snapshot()),
                    self._camera_overlay_error)
            result.diagnostics["cache_generation"] = generation
        else:
            self._publish_event_result(message.header, result)

    def _publish_event_result(self, header, result):
        if result.valid:
            self.path_pub.publish(self._path_message(header, result.points))
        self.valid_pub.publish(Bool(data=result.valid))
        self.confidence_pub.publish(Float32(data=float(result.confidence)))
        text = json.dumps(result.diagnostics, separators=(",", ":"))
        self.state_pub.publish(String(data=text))
        self.diag_pub.publish(String(data=text))
        if result.valid:
            safely_render_camera_overlay(
                lambda: self._maybe_publish_camera_overlay(
                    int(result.diagnostics["source_stamp_ns"]), result.points,
                    result.diagnostics),
                self._camera_overlay_error)

    def _maybe_publish_camera_overlay(self, stamp_ns, points, diagnostics):
        if (not self.camera_overlay_enabled or
                not self.camera_overlay_pub.get_subscription_count() or
                not self.camera_overlay_limiter.ready(time.monotonic())):
            return
        source = self.camera_overlay_cache.nearest(
            stamp_ns, self.get_parameter(
                "camera_overlay_sync_tolerance_sec").value)
        if source is None:
            return
        if (int(source.width) != self.camera_model.width or
                int(source.height) != self.camera_model.height):
            self._camera_overlay_error(ValueError(
                "camera overlay image dimensions do not match CameraInfo"))
            return
        overlay = safely_render_camera_overlay(
            lambda: self._render_camera_overlay(source, points, diagnostics),
            self._camera_overlay_error)
        if overlay is not None:
            self.camera_overlay_pub.publish(
                image_message(overlay, source.header, "bgr8"))

    def _render_camera_overlay(self, source, points, diagnostics=None):
        overlay = render_camera_path_overlay(
            source, points, self.camera_model, self.rotation, self.position,
            self.get_parameter("camera_overlay_line_thickness").value,
            self.get_parameter("camera_overlay_point_radius").value,
            self.get_parameter("camera_overlay_anchor_x_ratio").value,
            self.get_parameter("camera_overlay_anchor_y_ratio").value, True)
        data = diagnostics or {}
        reasons = data.get("reasons") or []
        lines = [
            f"{self.planner_variant} {data.get('mode', '')}/{data.get('state', '')}",
            f"reason={reasons[0] if reasons else 'NONE'}",
            f"steer={float(data.get('required_steering_deg') or 0.0):.1f} drive={self.latest_camera_drive:.1f} wheel={self.latest_camera_wheel}",
            f"sem={self.semantic_input_rate.rate():.1f} plan={self.planner_processing_rate.rate():.1f} FPS lat={float(data.get('end_to_end_latency_ms') or 0.0):.1f}ms",
        ]
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, 68), (0, 0, 0), -1)
        for index, value in enumerate(lines):
            cv2.putText(overlay, value, (5, 15+16*index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return overlay

    def _refresh_fixed_overlay(self, snapshot):
        with self.camera_overlay_render_lock:
            current = self.result_cache.snapshot()
            if (snapshot is None or current is None or
                    current.generation != snapshot.generation):
                return current
            return self._refresh_fixed_overlay_locked(current)

    def _refresh_fixed_overlay_locked(self, snapshot):
        if (snapshot is None or not snapshot.valid or
                not self.camera_overlay_enabled or
                not self.camera_overlay_pub.get_subscription_count()):
            return snapshot
        source = self.camera_overlay_cache.nearest(
            snapshot.source_stamp_ns, self.get_parameter(
                "camera_overlay_sync_tolerance_sec").value)
        if source is None:
            return snapshot
        source_stamp_ns = message_stamp_ns(source)
        key = (snapshot.generation, source_stamp_ns)
        if snapshot.overlay_key == key:
            return snapshot
        if (int(source.width) != self.camera_model.width or
                int(source.height) != self.camera_model.height):
            raise ValueError(
                "camera overlay image dimensions do not match CameraInfo")
        overlay = self._render_camera_overlay(
            source, snapshot.points, snapshot.diagnostics)
        message = image_message(overlay, source.header, "bgr8")
        self.result_cache.set_overlay(
            snapshot.generation, message, source_stamp_ns, key)
        return self.result_cache.snapshot()

    def _fixed_output_tick(self):
        now_wall = time.monotonic()
        now_message = self.get_clock().now().to_msg()
        publish_stamp_ns = (int(now_message.sec)*1_000_000_000+
                            int(now_message.nanosec))
        snapshot = self.result_cache.snapshot()
        generation = 0 if snapshot is None else snapshot.generation
        # A fixed-rate timer may poll the atomic cache, but it must never turn
        # one source frame into many apparently new path/overlay frames.
        if self.last_output_generation == generation:
            return
        if snapshot is not None and snapshot.valid:
            snapshot = safely_render_camera_overlay(
                lambda: self._refresh_fixed_overlay(snapshot),
                self._camera_overlay_error) or snapshot
        path_valid, reason, age = evaluate_fixed_result(
            snapshot, now_wall, self._calibration_ready(),
            self.get_parameter("path_stale_timeout_sec").value,
            self.get_parameter("state_stale_timeout_sec").value,
            self.planner.config.maximum_steering_deg)
        overlay_valid = bool(
            path_valid and snapshot.overlay_message is not None and
            age <= float(self.get_parameter(
                "camera_overlay_stale_timeout_sec").value) and
            self.camera_overlay_enabled and
            self.camera_overlay_pub.get_subscription_count())

        self.state_output_rate.tick(now_wall)
        if path_valid:
            self.path_output_rate.tick(now_wall)
        if overlay_valid:
            self.camera_overlay_output_rate.tick(now_wall)
        reused = False
        diagnostics = ({} if snapshot is None else
                       dict(snapshot.diagnostics))
        diagnostics.update({
            "stamp_ns": publish_stamp_ns,
            "source_stamp_ns": 0 if snapshot is None else
            snapshot.source_stamp_ns,
            "publish_stamp_ns": publish_stamp_ns,
            "source_age_sec": age,
            "semantic_input_fps": self.semantic_input_rate.rate(),
            "planner_processing_fps": self.planner_processing_rate.rate(),
            "path_output_fps": self.path_output_rate.rate(),
            "state_output_fps": self.state_output_rate.rate(),
            "camera_overlay_output_fps": self.camera_overlay_output_rate.rate(),
            "reused_cached_result": reused,
            "cache_generation": generation,
            "overlay_source_stamp_ns": (0 if snapshot is None else
                                         snapshot.overlay_source_stamp_ns),
            "path_valid": path_valid,
            "confidence": (float(snapshot.confidence) if path_valid else 0.0),
        })
        if not path_valid:
            diagnostics.update({"mode": "INVALID", "state": "INVALID",
                                "reasons": [reason]})
        self.last_output_generation = generation

        header = Header(); header.stamp = now_message
        if path_valid:
            self.path_pub.publish(self._path_message(header, snapshot.points))
        self.valid_pub.publish(Bool(data=path_valid))
        self.confidence_pub.publish(Float32(
            data=float(snapshot.confidence) if path_valid else 0.0))
        text = json.dumps(diagnostics, separators=(",", ":"))
        self.state_pub.publish(String(data=text))
        self.diag_pub.publish(String(data=text))
        if overlay_valid:
            snapshot.overlay_message.header.stamp = now_message
            self.camera_overlay_pub.publish(snapshot.overlay_message)

    def _camera_overlay_error(self, error):
        now = time.monotonic()
        if (self.camera_overlay_last_error_wall is None or
                now-self.camera_overlay_last_error_wall >= 5.0):
            self.get_logger().error(f"camera overlay rendering skipped: {error}")
            self.camera_overlay_last_error_wall = now

    @staticmethod
    def _path_message(header, points):
        path = Path(); path.header = header; path.header.frame_id = "base_link"
        for index, (x_m, y_m) in enumerate(points):
            pose = PoseStamped(); pose.header = path.header
            pose.pose.position.x = float(x_m); pose.pose.position.y = float(y_m)
            yaw = path_heading(points, index)
            pose.pose.orientation.z = math.sin(0.5*yaw)
            pose.pose.orientation.w = math.cos(0.5*yaw)
            path.poses.append(pose)
        return path

    def _publish_debug(self, header, raw, result):
        overlay = np.zeros((*raw.shape, 3), np.uint8)
        overlay[raw > 0] = (70, 20, 20)
        overlay[result.road > 0] = (120, 60, 20)
        overlay[result.component > 0] = (50, 100, 50)
        overlay[result.safe_road > 0] = (20, 150, 20)
        for points, color in ((result.left, (255, 255, 255)),
                              (result.right, (0, 255, 255)),
                              (result.rejected, (0, 0, 255)),
                              (result.points, (255, 0, 255))):
            for row, col in self.planner.metric_to_grid(points):
                if 0 <= row < overlay.shape[0] and 0 <= col < overlay.shape[1]:
                    cv2.circle(overlay, (int(col), int(row)), 2, color, -1)
        for key, color in (("road_left_boundary", (255, 80, 0)),
                           ("road_right_boundary", (0, 120, 255))):
            boundary = np.asarray(result.diagnostics.get(key, []), float).reshape(-1, 2)
            grid = self.planner.metric_to_grid(boundary) if len(boundary) else []
            if len(grid):
                pixels = np.column_stack((grid[:, 1], grid[:, 0])).astype(np.int32)
                cv2.polylines(overlay, [pixels], False, color, 2, cv2.LINE_AA)
        target = result.diagnostics.get("target_point")
        if target is not None:
            row, col = self.planner.metric_to_grid([target])[0]
            if 0 <= row < overlay.shape[0] and 0 <= col < overlay.shape[1]:
                cv2.circle(overlay, (int(col), int(row)), 5, (255, 255, 0), 1)
        # Vehicle footprint at the near edge of the configured raster.
        half = int(round(0.5*self.planner.config.vehicle_width_m /
                         self.planner.config.resolution_m))
        center = int(round((0.0-self.planner.config.y_min_m) /
                           self.planner.config.resolution_m))
        cv2.rectangle(overlay, (center-half, overlay.shape[0]-8),
                      (center+half, overlay.shape[0]-1), (255, 0, 0), 1)
        steering_value = result.diagnostics.get("required_steering_deg")
        try:
            steering_value = float(steering_value)
        except (TypeError, ValueError):
            steering_value = 0.0
        if not math.isfinite(steering_value):
            steering_value = 0.0
        cv2.putText(
            overlay,
            f"{result.diagnostics.get('path_source', result.mode)}/{result.state} steer={steering_value:.1f}",
            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (255, 255, 255), 1)
        self.overlay_pub.publish(image_message(overlay, header, "bgr8"))
        masks = np.dstack((raw*255, result.road*255, result.component*255))
        self.mask_pub.publish(image_message(masks, header, "bgr8"))
        self.safe_pub.publish(image_message(result.safe_road*255, header, "mono8"))

    @staticmethod
    def _blend_mask(image, mask, color, alpha):
        selected = np.asarray(mask) > 0
        if np.any(selected):
            values = image[selected].astype(np.float32)
            image[selected] = np.clip(
                (1.0-alpha)*values+alpha*np.asarray(color, np.float32),
                0, 255).astype(np.uint8)

    def _publish_bev_overlay(self, job):
        """Warp RGB with the exact planner remap and draw aligned BEV layers."""
        if (not self.bev_overlay_enabled or
                self.bev_overlay_pub.get_subscription_count() == 0):
            return
        started = time.perf_counter()
        try:
            source = image_to_bgr(job["source"])
            overlay = warp_rgb_to_bev(source, job["map_x"], job["map_y"])
            result = job["result"]
            self._blend_mask(overlay, job["raw_road"], (180, 70, 30), 0.18)
            self._blend_mask(overlay, job["refined_road"], (30, 170, 40), 0.18)
            self._blend_mask(overlay, result.safe_road, (20, 220, 20), 0.22)
            self._blend_mask(overlay, job["restored"], (255, 180, 0), 0.55)
            self._blend_mask(overlay, job["white"], (255, 255, 255), 0.85)
            self._blend_mask(overlay, job["yellow"], (0, 230, 255), 0.85)
            grid = self.planner.metric_to_grid(result.points)
            if len(grid):
                pixels = np.column_stack((grid[:, 1], grid[:, 0])).astype(np.int32)
                swept = np.zeros(overlay.shape[:2], np.uint8)
                swept_half = max(1, int(round(
                    (0.5*self.planner.config.vehicle_width_m+
                     self.planner.config.lateral_safety_margin_m) /
                    self.planner.config.resolution_m)))
                cv2.polylines(swept, [pixels], False, 255,
                              2*swept_half+1, cv2.LINE_AA)
                contours, _ = cv2.findContours(swept, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 180, 40), 1,
                                 cv2.LINE_AA)
                cv2.polylines(overlay, [pixels], False, (255, 0, 255), 3,
                              cv2.LINE_AA)
            for key, color in (("road_left_boundary", (255, 80, 0)),
                               ("road_right_boundary", (0, 120, 255))):
                boundary = np.asarray(
                    result.diagnostics.get(key, []), float).reshape(-1, 2)
                boundary_grid = (self.planner.metric_to_grid(boundary)
                                 if len(boundary) else np.empty((0, 2), int))
                if len(boundary_grid):
                    boundary_pixels = np.column_stack(
                        (boundary_grid[:, 1], boundary_grid[:, 0])).astype(np.int32)
                    cv2.polylines(overlay, [boundary_pixels], False, color, 2,
                                  cv2.LINE_AA)
            target = result.diagnostics.get("target_point")
            if target is not None:
                target_grid = self.planner.metric_to_grid([target])[0]
                row, col = int(target_grid[0]), int(target_grid[1])
                if 0 <= row < overlay.shape[0] and 0 <= col < overlay.shape[1]:
                    cv2.circle(overlay, (col, row), 5, (255, 255, 0), 1,
                               cv2.LINE_AA)
            center = int(round((0.0-self.planner.config.y_min_m)/
                               self.planner.config.resolution_m))
            base_y = overlay.shape[0]-2
            cv2.circle(overlay, (center, base_y), 5, (255, 80, 0), -1)
            cv2.arrowedLine(overlay, (center, base_y),
                            (center, max(0, base_y-24)), (255, 80, 0), 2,
                            tipLength=0.3)
            reasons = result.diagnostics.get("reasons", [])
            steering = result.diagnostics.get("required_steering_deg", 0.0)
            lines = [f"{self.planner_variant} {result.diagnostics.get('path_source', result.mode)}/{result.state}",
                     f"steer={float(steering or 0.0):.1f} look={float(result.diagnostics.get('lookahead_m') or 0.0):.1f}m",
                     f"drive={self.latest_camera_drive:.1f} wheel={self.latest_camera_wheel}",
                     f"sem={self.semantic_input_rate.rate():.1f} plan={self.planner_processing_rate.rate():.1f} FPS",
                     f"lat={float(result.diagnostics.get('end_to_end_latency_ms') or 0.0):.1f}ms",
                     "reason="+(str(reasons[0]) if reasons else "NONE")]
            for index, value in enumerate(lines):
                cv2.putText(overlay, value, (5, 16+16*index),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                            cv2.LINE_AA)
            self.bev_overlay_pub.publish(
                image_message(overlay, job["header"], "bgr8"))
            self.bev_overlay_rate.tick()
            self.last_bev_overlay_wall = time.monotonic()
            self.bev_overlay_times.append(
                (time.perf_counter()-started)*1000.0)
        except Exception as error:
            self.bev_overlay_errors += 1
            self.bev_diagnostic_reason = "OVERLAY_RENDER_ERROR"
            self._camera_overlay_error(error)

    def _calibration_wait_reason(self):
        if self.camera_model is None:
            return "WAITING_CAMERA_INFO"
        if not self.estimator.locked and not self._imu_available():
            return "WAITING_IMU"
        return "CALIBRATION_INVALID"

    def _current_diagnostic_reason(self):
        if self.camera_model is None:
            return "WAITING_CAMERA_INFO"
        if not self._calibration_ready():
            return self._calibration_wait_reason()
        if self.last_semantic_wall is None:
            return "WAITING_SEMANTIC"
        return self.bev_diagnostic_reason

    def _publish_diagnostic_overlay(self):
        if (not self.bev_overlay_enabled or
                self.bev_overlay_pub.get_subscription_count() == 0):
            return
        reason = self._current_diagnostic_reason()
        if reason is None:
            return
        latest = self.camera_overlay_cache.latest(
            self.get_parameter("rgb_stale_timeout_sec").value)
        try:
            if latest is None:
                overlay = render_diagnostic_overlay(
                    None, (self.planner.rows, self.planner.cols), reason)
                header = Header(); header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "bev"
            else:
                source = image_to_bgr(latest)
                if self._calibration_ready():
                    self._ensure_remap()
                    overlay = warp_rgb_to_bev(source, self.map_x, self.map_y)
                    overlay = (overlay.astype(np.float32)*0.30).astype(np.uint8)
                    cv2.rectangle(overlay, (0, 0),
                                  (overlay.shape[1]-1, 34), (0, 0, 0), -1)
                    cv2.putText(overlay, reason, (5, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 180, 255), 1, cv2.LINE_AA)
                else:
                    overlay = render_diagnostic_overlay(
                        latest, (self.planner.rows, self.planner.cols), reason)
                header = latest.header
            self.bev_overlay_pub.publish(image_message(overlay, header, "bgr8"))
        except Exception as error:
            self.bev_overlay_errors += 1
            self._camera_overlay_error(error)

    def _publish_invalid(self, stamp_ns, reason):
        self.valid_pub.publish(Bool(data=False))
        self.confidence_pub.publish(Float32(data=0.0))
        data = {"stamp_ns": int(stamp_ns), "mode": "INVALID",
                "state": "INVALID", "reasons": [reason],
                "planner_variant": self.planner_variant,
                "camera_drive": float(self.latest_camera_drive),
                "camera_wheel": int(self.latest_camera_wheel),
                "calibration_state": self.state,
                "calibration_reasons": self.state_reasons,
                "bev_overlay_diagnostic": self._current_diagnostic_reason(),
                "rgb_sync_exact": self.bev_sync_exact,
                "rgb_sync_nearest": self.bev_sync_nearest,
                "rgb_sync_miss": self.bev_sync_miss,
                "bev_overlay_errors": self.bev_overlay_errors,
                # Stage-diagnostics: this path fires before any semantic
                # frame is decoded/projected/planned (INPUT_TIMEOUT,
                # CALIBRATION_INVALID, FRAME_CONTRACT_INVALID,
                # PROCESSING_ERROR) -- no mask exists for any of these six
                # stages, so they are explicitly null rather than fabricated.
                # semantic_input_fps/planner_processing_fps are still real
                # (rolling rate trackers, not tied to this specific call).
                "raw_road_pixels": None, "refined_road_pixels": None,
                "decoded_road_pixels": None, "projected_road_pixels": None,
                "ego_component_pixels": None, "safe_road_pixels": None,
                "planner_state": "INVALID",
                "semantic_input_fps": self.semantic_input_rate.rate(),
                "planner_processing_fps": self.planner_processing_rate.rate(),
                "end_to_end_latency_ms": None}
        text = json.dumps(data, separators=(",", ":"))
        self.state_pub.publish(String(data=text))
        self.diag_pub.publish(String(data=text))

    def _handle_invalid(self, stamp_ns, reason):
        if self.fixed_output_rate_enabled:
            data = {"stamp_ns": int(stamp_ns),
                    "source_stamp_ns": int(stamp_ns),
                    "mode": "INVALID", "state": "INVALID",
                    "reasons": [reason],
                    "calibration_state": self.state,
                    "calibration_reasons": self.state_reasons,
                    "path_valid": False, "confidence": 0.0}
            self.result_cache.replace(
                np.empty((0, 2), float), "INVALID", False, 0.0,
                data, stamp_ns)
        else:
            self._publish_invalid(stamp_ns, reason)

    def _check_timeout(self):
        if (self.last_semantic_wall is not None and
                time.monotonic()-self.last_semantic_wall >
                float(self.get_parameter("input_timeout_sec").value)):
            if self.fixed_output_rate_enabled:
                self._handle_invalid(0, "INPUT_TIMEOUT")
            else:
                self._publish_invalid(0, "INPUT_TIMEOUT")
            self.last_semantic_wall = None

    def destroy_node(self):
        self.worker.close()
        self.bev_overlay_worker.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DirectBevPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
