"""ROS adapter for synchronized semantic masks and image-space paths."""
import json
import time
from collections import OrderedDict, deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int8, String
from race_interfaces.msg import ImagePath, ImagePathPoint, SemanticPathFrame
from .semantic_path_contract import decode_binary_rle
from .unique_frame_rates import UniqueFrameRates
from .overlay_worker import LatestOnlyWorker, OverlayRateLimiter
from .timestamp_sync import (TimestampedMessageCache, message_stamp_ns,
                             subscription_transition)


def path_computation_enabled(control_mode, visualization_only,
                             require_control_mode=True):
    return (not bool(require_control_mode) or int(control_mode) == 1 or
            bool(visualization_only))

from .image_path_planner import (ImagePathPlanner, PlannerConfig, INACTIVE,
                                 BOTH_BOUNDARIES, LEFT_BOUNDARY, RIGHT_BOUNDARY,
                                 ROAD_CENTER, TEMPORAL_FALLBACK)


class CameraImagePathNode(Node):
    def __init__(self):
        super().__init__("camera_image_path_node")
        defaults = {
            "vehicle_center_x_px": 320.0,
            "roi_top": 120, "roi_bottom": 475, "sample_interval_px": 10,
            "sample_band_half_height_px": 2, "seed_half_width_px": 80,
            "seed_height_px": 24, "minimum_component_pixels": 20,
            "maximum_lateral_jump_px": 90.0, "road_containment_tolerance_px": 8,
            "exclusion_overlap_ratio": 0.35, "polynomial_degree": 2,
            "temporal_alpha": 0.55, "maximum_temporal_shift_px": 55.0,
            "temporal_straight_alpha": 0.28,
            "temporal_curve_alpha": 0.72,
            "temporal_small_shift_px": 6.0,
            "temporal_hysteresis_frames": 3,
            "temporal_consistent_boost": 0.30,
            "temporal_reacquire_alpha": 0.40,
            "temporal_state_timeout_sec": 0.5,
            "max_temporal_fallback_frames": 5,
            "temporal_fallback_confidence_decay": 0.15,
            "max_single_boundary_fallback_frames": 5, "both_weight": 1.0,
            "single_weight": 0.72, "road_weight": 0.48, "temporal_weight": 0.25,
            "valid_min_points": 6, "valid_min_both_ratio": 0.0,
            "valid_min_single_boundary_points": 5,
            "valid_min_confidence": 0.45,
            "valid_min_continuity_score": 0.65,
            "road_validation_min_rows": 3,
            "valid_min_road_containment_ratio": 0.5,
            "fit_overshoot_margin_px": 20.0,
            "center_direction_outlier_px": 30.0,
            "road_gross_outlier_margin_px": 35.0,
            "polynomial_residual_scale_px": 18.0,
            "curve_moderate_threshold_px": 0.2,
            "curve_sharp_threshold_px": 1.5,
            "lookahead_straight_ratio": 0.75,
            "lookahead_curve_ratio": 0.52,
            "lookahead_sharp_ratio": 0.32,
            "lane_width_min_px": 20.0,
            "lane_width_max_px": 600.0,
            "lane_width_update_alpha": 0.2,
            "nominal_lane_width_m": 3.0,
            "minimum_boundary_clearance_m": 1.5,
            "vehicle_width_m": 0.46,
            "vehicle_boundary_margin_m": 0.1,
            "width_profile_lookup_radius_px": 30.0,
            "lane_width_relative_tolerance": 0.5,
            "lane_width_absolute_tolerance_px": 30.0,
            "lane_width_seed_px": 0.0,
            "road_edge_clip_margin_px": 3,
            "road_clipped_weight_scale": 0.55,
            "road_near_field_height_px": 80,
            "road_minimum_near_coverage_ratio": 0.55,
            "road_minimum_near_width_px": 60.0,
            "road_width_max_relative_deviation": 0.45,
            "road_width_outlier_window_rows": 2,
            "road_center_spike_px": 35.0,
            "road_single_center_disagreement_ratio": 0.10,
            "road_single_boundary_blend": 0.70,
            "road_transition_smoothing_alpha": 0.35,
            "road_transition_smoothing_max_px": 24.0,
            "road_min_width_stability_score": 0.55,
            "road_min_center_score": 0.45,
            "road_branch_min_rows": 3,
            "road_branch_expansion_ratio": 2.2,
            "road_branch_gap_tolerance_px": 14,
            "branch_confirm_frames": 1,
            "ego_exclusion_enabled": True,
            "ego_exclusion_bottom_ratio": 0.12,
            "ego_exclusion_polygon": [0.16, 1.00, 0.84, 1.00,
                                      0.72, 0.86, 0.28, 0.86],
            "ego_exclusion_branch_only": False,
            "road_hole_fill_enabled": True,
            "road_hole_close_kernel_px": 9,
            "road_hole_max_area_px": 8000,
            "marking_suppression_enabled": True,
            "marking_max_row_width_px": 18.0,
            "marking_min_length_px": 40.0,
            "marking_edge_margin_px": 18.0,
            "heading_jump_threshold_deg": 35.0,
            "curvature_jump_threshold_px": 45.0,
            "max_local_outlier_ratio": 0.35,
            "temporal_stale_lock_max_frames": 3,
            "temporal_stale_recovery_alpha": 0.85,
            "path_corridor_enabled": True,
            "path_corridor_near_half_width_ratio": 0.14,
            "path_corridor_mid_half_width_ratio": 0.22,
            "path_corridor_far_half_width_ratio": 0.32,
            "path_corridor_min_half_width_px": 28.0,
            "path_corridor_previous_path_weight": 0.7,
            "path_corridor_expand_when_lost": True,
            "path_corridor_max_expand_ratio": 2.5,
            "path_corridor_expand_step": 0.25,
            "path_corridor_shrink_step": 0.15,
            "boundary_track_filter_enabled": True,
            "boundary_min_vertical_span_ratio": 0.12,
            "boundary_max_lateral_slope_change": 0.9,
            "boundary_temporal_tolerance_px": 40.0,
            "boundary_min_track_score": 0.5,
            "final_curvature_jump_px": 60.0,
            "target_path_points": 24,
            "minimum_drivable_horizon_ratio": 0.50,
            "max_gap_repair_rows": 4,
            "maximum_steering_deg": 22.0,
            "steering_lookahead_y_ratio": 0.50,
            "steering_proportional_gain_deg_per_norm": 22.0,
            "steering_derivative_gain_deg_per_norm_per_s": 2.0,
            "max_steering_rate_deg_per_sec": 180.0,
            "max_steering_delta_deg_per_frame": 12.0,
            "max_steering_delta_deg_per_segment": 12.0,
            "steering_repair_previous_weight": 0.65,
            "nominal_frame_period_sec": 0.05,
            "valid_min_road_only_points": 4,
            "source_confirm_frames": 2,
            "source_release_frames": 3,
            "source_switch_max_lateral_px": 45.0,
            "final_path_safety_margin_near_px": 12.0,
            "final_path_safety_margin_mid_px": 8.0,
            "final_path_safety_margin_far_px": 4.0,
            "continuity_hard_invalid_floor": 0.15,
            "steering_only_validity": False,
            "semantic_path_frame_topic": "/perception/semantic_path_frame",
            "control_mode_topic": "/mission/control_mode",
            "require_control_mode": True,
            "visualization_only": False,
            "input_image_topic": "/camera/image_raw",
            "path_overlay_max_fps": 45.0,
            "rgb_sync_tolerance_sec": 0.08,
            "rgb_cache_frames": 60,
            "rgb_stale_timeout_sec": 0.50,
            "rgb_pending_frames": 12,
        }
        for name, value in defaults.items(): self.declare_parameter(name, value)
        config_names = [name for name in defaults if name not in
                        ("semantic_path_frame_topic", "control_mode_topic",
                         "require_control_mode", "visualization_only", "input_image_topic",
                         "path_overlay_max_fps", "rgb_sync_tolerance_sec",
                         "rgb_cache_frames", "rgb_stale_timeout_sec",
                         "rgb_pending_frames")]
        config_values = {name: self.get_parameter(name).value for name in config_names}
        # ROS parameters cannot carry a list-of-pairs; ego_exclusion_polygon
        # is declared as a flat [x0, y0, x1, y1, ...] double array and
        # reassembled into the (x_ratio, y_ratio) tuple PlannerConfig expects.
        flat_polygon = list(config_values.get("ego_exclusion_polygon", ()))
        if len(flat_polygon) % 2 != 0:
            raise ValueError("ego_exclusion_polygon must have an even number of values")
        config_values["ego_exclusion_polygon"] = tuple(
            (float(flat_polygon[i]), float(flat_polygon[i+1]))
            for i in range(0, len(flat_polygon), 2))
        self.planner = ImagePathPlanner(PlannerConfig(**config_values))
        self.control_mode = 0
        self.latencies = deque(maxlen=300)
        self.processing_latencies = deque(maxlen=300); self.frame_count = 0
        self.overlay_latencies = deque(maxlen=300)
        self.end_to_end_latencies = deque(maxlen=300)
        self.overlay_missing_rgb = 0
        self.overlay_sync_exact = 0
        self.overlay_sync_nearest = 0
        self.overlay_sync_deltas_ms = deque(maxlen=600)
        self.overlay_errors = 0
        self.started = None
        self.received_count = 0; self.accepted_count = 0; self.published_count = 0
        self.duplicate_count = 0; self.stale_count = 0
        self.last_processed_stamp = None; self.last_near = None
        self.rates = UniqueFrameRates()
        self.rgb_cache = TimestampedMessageCache(
            self.get_parameter("rgb_cache_frames").value)
        self.pending_overlays = OrderedDict()
        self.upstream_fps = {}
        semantic_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.rgb_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(
            SemanticPathFrame,
            self.get_parameter("semantic_path_frame_topic").value,
            self.on_semantic_frame, semantic_qos)
        self.create_subscription(Int8, self.get_parameter("control_mode_topic").value,
                                 self.on_control_mode, 10)
        if float(self.get_parameter("path_overlay_max_fps").value) <= 0.0:
            raise ValueError("path_overlay_max_fps must be positive")
        self._rgb_subscription = None
        self.create_subscription(String, "/camera/realtime_fps", self.on_upstream_fps, 10)
        self.typed_path_pub = self.create_publisher(ImagePath, "/camera/image_path_typed", 10)
        self.path_pub = self.create_publisher(String, "/camera/image_path", 10)
        self.valid_pub = self.create_publisher(Bool, "/camera/image_path_valid", 10)
        self.confidence_pub = self.create_publisher(Float32, "/camera/image_path_confidence", 10)
        self.state_pub = self.create_publisher(String, "/camera/image_path_state", 10)
        self.owner_pub = self.create_publisher(String, "/camera/path_ownership", 10)
        # Canonical scalar-health aliases consumed by autonomy_output_node /
        # course_mission_node / camera_path_controller_node. These carry real,
        # non-fabricated signals (validity/confidence/mode derived from the same
        # result as /camera/image_path_*); they do NOT imply a metric geometric
        # /camera/path (nav_msgs/Path) exists — see camera_pure_pursuit.launch.py
        # for why that leg is intentionally left unwired pending calibration.
        self.canonical_valid_pub = self.create_publisher(Bool, "/camera/path_valid", 10)
        self.canonical_confidence_pub = self.create_publisher(Float32, "/camera/path_confidence", 10)
        self.canonical_mode_pub = self.create_publisher(Int8, "/camera/path_mode", 10)
        visualization_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.debug_pub = self.create_publisher(
            Image, "/camera/path_debug_image", visualization_qos)
        overlay_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=QoSReliabilityPolicy.RELIABLE,
                                 durability=QoSDurabilityPolicy.VOLATILE)
        self.overlay_pub = self.create_publisher(
            Image, "/camera/path_overlay_image", overlay_qos)
        self.metrics_pub = self.create_publisher(String, "/camera/path_metrics", 10)
        self.realtime_pub = self.create_publisher(String, "/camera/path_realtime_fps", 10)
        self.overlay_limiter = OverlayRateLimiter(
            self.get_parameter("path_overlay_max_fps").value)
        self.overlay_worker = LatestOnlyWorker(
            self._publish_path_overlay, "path-overlay")
        self.create_timer(1.0, self.publish_realtime_fps)
        self.create_timer(0.25, self._update_overlay_subscription)

    def on_control_mode(self, message):
        self.control_mode = int(message.data)
        if (self.get_parameter("require_control_mode").value and
                self.control_mode != 1):
            self.planner.reset()

    def on_rgb(self, message):
        self.rgb_cache.add(message)
        self._resolve_pending_overlays(message_stamp_ns(message))

    def _rgb_match(self, stamp_ns):
        return self.rgb_cache.nearest_match(
            stamp_ns, self.get_parameter("rgb_sync_tolerance_sec").value,
            self.get_parameter("rgb_stale_timeout_sec").value)

    def _record_rgb_match(self, match):
        self.overlay_sync_deltas_ms.append(match.delta_ns*1.0e-6)
        if match.exact:
            self.overlay_sync_exact += 1
        else:
            self.overlay_sync_nearest += 1

    def _queue_overlay(self, semantic_message, result):
        key = message_stamp_ns(semantic_message)
        self.pending_overlays[key] = (semantic_message, result, time.monotonic())
        maximum = int(self.get_parameter("rgb_pending_frames").value)
        while len(self.pending_overlays) > maximum:
            self.pending_overlays.popitem(last=False)
            self.overlay_missing_rgb += 1

    def _resolve_pending_overlays(self, newest_rgb_stamp_ns=None):
        tolerance_ns = int(float(self.get_parameter(
            "rgb_sync_tolerance_sec").value)*1.0e9)
        stale_sec = float(self.get_parameter("rgb_stale_timeout_sec").value)
        now = time.monotonic()
        for key, (_semantic, result, queued_wall) in list(
                self.pending_overlays.items()):
            match = self._rgb_match(key)
            if match is not None:
                self._record_rgb_match(match)
                if self.overlay_limiter.ready(now):
                    self.overlay_worker.submit((match.message, result))
                del self.pending_overlays[key]
            elif (now-queued_wall > stale_sec or
                  (newest_rgb_stamp_ns is not None and
                   newest_rgb_stamp_ns > key+tolerance_ns)):
                self.overlay_missing_rgb += 1
                del self.pending_overlays[key]

    def _update_overlay_subscription(self):
        action = subscription_transition(
            self.overlay_pub.get_subscription_count(),
            self._rgb_subscription is not None)
        if action == "CREATE":
            self._rgb_subscription = self.create_subscription(
                Image, self.get_parameter("input_image_topic").value,
                self.on_rgb, self.rgb_qos)
        elif action == "DESTROY":
            self.destroy_subscription(self._rgb_subscription)
            self._rgb_subscription = None
            self.rgb_cache.clear()
            self.pending_overlays.clear()
            self.overlay_worker.clear()
            self.overlay_limiter.reset()

    def on_upstream_fps(self, message):
        try: self.upstream_fps = json.loads(message.data)
        except (ValueError, TypeError): self.upstream_fps = {}

    def on_semantic_frame(self, message):
        callback_started = time.perf_counter()
        self.received_count += 1
        key = int(message.header.stamp.sec)*1_000_000_000 + int(message.header.stamp.nanosec)
        if self.last_processed_stamp is not None:
            if key == self.last_processed_stamp: self.duplicate_count += 1; return
            if key < self.last_processed_stamp: self.stale_count += 1; return
        self.last_processed_stamp = key
        self.accepted_count += 1
        self.rates.mark("semantic_received", message.header.stamp.sec,
                        message.header.stamp.nanosec)
        height, width = int(message.image_height), int(message.image_width)
        decode_started = time.perf_counter()
        masks = [decode_binary_rle(values, height, width) for values in
                 (message.road_rle, message.lane_rle, message.words_rle,
                  message.stop_line_rle, message.c_line_rle)]
        decode_ms = (time.perf_counter()-decode_started)*1000.0
        self._current_rgb_match = self._rgb_match(key)
        self._path_callback_started = callback_started
        self.process(message, masks, decode_ms)

    def process(self, semantic_message, masks, decode_ms):
        callback_started = getattr(
            self, "_path_callback_started", time.perf_counter())
        rgb_match = getattr(self, "_current_rgb_match", None)
        rgb_message = None if rgb_match is None else rgb_match.message
        if self.started is None: self.started = time.monotonic()
        road, lane, words, stop, c_line = masks
        compute_enabled = path_computation_enabled(
            self.control_mode, self.get_parameter("visualization_only").value,
            self.get_parameter("require_control_mode").value)
        if not compute_enabled:
            result = self.planner.inactive(road.shape); owner = "GPS_PATH_OWNER"
        else:
            timestamp_sec = (float(semantic_message.header.stamp.sec) +
                             float(semantic_message.header.stamp.nanosec)*1.0e-9)
            result = self.planner.plan(
                road, lane, np.zeros_like(lane), words, stop, c_line,
                timestamp_sec=timestamp_sec)
            owner = "CAMERA_PATH_OWNER"
        diagnostics = result.diagnostics or {}
        self.frame_count += 1; self.latencies.append(result.latency_ms)
        self.rates.mark("path_processed", semantic_message.header.stamp.sec,
                        semantic_message.header.stamp.nanosec)
        points = [{"x_px": float(x), "y_px": float(y),
                   "x_norm": float(x/road.shape[1]), "y_norm": float(y/road.shape[0]),
                   "source": result.sources[i] if i < len(result.sources) else "UNKNOWN"}
                  for i, (x, y) in enumerate(result.points)]
        payload = {"header": {"stamp": {"sec": semantic_message.header.stamp.sec,
                                          "nanosec": semantic_message.header.stamp.nanosec},
                               "frame_id": semantic_message.header.frame_id},
                   "coordinate_frame": "IMAGE_PIXELS", "image_width": road.shape[1],
                   "image_height": road.shape[0], "path_valid": result.valid,
                   "path_confidence": result.confidence, "path_state": result.state,
                   "ownership": owner, "control_mode": self.control_mode,
                   "path_curvature_px": diagnostics.get("path_curvature_px", 0.0),
                   "recommended_lookahead_y_ratio": diagnostics.get(
                       "recommended_lookahead_y_ratio", 0.5),
                   "points": points}
        self.typed_path_pub.publish(self.typed_message(semantic_message, result, owner))
        self.path_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.valid_pub.publish(Bool(data=result.valid)); self.confidence_pub.publish(Float32(data=result.confidence))
        self.state_pub.publish(String(data=result.state)); self.owner_pub.publish(String(data=owner))
        state_mode = {"INACTIVE": 0, "INVALID": 1, "DEGRADED": 2, "VALID": 3}.get(result.state, 0)
        self.canonical_valid_pub.publish(Bool(data=result.valid))
        self.canonical_confidence_pub.publish(Float32(data=result.confidence))
        self.canonical_mode_pub.publish(Int8(data=state_mode))
        self.published_count += 1
        self.rates.mark("path_published", semantic_message.header.stamp.sec,
                        semantic_message.header.stamp.nanosec)
        metrics = self.metrics(result, owner); metrics["semantic_decode_ms"] = decode_ms
        metrics["candidate_survival"] = diagnostics
        metrics["confidence_components"] = result.confidence_components
        metrics["semantic_encoded_bytes"] = int(semantic_message.encoded_bytes)
        metrics["semantic_uncompressed_bytes"] = int(semantic_message.uncompressed_bytes)
        try:
            metrics["refinement"] = json.loads(
                getattr(semantic_message, "refinement_diagnostics_json", "") or "{}")
        except (TypeError, ValueError):
            metrics["refinement"] = {"error": "INVALID_REFINEMENT_DIAGNOSTICS"}
        metrics["stop_line_confidence"] = float(getattr(
            semantic_message, "stop_line_confidence", 0.0))
        metrics["crosswalk_confidence"] = float(getattr(
            semantic_message, "crosswalk_confidence", 0.0))
        self.metrics_pub.publish(String(data=json.dumps(metrics, separators=(",", ":"))))
        self.processing_latencies.append(
            (time.perf_counter()-callback_started)*1000.0)
        stamp_ns = (int(semantic_message.header.stamp.sec)*1_000_000_000+
                    int(semantic_message.header.stamp.nanosec))
        self.end_to_end_latencies.append(max(
            0.0, (self.get_clock().now().nanoseconds-stamp_ns)*1.0e-6))
        # Debug rendering and 640x480 byte serialization are intentionally lazy;
        # RQT enables them simply by subscribing without taxing normal driving.
        if self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(self.debug_image(semantic_message, result, owner, masks[2:]))
        if (self.overlay_pub.get_subscription_count() > 0 and rgb_message is not None and
                self.overlay_limiter.ready(time.monotonic())):
            self._record_rgb_match(rgb_match)
            # Rendering and Image serialization never block path publication.
            self.overlay_worker.submit((rgb_message, result))
        elif self.overlay_pub.get_subscription_count() > 0 and rgb_message is None:
            # RGB may arrive after semantic processing. Keep a bounded pending
            # result and retry from the RGB callback instead of permanently
            # losing this overlay frame.
            self._queue_overlay(semantic_message, result)

    @staticmethod
    def typed_message(source, result, owner):
        message = ImagePath(); message.header = source.header
        message.image_width = source.image_width; message.image_height = source.image_height
        message.path_valid = result.valid; message.path_confidence = result.confidence
        message.path_state = {"INVALID": ImagePath.STATE_INVALID, "VALID": ImagePath.STATE_VALID,
                              "DEGRADED": ImagePath.STATE_DEGRADED,
                              "INACTIVE": ImagePath.STATE_INACTIVE}[result.state]
        message.ownership = (ImagePath.OWNER_CAMERA if owner == "CAMERA_PATH_OWNER"
                             else ImagePath.OWNER_GPS)
        source_codes = {BOTH_BOUNDARIES: ImagePathPoint.BOTH_BOUNDARIES,
                        LEFT_BOUNDARY: ImagePathPoint.LEFT_BOUNDARY,
                        RIGHT_BOUNDARY: ImagePathPoint.RIGHT_BOUNDARY,
                        ROAD_CENTER: ImagePathPoint.ROAD_CENTER,
                        TEMPORAL_FALLBACK: ImagePathPoint.TEMPORAL_FALLBACK}
        for index, (x, y) in enumerate(result.points):
            point = ImagePathPoint(x_px=float(x), y_px=float(y),
                                   x_norm=float(x/source.image_width),
                                   y_norm=float(y/source.image_height),
                                   confidence=float(result.confidence),
                                   source=source_codes[result.sources[index]])
            message.points.append(point)
        return message

    def publish_realtime_fps(self):
        self._resolve_pending_overlays()
        path_rates=self.rates.snapshot("path_published")
        processed_rates=self.rates.snapshot("path_processed")
        overlay_rates=self.rates.snapshot("overlay_published")
        processing=np.asarray(self.processing_latencies,dtype=float)
        overlay_processing=np.asarray(self.overlay_latencies,dtype=float)
        end_to_end=np.asarray(self.end_to_end_latencies,dtype=float)
        sync=np.asarray(self.overlay_sync_deltas_ms,dtype=float)
        processing_summary=(
            {"count":int(processing.size),
             "p50":float(np.percentile(processing,50)),
             "p95":float(np.percentile(processing,95)),
             "max":float(processing.max())}
            if processing.size else {"count":0,"p50":0.0,"p95":0.0,"max":0.0})
        document={"semantic_received":self.rates.snapshot("semantic_received"),
                  "path_processed":processed_rates,
                  "path_published":path_rates,
                  "overlay_published":overlay_rates,
                  "path_unique_fps":path_rates,
                  "path_overlay_fps":overlay_rates,
                  "received_count":self.received_count,
                  "accepted_count":self.accepted_count,
                  "processed_count":self.frame_count,
                  "published_count":self.published_count,"stale":self.stale_count,
                  "duplicate":self.duplicate_count,
                  "backlog":max(0,self.accepted_count-self.frame_count),
                  "path_processing_ms":processing_summary,
                  "end_to_end_latency_ms":({"count":int(end_to_end.size),
                      "p50":float(np.percentile(end_to_end,50)),
                      "p95":float(np.percentile(end_to_end,95)),
                      "max":float(end_to_end.max())} if end_to_end.size else
                      {"count":0,"p50":0.0,"p95":0.0,"max":0.0}),
                  "overlay_worker":self.overlay_worker.snapshot(),
                  "overlay_missing_rgb":self.overlay_missing_rgb,
                  "rgb_sync_exact":self.overlay_sync_exact,
                  "rgb_sync_nearest":self.overlay_sync_nearest,
                  "rgb_sync_miss":self.overlay_missing_rgb,
                  "rgb_sync_delta_ms":({"count":int(sync.size),
                      "p50":float(np.percentile(sync,50)),
                      "p95":float(np.percentile(sync,95)),
                      "max":float(sync.max())} if sync.size else
                      {"count":0,"p50":0.0,"p95":0.0,"max":0.0}),
                  "overlay_errors":self.overlay_errors,
                  "overlay_render_ms":({"count":int(overlay_processing.size),
                      "p50":float(np.percentile(overlay_processing,50)),
                      "p95":float(np.percentile(overlay_processing,95)),
                      "max":float(overlay_processing.max())} if overlay_processing.size
                      else {"count":0,"p50":0.0,"p95":0.0,"max":0.0}),
                  "mission_owner":"CAMERA" if self.control_mode==1 else "GPS",
                  "path_computation_enabled":path_computation_enabled(
                      self.control_mode,
                      self.get_parameter("visualization_only").value,
                      self.get_parameter("require_control_mode").value),
                  "vehicle_command_enabled":False}
        self.realtime_pub.publish(String(data=json.dumps(document,separators=(",",":"))))

    def overlay_image(self, source, result):
        raw=np.frombuffer(source.data,np.uint8).reshape(source.height,source.step)[:,:source.width*3]
        image=np.ascontiguousarray(raw.reshape(source.height,source.width,3))
        if source.encoding.lower()=="rgb8": image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        else: image=image.copy()
        self._draw_path_layers(image, result)
        upstream=self.upstream_fps
        def upstream_rate(stage):
            value=upstream.get(stage,{})
            if isinstance(value,dict): return float(value.get("5s",{}).get("header_fps",0.0))
            return float(value or 0.0)
        lines=[f"CAM  : {float(upstream.get('camera_input_fps_5s',0.0)):4.1f}",
               f"INF  : {upstream_rate('inference_unique_fps'):4.1f}",
               f"PATH : {self.rates.snapshot('path_published')['5s']['header_fps']:4.1f}",
               f"STATE: {result.state}"]
        for index,text in enumerate(lines):
            cv2.putText(image,text,(8,20+18*index),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,0),2,cv2.LINE_AA)
        msg=Image();msg.header=source.header;msg.height=source.height;msg.width=source.width
        msg.encoding="bgr8";msg.is_bigendian=False;msg.step=source.width*3;msg.data=image.tobytes();return msg

    def _publish_path_overlay(self, job):
        if self.overlay_pub.get_subscription_count() == 0:
            return
        source, result = job
        started=time.perf_counter()
        try:
            self.overlay_pub.publish(self.overlay_image(source, result))
            self.rates.mark("overlay_published", source.header.stamp.sec,
                            source.header.stamp.nanosec)
            self.overlay_latencies.append((time.perf_counter()-started)*1000.0)
        except Exception as error:
            self.overlay_errors += 1
            self.get_logger().error(f"Path overlay failed: {error}")

    def destroy_node(self):
        self.overlay_worker.close()
        return super().destroy_node()

    def metrics(self, result, owner):
        points = result.points
        diagnostics = result.diagnostics or {}
        near = float(points[0, 0]) if len(points) else None
        middle = float(points[len(points)//2, 0]) if len(points) else None
        far = float(points[-1, 0]) if len(points) else None
        heading = float(np.degrees(np.arctan2(points[-1, 0]-points[0, 0],
                                              max(1.0, points[0, 1]-points[-1, 1])))) if len(points)>1 else None
        curvature = diagnostics.get("path_curvature_px")
        if curvature is None and len(points)>2:
            curvature = float(np.max(np.abs(np.diff(points[:, 0], n=2))))
        array = np.asarray(self.latencies)
        lateral_shift = None if near is None or self.last_near is None else near-self.last_near
        if near is not None: self.last_near = near
        return {"frame_count": self.frame_count,
                "received_count": self.received_count,
                "accepted_count": self.accepted_count,
                "processed_count": self.frame_count,
                "published_count": self.published_count,
                "processed_fps_observed": self.frame_count/max(1e-9, time.monotonic()-self.started),
                "path_valid": result.valid,
                "path_state": result.state, "path_confidence": result.confidence,
                "point_count": len(points), "near_x": near, "middle_x": middle,
                "far_x": far, "heading_deg": heading, "curvature_px": curvature,
                "recommended_lookahead_y_ratio": diagnostics.get(
                    "recommended_lookahead_y_ratio"),
                "polynomial_residual_px": diagnostics.get(
                    "polynomial_residual_px"),
                "required_steering_deg": diagnostics.get(
                    "required_steering_deg"),
                "max_required_steering_deg": diagnostics.get(
                    "max_required_steering_deg"),
                "steering_rate_deg_per_sec": diagnostics.get(
                    "steering_rate_deg_per_sec"),
                "steering_repaired": diagnostics.get(
                    "steering_repaired", False),
                "path_horizon_ratio": diagnostics.get(
                    "path_horizon_ratio"),
                "fallback_ratio": diagnostics.get("fallback_ratio"),
                "temporal_alpha_used": diagnostics.get(
                    "temporal_alpha_used"),
                "lateral_shift_px": lateral_shift,
                "source_mode": max(set(result.sources), key=result.sources.count) if result.sources else "NONE",
                "ownership": owner, "latency_ms": result.latency_ms,
                "latency_mean_ms": float(array.mean()),
                "latency_p95_ms": float(np.percentile(array, 95)),
                "stale_count": self.stale_count, "duplicate_count": self.duplicate_count,
                "backlog_drop_count": max(0, self.accepted_count-self.frame_count)}

    def debug_image(self, source, result, owner, exclusions):
        road = result.road_component
        image = np.zeros((*road.shape, 3), np.uint8); image[road > 0] = (40, 70, 40)
        for mask, color in zip(exclusions, ((180, 60, 180), (0, 0, 255), (255, 80, 200))): image[mask > 0] = color
        self._draw_path_layers(image, result)
        text = ("INTERSECTION | GPS PATH OWNER | CAMERA PATH INACTIVE" if result.state == INACTIVE
                else f"{result.state} conf={result.confidence:.2f} {owner}")
        cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2, cv2.LINE_AA)
        message = Image(); message.header = source.header; message.height, message.width = image.shape[:2]
        message.encoding = "bgr8"; message.is_bigendian = False; message.step = message.width*3
        message.data = image.tobytes(); return message

    @staticmethod
    def _draw_corridor_layer(image, result):
        """Thin PATH ROI boundary lines so a candidate rejected by the
        dynamic corridor is visually obvious (req 9)."""
        bounds = (result.diagnostics or {}).get("path_corridor_bounds") or []
        if not bounds:
            return
        left_pts = np.asarray([[b["lo"], b["y"]] for b in bounds])
        right_pts = np.asarray([[b["hi"], b["y"]] for b in bounds])
        for pts in (left_pts, right_pts):
            pts = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], False, (0, 165, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_path_layers(image, result):
        colors = {
            "LEFT": (255, 80, 0),
            "RIGHT": (0, 200, 255),
            "RAW CENTER": (255, 0, 255),
            "VIRTUAL CENTER": (255, 255, 0),
            "FINAL PATH": (0, 255, 0),
        }
        CameraImagePathNode._draw_corridor_layer(image, result)
        for points, key in ((result.left, "LEFT"), (result.right, "RIGHT"),
                            (result.raw, "RAW CENTER")):
            for x, y in points:
                cv2.circle(image, (int(round(x)), int(round(y))), 3,
                           colors[key], -1, cv2.LINE_AA)
        virtual = result.virtual
        if virtual is not None:
            for x, y in virtual:
                cv2.circle(image, (int(round(x)), int(round(y))), 3,
                           colors["VIRTUAL CENTER"], -1, cv2.LINE_AA)
        if len(result.points) > 1:
            points = np.rint(result.points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, colors["FINAL PATH"], 3,
                          cv2.LINE_AA)
        for index, key in enumerate(("LEFT", "RIGHT", "RAW CENTER",
                                     "VIRTUAL CENTER", "FINAL PATH")):
            x, y = 8 + 145*(index % 2), image.shape[0]-48+18*(index // 2)
            cv2.circle(image, (x, y-4), 3, colors[key], -1, cv2.LINE_AA)
            cv2.putText(image, key, (x+8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, colors[key], 1, cv2.LINE_AA)
        CameraImagePathNode._draw_corridor_status_text(image, result)

    @staticmethod
    def _draw_corridor_status_text(image, result):
        """PATH ROI / corridor / source / ROAD_ONLY / track-score readout (req 9)."""
        d = result.diagnostics or {}
        road_only = bool(d.get("road_only_path"))
        lines = [
            f"CORRIDOR: {d.get('path_corridor_state', '?')} "
            f"w={d.get('path_corridor_near_half_width_px', 0.0):.0f}px",
            f"SOURCE: {d.get('source_mode_confirmed', d.get('source_mode', '?'))} "
            f"{'[ROAD_ONLY]' if road_only else ''}",
            f"TRACK L/R: {d.get('left_boundary_track_score', 1.0):.2f}/"
            f"{d.get('right_boundary_track_score', 1.0):.2f}",
            f"STEER REQ/MAX: {d.get('required_steering_deg', 0.0):.1f}/"
            f"{d.get('max_required_steering_deg', 0.0):.1f}deg "
            f"{'REPAIRED' if d.get('steering_repaired') else ''}",
        ]
        for index, text in enumerate(lines):
            cv2.putText(image, text, (image.shape[1]-230, 20+16*index),
                       cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 165, 255), 1, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO); node = CameraImagePathNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
