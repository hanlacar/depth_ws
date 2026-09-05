"""ROS adapter for D456 RGB to CUDA YOLO-Seg perception outputs."""

import json
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import (QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
                       QoSDurabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from race_interfaces.msg import SemanticPathFrame

from .class_mapper import SemanticClassMapper
from .debug_renderer import DebugRenderer, DebugRenderingConfig
from .detection_schema import serialize_detection_document
from .image_contract import LatestFrameBuffer, validate_image_contract
from .inference_backend import CudaValidationError, create_inference_backend
from .inference_diagnostics import (EventRateTracker, LatencyTracker,
                                    ScalarLatencyTracker, UniqueFrameRateTracker)
from .mask_postprocessor import (build_semantic_masks, has_navigation_mask,
                                 validate_output_mask)
from .lane_temporal_tracker import (LaneMaskTemporalTracker,
                                    LaneTemporalConfig)
from .model_manifest import load_manifest
from .overlay_worker import LatestOnlyWorker, OverlayRateLimiter
from .perception_outputs import (COMPATIBILITY_MASK_TOPICS,
                                 SEMANTIC_MASK_TOPICS)
from .perception_refinement import (CommonPerceptionRefiner,
                                    RefinementConfig)
from .ros_image import bgr8_to_image, image_to_bgr8, mono8_to_image
from .semantic_path_contract import encode_binary_rle, encoded_payload_bytes
from .traffic_light_state import TrafficLightConfig, TrafficLightFilter, UNKNOWN


DEFAULTS = {
    "backend": "tensorrt", "segmentation_model_path": "", "class_manifest_path": "",
    "expected_model_sha256": "",
    "backend_input_probe_dir": "", "backend_input_probe_max_frames": 0,
    "device": "cuda:0", "require_cuda": True, "input_size": 640,
    "confidence_threshold": 0.25, "mask_threshold": 0.5,
    "max_image_age_sec": 0.4, "max_inference_latency_ms": 300.0,
    "input_image_topic": "/camera/image_raw",
    "input_reliability": "best_effort",
    "input_depth": 1,
    "input_camera_info_topic": "/camera/camera_info",
    "expected_image_width": 640, "expected_image_height": 480,
    "stop_detected_topic": "/perception/stop_detected",
    "stop_confidence_threshold": 0.25,
    "traffic20_confidence_threshold": 0.25, "publish_diagnostics": True,
    "publish_debug_image": True, "debug_overlay_alpha": 0.35,
    "debug_draw_contours": True, "debug_draw_boxes": True,
    "debug_draw_labels": True, "debug_draw_confidence": True,
    "publish_optional_masks": False,
    "semantic_path_frame_topic": "/perception/semantic_path_frame",
    "warmup_count": 5,
    "debug_overlay_show_fps": False,
    "external_input_fps_topic": "",
    "perception_overlay_max_fps": 45.0,
    "detections_image_max_fps": 50.0,
    # Zero preserves the established production behavior (one Image mask per
    # semantic result). Validation launches may set a finite display rate;
    # SemanticPathFrame/RLE output is never throttled by this parameter.
    "mask_image_publish_hz": 0.0,
    "traffic_light_topic": "/camera_traffic_light",
    "traffic_light_confirmation_frames": 3,
    "traffic_light_lost_timeout_sec": 0.5,
    "traffic_light_minimum_confidence": 0.5,
    "traffic_light_conflict_score_margin": 0.15,
    "traffic_light_mask_area_weight": 0.2,
    "traffic_light_mask_threshold": 0.5,
    "line_track_mode": "none",
    "line_track_max_hold_sec": 0.4,
    "line_track_max_hold_frames": 12,
    "line_track_confidence_decay": 0.88,
    "line_track_min_flow_points": 18,
    "line_track_max_motion_px": 45.0,
    "line_track_min_road_overlap": 0.65,
    "line_track_scene_change_threshold": 0.30,
}

REFINED_MASK_TOPICS = {
    "road": "/perception/refined/road",
    "white_line": "/perception/refined/white_line",
    "yellow_line": "/perception/refined/yellow_line",
    "unknown_line": "/perception/refined/unknown_line",
    "stop_line": "/perception/refined/stop_line_candidate",
    "crosswalk": "/perception/refined/crosswalk_candidate",
    "words": "/perception/refined/words",
    "road_marking_unknown": "/perception/refined/road_marking_unknown",
    "restored_markings": "/perception/refined/restored_markings",
}


def stable_failure_reason(error):
    text = str(error).lower()
    if "model sha256 mismatch" in text:
        return "model_sha256_mismatch"
    if isinstance(error, CudaValidationError):
        if "index" in text:
            return "invalid_cuda_device"
        return "cuda_unavailable"
    if "task must be segment" in text:
        return "model_task_not_segment"
    if isinstance(error, FileNotFoundError) and ".engine" in text:
        return "engine_missing"
    if "tensorrt" in text and ("import" in text or "unavailable" in text):
        return "tensorrt_unavailable"
    if "engine" in text and ("load" in text or "deserialize" in text or "incompatible" in text):
        return "engine_load_failed"
    return "model_load_failed"


class CameraYoloInferenceNode(Node):
    def __init__(self, backend=None):
        super().__init__("camera_yolo_inference_node")
        # Full-frame OpenCV operations are already parallel with TensorRT via
        # the latest-only debug worker.  Internal OpenCV pools only add CPU
        # oversubscription and tail latency here.
        cv2.setNumThreads(1)
        self.frames = LatestFrameBuffer()
        self.camera_info = None
        self.busy = False
        self.latencies = LatencyTracker()
        self.events = EventRateTracker(window_sec=10.0)
        # Second, short-window tracker purely for the optional on-screen FPS
        # overlay (requirement: 1s-ish rolling window, separate from the
        # 10s diagnostics window used elsewhere). marks are O(1) deque ops,
        # not a hot-path cost.
        self.unique_rates = UniqueFrameRateTracker()
        self.callback_store_latency = ScalarLatencyTracker()
        self.debug_render_latency = ScalarLatencyTracker()
        self.debug_publish_latency = ScalarLatencyTracker()
        self.semantic_path_pack_latency = ScalarLatencyTracker()
        self.semantic_path_publish_latency = ScalarLatencyTracker()
        self.refinement_latency = ScalarLatencyTracker()
        self._external_input_fps = None
        self._path_realtime = {}
        self.ready = False
        self.backend_input_probe_count = 0
        self.failure = "not_initialized"
        for key, value in DEFAULTS.items():
            self.declare_parameter(key, value)
        refinement_defaults = {name: field.default for name, field in
                               RefinementConfig.__dataclass_fields__.items()}
        for name, value in refinement_defaults.items():
            self.declare_parameter("refinement."+name, value)
        self.refiner = CommonPerceptionRefiner(RefinementConfig(**{
            name: self.p("refinement."+name) for name in refinement_defaults}))
        self.lane_tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
            mode=str(self.p("line_track_mode")),
            max_hold_sec=float(self.p("line_track_max_hold_sec")),
            max_hold_frames=int(self.p("line_track_max_hold_frames")),
            confidence_decay=float(self.p("line_track_confidence_decay")),
            min_flow_points=int(self.p("line_track_min_flow_points")),
            max_motion_px=float(self.p("line_track_max_motion_px")),
            min_road_overlap=float(self.p("line_track_min_road_overlap")),
            scene_change_threshold=float(
                self.p("line_track_scene_change_threshold"))))
        self._validate_parameters()
        self.traffic_light_filter = TrafficLightFilter(TrafficLightConfig(**{
            name: self.p(name)
            for name in TrafficLightConfig.__dataclass_fields__
        }))

        output_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                reliability=QoSReliabilityPolicy.RELIABLE)
        self.semantic_mask_pubs = {
            role: self.create_publisher(Image, topic, output_qos)
            for role, topic in SEMANTIC_MASK_TOPICS.items()
        }
        self.compatibility_mask_pubs = {
            role: self.create_publisher(Image, topic, output_qos)
            for role, topic in COMPATIBILITY_MASK_TOPICS.items()
        }
        self.refined_mask_pubs = {
            role: self.create_publisher(Image, topic, output_qos)
            for role, topic in REFINED_MASK_TOPICS.items()
        }
        self.tracked_line_pubs = {
            "white_line": self.create_publisher(
                Image, "/perception/tracked/white_line", output_qos),
            "yellow_line": self.create_publisher(
                Image, "/perception/tracked/yellow_line", output_qos),
        }
        self.refinement_diag_pub = self.create_publisher(
            String, "/camera/perception_refinement_diagnostics", output_qos)
        self.detections_image_pub = self.create_publisher(
            Image, "/perception/detections_image", output_qos)
        overlay_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=QoSReliabilityPolicy.RELIABLE,
                                 durability=QoSDurabilityPolicy.VOLATILE)
        self.perception_overlay_pub = self.create_publisher(
            Image, "/camera/perception_overlay_image", overlay_qos)
        self.detections_pub = self.create_publisher(
            String, "/perception/detections_json", output_qos)
        self.stop_pub = self.create_publisher(
            Bool, self.p("stop_detected_topic"), output_qos)
        self.traffic20_pub = self.create_publisher(
            Bool, "/perception/traffic20_detected", output_qos)
        self.traffic_light_pub = self.create_publisher(
            String, self.p("traffic_light_topic"), output_qos)
        self.valid_pub = self.create_publisher(
            Bool, "/camera/perception_valid", output_qos)
        self.navigation_available_pub = self.create_publisher(
            Bool, "/camera/navigation_mask_available", output_qos)
        self.latency_pub = self.create_publisher(
            Float32, "/camera/inference_latency_ms", output_qos)
        self.status_pub = self.create_publisher(
            String, "/camera/inference_status", output_qos)
        self.pipeline_latency_pub = self.create_publisher(
            Float32, "/camera/pipeline_latency_ms", output_qos)
        self.performance_pub = self.create_publisher(
            String, "/camera/performance_diagnostics", output_qos)
        self.semantic_path_pub = self.create_publisher(
            SemanticPathFrame, self.p("semantic_path_frame_topic"), output_qos)
        # Lightweight (small String, not an Image) 1s-rolling realtime chain
        # FPS report, driven by internal unique-stamp event counters —
        # never by ros2 topic hz. Intended as the primary way to verify
        # INPUT/RECEIVED/INFERENCE/SEMANTIC/OUTPUT without deserializing
        # sensor_msgs/Image over the CLI.
        self.realtime_fps_pub = self.create_publisher(
            String, "/camera/realtime_fps", output_qos)

        self.input_callbacks = MutuallyExclusiveCallbackGroup()
        self.processing_callbacks = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.p("input_image_topic"),
                                 self._image_callback, self._image_input_qos(),
                                 callback_group=self.input_callbacks)
        self.create_subscription(CameraInfo, self.p("input_camera_info_topic"),
                                 self._camera_info_callback, qos_profile_sensor_data)
        external_fps_topic = str(self.p("external_input_fps_topic"))
        if external_fps_topic:
            # Debug-only, opt-in: reads the upstream source's own observed
            # publish rate (e.g. video_perception_test's
            # /video_test/publisher_diagnostics) so the overlay can show a
            # true "INPUT FPS" distinct from this node's own RECEIVED FPS.
            # Disabled by default (empty topic name -> no subscription),
            # so production launches are unaffected.
            self.create_subscription(String, external_fps_topic,
                                     self._external_fps_callback, 10)
        self.create_subscription(String, "/camera/path_realtime_fps",
                                 self._path_realtime_callback, 10)
        self.renderer = DebugRenderer(DebugRenderingConfig(
            overlay_alpha=self.p("debug_overlay_alpha"),
            draw_contours=self.p("debug_draw_contours"),
            draw_boxes=self.p("debug_draw_boxes"),
            draw_labels=self.p("debug_draw_labels"),
            draw_confidence=self.p("debug_draw_confidence"),
            mask_threshold=self.p("mask_threshold"),
        ))
        self.perception_renderer = DebugRenderer(DebugRenderingConfig(
            overlay_alpha=self.p("debug_overlay_alpha"),
            draw_contours=True, draw_boxes=False, draw_labels=False,
            draw_confidence=False, mask_threshold=self.p("mask_threshold"),
        ))
        self.perception_overlay_limiter = OverlayRateLimiter(
            self.p("perception_overlay_max_fps"))
        self.perception_overlay_worker = LatestOnlyWorker(
            self._publish_perception_overlay, "perception-overlay")
        self.detections_image_limiter = OverlayRateLimiter(
            self.p("detections_image_max_fps"))
        self.detections_image_worker = LatestOnlyWorker(
            self._publish_detections_image, "detections-image")
        self.semantic_path_worker = LatestOnlyWorker(
            self._publish_semantic_path, "semantic-path")
        mask_hz = float(self.p("mask_image_publish_hz"))
        if mask_hz < 0.0:
            raise ValueError("mask_image_publish_hz must be >= 0")
        self.mask_image_limiter = (
            None if mask_hz == 0.0 else OverlayRateLimiter(mask_hz))
        self._initialize_backend(backend)
        self.create_timer(0.005, self.process_latest,
                          callback_group=self.processing_callbacks)
        self.create_timer(1.0, self.publish_health,
                          callback_group=self.processing_callbacks)
        self.create_timer(1.0, self.publish_realtime_fps,
                          callback_group=self.processing_callbacks)
        self.create_timer(1.0, self.publish_performance,
                          callback_group=self.processing_callbacks)

    def p(self, name):
        return self.get_parameter(name).value

    def _image_input_qos(self):
        reliability = str(self.p("input_reliability")).lower()
        depth = int(self.p("input_depth"))
        if reliability == "best_effort":
            return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=depth,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT,
                              durability=QoSDurabilityPolicy.VOLATILE)
        if reliability == "reliable":
            return QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=depth,
                              reliability=QoSReliabilityPolicy.RELIABLE,
                              durability=QoSDurabilityPolicy.VOLATILE)
        raise ValueError("input_reliability must be best_effort or reliable")

    def _camera_info_callback(self, message):
        self.camera_info = message

    def _image_callback(self, message):
        callback_entry = time.perf_counter()
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        self.events.mark("received_from_ros", callback_entry)
        unique = self.unique_rates.mark("camera_input", *stamp,
                                        arrival=callback_entry)
        outcome = self.frames.push(message)
        self.callback_store_latency.add(
            (time.perf_counter()-callback_entry)*1000.0)
        if unique:
            self.events.mark("callback_unique", callback_entry)
        if outcome in ("accepted", "replaced"):
            self.unique_rates.mark("received", *stamp)

    def _external_fps_callback(self, message):
        try:
            payload = json.loads(message.data)
            value = payload.get("observed_publish_fps")
            self._external_input_fps = float(value) if value is not None else None
        except (ValueError, TypeError, AttributeError):
            self._external_input_fps = None

    def _path_realtime_callback(self, message):
        try: self._path_realtime = json.loads(message.data)
        except (ValueError, TypeError, AttributeError): self._path_realtime = {}

    def _draw_fps_overlay(self, frame):
        """Debug-only text overlay driven by internal event counters (1s
        rolling window), never by ros2 topic hz. Runs only when the
        renderer already ran (subscriber-gated), so it adds no cost when
        nothing is watching /perception/detections_image."""
        def rate(event):
            return self.unique_rates.snapshot(event)["1s"]["header_fps"]

        lines = []
        if self._external_input_fps is not None:
            lines.append(f"INPUT     {self._external_input_fps:5.1f} FPS")
        lines.append(f"RECEIVED  {rate('received'):5.1f} FPS")
        lines.append(f"INFERENCE {rate('inference'):5.1f} FPS")
        lines.append(f"SEMANTIC  {rate('semantic'):5.1f} FPS")
        lines.append(f"OUTPUT    {rate('overlay'):5.1f} FPS")

        pad, line_h = 6, 18
        width = 210
        height = pad * 2 + line_h * len(lines)
        region = frame[0:height, 0:width]
        shaded = np.zeros_like(region)
        cv2.addWeighted(shaded, 0.55, region, 0.45, 0, dst=region)
        for index, text in enumerate(lines):
            y = pad + line_h * (index + 1) - 4
            cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (0, 255, 0), 1, cv2.LINE_AA)

    def _validate_parameters(self):
        expected_sha = str(self.p("expected_model_sha256")).strip().lower()
        if expected_sha and (len(expected_sha) != 64 or
                             any(char not in "0123456789abcdef" for char in expected_sha)):
            raise ValueError("expected_model_sha256 must be empty or 64 lowercase hex characters")
        if not 0.0 <= float(self.p("confidence_threshold")) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= float(self.p("mask_threshold")) <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        for name in ("input_size", "expected_image_width", "expected_image_height"):
            if int(self.p(name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.p("warmup_count")) < 0:
            raise ValueError("warmup_count must be non-negative")
        if int(self.p("input_depth")) <= 0:
            raise ValueError("input_depth must be positive")
        if int(self.p("backend_input_probe_max_frames")) < 0:
            raise ValueError("backend_input_probe_max_frames must be non-negative")
        for name in ("max_image_age_sec", "max_inference_latency_ms"):
            if float(self.p(name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("perception_overlay_max_fps", "detections_image_max_fps"):
            if float(self.p(name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        TrafficLightConfig(**{
            name: self.p(name)
            for name in TrafficLightConfig.__dataclass_fields__
        }).validate()

    def _initialize_backend(self, backend):
        try:
            manifest = load_manifest(self.p("class_manifest_path"))
            self.mapper = SemanticClassMapper(manifest)
            self.backend = backend or create_inference_backend(
                self.p("backend"), self.p("segmentation_model_path"),
                self.p("device"), self.p("input_size"),
                self.p("confidence_threshold"),
                self.p("require_cuda"))
            self.backend.load_model()
            self.mapper.resolve_model_classes(self.backend.get_model_names())
            self.get_logger().info(
                "Semantic class mapping: " + json.dumps(
                    self.mapper.describe_mapping(), separators=(",", ":")))
            for warning in self.mapper.warnings:
                self.get_logger().warning(warning)
            info = self.backend.get_device_info()
            expected_sha = str(self.p("expected_model_sha256")).strip().lower()
            if expected_sha and info["model_sha256"] != expected_sha:
                raise ValueError(
                    "model SHA256 mismatch: "
                    f"expected={expected_sha}, actual={info['model_sha256']}, "
                    f"path={info['model_path']}")
            self.backend.warmup(self.p("warmup_count"))
            self.ready = True
            self.failure = "ok"
            info = self.backend.get_device_info()
            self.get_logger().info(
                "Inference backend ready: " + ", ".join((
                    f"requested={info['requested_device']}",
                    f"backend={info['backend']}", f"precision={info['precision']}",
                    f"actual={info['actual_device']}", f"gpu={info['gpu_name']}",
                    f"model={info['model_path']}",
                    f"sha256={info['model_sha256']}", f"task={info['task']}",
                    f"imgsz={info['input_size']}",
                    f"confidence={info['confidence_threshold']}",
                    f"mask_threshold={self.p('mask_threshold')}",
                )))
        except Exception as error:
            self.failure = stable_failure_reason(error)
            self.get_logger().error(f"Perception initialization failed [{self.failure}]: {error}")

    def publish_status(self, text):
        if self.p("publish_diagnostics"):
            self.status_pub.publish(String(data=text))

    def publish_health(self):
        self.publish_traffic_light(
            self.traffic_light_filter.tick(time.monotonic()))
        if not self.ready:
            self.valid_pub.publish(Bool(data=False))
            self.navigation_available_pub.publish(Bool(data=False))
            self.stop_pub.publish(Bool(data=False))
            self.traffic20_pub.publish(Bool(data=False))
            self.publish_status(self.failure)

    def publish_traffic_light(self, decision):
        state = decision.state if decision.state in ("R", "G") else UNKNOWN
        self.traffic_light_pub.publish(String(data=state))
        self.events.mark(f"traffic_light.{state}")

    def publish_realtime_fps(self):
        """1s-rolling-window chain FPS, from internal unique-stamp event
        counters only. Never derived from ros2 topic hz. Small String
        message so subscribing to it costs nothing like subscribing to
        the raw Image topics does."""
        stages = {name: self.unique_rates.snapshot(name) for name in
                  ("camera_input", "received", "inference", "refinement", "semantic", "overlay",
                   "perception_overlay")}
        document = {
            "input_fps": stages["camera_input"]["10s"]["header_fps"],
            "camera_input_fps_1s": stages["camera_input"]["1s"]["header_fps"],
            "camera_input_fps_5s": stages["camera_input"]["5s"]["header_fps"],
            "camera_input_fps_10s": stages["camera_input"]["10s"]["header_fps"],
            "received_unique_fps": stages["received"],
            "image_callback": stages["camera_input"],
            "latest_buffer": self.frames.snapshot(),
            "callback_store_ms": self.callback_store_latency.summary(),
            "inference_unique_fps": stages["inference"],
            "refinement_unique_fps": stages["refinement"],
            "semantic_unique_fps": stages["semantic"],
            "overlay_unique_fps": stages["overlay"],
            "perception_overlay_unique_fps": stages["perception_overlay"],
            "perception_overlay_fps": stages["perception_overlay"],
            "path_unique_fps": self._path_realtime.get("path_unique_fps", {}),
            "path_overlay_fps": self._path_realtime.get("path_overlay_fps", {}),
            "path": self._path_realtime,
            "external_debug_input_fps": self._external_input_fps,
        }
        self.realtime_fps_pub.publish(String(data=json.dumps(document, separators=(",", ":"))))

    def publish_performance(self):
        if self.ready and self.p("publish_diagnostics"):
            info = self.backend.get_device_info()
            document = {"backend": info["backend"], "precision": info["precision"],
                        "dropped_frames": self.frames.dropped_count,
                        "latest_buffer": self.frames.snapshot(),
                        "callback_store_ms":self.callback_store_latency.summary(),
                        "input_qos":{"reliability":str(self.p("input_reliability")),
                                     "history":"keep_last","depth":int(self.p("input_depth"))},
                        "executor":{"type":"MultiThreadedExecutor","threads":2,
                                    "image_callback_group":"MutuallyExclusiveCallbackGroup",
                                    "inference_callback_group":"MutuallyExclusiveCallbackGroup",
                                    "blocking_relationship":"separate_callback_groups"},
                        "statistics": self.latencies.summary(),
                        "async_stages": {
                            "refinement_ms": self.refinement_latency.summary(),
                            "debug_render_ms": self.debug_render_latency.summary(),
                            "debug_publish_ms": self.debug_publish_latency.summary(),
                            "semantic_path_pack_ms": self.semantic_path_pack_latency.summary(),
                            "semantic_path_publish_ms": self.semantic_path_publish_latency.summary(),
                        },
                        "workers": {
                            "detections_image": self.detections_image_worker.snapshot(),
                            "semantic_path": self.semantic_path_worker.snapshot(),
                            "perception_overlay": self.perception_overlay_worker.snapshot(),
                        },
                        "event_rates": self.events.snapshot()}
            self.performance_pub.publish(String(data=json.dumps(document, separators=(",", ":"))))

    def publish_invalid(self, image, reason):
        self.valid_pub.publish(Bool(data=False))
        self.navigation_available_pub.publish(Bool(data=False))
        self.stop_pub.publish(Bool(data=False))
        self.traffic20_pub.publish(Bool(data=False))
        self.publish_traffic_light(
            self.traffic_light_filter.tick(time.monotonic()))
        self.publish_status(reason)
        publish_masks = (self.mask_image_limiter is None or
                         self.mask_image_limiter.ready(time.monotonic()))
        if image is not None and publish_masks:
            zero = np.zeros((image.height, image.width), np.uint8)
            message = mono8_to_image(zero, image.header)
            for publisher in (*self.semantic_mask_pubs.values(),
                              *self.compatibility_mask_pubs.values(),
                              *self.refined_mask_pubs.values(),
                              *self.tracked_line_pubs.values()):
                publisher.publish(message)

    def _build_masks(self, instances, shape):
        role_class_ids = {role: self.mapper.class_ids_for_role(role)
                          for role in SEMANTIC_MASK_TOPICS}
        masks = build_semantic_masks(instances, role_class_ids, shape,
                                     self.p("mask_threshold"))
        for role in SEMANTIC_MASK_TOPICS:
            if masks[role].shape != shape or masks[role].dtype != np.uint8:
                raise ValueError(f"invalid_{role}_mask")
        return masks

    def _publish_masks(self, masks, header, refined=None, temporal=None):
        if (self.mask_image_limiter is not None and
                not self.mask_image_limiter.ready(time.monotonic())):
            return
        core_roles = frozenset(COMPATIBILITY_MASK_TOPICS)
        for role, publisher in self.semantic_mask_pubs.items():
            if (role not in core_roles and not self.p("publish_optional_masks") and
                    publisher.get_subscription_count() == 0):
                continue
            publisher.publish(mono8_to_image(masks[role], header))
            self.events.mark(f"semantic_mask_published.{role}")
        for role, publisher in self.compatibility_mask_pubs.items():
            source = masks[role] if refined is None else refined[role]
            publisher.publish(mono8_to_image(source, header))
            self.events.mark(f"compatibility_mask_published.{role}")
        if refined is not None:
            for role, publisher in self.refined_mask_pubs.items():
                if publisher.get_subscription_count() > 0:
                    publisher.publish(mono8_to_image(refined[role], header))
        if temporal is not None:
            tracked = {"white_line": temporal.tracked_white_line,
                       "yellow_line": temporal.tracked_yellow_line}
            for role, publisher in self.tracked_line_pubs.items():
                if publisher.get_subscription_count() > 0:
                    publisher.publish(mono8_to_image(tracked[role], header))

    def _publish_detections(self, instances, image):
        text = serialize_detection_document(
            instances, self.backend.get_model_names(), image.header.stamp.sec,
            image.header.stamp.nanosec, image.header.frame_id)
        self.detections_pub.publish(String(data=text))

    def _build_semantic_path_frame(self, refined, header):
        message = SemanticPathFrame()
        message.header = header
        message.image_height, message.image_width = refined["road"].shape
        lane = np.bitwise_or.reduce((refined["white_line"],
                                    refined["yellow_line"],
                                    refined["unknown_line"]))
        arrays = [encode_binary_rle(mask) for mask in
                  (refined["road"], lane, refined["words"],
                   refined["stop_line"], refined["crosswalk"])]
        (message.road_rle, message.lane_rle, message.words_rle,
         message.stop_line_rle, message.c_line_rle) = [array.tolist() for array in arrays]
        extra_arrays = [encode_binary_rle(refined[name]) for name in
                        ("white_line", "yellow_line", "unknown_line",
                         "road_marking_unknown", "restored_markings", "raw_road")]
        (message.white_line_rle, message.yellow_line_rle,
         message.unknown_line_rle, message.road_marking_unknown_rle,
         message.restored_markings_rle,
         message.raw_road_rle) = [array.tolist() for array in extra_arrays]
        message.stop_line_confidence = float(refined["stop_confidence"])
        message.crosswalk_confidence = float(refined["crosswalk_confidence"])
        message.refinement_diagnostics_json = json.dumps(
            refined["diagnostics"], separators=(",", ":"))
        message.uncompressed_bytes = int(message.image_width*message.image_height*11)
        message.encoded_bytes = int(encoded_payload_bytes(*(arrays+extra_arrays)))
        return message

    def process_latest(self):
        if self.busy:
            return
        image = self.frames.take()
        if image is None:
            return
        if not self.ready:
            self.publish_invalid(image, self.failure)
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        valid, reason = validate_image_contract(
            image, self.camera_info, now, self.p("max_image_age_sec"),
            self.p("expected_image_width"), self.p("expected_image_height"))
        if not valid:
            mapped = "camera_info_missing" if reason == "missing_image_or_camera_info" else reason
            mapped = "image_size_mismatch" if reason == "resolution_mismatch" else mapped
            self.publish_invalid(image, mapped)
            return

        self.busy = True
        self.events.mark("inference_started")
        pipeline_started = time.perf_counter()
        try:
            stage_started = time.perf_counter()
            bgr = image_to_bgr8(image)
            probe_dir = str(self.p("backend_input_probe_dir")).strip()
            probe_limit = int(self.p("backend_input_probe_max_frames"))
            if (probe_dir and self.backend_input_probe_count < probe_limit):
                directory = Path(probe_dir).expanduser()
                directory.mkdir(parents=True, exist_ok=True)
                stem = f"{image.header.stamp.sec}_{image.header.stamp.nanosec}"
                cv2.imwrite(str(directory / f"C_backend_{stem}.png"), bgr)
                with (directory / "backend_inputs.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "stamp_sec": image.header.stamp.sec,
                        "stamp_nanosec": image.header.stamp.nanosec,
                        "shape": list(bgr.shape), "dtype": str(bgr.dtype),
                        "mean_bgr": bgr.reshape(-1, 3).mean(axis=0).tolist(),
                        "std_bgr": bgr.reshape(-1, 3).std(axis=0).tolist(),
                    }, separators=(",", ":")) + "\n")
                self.backend_input_probe_count += 1
            conversion_ms = (time.perf_counter() - stage_started) * 1000.0
            backend_started = time.perf_counter()
            instances = self.backend.infer(bgr)
            self.events.mark("inference_completed")
            self.unique_rates.mark("inference", image.header.stamp.sec,
                                   image.header.stamp.nanosec)
            backend_wall_ms = (time.perf_counter() - backend_started) * 1000.0
            backend_timings = self.backend.get_last_timings()
            inference_ms = backend_timings["inference_ms"]
            stamp_age = self.get_clock().now().nanoseconds * 1e-9 - (
                image.header.stamp.sec + image.header.stamp.nanosec * 1e-9)
            if inference_ms > self.p("max_inference_latency_ms"):
                raise TimeoutError("latency_limit_exceeded")
            if stamp_age > self.p("max_image_age_sec"):
                raise TimeoutError("stale_image")

            stage_started = time.perf_counter()
            masks = self._build_masks(instances, (image.height, image.width))
            self.events.mark("core_semantic_completed")
            mask_ms = (time.perf_counter() - stage_started) * 1000.0
            stage_started = time.perf_counter()
            role_class_ids = {role: self.mapper.class_ids_for_role(role)
                              for role in SEMANTIC_MASK_TOPICS}
            stamp_sec = (float(image.header.stamp.sec) +
                         float(image.header.stamp.nanosec)*1.0e-9)
            refined_result = self.refiner.refine(
                bgr, instances, masks, role_class_ids, stamp_sec,
                self.p("mask_threshold"))
            temporal_result = self.lane_tracker.update(
                bgr, masks["white_line"], masks["yellow_line"], masks["road"],
                stamp_sec)
            tracked_enabled = str(self.p("line_track_mode")) != "none"
            refined_result.diagnostics["lane_temporal"] = \
                temporal_result.diagnostics
            refined = {
                "raw_road": masks["road"],
                "road": refined_result.road,
                "white_line": (temporal_result.effective_white_line
                               if tracked_enabled else refined_result.white_line),
                "yellow_line": (temporal_result.effective_yellow_line
                                if tracked_enabled else refined_result.yellow_line),
                "unknown_line": refined_result.unknown_line,
                "stop_line": refined_result.stop_line,
                "crosswalk": refined_result.crosswalk,
                "words": refined_result.words,
                "road_marking_unknown": refined_result.road_marking_unknown,
                "restored_markings": refined_result.restored_markings,
                "stop_confidence": refined_result.stop_confidence,
                "crosswalk_confidence": refined_result.crosswalk_confidence,
                "diagnostics": refined_result.diagnostics,
            }
            refinement_ms = (time.perf_counter()-stage_started)*1000.0
            self.refinement_latency.add(refinement_ms)
            self.unique_rates.mark("refinement", image.header.stamp.sec,
                                   image.header.stamp.nanosec)
            navigation_available = has_navigation_mask(refined)
            # RLE conversion and ROS sequence construction are CPU-heavy for
            # fragmented masks.  A single-slot worker preserves the newest
            # semantic frame without blocking the next TensorRT invocation.
            if self.semantic_path_pub.get_subscription_count() > 0:
                self.semantic_path_worker.submit((refined, image.header))
            stage_started = time.perf_counter()
            detection_text = serialize_detection_document(
                instances, self.backend.get_model_names(), image.header.stamp.sec,
                image.header.stamp.nanosec, image.header.frame_id)
            json_ms = (time.perf_counter() - stage_started) * 1000.0
            names = self.backend.get_model_names()
            traffic_role_ids = {
                role: self.mapper.class_ids_for_role(role)
                for role in ("red_light", "green_light")
            }
            traffic_decision = self.traffic_light_filter.update(
                instances, traffic_role_ids, time.monotonic())
            self.publish_traffic_light(traffic_decision)
            if (self.p("publish_debug_image") and
                    self.perception_overlay_pub.get_subscription_count() > 0 and
                    self.perception_overlay_limiter.ready(time.monotonic())):
                # References are handed to a single-slot worker. No image/mask
                # copy or rendering occurs in this production callback.
                self.perception_overlay_worker.submit(
                    (bgr, masks, image.header))
            self.stop_pub.publish(Bool(data=bool(
                np.any(refined_result.stop_line) and
                refined_result.stop_confidence >=
                self.p("stop_confidence_threshold"))))
            traffic_ids = self.mapper.class_ids_for_role("traffic20")
            traffic20 = any(int(item["class_id"]) in traffic_ids and
                            float(item.get("confidence", 0.0)) >=
                            self.p("traffic20_confidence_threshold") for item in instances)
            self.traffic20_pub.publish(Bool(data=traffic20))
            # Debug rendering / image serialization run in a latest-only
            # worker.  Slow RQT or ros2 topic hz subscribers therefore cannot
            # create inference backlog.
            if (self.p("publish_debug_image") and
                    self.detections_image_pub.get_subscription_count() > 0 and
                    self.detections_image_limiter.ready(time.monotonic())):
                info = self.backend.get_device_info()
                self.detections_image_worker.submit((
                    bgr, instances, names,
                    (info["actual_device"], f"{inference_ms:.1f} ms",
                     f"{len(instances)} instances"),
                    masks, image.header))
            stage_started = time.perf_counter()
            self._publish_masks(masks, image.header, refined, temporal_result)
            if self.p("publish_diagnostics"):
                self.refinement_diag_pub.publish(String(data=json.dumps(
                    refined_result.diagnostics, separators=(",", ":"))))
            self.unique_rates.mark("semantic", image.header.stamp.sec,
                                   image.header.stamp.nanosec)
            self.detections_pub.publish(String(data=detection_text))
            self.events.mark("detections_json_published")
            self.valid_pub.publish(Bool(data=True))
            self.navigation_available_pub.publish(Bool(data=navigation_available))
            self.latency_pub.publish(Float32(data=float(inference_ms)))
            self.publish_status("ok")
            publish_ms = (time.perf_counter() - stage_started) * 1000.0
            total_ms = (time.perf_counter() - pipeline_started) * 1000.0
            stages = {"image_conversion_ms": conversion_ms,
                      "preprocess_ms": backend_timings["preprocess_ms"],
                      "inference_ms": inference_ms,
                      "decode_ms": backend_timings["decode_ms"],
                      "result_transfer_ms": backend_timings.get("result_transfer_ms", 0.0),
                      "backend_wall_ms": backend_wall_ms,
                      "mask_postprocess_ms": mask_ms, "renderer_ms": 0.0,
                      "refinement_ms": refinement_ms,
                      "semantic_path_pack_ms": 0.0,
                      "json_ms": json_ms, "ros_publish_ms": publish_ms,
                      "total_pipeline_ms": total_ms}
            self.latencies.add_stages(stages)
            self.pipeline_latency_pub.publish(Float32(data=float(total_ms)))
        except TimeoutError as error:
            self.publish_invalid(image, str(error))
        except Exception as error:
            self.get_logger().error(f"Inference failed: {error}")
            self.publish_invalid(image, "inference_failed")
        finally:
            self.busy = False

    def _publish_perception_overlay(self, job):
        if self.perception_overlay_pub.get_subscription_count() == 0:
            return
        bgr, masks, header = job
        try:
            rendered = self.perception_renderer.render_perception(bgr, masks)
            self.perception_overlay_pub.publish(bgr8_to_image(rendered, header))
            self.events.mark("perception_overlay_published")
            self.unique_rates.mark("perception_overlay", header.stamp.sec,
                                   header.stamp.nanosec)
        except Exception as error:
            self.get_logger().error(f"Perception overlay failed: {error}")

    def _publish_detections_image(self, job):
        if self.detections_image_pub.get_subscription_count() == 0:
            return
        bgr, instances, names, diagnostics, masks, header = job
        try:
            started = time.perf_counter()
            debug = self.renderer.render(
                bgr, instances, names, diagnostics, semantic_masks=masks)
            if self.p("debug_overlay_show_fps"):
                self._draw_fps_overlay(debug)
            message = bgr8_to_image(debug, header)
            self.debug_render_latency.add(
                (time.perf_counter() - started) * 1000.0)
            self.events.mark("detections_image_rendered")
            started = time.perf_counter()
            self.detections_image_pub.publish(message)
            self.debug_publish_latency.add(
                (time.perf_counter() - started) * 1000.0)
            self.events.mark("detections_image_published")
            self.unique_rates.mark("overlay", header.stamp.sec,
                                   header.stamp.nanosec)
        except Exception as error:
            self.get_logger().error(f"Detections image rendering failed: {error}")

    def _publish_semantic_path(self, job):
        masks, header = job
        try:
            started = time.perf_counter()
            message = self._build_semantic_path_frame(masks, header)
            self.semantic_path_pack_latency.add(
                (time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            self.semantic_path_pub.publish(message)
            self.semantic_path_publish_latency.add(
                (time.perf_counter() - started) * 1000.0)
            self.events.mark("semantic_path_frame_published")
        except Exception as error:
            self.get_logger().error(f"Semantic path publish failed: {error}")

    def destroy_node(self):
        self.semantic_path_worker.close()
        self.detections_image_worker.close()
        self.perception_overlay_worker.close()
        return super().destroy_node()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CameraYoloInferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
