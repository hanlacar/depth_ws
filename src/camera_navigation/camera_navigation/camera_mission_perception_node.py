#!/usr/bin/env python3
"""Advisory-only stop-line, sign, traffic-light and uphill perception."""

from collections import deque
import json
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from race_interfaces.msg import SemanticPathFrame
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
from tf2_ros import (Buffer, TransformBroadcaster, TransformException,
                     TransformListener)

from .mission_perception_core import (
    DebouncedPresence, MissionTrafficConfig, MissionTrafficFilter,
    PresenceConfig, StopLineDepthConfig, UphillConfig, UphillDetector,
    front_axle_distance_from_base_point, pitch_deg_from_quaternion,
    robust_stop_line_point, transform_point)
from .semantic_path_contract import decode_binary_rle


def stamp_sec(stamp):
    return float(stamp.sec)+float(stamp.nanosec)*1.0e-9


def json_number(value):
    return float(value) if value is not None and math.isfinite(value) else None


def image_array(message):
    enc = message.encoding.upper()
    if enc == "16UC1":
        dtype, channels = np.dtype("<u2"), 1
    elif enc == "32FC1":
        dtype, channels = np.dtype("<f4"), 1
    elif enc in ("BGR8", "RGB8"):
        dtype, channels = np.uint8, 3
    else:
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    row_values = message.step//np.dtype(dtype).itemsize
    raw = np.frombuffer(message.data, dtype=dtype)
    rows = raw[:message.height*row_values].reshape(message.height, row_values)
    used = rows[:, :message.width*channels]
    shape = ((message.height, message.width) if channels == 1 else
             (message.height, message.width, channels))
    return used.reshape(shape).copy()


class CameraMissionPerceptionNode(Node):
    def __init__(self):
        super().__init__("camera_mission_perception_node")
        defaults = {
            "semantic_topic": "/perception/semantic_path_frame",
            "detections_topic": "/perception/detections_json",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera_info",
            "imu_topic": "/imu/data", "imu_valid_topic": "/imu/valid",
            "color_topic": "/camera/image_raw",
            "base_frame": "base_link", "wheelbase_m": 0.73,
            "front_axle_x_m": 0.365,
            "sync_slop_sec": 0.05, "max_input_age_sec": 0.30,
            "max_tf_age_sec": 0.20, "stop_line_min_confidence": 0.50,
            "stop_line_min_depth_pixels": 30,
            "stop_line_depth_min_m": 0.20,
            "stop_line_depth_max_m": 12.0,
            "stop_line_depth_mad_threshold": 3.5,
            "stop_line_max_depth_mad_m": 0.45,
            "stop_line_component_merge_px": 8,
            "stop_line_distance_merge_m": 0.20,
            "sign_min_confidence": 0.50, "sign_min_area_px": 25.0,
            "sign_on_frames": 3, "sign_off_frames": 2,
            "sign_timeout_sec": 0.50,
            "traffic_light_min_confidence": 0.50,
            "traffic_light_on_frames": 3,
            "traffic_light_switch_frames": 3,
            "traffic_light_timeout_sec": 0.50,
            "traffic_light_conflict_margin": 0.15,
            "uphill_on_deg": 15.0, "uphill_off_deg": 12.0,
            "uphill_min_duration_sec": 0.25,
            "imu_timeout_sec": 0.20, "uphill_pitch_sign": 1.0,
            # /imu/data is already startup-level-calibrated in base_link by
            # imu_manager. Zero means use that calibrated vehicle pitch
            # directly; do not learn another relative reference here.
            "imu_reference_pitch_deg": 0.0,
            "publish_stop_line_tf": True,
            "debug_overlay_enabled": False,
            "publish_rate_hz": 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.depth_config = StopLineDepthConfig(
            int(self.p("stop_line_min_depth_pixels")),
            float(self.p("stop_line_depth_min_m")),
            float(self.p("stop_line_depth_max_m")),
            float(self.p("stop_line_depth_mad_threshold")),
            float(self.p("stop_line_max_depth_mad_m")))
        self.sign_filter = DebouncedPresence(PresenceConfig(
            float(self.p("sign_min_confidence")),
            float(self.p("sign_min_area_px")), int(self.p("sign_on_frames")),
            int(self.p("sign_off_frames")), float(self.p("sign_timeout_sec"))))
        self.traffic_filter = MissionTrafficFilter(MissionTrafficConfig(
            float(self.p("traffic_light_min_confidence")),
            int(self.p("traffic_light_on_frames")),
            int(self.p("traffic_light_switch_frames")),
            float(self.p("traffic_light_timeout_sec")),
            float(self.p("traffic_light_conflict_margin"))))
        reference = float(self.p("imu_reference_pitch_deg"))
        self.uphill = UphillDetector(UphillConfig(
            float(self.p("uphill_on_deg")), float(self.p("uphill_off_deg")),
            float(self.p("uphill_min_duration_sec")),
            float(self.p("uphill_pitch_sign"))),
            None if not math.isfinite(reference) else reference)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.depth_cache = deque(maxlen=12)
        self.camera_info = None
        self.camera_info_stamp = None
        self.color_cache = deque(maxlen=30)
        self.latest_imu_valid = False
        self.latest_imu_valid_wall = None
        self.latest_imu_wall = None
        self.latest_input_stamp = None
        self.latest_detection_wall = None
        self.stop_mask = None
        self.stop_samples = ()
        self.stop_distances = []
        self.stop = self._empty_stop("NOT_OBSERVED")
        self.sign_raw = False
        self.traffic_raw = "UNKNOWN"
        self.red_score = self.green_score = self.left_score = self.other_score = 0.0
        self.imu_pitch = self.imu_relative = math.nan
        self.imu_valid = False
        self.imu_reason = "NOT_OBSERVED"
        self.debug_enabled = bool(self.p("debug_overlay_enabled"))

        self.stop_detected_pub = self.create_publisher(
            Bool, "/camera/mission/stop_line_detected", 10)
        self.stop_distance_pub = self.create_publisher(
            Float32, "/camera/mission/stop_line_distance_m", 10)
        self.stop_count_pub = self.create_publisher(
            Int32, "/camera/mission/stop_line_count", 10)
        self.stop_distances_pub = self.create_publisher(
            Float32MultiArray, "/camera/mission/stop_line_distances_m", 10)
        self.sign_pub = self.create_publisher(
            Bool, "/camera/mission/sign_detected", 10)
        self.traffic_pub = self.create_publisher(
            String, "/camera/mission/traffic_light", 10)
        self.uphill_pub = self.create_publisher(
            Bool, "/camera/mission/uphill_detected", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/mission/diagnostics", 10)
        self.overlay_pub = self.create_publisher(
            Image, "/camera/mission/debug_overlay", 1)
        self.create_subscription(SemanticPathFrame, self.p("semantic_topic"),
                                 self.on_semantic, qos_profile_sensor_data)
        self.create_subscription(String, self.p("detections_topic"),
                                 self.on_detections, qos_profile_sensor_data)
        self.create_subscription(Image, self.p("depth_topic"),
                                 self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.p("camera_info_topic"),
                                 self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Imu, self.p("imu_topic"),
                                 self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Bool, self.p("imu_valid_topic"),
                                 self.on_imu_valid, 10)
        if self.debug_enabled:
            self.create_subscription(Image, self.p("color_topic"),
                                     self.on_color, qos_profile_sensor_data)
        self.create_timer(1.0/float(self.p("publish_rate_hz")), self.publish)

    def p(self, name):
        return self.get_parameter(name).value

    @staticmethod
    def _empty_stop(reason, raw=False, stamp=None):
        return {"raw": raw, "detected": False, "valid": False,
                "distance": math.nan, "pixels": 0, "median": math.nan,
                "tf_valid": False, "reason": reason, "stamp": stamp,
                "wall": time.monotonic(), "optical": None}

    def on_depth(self, message):
        try:
            self.depth_cache.append((stamp_sec(message.header.stamp), message,
                                     image_array(message)))
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)

    def on_camera_info(self, message):
        self.camera_info = message
        self.camera_info_stamp = stamp_sec(message.header.stamp)

    def on_color(self, message):
        try:
            array = image_array(message)
            if message.encoding.upper() == "RGB8":
                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            self.color_cache.append(
                (stamp_sec(message.header.stamp), message, array))
        except ValueError:
            pass

    def on_semantic(self, message):
        started = time.perf_counter()
        self.stop_distances = []
        stamp = stamp_sec(message.header.stamp)
        self.latest_input_stamp = stamp
        ros_now = self.get_clock().now().nanoseconds*1.0e-9
        if (stamp <= 0.0 or abs(ros_now-stamp) >
                float(self.p("max_input_age_sec"))):
            self.stop = self._empty_stop("STOP_INPUT_STALE", stamp=stamp)
            return
        try:
            mask = decode_binary_rle(message.stop_line_rle,
                                     message.image_height, message.image_width)
        except ValueError:
            self.stop = self._empty_stop("STOP_MASK_INVALID", stamp=stamp)
            return
        self.stop_mask = mask
        raw = bool(np.any(mask) and float(message.stop_line_confidence) >=
                   float(self.p("stop_line_min_confidence")))
        if not raw:
            self.stop = self._empty_stop("NO_STOP_LINE", stamp=stamp)
            return
        if not self.depth_cache:
            self.stop = self._empty_stop("DEPTH_MISSING", True, stamp)
            return
        depth_stamp, depth_message, depth = min(
            self.depth_cache, key=lambda item: abs(item[0]-stamp))
        if abs(depth_stamp-stamp) > float(self.p("sync_slop_sec")):
            self.stop = self._empty_stop("DEPTH_SYNC_MISMATCH", True, stamp)
            return
        if self.camera_info is None or self.camera_info_stamp is None:
            self.stop = self._empty_stop("CAMERA_INFO_MISSING", True, stamp)
            return
        if abs(self.camera_info_stamp-stamp) > float(self.p("max_input_age_sec")):
            self.stop = self._empty_stop("CAMERA_INFO_STALE", True, stamp)
            return
        estimates = self._component_distances(
            mask, depth, depth_message, message.header, stamp)
        if not estimates:
            whole = robust_stop_line_point(
                mask, depth, depth_message.encoding, self.camera_info.k,
                self.depth_config)
            self.stop = self._empty_stop(whole.reason, True, stamp)
            self.stop.update({"pixels": whole.valid_pixels,
                              "median": whole.median_depth_m})
            self.stop_distances = []
            return
        self.stop_distances = [item["distance"] for item in estimates]
        positive = [item for item in estimates if item["distance"] > 0.0]
        target = min(positive, key=lambda item: item["distance"]) \
            if positive else min(estimates, key=lambda item: abs(item["distance"]))
        distance = float(target["distance"])
        detected = distance > 0.0
        reason = "OK" if detected else "STOP_LINE_BEHIND_FRONT_AXLE"
        self.stop = {"raw": True, "detected": detected, "valid": detected,
                     "distance": distance if detected else math.nan,
                     "pixels": sum(item["pixels"] for item in estimates),
                     "median": target["median"],
                     "tf_valid": True, "reason": reason, "stamp": stamp,
                     "wall": time.monotonic(),
                     "optical": target["optical"],
                     "latency_ms": (time.perf_counter()-started)*1000.0}
        if detected and bool(self.p("publish_stop_line_tf")):
            self._publish_stop_tf(message.header.stamp, distance)
        self._publish_overlay(stamp)

    def _component_distances(self, mask, depth, depth_message, header, stamp):
        merge = max(1, int(self.p("stop_line_component_merge_px")))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*merge+1, 3))
        joined = cv2.morphologyEx((mask > 0).astype(np.uint8),
                                  cv2.MORPH_CLOSE, kernel)
        count, labels = cv2.connectedComponents(joined, 8)
        items = []
        samples = []
        source_frame = header.frame_id or depth_message.header.frame_id
        for label in range(1, count):
            # Closing only assigns fragments to a physical line. Synthetic
            # pixels introduced by closing must not contribute depth samples.
            component = np.where((labels == label) & (mask > 0),
                                 255, 0).astype(np.uint8)
            estimate = robust_stop_line_point(
                component, depth, depth_message.encoding, self.camera_info.k,
                self.depth_config)
            if not estimate.valid:
                continue
            distance, tf_valid, _ = self._front_axle_distance(
                estimate.optical_xyz_m, source_frame, header.stamp)
            if not tf_valid or not math.isfinite(distance):
                continue
            samples.extend(estimate.sample_pixels)
            items.append({"distance": float(distance),
                          "pixels": estimate.valid_pixels,
                          "median": estimate.median_depth_m,
                          "optical": estimate.optical_xyz_m})
        self.stop_samples = tuple(samples)
        items.sort(key=lambda item: abs(item["distance"]))
        distinct = []
        threshold = float(self.p("stop_line_distance_merge_m"))
        for item in items:
            if any(abs(item["distance"]-old["distance"]) <= threshold
                   for old in distinct):
                continue
            distinct.append(item)
        return distinct

    def _front_axle_distance(self, point, source_frame, stamp):
        target = str(self.p("base_frame"))
        try:
            transform = self.tf_buffer.lookup_transform(
                target, source_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=float(self.p("max_tf_age_sec"))))
            header_sec = stamp_sec(transform.header.stamp)
            source_sec = stamp_sec(stamp)
            if header_sec > 0 and abs(header_sec-source_sec) > float(
                    self.p("max_tf_age_sec")):
                return math.nan, False, "TF_STALE"
            t, q = transform.transform.translation, transform.transform.rotation
            output = transform_point(point, (t.x, t.y, t.z),
                                     (q.x, q.y, q.z, q.w))
            wheelbase = float(self.p("wheelbase_m"))
            configured_axle_x = float(self.p("front_axle_x_m"))
            expected_axle_x = 0.5*wheelbase
            if (not math.isfinite(configured_axle_x) or
                    abs(configured_axle_x-expected_axle_x) > 1.0e-9):
                return math.nan, False, "FRONT_AXLE_GEOMETRY_INVALID"
            distance = front_axle_distance_from_base_point(output, wheelbase)
            return distance, math.isfinite(distance), "OK"
        except TransformException:
            return math.nan, False, "BASE_LINK_TF_MISSING"

    def _publish_stop_tf(self, stamp, distance):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = str(self.p("base_frame"))
        message.child_frame_id = "detected_stop_line"
        message.transform.translation.x = (
            float(self.p("front_axle_x_m")) + float(distance))
        message.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(message)

    def on_detections(self, message):
        now = time.monotonic()
        self.latest_detection_wall = now
        try:
            document = json.loads(message.data)
            detections = document.get("detections", [])
            timestamp = document.get("timestamp", {})
            self.latest_input_stamp = float(timestamp.get("sec", 0)) + \
                float(timestamp.get("nanosec", 0))*1.0e-9
        except (ValueError, TypeError):
            detections = []
        sign_confidence = sign_area = 0.0
        scores = {"R": 0.0, "G": 0.0, "LEFT": 0.0, "OTHER": 0.0}
        for item in detections:
            name = str(item.get("class_name", "")).strip().lower()
            confidence = float(item.get("confidence", 0.0))
            box = item.get("xyxy", [])
            area = (max(0.0, float(box[2])-float(box[0]))*
                    max(0.0, float(box[3])-float(box[1]))
                    if len(box) == 4 else 0.0)
            # class_manifest.yaml declares only the real dataset class
            # ``traffic20``. Do not arm acceleration from guessed aliases.
            if name == "traffic20" and confidence > sign_confidence:
                sign_confidence, sign_area = confidence, area
            if name in ("r_light", "red_light"):
                scores["R"] = max(scores["R"], confidence)
            elif name in ("g_light", "green_light"):
                scores["G"] = max(scores["G"], confidence)
            elif name in ("left", "left_light", "left_sign"):
                scores["LEFT"] = max(scores["LEFT"], confidence)
            elif name in ("etc_light", "other_light"):
                scores["OTHER"] = max(scores["OTHER"], confidence)
        self.sign_raw = sign_confidence > 0.0
        self.sign_filter.update(self.sign_raw, sign_confidence, sign_area, now)
        self.red_score, self.green_score = scores["R"], scores["G"]
        self.left_score = scores["LEFT"]
        self.other_score = scores["OTHER"]
        if self.red_score and self.green_score:
            self.traffic_raw = "CONFLICT"
        elif self.red_score and self.left_score:
            self.traffic_raw = "LEFT"
        elif self.red_score:
            self.traffic_raw = "R"
        elif self.green_score:
            self.traffic_raw = "G"
        elif self.left_score or self.other_score:
            self.traffic_raw = "UNSUPPORTED"
        else:
            self.traffic_raw = "UNKNOWN"
        self.traffic_filter.update(
            self.red_score, self.green_score, self.left_score,
            self.other_score, now)

    def on_imu_valid(self, message):
        self.latest_imu_valid = bool(message.data)
        self.latest_imu_valid_wall = time.monotonic()

    def on_imu(self, message):
        now = time.monotonic(); self.latest_imu_wall = now
        covariance_valid = (len(message.orientation_covariance) == 9 and
                            message.orientation_covariance[0] >= 0.0 and
                            all(math.isfinite(v) for v in
                                message.orientation_covariance))
        valid_flag = (self.latest_imu_valid and
                      self.latest_imu_valid_wall is not None and
                      now-self.latest_imu_valid_wall <= float(
                          self.p("imu_timeout_sec")))
        if message.header.frame_id not in ("", str(self.p("base_frame"))):
            self.imu_valid = False; self.imu_reason = "IMU_FRAME_NOT_BASE"
            self.uphill.update(math.nan, False, now); return
        q = message.orientation
        try:
            pitch = pitch_deg_from_quaternion((q.x, q.y, q.z, q.w))
        except ValueError:
            pitch = math.nan
        self.imu_valid = bool(valid_flag and covariance_valid and
                              math.isfinite(pitch))
        self.imu_reason = "OK" if self.imu_valid else "IMU_INVALID"
        self.imu_pitch = pitch
        self.uphill.update(pitch, self.imu_valid, now)
        self.imu_relative = self.uphill.relative_deg

    def _publish_overlay(self, stamp):
        if not self.debug_enabled or not self.color_cache:
            return
        color_stamp, source, image = min(
            self.color_cache, key=lambda item: abs(item[0]-stamp))
        if abs(color_stamp-stamp) > float(self.p("sync_slop_sec")):
            return
        canvas = image.copy()
        if self.stop_mask is not None and self.stop_mask.shape == canvas.shape[:2]:
            canvas[self.stop_mask > 0] = (
                0.45*canvas[self.stop_mask > 0]+0.55*np.array([0, 255, 255])
            ).astype(np.uint8)
        for u, v in self.stop_samples:
            cv2.circle(canvas, (u, v), 1, (255, 0, 255), -1)
        lines = [
            f"STOP raw={self.stop['raw']} valid={self.stop['valid']} "
            f"distance={self.stop['distance']:.2f} reason={self.stop['reason']}",
            f"SIGN={self.sign_filter.detected} TL={self.traffic_filter.state}",
            f"pitch={self.imu_pitch:.1f} ref={self.uphill.reference_pitch_deg} "
            f"relative={self.imu_relative:.1f} uphill={self.uphill.uphill}",
        ]
        for index, text in enumerate(lines):
            cv2.putText(canvas, text, (10, 25+22*index),
                        cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2,
                        cv2.LINE_AA)
        output = Image()
        output.header = source.header; output.height, output.width = canvas.shape[:2]
        output.encoding = "bgr8"; output.is_bigendian = False
        output.step = output.width*3; output.data = canvas.tobytes()
        self.overlay_pub.publish(output)

    def publish(self):
        now = time.monotonic()
        self.sign_filter.tick(now); self.traffic_filter.tick(now)
        if (self.stop.get("wall") is None or
                now-self.stop["wall"] > float(self.p("max_input_age_sec"))):
            self.stop = self._empty_stop("STOP_INPUT_TIMEOUT")
            self.stop_distances = []
        if (self.latest_imu_wall is None or
                now-self.latest_imu_wall > float(self.p("imu_timeout_sec"))):
            self.imu_valid = False; self.imu_reason = "IMU_TIMEOUT"
            self.uphill.update(math.nan, False, now)
            self.imu_relative = math.nan
        self.stop_detected_pub.publish(Bool(data=bool(self.stop["detected"])))
        self.stop_distance_pub.publish(Float32(data=float(self.stop["distance"])))
        self.stop_count_pub.publish(Int32(data=len(self.stop_distances)))
        self.stop_distances_pub.publish(Float32MultiArray(
            data=[float(value) for value in self.stop_distances]))
        self.sign_pub.publish(Bool(data=bool(self.sign_filter.detected)))
        self.traffic_pub.publish(String(data=self.traffic_filter.state))
        self.uphill_pub.publish(Bool(data=bool(self.uphill.uphill)))
        if self.debug_enabled and self.latest_input_stamp is not None:
            self._publish_overlay(self.latest_input_stamp)
        sign_age = (math.inf if self.sign_filter.last_raw_at is None else
                    now-self.sign_filter.last_raw_at)
        traffic_age = (math.inf if self.traffic_filter.last_reliable_at is None
                       else now-self.traffic_filter.last_reliable_at)
        diagnostics = {
            "stop_line_raw_detected": self.stop["raw"],
            "stop_line_detected": self.stop["detected"],
            "stop_line_distance_valid": self.stop["valid"],
            "stop_line_distance_m": json_number(self.stop["distance"]),
            "stop_line_depth_valid_pixels": self.stop["pixels"],
            "stop_line_count": len(self.stop_distances),
            "stop_line_distances_m": self.stop_distances,
            "stop_line_depth_median_m": json_number(self.stop["median"]),
            "stop_line_tf_valid": self.stop["tf_valid"],
            "stop_line_failure_reason": self.stop["reason"],
            "sign_raw_detected": self.sign_raw,
            "sign_detected": self.sign_filter.detected,
            "sign_confidence": self.sign_filter.last_confidence,
            "sign_age_sec": json_number(sign_age),
            "traffic_light_raw_state": self.traffic_raw,
            "traffic_light_state": self.traffic_filter.state,
            "traffic_light_red_score": self.red_score,
            "traffic_light_green_score": self.green_score,
            "traffic_light_left_score": self.left_score,
            "traffic_light_left_requires_red": True,
            "traffic_light_conflict": self.traffic_filter.conflict,
            "traffic_light_age_sec": json_number(traffic_age),
            "imu_valid": self.imu_valid,
            "imu_pitch_deg": json_number(self.imu_pitch),
            "imu_vehicle_pitch_deg": json_number(self.imu_pitch),
            "imu_reference_pitch_deg": json_number(
                self.uphill.reference_pitch_deg),
            "imu_relative_uphill_deg": json_number(self.imu_relative),
            "uphill_detected": self.uphill.uphill,
            "imu_failure_reason": self.imu_reason,
            "input_timestamp": self.latest_input_stamp,
            "processing_latency_ms": self.stop.get("latency_ms", 0.0),
        }
        self.diag_pub.publish(String(data=json.dumps(
            diagnostics, separators=(",", ":"), allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = CameraMissionPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
