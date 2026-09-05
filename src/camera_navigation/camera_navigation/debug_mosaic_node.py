#!/usr/bin/env python3
"""Validation-only latest-frame mosaic for camera/BEV/command inspection."""

from collections import deque
import json
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32, String


PANELS = (
    ("ORIGINAL", "/camera/image_raw"),
    ("DETECTIONS", "/perception/detections_image"),
    ("PERCEPTION OVERLAY", "/camera/perception_overlay_image"),
    ("RAW ROAD", "/perception/masks/road"),
    ("REFINED ROAD", "/perception/refined/road"),
    ("WHITE LINE", "/perception/masks/white_line"),
    ("YELLOW LINE", "/perception/masks/yellow_line"),
    ("BEV OVERLAY", "/camera/bev/overlay_image"),
    ("CAMERA OVERLAY", "/camera/bev/camera_overlay"),
)


class FrameSlot:
    def __init__(self):
        self.image = None
        self.error = "NO DATA"
        self.encoding = ""
        self.stamp_ns = 0
        self.wall_time = 0.0
        self.arrivals = deque(maxlen=60)

    def fps(self):
        if len(self.arrivals) < 2:
            return 0.0
        elapsed = self.arrivals[-1] - self.arrivals[0]
        return 0.0 if elapsed <= 0.0 else (len(self.arrivals) - 1) / elapsed


def status_text(slot, now, stale_sec):
    if slot.image is None:
        return slot.error or "NO DATA"
    age = now - slot.wall_time
    if age > stale_sec:
        return "STALE"
    if slot.image.size == 0:
        return "ENCODING ERROR"
    if slot.encoding.lower() in ("mono8", "8uc1") and not np.any(slot.image):
        return "EMPTY MASK"
    return "LIVE"


class DebugMosaicNode(Node):
    def __init__(self):
        super().__init__("camera_debug_mosaic")
        self.declare_parameter("publish_hz", 4.0)
        self.declare_parameter("stale_sec", 1.0)
        self.declare_parameter("panel_width", 426)
        self.declare_parameter("panel_height", 240)
        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0 or publish_hz > 5.0:
            raise ValueError("publish_hz must be in (0, 5]")
        self.stale_sec = float(self.get_parameter("stale_sec").value)
        self.panel_size = (int(self.get_parameter("panel_width").value),
                           int(self.get_parameter("panel_height").value))
        self.bridge = CvBridge()
        self.slots = {topic: FrameSlot() for _, topic in PANELS}
        self.values = {
            "planner_variant": "-", "planner_state": "-",
            "failure_reason": "-", "camera_drive": "-", "camera_wheel": "-",
            "mcu_drive": "-", "mcu_wheel": "-", "drive_owner": "-",
            "wheel_owner": "-", "semantic_fps": "-", "planner_fps": "-",
            "pipeline_latency_ms": "-", "source_stamp_ns": "-",
        }
        input_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        output_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self._subscriptions = []
        for _, topic in PANELS:
            self._subscriptions.append(self.create_subscription(
                Image, topic, lambda message, name=topic: self._image(name, message),
                input_qos))
        self._subscribe_values()
        self.publisher = self.create_publisher(
            Image, "/camera/debug/mosaic", output_qos)
        self.create_timer(1.0 / publish_hz, self._publish)

    def _subscribe_values(self):
        self.create_subscription(Float32, "/camera_drive",
                                 lambda m: self._set("camera_drive", f"{m.data:.1f}"), 10)
        self.create_subscription(Int32, "/camera_wheel",
                                 lambda m: self._set("camera_wheel", str(m.data)), 10)
        self.create_subscription(Float32, "/mcu/cmd_drive",
                                 lambda m: self._set("mcu_drive", f"{m.data:.1f}"), 10)
        self.create_subscription(Int32, "/mcu/cmd_wheel",
                                 lambda m: self._set("mcu_wheel", str(m.data)), 10)
        self.create_subscription(String, "/mcu/active_drive_source",
                                 lambda m: self._set("drive_owner", m.data), 10)
        self.create_subscription(String, "/mcu/active_wheel_source",
                                 lambda m: self._set("wheel_owner", m.data), 10)
        self.create_subscription(Float32, "/camera/pipeline_latency_ms",
                                 lambda m: self._set("pipeline_latency_ms", f"{m.data:.1f}"), 10)
        self.create_subscription(String, "/camera/bev/diagnostics", self._diagnostics, 10)
        self.create_subscription(String, "/camera/realtime_fps", self._realtime, 10)

    def _set(self, key, value):
        self.values[key] = value

    def _image(self, topic, message):
        slot = self.slots[topic]
        now = time.monotonic()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            if image.ndim not in (2, 3):
                raise ValueError(f"unsupported shape {image.shape}")
            slot.image = np.asarray(image).copy()
            slot.error = ""
        except Exception as error:  # CvBridge reports malformed step/data here.
            slot.image = None
            slot.error = "ENCODING ERROR"
            self.get_logger().warning(f"{topic}: {error}", throttle_duration_sec=2.0)
        slot.encoding = str(message.encoding)
        slot.stamp_ns = (int(message.header.stamp.sec) * 1_000_000_000 +
                         int(message.header.stamp.nanosec))
        slot.wall_time = now
        slot.arrivals.append(now)

    def _diagnostics(self, message):
        try:
            data = json.loads(message.data)
        except (TypeError, ValueError):
            return
        reasons = data.get("reasons") or []
        self.values.update({
            "planner_variant": str(data.get("planner_variant", data.get("variant", "-"))),
            "planner_state": str(data.get("state", "-")),
            "failure_reason": "|".join(map(str, reasons)) if reasons else "-",
            "planner_fps": f"{float(data.get('planner_processing_fps', 0.0)):.1f}",
            "source_stamp_ns": str(data.get("source_stamp_ns", data.get("stamp_ns", "-"))),
        })
        semantic_fps = data.get("semantic_input_fps")
        if semantic_fps is not None:
            self.values["semantic_fps"] = f"{float(semantic_fps):.1f}"

    def _realtime(self, message):
        try:
            data = json.loads(message.data)
            semantic = data.get("semantic_unique_fps", {}).get("1s", {})
            value = semantic.get("header_fps")
            if value is not None:
                self.values["semantic_fps"] = f"{float(value):.1f}"
        except (TypeError, ValueError):
            return

    def _panel(self, title, topic, now):
        width, height = self.panel_size
        slot = self.slots[topic]
        state = status_text(slot, now, self.stale_sec)
        if state in ("NO DATA", "STALE", "ENCODING ERROR"):
            panel = np.full((height, width, 3), 72, np.uint8)
        elif state == "EMPTY MASK":
            panel = np.full((height, width, 3), 48, np.uint8)
        else:
            image = slot.image
            if image.ndim == 2:
                image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
            elif slot.encoding.lower() == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        age = float("inf") if not slot.wall_time else max(0.0, now-slot.wall_time)
        cv2.rectangle(panel, (0, 0), (width, 43), (0, 0, 0), -1)
        cv2.putText(panel, f"{title}  {state}", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (60, 255, 60) if state == "LIVE" else (40, 210, 255), 1,
                    cv2.LINE_AA)
        age_text = "inf" if not np.isfinite(age) else f"{age:.2f}s"
        cv2.putText(panel, f"{topic}  {slot.fps():.1f}Hz age={age_text}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1,
                    cv2.LINE_AA)
        return panel

    def _publish(self):
        now = time.monotonic()
        panels = [self._panel(title, topic, now) for title, topic in PANELS]
        rows = [np.hstack(panels[index:index+3]) for index in (0, 3, 6)]
        mosaic = np.vstack(rows)
        width = mosaic.shape[1]
        footer = np.full((104, width, 3), 20, np.uint8)
        line1 = (f"variant={self.values['planner_variant']}  state={self.values['planner_state']}  "
                 f"reason={self.values['failure_reason']}")
        line2 = (f"camera drive/wheel={self.values['camera_drive']}/{self.values['camera_wheel']}  "
                 f"mcu={self.values['mcu_drive']}/{self.values['mcu_wheel']}  "
                 f"owners={self.values['drive_owner']}/{self.values['wheel_owner']}")
        line3 = (f"semantic={self.values['semantic_fps']}Hz  planner={self.values['planner_fps']}Hz  "
                 f"latency={self.values['pipeline_latency_ms']}ms  "
                 f"source_stamp_ns={self.values['source_stamp_ns']}")
        for y, text in ((26, line1), (57, line2), (88, line3)):
            cv2.putText(footer, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
        mosaic = np.vstack((mosaic, footer))
        message = self.bridge.cv2_to_imgmsg(mosaic, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_debug_mosaic"
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = DebugMosaicNode()
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
