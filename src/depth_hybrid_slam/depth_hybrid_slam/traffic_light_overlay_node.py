"""Render YOLO, independent RGB, and fused traffic-light evidence on D456 RGB."""

import json
import math

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .perception_vslam_core import json_stamp_seconds, select_traffic_detection
from .ros_helpers import safe_shutdown


def image_to_bgr(message):
    if message.encoding.upper() not in ("RGB8", "BGR8"):
        raise ValueError(f"unsupported color encoding: {message.encoding}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    rows = raw[:message.height*message.step].reshape(message.height, message.step)
    image = rows[:, :message.width*3].reshape(
        message.height, message.width, 3).copy()
    if message.encoding.upper() == "RGB8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class TrafficLightOverlayNode(Node):
    def __init__(self):
        super().__init__("traffic_light_live_overlay")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("detections_topic", "/perception/detections_json")
        self.declare_parameter(
            "rgb_diagnostics_topic", "/camera/traffic_light_rgb/diagnostics")
        self.declare_parameter(
            "fusion_diagnostics_topic", "/camera/traffic_light_fused/diagnostics")
        self.declare_parameter("maximum_age_sec", 0.25)
        self.detection = None
        self.rgb = None
        self.fusion = None
        self.output_pub = self.create_publisher(
            Image, "/depth_slam/traffic_light/debug_overlay", 1)
        self.status_pub = self.create_publisher(
            String, "/depth_slam/traffic_light/live_status", 10)
        self.create_subscription(
            Image, self.p("color_topic"), self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            String, self.p("detections_topic"), self.on_detection, 10)
        self.create_subscription(
            String, self.p("rgb_diagnostics_topic"), self.on_rgb, 10)
        self.create_subscription(
            String, self.p("fusion_diagnostics_topic"), self.on_fusion, 10)

    def p(self, name):
        return str(self.get_parameter(name).value)

    @staticmethod
    def _read(message):
        try:
            document = json.loads(message.data)
            return document, json_stamp_seconds(document)
        except (KeyError, TypeError, ValueError):
            return None

    def on_detection(self, message):
        value = self._read(message)
        if value is not None:
            self.detection = value

    def on_rgb(self, message):
        value = self._read(message)
        if value is not None:
            self.rgb = value

    def on_fusion(self, message):
        value = self._read(message)
        if value is not None:
            self.fusion = value

    @staticmethod
    def _draw_text(image, text, position, color):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)

    def _fresh(self, item, stamp):
        return (item is not None and
                abs(float(item[1])-float(stamp)) <= float(
                    self.get_parameter("maximum_age_sec").value))

    def on_image(self, message):
        try:
            canvas = image_to_bgr(message)
        except ValueError:
            return
        stamp = float(message.header.stamp.sec)+float(
            message.header.stamp.nanosec)*1.0e-9
        fusion = self.fusion[0] if self._fresh(self.fusion, stamp) else {}
        rgb = self.rgb[0] if self._fresh(self.rgb, stamp) else {}
        selected = None
        if self._fresh(self.detection, stamp):
            selected = select_traffic_detection(
                self.detection[0], fusion.get("fused_state", ""))
            if selected is None:
                selected = select_traffic_detection(self.detection[0])
        fused_state = str(fusion.get("fused_state", "UNKNOWN"))
        fused_confidence = float(fusion.get("fused_confidence", 0.0) or 0.0)
        agreed = bool(fusion.get("sources_agree"))
        valid = fused_state in ("R", "G") and agreed
        color = (50, 230, 50) if valid else (0, 190, 255)
        if selected is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in selected["bbox"])
            x1, x2 = max(0, x1), min(canvas.shape[1]-1, x2)
            y1, y2 = max(0, y1), min(canvas.shape[0]-1, y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            self._draw_text(
                canvas,
                f"YOLO {selected['class_name']} {selected['confidence']:.2f}",
                (x1, max(18, y1-6)), color)
        lines = [
            (f"YOLO: {fusion.get('yolo_state', 'UNKNOWN')} "
             f"confidence={float(fusion.get('yolo_confidence', 0.0) or 0.0):.2f}"),
            (f"RGB: {rgb.get('state', fusion.get('rgb_state', 'UNKNOWN'))} "
             f"aspect={rgb.get('aspect', fusion.get('rgb_aspect', 'UNKNOWN'))} "
             f"confidence={float(rgb.get('confidence', fusion.get('rgb_confidence', 0.0)) or 0.0):.2f}"),
            (f"FUSED: {fused_state} confidence={fused_confidence:.2f} "
             f"decision={'VALID' if valid else 'UNCONFIRMED'} "
             f"reason={fusion.get('fusion_reason', 'WAITING_FOR_INPUT')}"),
        ]
        cv2.rectangle(canvas, (0, canvas.shape[0]-78),
                      (canvas.shape[1]-1, canvas.shape[0]-1), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            self._draw_text(canvas, line,
                            (8, canvas.shape[0]-55+24*index),
                            color if index == 2 else (255, 255, 255))
        status = {
            "stamp": {"sec": int(message.header.stamp.sec),
                      "nanosec": int(message.header.stamp.nanosec)},
            "yolo_bbox_present": selected is not None,
            "rgb_fresh": bool(rgb), "fusion_fresh": bool(fusion),
            "fused_state": fused_state,
            "fused_confidence": (fused_confidence
                                 if math.isfinite(fused_confidence) else 0.0),
            "sources_agree": agreed,
            "decision": "VALID" if valid else "UNCONFIRMED",
            "reason": fusion.get("fusion_reason", "WAITING_FOR_INPUT"),
        }
        output = Image()
        output.header = message.header
        output.height, output.width = canvas.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width*3
        output.data = canvas.tobytes()
        self.output_pub.publish(output)
        self.status_pub.publish(String(data=json.dumps(
            status, separators=(",", ":"), sort_keys=True, allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightOverlayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
