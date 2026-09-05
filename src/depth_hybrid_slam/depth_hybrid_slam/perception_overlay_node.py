"""Compose existing mission overlay, traffic bbox, and VSLAM verdict."""

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


def bgr_image(message):
    if message.encoding.upper() not in ("BGR8", "RGB8"):
        raise ValueError(f"unsupported overlay encoding: {message.encoding}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    rows = raw[:message.height*message.step].reshape(message.height, message.step)
    image = rows[:, :message.width*3].reshape(
        message.height, message.width, 3).copy()
    if message.encoding.upper() == "RGB8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class PerceptionOverlayNode(Node):
    def __init__(self):
        super().__init__("perception_vslam_overlay")
        self.declare_parameter("input_overlay_topic",
                               "/camera/mission/debug_overlay")
        self.declare_parameter("detections_topic", "/perception/detections_json")
        self.declare_parameter(
            "status_topic", "/depth_slam/perception/cross_validation_status")
        self.declare_parameter("maximum_stamp_delta_sec", 0.08)
        self.detection = None
        self.status = {}
        self.publisher = self.create_publisher(
            Image, "/depth_slam/perception/debug_overlay", 1)
        self.create_subscription(
            Image, str(self.get_parameter("input_overlay_topic").value),
            self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            String, str(self.get_parameter("detections_topic").value),
            self.on_detection, 10)
        self.create_subscription(
            String, str(self.get_parameter("status_topic").value),
            self.on_status, 10)

    def on_detection(self, message):
        try:
            document = json.loads(message.data)
            stamp = json_stamp_seconds(document)
        except (KeyError, TypeError, ValueError):
            return
        self.detection = (stamp, document)

    def on_status(self, message):
        try:
            self.status = json.loads(message.data)
        except (TypeError, ValueError):
            self.status = {}

    @staticmethod
    def _text(canvas, value, origin, color):
        cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, color, 1, cv2.LINE_AA)

    def on_image(self, message):
        try:
            canvas = bgr_image(message)
        except ValueError:
            return
        stamp = float(message.header.stamp.sec)+float(
            message.header.stamp.nanosec)*1.0e-9
        traffic = self.status.get("traffic_light", {})
        expected = traffic.get("state", "")
        selected = None
        if self.detection is not None:
            detection_stamp, document = self.detection
            if abs(detection_stamp-stamp) <= float(
                    self.get_parameter("maximum_stamp_delta_sec").value):
                selected = select_traffic_detection(document, expected)
                if selected is None:
                    selected = select_traffic_detection(document)
        valid = bool(traffic.get("valid"))
        color = (70, 230, 70) if valid else (0, 190, 255)
        if selected is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in selected["bbox"])
            x1, x2 = max(0, x1), min(canvas.shape[1]-1, x2)
            y1, y2 = max(0, y1), min(canvas.shape[0]-1, y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = (f"TL bbox={selected['class_name']} state={expected or '?'} "
                     f"conf={float(traffic.get('confidence', 0.0)):.2f}")
            self._text(canvas, label, (x1, max(18, y1-6)), color)
        stop = self.status.get("stop_line", {})
        distance = stop.get("front_axle_distance_m")
        distance_text = ("n/a" if distance is None or not isinstance(
            distance, (int, float)) or not math.isfinite(float(distance))
            else f"{float(distance):.2f} m")
        lines = [
            (f"FINAL TL: {'VALID' if valid else 'UNCONFIRMED'} "
             f"state={traffic.get('state', 'UNKNOWN')} "
             f"confidence={float(traffic.get('confidence', 0.0)):.2f} "
             f"reason={traffic.get('reason', 'NO_INPUT')}"),
            (f"STOP LINE: {'VALID' if stop.get('valid') else 'UNCONFIRMED'} "
             f"front_axle_distance={distance_text} "
             f"reason={stop.get('reason', 'NO_INPUT')}"),
            (f"VSLAM: tracking={self.status.get('cuvslam_tracking_valid', False)} "
             f"mapping={self.status.get('mapping_state', 'MISSING')} "
             f"localization={self.status.get('localization_state', 'MISSING')}"),
        ]
        y0 = max(20, canvas.shape[0]-62)
        cv2.rectangle(canvas, (0, y0-18),
                      (canvas.shape[1]-1, canvas.shape[0]-1), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            line_color = color if index == 0 else (255, 255, 255)
            self._text(canvas, line, (8, y0+20*index), line_color)
        output = Image()
        output.header = message.header
        output.height, output.width = canvas.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width*3
        output.data = canvas.tobytes()
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionOverlayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
