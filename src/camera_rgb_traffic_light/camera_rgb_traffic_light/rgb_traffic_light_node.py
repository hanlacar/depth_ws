"""Latest-frame-only ROS 2 wrapper for CPU color traffic-light detection."""

import json
import math
import signal
import threading
import time

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .detector import (
    STATES, ColorTrafficLightDetector, DetectorConfig, TemporalConfig,
    TemporalTrafficLightFilter, build_diagnostics, normalize_to_bgr)


class EventRate:
    def __init__(self, window_sec=2.0):
        self.window = float(window_sec)
        self.events = []

    def tick(self, now):
        self.events.append(float(now))
        cutoff = now-self.window
        while self.events and self.events[0] < cutoff:
            self.events.pop(0)

    def rate(self):
        if len(self.events) < 2:
            return 0.0
        return (len(self.events)-1)/max(self.events[-1]-self.events[0], 1.0e-6)


class RgbTrafficLightNode(Node):
    def __init__(self):
        super().__init__("rgb_traffic_light_node")
        self.declare_parameter("input_image_topic", "/camera/image_raw")
        self.declare_parameter("state_topic", "/camera/traffic_light_rgb/state")
        self.declare_parameter("confidence_topic", "/camera/traffic_light_rgb/confidence")
        self.declare_parameter("aspect_topic", "/camera/traffic_light_rgb/aspect")
        self.declare_parameter("diagnostics_topic", "/camera/traffic_light_rgb/diagnostics")
        self.declare_parameter("overlay_topic", "/camera/traffic_light_rgb/overlay_image")
        self.declare_parameter("watchdog_rate_hz", 10.0)
        detector_defaults = DetectorConfig()
        temporal_defaults = TemporalConfig()
        for name, default in detector_defaults.__dict__.items():
            self.declare_parameter(name, default)
        for name, default in temporal_defaults.__dict__.items():
            self.declare_parameter(name, default)
        detector_config = DetectorConfig(**{
            name: self.get_parameter(name).value
            for name in detector_defaults.__dict__})
        temporal_config = TemporalConfig(**{
            name: self.get_parameter(name).value
            for name in temporal_defaults.__dict__})
        self.detector = ColorTrafficLightDetector(detector_config)
        self.temporal = TemporalTrafficLightFilter(temporal_config)
        self.bridge = CvBridge()
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10)
        self.confidence_pub = self.create_publisher(
            Float32, str(self.get_parameter("confidence_topic").value), 10)
        self.aspect_pub = self.create_publisher(
            String, str(self.get_parameter("aspect_topic").value), 10)
        self.diagnostics_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10)
        self.overlay_pub = self.create_publisher(
            Image, str(self.get_parameter("overlay_topic").value), 1)
        self.create_subscription(
            Image, str(self.get_parameter("input_image_topic").value),
            self._on_image, qos_profile_sensor_data)
        watchdog_hz = float(self.get_parameter("watchdog_rate_hz").value)
        if not math.isfinite(watchdog_hz) or watchdog_hz <= 0.0:
            raise ValueError("watchdog_rate_hz must be finite and positive")
        self.create_timer(1.0/watchdog_hz, self._watchdog)

        self.slot_condition = threading.Condition()
        self.latest_message = None
        self.latest_receive_monotonic = None
        self.stop_worker = False
        self.processing_lock = threading.Lock()
        self.input_rate = EventRate()
        self.processing_rate = EventRate()
        self.last_timeout_publish = 0.0
        self.worker = threading.Thread(
            target=self._worker_loop, name="rgb-traffic-light-latest", daemon=True)
        self.worker.start()
        self.get_logger().info(
            "CPU RGB/HSV/Lab traffic-light detector ready; YOLO and vehicle "
            "control topics are not used")

    @staticmethod
    def _stamp_ns(message):
        return (int(message.header.stamp.sec)*1_000_000_000+
                int(message.header.stamp.nanosec))

    def _on_image(self, message):
        now = time.monotonic()
        with self.slot_condition:
            self.latest_message = message
            self.latest_receive_monotonic = now
            self.input_rate.tick(now)
            self.slot_condition.notify()

    def _take_latest(self):
        with self.slot_condition:
            while self.latest_message is None and not self.stop_worker:
                self.slot_condition.wait(timeout=0.2)
            if self.stop_worker:
                return None, None
            message = self.latest_message
            received = self.latest_receive_monotonic
            self.latest_message = None
            return message, received

    def _worker_loop(self):
        while True:
            message, received = self._take_latest()
            if message is None:
                return
            started = time.monotonic()
            try:
                array = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
                bgr = normalize_to_bgr(array, message.encoding)
                result = self.detector.detect(bgr)
                finished = time.monotonic()
                with self.processing_lock:
                    decision = self.temporal.update(result, received, bgr.shape)
                    self.processing_rate.tick(finished)
                    self._publish_result(
                        message, bgr, result, decision, received,
                        (finished-started)*1000.0)
            except Exception as error:
                with self.processing_lock:
                    decision = self.temporal.force_unknown("UNKNOWN")
                    self._publish_unknown(
                        "processing_exception", received, str(error), message)
                self.get_logger().error(f"RGB traffic-light processing failed: {error!r}")

    def _publish_result(self, message, bgr, result, decision, received, latency_ms):
        if decision.state not in STATES:
            self.temporal.force_unknown("UNKNOWN")
            self._publish_unknown("invalid_control_state", received, "", message)
            return
        now = time.monotonic()
        input_age_ms = max(0.0, (now-received)*1000.0)
        try:
            payload = build_diagnostics(
                self._stamp_ns(message), result, decision, input_age_ms,
                latency_ms, self.input_rate.rate(), self.processing_rate.rate())
        except ValueError:
            self.temporal.force_unknown("UNKNOWN")
            self._publish_unknown("non_finite_diagnostics", received, "", message)
            return
        self.state_pub.publish(String(data=decision.state))
        self.aspect_pub.publish(String(data=decision.aspect))
        self.confidence_pub.publish(Float32(data=float(decision.confidence)))
        self.diagnostics_pub.publish(String(
            data=json.dumps(payload, separators=(",", ":"), sort_keys=True)))
        if self.overlay_pub.get_subscription_count() > 0:
            overlay = self.detector.render_overlay(
                bgr, result, decision.state, decision.confirmation_count,
                self.input_rate.rate(), self.processing_rate.rate(), latency_ms)
            output = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            output.header = message.header
            self.overlay_pub.publish(output)

    def _publish_unknown(self, reason, received=None, detail="", message=None):
        now = time.monotonic()
        age_ms = (None if received is None else max(0.0, (now-received)*1000.0))
        payload = {
            "stamp": 0 if message is None else self._stamp_ns(message),
            "state": "UNKNOWN", "aspect": "UNKNOWN",
            "raw_aspect": "UNKNOWN", "confidence": 0.0,
            "input_age_ms": age_ms,
            "processing_latency_ms": 0.0,
            "candidate_count": 0, "red_candidates": 0,
            "yellow_candidates": 0, "green_candidates": 0,
            "left_candidates": 0, "down_candidates": 0,
            "raw_color": "UNKNOWN", "raw_shape": "UNKNOWN_SHAPE",
            "raw_state": "UNKNOWN", "green_shape_score": 0.0,
            "circle_score": 0.0, "left_arrow_score": 0.0,
            "down_arrow_score": 0.0,
            "selected_bbox": [],
            "confirmed_state": "UNKNOWN", "confirmation_count": 0,
            "rejection_reasons": {reason: 1}, "roi": [],
            "detail": detail[:160], "input_fps": self.input_rate.rate(),
            "processing_fps": self.processing_rate.rate(),
        }
        self.state_pub.publish(String(data="UNKNOWN"))
        self.aspect_pub.publish(String(data="UNKNOWN"))
        self.confidence_pub.publish(Float32(data=0.0))
        self.diagnostics_pub.publish(String(
            data=json.dumps(payload, separators=(",", ":"), sort_keys=True)))

    def _watchdog(self):
        now = time.monotonic()
        with self.processing_lock:
            decision = self.temporal.tick(now)
            if decision.state != "UNKNOWN":
                return
            if now-self.last_timeout_publish < 0.5:
                return
            self.last_timeout_publish = now
            reason = ("waiting_for_input" if self.latest_receive_monotonic is None
                      else "input_timeout")
            self._publish_unknown(reason, self.latest_receive_monotonic)

    def destroy_node(self):
        with self.slot_condition:
            self.stop_worker = True
            self.latest_message = None
            self.slot_condition.notify_all()
        self.worker.join(timeout=2.0)
        with self.processing_lock:
            self.temporal.force_unknown()
            self._publish_unknown("shutdown", self.latest_receive_monotonic)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = None
    try:
        node = RgbTrafficLightNode()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
