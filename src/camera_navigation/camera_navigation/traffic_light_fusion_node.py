#!/usr/bin/env python3
"""Fuse independent YOLO and RGB traffic-light results; no control output."""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from .traffic_light_fusion_core import (
    FusionConfig, TrafficLightFusion, normalize_rgb_diagnostics,
    normalize_yolo_document)


class TrafficLightFusionNode(Node):
    def __init__(self):
        super().__init__("traffic_light_fusion_node")
        defaults = {
            "yolo_detections_topic": "/perception/detections_json",
            "yolo_state_topic": "/camera_traffic_light",
            "rgb_state_topic": "/camera/traffic_light_rgb/state",
            "rgb_aspect_topic": "/camera/traffic_light_rgb/aspect",
            "rgb_confidence_topic": "/camera/traffic_light_rgb/confidence",
            "rgb_diagnostics_topic": "/camera/traffic_light_rgb/diagnostics",
            "route_mode_topic": "/mcu/current_mode",
            "fused_state_topic": "/camera/traffic_light_fused/state",
            "fused_aspect_topic": "/camera/traffic_light_fused/aspect",
            "fused_confidence_topic": "/camera/traffic_light_fused/confidence",
            "fused_diagnostics_topic": "/camera/traffic_light_fused/diagnostics",
            "publish_rate_hz": 20.0,
        }
        config_defaults = FusionConfig()
        defaults.update(vars(config_defaults))
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.fusion = TrafficLightFusion(FusionConfig(**{
            name: self.get_parameter(name).value
            for name in vars(config_defaults)
        }))
        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.yolo_sequence = 0
        self.rgb_sequence = 0
        self.route_mode = None
        self.raw_yolo_state = "UNKNOWN"
        self.raw_rgb_state = "UNKNOWN"
        self.raw_rgb_aspect = "UNKNOWN"
        self.raw_rgb_confidence = 0.0

        self.state_pub = self.create_publisher(
            String, self.p("fused_state_topic"), 10)
        self.aspect_pub = self.create_publisher(
            String, self.p("fused_aspect_topic"), 10)
        self.confidence_pub = self.create_publisher(
            Float32, self.p("fused_confidence_topic"), 10)
        self.diagnostics_pub = self.create_publisher(
            String, self.p("fused_diagnostics_topic"), 10)
        self.create_subscription(
            String, self.p("yolo_detections_topic"), self.on_yolo, 10)
        self.create_subscription(
            String, self.p("yolo_state_topic"),
            lambda message: setattr(self, "raw_yolo_state",
                                    str(message.data).upper()), 10)
        self.create_subscription(
            String, self.p("rgb_state_topic"),
            lambda message: setattr(self, "raw_rgb_state",
                                    str(message.data).upper()), 10)
        self.create_subscription(
            String, self.p("rgb_aspect_topic"),
            lambda message: setattr(self, "raw_rgb_aspect",
                                    str(message.data).upper()), 10)
        self.create_subscription(
            Float32, self.p("rgb_confidence_topic"),
            lambda message: setattr(self, "raw_rgb_confidence",
                                    float(message.data)), 10)
        self.create_subscription(
            String, self.p("rgb_diagnostics_topic"), self.on_rgb, 10)
        self.create_subscription(
            String, self.p("route_mode_topic"),
            lambda message: setattr(self, "route_mode", str(message.data)), 10)
        self.create_timer(1.0/rate, self.publish)
        self.get_logger().info(
            "traffic-light fusion ready: YOLO detections and RGB diagnostics "
            "are independent latest-result inputs")

    def p(self, name):
        return str(self.get_parameter(name).value)

    def on_yolo(self, message):
        self.yolo_sequence += 1
        try:
            document = json.loads(message.data)
        except (TypeError, ValueError):
            self.fusion._reject("YOLO_JSON_INVALID")
            return
        observation, reason = normalize_yolo_document(
            document, time.monotonic(), self.yolo_sequence)
        if reason != "OK":
            self.fusion._reject(reason)
        if observation is not None:
            self.fusion.ingest(observation)

    def on_rgb(self, message):
        self.rgb_sequence += 1
        try:
            document = json.loads(message.data)
        except (TypeError, ValueError):
            self.fusion._reject("RGB_JSON_INVALID")
            return
        observation, reason = normalize_rgb_diagnostics(
            document, time.monotonic(), self.rgb_sequence)
        if reason != "OK":
            self.fusion._reject(reason)
        if observation is not None:
            self.fusion.ingest(observation)

    def publish(self):
        decision = self.fusion.evaluate(time.monotonic(), self.route_mode)
        diagnostics = decision.diagnostics
        diagnostics["yolo_published_state"] = self.raw_yolo_state
        diagnostics["rgb_published_state"] = self.raw_rgb_state
        diagnostics["rgb_published_aspect"] = self.raw_rgb_aspect
        diagnostics["rgb_published_confidence"] = (
            self.raw_rgb_confidence if self.raw_rgb_confidence ==
            self.raw_rgb_confidence and abs(self.raw_rgb_confidence) != float("inf")
            else None)
        self.state_pub.publish(String(data=decision.state))
        self.aspect_pub.publish(String(data=decision.aspect))
        self.confidence_pub.publish(Float32(data=decision.confidence))
        self.diagnostics_pub.publish(String(data=json.dumps(
            diagnostics, separators=(",", ":"), sort_keys=True,
            allow_nan=False)))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightFusionNode()
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
