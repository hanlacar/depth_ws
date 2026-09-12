"""Explicit /slam_* candidate to current MCU /gps_* source-topic adapter."""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .mcu_source_adapter_core import adapt_slam_command


class McuSourceAdapterNode(Node):
    def __init__(self):
        super().__init__("depth_slam_mcu_source_adapter")
        for name, default in (
                ("armed", False), ("publish_hz", 20.0),
                ("command_timeout_s", 0.5), ("max_wheel_deg", 22),
                ("input_drive_topic", "/slam_drive"),
                ("input_wheel_topic", "/slam_wheel"),
                ("input_stop_topic", "/slam_stop"),
                ("output_drive_topic", "/gps_drive"),
                ("output_wheel_topic", "/gps_wheel"),
                ("output_stop_topic", "/gps_stop")):
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value
        self.armed = bool(p("armed"))
        self.timeout = float(p("command_timeout_s"))
        self.max_wheel = int(p("max_wheel_deg"))
        self.values = {"drive": 0.0, "wheel": 0, "stop": True}
        self.received = {"drive": None, "wheel": None, "stop": None}
        self.publisher_conflicts = []
        self.create_subscription(
            Float32, str(p("input_drive_topic")),
            lambda msg: self._update("drive", float(msg.data)), 10)
        self.create_subscription(
            Int32, str(p("input_wheel_topic")),
            lambda msg: self._update("wheel", int(msg.data)), 10)
        self.create_subscription(
            Bool, str(p("input_stop_topic")),
            lambda msg: self._update("stop", bool(msg.data)), 10)
        self.output_topics = (str(p("output_drive_topic")),
                              str(p("output_wheel_topic")),
                              str(p("output_stop_topic")))
        self.pub_drive = self.create_publisher(Float32, self.output_topics[0], 10)
        self.pub_wheel = self.create_publisher(Int32, self.output_topics[1], 10)
        self.pub_stop = self.create_publisher(Bool, self.output_topics[2], 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/mcu_adapter/state", 10)
        self.pub_diag = self.create_publisher(
            String, "/depth_slam/mcu_adapter/diagnostics", 10)
        hz = float(p("publish_hz"))
        if hz <= 0.0 or self.timeout <= 0.0 or self.max_wheel != 22:
            raise ValueError("invalid current MCU adapter parameters")
        self.create_timer(1.0/hz, self._tick)
        self.create_timer(2.0, self._audit_publishers)
        if not self.armed:
            self.get_logger().warn(
                "adapter disarmed: no /gps_* messages will be published")

    def _update(self, name, value):
        self.values[name] = value
        self.received[name] = time.monotonic()

    def _audit_publishers(self):
        conflicts = []
        for topic in self.output_topics:
            for info in self.get_publishers_info_by_topic(topic):
                if info.node_name != self.get_name():
                    namespace = (info.node_namespace or "").rstrip("/")
                    conflicts.append(f"{topic}:{namespace}/{info.node_name}")
        self.publisher_conflicts = sorted(set(conflicts))

    def _tick(self):
        if not self.armed:
            self.pub_state.publish(String(data="DISARMED"))
            return
        now = time.monotonic()
        ages = tuple(float("inf") if self.received[name] is None else
                     now-self.received[name]
                     for name in ("drive", "wheel", "stop"))
        result = adapt_slam_command(
            self.values["drive"], self.values["wheel"], self.values["stop"],
            ages, self.timeout, self.max_wheel)
        if self.publisher_conflicts:
            result = type(result)(0.0, 0, True, False,
                                  "DUPLICATE_GPS_SOURCE_PUBLISHER")
        self.pub_drive.publish(Float32(data=result.drive))
        self.pub_wheel.publish(Int32(data=result.wheel))
        self.pub_stop.publish(Bool(data=result.stop))
        self.pub_state.publish(String(data=result.state))
        self.pub_diag.publish(String(data=json.dumps({
            "state": result.state, "valid": result.valid,
            "armed": self.armed, "input_convention": "+LEFT/-RIGHT",
            "gps_output_convention": "+RIGHT/-LEFT",
            "allowed_drive_stages": [-1, 0, 1, 2, 3],
            "max_wheel_deg": self.max_wheel,
            "input_ages_s": list(ages),
            "publisher_conflicts": self.publisher_conflicts,
            "mcu_python_import": False}, sort_keys=True)))


def main(args=None):
    rclpy.init(args=args)
    node = McuSourceAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
