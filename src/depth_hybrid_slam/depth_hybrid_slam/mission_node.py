"""Adapter from verified camera_ws topics to the depth_slam mission contract."""

import json
import math
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from .mission_core import MissionMachine
from .models import MissionInputs
from .ros_helpers import safe_shutdown, status


class MissionNode(Node):
    def __init__(self):
        super().__init__("depth_mission_manager")
        for name, default in (("ramp_sections", [""]),
                              ("acceleration_sections", [""]),
                              ("finish_sections", [""])):
            self.declare_parameter(name, default)

        def values(name):
            return [str(v) for v in self.get_parameter(name).value if str(v)]
        self.core = MissionMachine(values("ramp_sections"),
                                   values("acceleration_sections"),
                                   values("finish_sections"))
        self.value = MissionInputs(now=time.monotonic())
        self.traffic_received = None
        self.create_subscription(String, "/camera/traffic_light_fused/state", self.on_traffic, 10)
        self.create_subscription(String, "/camera/traffic_light_fused/aspect",
                                 lambda m: setattr(self.value, "traffic_aspect", m.data.upper()), 10)
        self.create_subscription(Float32, "/camera/traffic_light_fused/confidence",
                                 lambda m: setattr(self.value, "traffic_confidence", float(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/stop_line_detected",
                                 lambda m: setattr(self.value, "stop_line_detected", bool(m.data)), 10)
        self.create_subscription(Float32, "/camera/mission/stop_line_distance_m",
                                 lambda m: setattr(self.value, "stop_line_distance_m", float(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/sign_detected",
                                 lambda m: setattr(self.value, "sign_detected", bool(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/uphill_detected",
                                 lambda m: setattr(self.value, "uphill_detected", bool(m.data)), 10)
        self.create_subscription(String, "/camera/mission/diagnostics",
                                 self.on_camera_diagnostics, 10)
        self.create_subscription(Float32, "/mcu/steer_deg",
                                 lambda m: setattr(self.value, "steering_deg", float(m.data)), 10)
        self.create_subscription(String, "/camera/mission/section",
                                 lambda m: setattr(self.value, "section_id", str(m.data)), 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/mission/state", 10)
        self.pub_section = self.create_publisher(String, "/depth_slam/mission/section", 10)
        self.pub_stop = self.create_publisher(Bool, "/depth_slam/mission/stop_required", 10)
        self.pub_reason = self.create_publisher(String, "/depth_slam/mission/stop_reason", 10)
        self.pub_speed = self.create_publisher(Float32, "/depth_slam/mission/speed_limit", 10)
        self.pub_permission = self.create_publisher(String, "/depth_slam/mission/traffic_permission", 10)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, "/depth_slam/mission/diagnostics", 10)
        self.create_timer(1.0/30.0, self.tick)

    def on_traffic(self, message):
        self.value.traffic_state = message.data.upper()
        self.traffic_received = time.monotonic()

    def on_camera_diagnostics(self, message):
        try:
            values = json.loads(message.data)
            pitch = values.get("imu_relative_uphill_deg", values.get("imu_pitch_deg"))
            if pitch is not None and math.isfinite(float(pitch)):
                self.value.pitch_deg = float(pitch)
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def tick(self):
        now = time.monotonic()
        self.value.now = now
        self.value.traffic_age = float("inf") if self.traffic_received is None else now-self.traffic_received
        decision = self.core.update(self.value)
        self.pub_state.publish(String(data=decision.state))
        self.pub_section.publish(String(data=self.value.section_id))
        self.pub_stop.publish(Bool(data=decision.stop_required))
        self.pub_reason.publish(String(data=decision.stop_reason))
        self.pub_speed.publish(Float32(data=float(decision.speed_limit)))
        self.pub_permission.publish(String(data=decision.traffic_permission))
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        diag.status = [status(
            "depth_slam/mission",
            DiagnosticStatus.WARN if decision.stop_required else DiagnosticStatus.OK,
            decision.state,
            (("section", self.value.section_id), ("stop_reason", decision.stop_reason),
             ("traffic_age_s", self.value.traffic_age),
             ("pitch_deg", self.value.pitch_deg),
             ("uphill_detected", self.value.uphill_detected)))]
        self.pub_diag.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
