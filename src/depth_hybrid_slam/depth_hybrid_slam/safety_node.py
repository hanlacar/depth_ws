"""ROS adapter for the fail-closed safety gate."""

import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from .models import SafetyInputs
from .ros_helpers import safe_shutdown, stamp_seconds, status
from .safety_core import SafetyGate


class SafetyNode(Node):
    def __init__(self):
        super().__init__("depth_safety_monitor")
        for name, default in (("enable_control", False), ("dry_run", True),
                              ("user_approved", False), ("map_route_match", False),
                              ("within_map", False)):
            self.declare_parameter(name, default)
        self.core = SafetyGate()
        self.value = SafetyInputs(now=time.monotonic())
        self.value.enable_control = bool(self.get_parameter("enable_control").value)
        self.value.dry_run = bool(self.get_parameter("dry_run").value)
        self.value.user_approved = bool(self.get_parameter("user_approved").value)
        self.value.map_route_match = bool(self.get_parameter("map_route_match").value)
        self.value.within_map = bool(self.get_parameter("within_map").value)
        self.mission_reason = ""
        self.create_subscription(Bool, "/depth_slam/cuvslam/tracking_valid",
                                 lambda m: setattr(self.value, "tracking_valid", bool(m.data)), 10)
        self.create_subscription(String, "/depth_slam/localization/state",
                                 lambda m: setattr(self.value, "localization_state", m.data), 10)
        self.create_subscription(Float32, "/depth_slam/localization/confidence",
                                 lambda m: setattr(self.value, "localization_confidence", float(m.data)), 10)
        self.create_subscription(PoseWithCovarianceStamped, "/depth_slam/localization/pose",
                                 lambda m: setattr(self.value, "pose_stamp", stamp_seconds(m.header.stamp)), 10)
        self.create_subscription(Bool, "/depth_slam/localization/pose_jump",
                                 lambda m: setattr(self.value, "pose_jump", bool(m.data)), 10)
        self.create_subscription(Bool, "/depth_slam/route/map_route_verified",
                                 lambda m: setattr(self.value, "map_route_match", bool(m.data)), 10)
        self.create_subscription(Bool, "/depth_slam/route/within_map",
                                 lambda m: setattr(self.value, "within_map", bool(m.data)), 10)
        self.create_subscription(Float32, "/depth_slam/route/cross_track_error",
                                 lambda m: setattr(self.value, "cross_track_error", float(m.data)), 10)
        self.create_subscription(Float32, "/depth_slam/route/heading_error", self.on_heading, 10)
        self.create_subscription(String, "/depth_slam/route/controller_state",
                                 lambda _m: setattr(self.value, "controller_stamp", self.ros_now()), 10)
        self.create_subscription(Bool, "/depth_slam/mission/stop_required",
                                 lambda m: setattr(self.value, "mission_stop", bool(m.data)), 10)
        self.create_subscription(String, "/depth_slam/mission/stop_reason",
                                 lambda m: setattr(self.value, "mission_reason", m.data), 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/safety/state", 10)
        self.pub_stop = self.create_publisher(Bool, "/depth_slam/safety/stop_required", 10)
        self.pub_reason = self.create_publisher(String, "/depth_slam/safety/stop_reason", 10)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, "/depth_slam/safety/diagnostics", 10)
        self.create_timer(1.0/30.0, self.tick)

    def ros_now(self):
        return self.get_clock().now().nanoseconds*1.0e-9

    def on_heading(self, message):
        self.value.heading_error = float(message.data)
        self.value.controller_stamp = self.ros_now()

    def tick(self):
        self.value.now = self.ros_now()
        decision = self.core.evaluate(self.value)
        self.pub_state.publish(String(data="READY" if decision.ready else "STOP_REQUIRED"))
        self.pub_stop.publish(Bool(data=decision.stop_required))
        self.pub_reason.publish(String(data=",".join(decision.reasons)))
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        diag.status = [status(
            "depth_slam/safety",
            DiagnosticStatus.OK if decision.ready else DiagnosticStatus.ERROR,
            "READY" if decision.ready else "STOP_REQUIRED",
            (("reasons", ",".join(decision.reasons)),
             ("control_enabled", self.value.enable_control),
             ("dry_run", self.value.dry_run)))]
        self.pub_diag.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
