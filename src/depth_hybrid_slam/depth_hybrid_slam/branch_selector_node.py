"""Default-A branch selection with a mode-11 wait/fallback contract."""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .prehardware_core import BranchSelector


class BranchSelectorNode(Node):
    def __init__(self):
        super().__init__("depth_route_branch_selector")
        self.declare_parameter("command_timeout_s", 3.0)
        self.declare_parameter("mode_11_wait_s", 3.0)
        self.declare_parameter("publish_hz", 20.0)
        self.core = BranchSelector(
            self.get_parameter("command_timeout_s").value,
            self.get_parameter("mode_11_wait_s").value)
        self.create_subscription(
            String, "/depth_slam/route/branch_command", self._command, 10)
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.branch_pub = self.create_publisher(
            String, "/depth_slam/route/selected_branch", 10)
        self.stop_pub = self.create_publisher(
            Bool, "/depth_slam/route/branch_stop", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/route/branch_state", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/route/branch_diagnostics", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    def _command(self, message):
        self.core.set_command(message.data, time.monotonic())

    def _mode(self, message):
        self.core.set_mode(message.data, time.monotonic())

    def _tick(self):
        decision = self.core.evaluate(time.monotonic())
        self.branch_pub.publish(String(data=decision.branch))
        self.stop_pub.publish(Bool(data=decision.stop))
        self.state_pub.publish(String(data=decision.state))
        self.diag_pub.publish(String(data=json.dumps({
            "branch": decision.branch,
            "stop": decision.stop,
            "state": decision.state,
            "default": "A",
            "mode": self.core.mode,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = BranchSelectorNode()
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
