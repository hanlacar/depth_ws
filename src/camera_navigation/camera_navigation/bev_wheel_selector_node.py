"""Explicitly bridge BEV path and steering to the legacy vehicle topics."""

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Int32


def selected_wheel(active_planner, value):
    """Return the bounded vehicle command, or None for an inactive source."""
    if str(active_planner) != "bev":
        return None
    return max(-22, min(22, int(value)))


class BevWheelSelector(Node):
    def __init__(self):
        super().__init__("bev_wheel_selector_node")
        self.declare_parameter("active_planner", "none")
        self.active = str(self.get_parameter("active_planner").value)
        self.wheel_publisher = self.create_publisher(
            Int32, "/camera/candidate/bev/wheel", 10)
        self.path_publisher = self.create_publisher(Path, "/camera/path", 10)
        self.create_subscription(Int32, "/camera/bev/wheel", self._on_wheel, 10)
        self.create_subscription(Path, "/camera/bev/path", self._on_path, 10)

    def _on_wheel(self, message):
        wheel = selected_wheel(self.active, message.data)
        if wheel is not None:
            self.wheel_publisher.publish(Int32(data=wheel))

    def _on_path(self, message):
        if self.active == "bev":
            # Preserve the source image timestamp and base_link frame contract.
            self.path_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = BevWheelSelector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
