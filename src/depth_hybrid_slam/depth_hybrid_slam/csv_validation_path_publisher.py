"""Synthetic base_link Path publisher for camera/CSV validator bench tests."""

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SyntheticValidationPathPublisher(Node):
    def __init__(self):
        super().__init__("csv_validation_path_publisher")
        for name, default in (("topic", "/depth_slam/csv_validation/local_path"),
                              ("lateral_offset_m", 0.0),
                              ("yaw_offset_deg", 0.0),
                              ("start_x_m", 0.5), ("length_m", 4.0),
                              ("spacing_m", 0.10), ("publish_hz", 10.0)):
            self.declare_parameter(name, default)
        self.publisher = self.create_publisher(
            Path, str(self.get_parameter("topic").value), 10)
        self.state = self.create_publisher(
            String, "/depth_slam/csv_validation/synthetic_state", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self.tick)

    def tick(self):
        offset = float(self.get_parameter("lateral_offset_m").value)
        yaw = math.radians(float(self.get_parameter("yaw_offset_deg").value))
        start = float(self.get_parameter("start_x_m").value)
        length = float(self.get_parameter("length_m").value)
        spacing = float(self.get_parameter("spacing_m").value)
        if spacing <= 0.0 or length <= 0.0:
            return
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        count = max(2, int(math.ceil(length/spacing))+1)
        for index in range(count):
            distance = min(length, index*spacing)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = start+distance*math.cos(yaw)
            pose.pose.position.y = offset+distance*math.sin(yaw)
            pose.pose.orientation.z = math.sin(yaw*0.5)
            pose.pose.orientation.w = math.cos(yaw*0.5)
            message.poses.append(pose)
        self.publisher.publish(message)
        self.state.publish(String(data=(
            f"SYNTHETIC offset_m={offset:.3f} yaw_deg={math.degrees(yaw):.1f}")))


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticValidationPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
