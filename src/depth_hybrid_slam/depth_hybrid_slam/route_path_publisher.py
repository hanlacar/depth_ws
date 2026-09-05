"""Publish a saved map-frame vehicle reference route for offline RViz."""

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .ros_helpers import quaternion_from_yaw, safe_shutdown
from .route_io import load_map_route


class RoutePathPublisher(Node):
    def __init__(self):
        super().__init__("route_path_publisher")
        self.declare_parameter("route_path", "")
        self.declare_parameter("include_invalid", False)
        route_path = str(self.get_parameter("route_path").value)
        if not route_path:
            raise ValueError("route_path is required")
        points = load_map_route(
            route_path, bool(self.get_parameter("include_invalid").value))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(Path, "/depth_slam/route/path", qos)
        self.message = Path()
        self.message.header.frame_id = "map"
        for point in points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp.sec = point["stamp_ns"] // 1_000_000_000
            pose.header.stamp.nanosec = point["stamp_ns"] % 1_000_000_000
            pose.pose.position.x = point["x"]
            pose.pose.position.y = point["y"]
            pose.pose.position.z = point["z"]
            pose.pose.orientation = quaternion_from_yaw(point["yaw"])
            self.message.poses.append(pose)
        self.create_timer(1.0, self.publish)
        self.publish()

    def publish(self):
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)


def main(args=None):
    rclpy.init(args=args)
    node = RoutePathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
