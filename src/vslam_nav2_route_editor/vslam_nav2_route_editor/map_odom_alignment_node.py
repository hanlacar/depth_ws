"""Align the initial rviz odometry pose to the first route pose in map."""
from __future__ import annotations

import math
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def map_to_odom_alignment(route_pose, odom_pose):
    """T_map_odom = T_map_route_start * inverse(T_odom_initial_base)."""
    map_x, map_y, map_yaw = route_pose
    odom_x, odom_y, odom_yaw = odom_pose
    yaw = map_yaw - odom_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    return map_x - (c * odom_x - s * odom_y), \
        map_y - (s * odom_x + c * odom_y), yaw


class MapOdomAlignment(Node):
    def __init__(self):
        super().__init__("route_test_map_odom_alignment")
        # Retain the launch parameter for compatibility, but use the ROS Path
        # itself as the authoritative route geometry.
        self.declare_parameter("route_path", "")
        self.declare_parameter("start_index", 0)
        self.start_index = int(self.get_parameter("start_index").value)
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.route_pose = None
        self.broadcaster = StaticTransformBroadcaster(self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.alignment_publisher = self.create_publisher(
            TransformStamped, "/route_debug/map_odom_alignment", qos)
        self.sent = False
        self.latest_odom = None
        self.create_subscription(Odometry, "/rviz_check/odom", self.odom, 10)
        self.create_subscription(Path, "/nav2_route/path", self.route, qos)

    def route(self, message):
        if message.header.frame_id not in ("", "map"):
            self.get_logger().error(
                f"route alignment rejected frame {message.header.frame_id!r}; expected map")
            return
        if message.poses and self.route_pose is None:
            if self.start_index >= len(message.poses):
                self.get_logger().error(
                    f"start_index {self.start_index} is outside Path with "
                    f"{len(message.poses)} poses")
                return
            pose = message.poses[self.start_index].pose
            self.route_pose = (
                float(pose.position.x), float(pose.position.y),
                yaw_from_quaternion(pose.orientation))
            self.try_send()

    def odom(self, message):
        self.latest_odom = message
        self.try_send()

    def try_send(self):
        if self.sent or self.route_pose is None or self.latest_odom is None:
            return
        message = self.latest_odom
        pose = message.pose.pose
        x, y, yaw = map_to_odom_alignment(
            self.route_pose,
            (pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation)))
        transform = TransformStamped(); transform.header.stamp = message.header.stamp
        transform.header.frame_id = "map"; transform.child_frame_id = "rviz_odom"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self.broadcaster.sendTransform(transform)
        self.alignment_publisher.publish(transform)
        self.sent = True
        self.get_logger().info(
            f"map->rviz_odom aligned to route index {self.start_index}: "
            f"x={x:.6f}, y={y:.6f}, "
            f"yaw={math.degrees(yaw):.6f}deg")


def main(args=None):
    rclpy.init(args=args); node = MapOdomAlignment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
