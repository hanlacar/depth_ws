"""Publish T870 measured odometry only as rviz_odom -> rviz_base_link."""
from __future__ import annotations

import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from tf2_ros import TransformBroadcaster

from .t870_odom_core import T870OdomModel


class T870TestOdom(Node):
    def __init__(self):
        super().__init__("t870_route_test_odom")
        self.model = T870OdomModel()
        self.publisher = self.create_publisher(Odometry, "/rviz_check/odom", 10)
        self.steer_publisher = self.create_publisher(
            Float32, "/rviz_check/measured_steer_deg", 10)
        self.distance_publisher = self.create_publisher(
            Float32, "/rviz_check/distance_m", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Int32, "/mcu/encoder", self.encoder, 20)
        self.create_subscription(Int32, "/mcu/steer_a0", self.steering, 20)
        self.create_subscription(Float32, "/mcu_drive", self.drive, 20)
        self.create_timer(0.05, self.publish)
        self.get_logger().info(
            "test-only odom: 797 count/m, center=484, 18 count/deg, L=0.730m")

    def encoder(self, message):
        self.model.update_encoder(message.data)

    def steering(self, message):
        value = self.model.set_steering_adc(message.data)
        self.steer_publisher.publish(Float32(data=float(value)))

    def drive(self, message):
        self.model.set_drive_stage(message.data)

    def publish(self):
        x, y, yaw = self.model.base_pose()
        stamp = self.get_clock().now().to_msg()
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        odom = Odometry(); odom.header.stamp = stamp
        odom.header.frame_id = "rviz_odom"; odom.child_frame_id = "rviz_base_link"
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz; odom.pose.pose.orientation.w = qw
        self.publisher.publish(odom)
        transform = TransformStamped(); transform.header.stamp = stamp
        transform.header.frame_id = "rviz_odom"
        transform.child_frame_id = "rviz_base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = qz; transform.transform.rotation.w = qw
        self.tf.sendTransform(transform)
        self.distance_publisher.publish(
            Float32(data=float(self.model.state.distance_m)))


def main(args=None):
    rclpy.init(args=args); node = T870TestOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
