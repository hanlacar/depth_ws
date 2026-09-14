"""TEST ONLY /odom from real T870 encoder and steering measurements."""

import math
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String
from tf2_ros import TransformBroadcaster

from .test_odom_core import EncoderSteeringOdom


class EncoderOdomPublisher(Node):
    def __init__(self):
        super().__init__("test_odom_publisher")
        self.declare_parameter("test_only_acknowledged", False)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("input_timeout_s", 0.5)
        if not bool(self.get_parameter("test_only_acknowledged").value):
            raise ValueError(
                "TEST ONLY ODOM requires test_only_acknowledged:=true")
        self.model = EncoderSteeringOdom()
        self.encoder_received = None
        self.steering_received = None
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/test_odom/state", 10)
        self.pub_distance = self.create_publisher(
            Float32, "/depth_slam/test_odom/distance_m", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Int32, "/mcu/encoder", self._encoder, 20)
        self.create_subscription(Int32, "/mcu/steer_a0", self._steering, 20)
        self.create_subscription(Float32, "/cmd_drive", self._drive, 20)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._publish)
        self.get_logger().warn(
            "TEST ONLY ODOM: do not run with a production /odom owner")

    def _encoder(self, message):
        self.model.update_encoder(message.data)
        self.encoder_received = time.monotonic()

    def _steering(self, message):
        self.model.set_steering_adc(message.data)
        self.steering_received = time.monotonic()

    def _drive(self, message):
        self.model.set_drive_stage(message.data)

    def _publish(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("input_timeout_s").value)
        fresh = (self.encoder_received is not None and
                 self.steering_received is not None and
                 now-self.encoder_received <= timeout and
                 now-self.steering_received <= timeout)
        self.pub_state.publish(String(data="TRACKING" if fresh else
                                      "WAITING_FOR_REAL_ENCODER_STEERING"))
        if not fresh:
            return
        x, y, yaw = self.model.base_pose()
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw/2.0)
        odom.pose.pose.orientation.w = math.cos(yaw/2.0)
        self.pub_odom.publish(odom)
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)
        self.pub_distance.publish(Float32(
            data=float(self.model.state.distance_m)))


def main(args=None):
    rclpy.init(args=args)
    node = EncoderOdomPublisher()
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
