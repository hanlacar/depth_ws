"""Convert the canonical real /odom stream into the map-pose contract."""

import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .ros_helpers import quaternion_from_yaw, yaw_from_quaternion


class OdomLocalizationNode(Node):
    def __init__(self):
        super().__init__("odom_localization")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_timeout_s", 0.5)
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.last_received = None
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose", 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/localization/state", 10)
        self.pub_confidence = self.create_publisher(
            Float32, "/depth_slam/localization/confidence", 10)
        self.pub_jump = self.create_publisher(
            Bool, "/depth_slam/localization/pose_jump", 10)
        self.pub_tracking = self.create_publisher(
            Bool, "/depth_slam/localization/tracking_valid", 10)
        self.create_subscription(Odometry, "/odom", self._odom, 20)
        self.create_timer(0.1, self._health)

    def _odom(self, message):
        source = str(message.header.frame_id).lstrip("/")
        child = str(message.child_frame_id).lstrip("/")
        target = str(self.get_parameter("map_frame").value).lstrip("/")
        expected_child = str(
            self.get_parameter("base_frame").value).lstrip("/")
        if not source or child != expected_child:
            self.get_logger().error(
                f"invalid /odom frames: {source!r} -> {child!r}")
            return
        pose = message.pose.pose
        x, y = float(pose.position.x), float(pose.position.y)
        yaw = yaw_from_quaternion(pose.orientation)
        if source != target:
            try:
                tf = self.buffer.lookup_transform(
                    target, source, rclpy.time.Time())
            except TransformException:
                self.get_logger().warning(
                    f"waiting for {target}->{source} transform",
                    throttle_duration_sec=2.0)
                return
            translation = tf.transform.translation
            tf_yaw = yaw_from_quaternion(tf.transform.rotation)
            c, s = math.cos(tf_yaw), math.sin(tf_yaw)
            x, y = (translation.x+c*x-s*y,
                    translation.y+s*x+c*y)
            yaw = math.atan2(math.sin(tf_yaw+yaw),
                             math.cos(tf_yaw+yaw))
        output = PoseWithCovarianceStamped()
        output.header = message.header
        output.header.frame_id = target
        output.pose = message.pose
        output.pose.pose.position.x = x
        output.pose.pose.position.y = y
        output.pose.pose.orientation = quaternion_from_yaw(yaw)
        self.pub_pose.publish(output)
        self.last_received = time.monotonic()
        self._publish_health(True)

    def _publish_health(self, valid):
        self.pub_state.publish(String(data="TRACKING" if valid else "STALE"))
        self.pub_confidence.publish(Float32(data=1.0 if valid else 0.0))
        self.pub_jump.publish(Bool(data=False))
        self.pub_tracking.publish(Bool(data=valid))

    def _health(self):
        timeout = float(self.get_parameter("odom_timeout_s").value)
        valid = (self.last_received is not None and
                 time.monotonic()-self.last_received <= timeout)
        if not valid:
            self._publish_health(False)


def main(args=None):
    rclpy.init(args=args)
    node = OdomLocalizationNode()
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
