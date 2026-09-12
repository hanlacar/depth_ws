"""Explicit test-only adapter from CSV virtual odometry to follower inputs."""

import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from .ros_helpers import safe_shutdown


class CsvOnlyLocalizationSourceNode(Node):
    def __init__(self):
        super().__init__("depth_csv_only_localization_source")
        self.declare_parameter("prehardware_test_only", True)
        self.declare_parameter("odom_timeout_s", 0.50)
        if not bool(self.get_parameter("prehardware_test_only").value):
            raise ValueError("CSV-only localization source is test-only")
        self.odom = None
        self.received = None
        self.branch_stop = True
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(
            Bool, "/depth_slam/csv_only/branch_stop",
            lambda message: setattr(self, "branch_stop", bool(message.data)), 10)
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose", 10)
        self.pub_localization = self.create_publisher(
            String, "/depth_slam/localization/state", 10)
        self.pub_safety = self.create_publisher(
            String, "/depth_slam/safety/state", 10)
        self.pub_mission_stop = self.create_publisher(
            Bool, "/depth_slam/mission/stop_required", 10)
        self.pub_speed_limit = self.create_publisher(
            Float32, "/depth_slam/mission/speed_limit", 10)
        self.pub_source = self.create_publisher(
            String, "/depth_slam/csv_only/localization_source", 10)
        self.create_timer(1.0/30.0, self.tick)
        self.get_logger().warn(
            "PREHARDWARE TEST ONLY: /odom is the sole CSV follower "
            "localization source")

    def on_odom(self, message):
        if message.header.frame_id != "map" or message.child_frame_id != "base_link":
            self.get_logger().error("CSV-only /odom must be map -> base_link")
            return
        self.odom = message
        self.received = time.monotonic()

    def tick(self):
        timeout = float(self.get_parameter("odom_timeout_s").value)
        fresh = (self.odom is not None and self.received is not None and
                 time.monotonic()-self.received <= timeout)
        if fresh:
            pose = PoseWithCovarianceStamped()
            pose.header = self.odom.header
            pose.pose = self.odom.pose
            self.pub_pose.publish(pose)
        self.pub_localization.publish(String(data="TRACKING" if fresh else "STALE"))
        self.pub_safety.publish(String(data="READY" if fresh else "WAITING_FOR_ODOM"))
        self.pub_mission_stop.publish(Bool(data=self.branch_stop))
        self.pub_speed_limit.publish(Float32(data=3.0))
        self.pub_source.publish(String(data=(
            "PREHARDWARE_CSV_ONLY_ODOM" if fresh else "WAITING_FOR_ODOM")))


def main(args=None):
    rclpy.init(args=args)
    node = CsvOnlyLocalizationSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            safe_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
