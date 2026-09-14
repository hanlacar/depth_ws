"""ROS adapter for real-VSLAM/user START A/B double validation."""

import json
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .csv_only_branching import (
    csv_only_network_segments, StartBranchClassifier)
from .models import Pose2D
from .ros_helpers import yaw_from_quaternion
from .start_validation import StartDoubleValidator


class StartValidationNode(Node):
    def __init__(self):
        super().__init__("depth_start_validation")
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        self.declare_parameter("user_branch", "A")
        self.declare_parameter("classification_timeout_s", 30.0)
        self.declare_parameter("pose_timeout_s", 0.5)
        segments = csv_only_network_segments(
            self.get_parameter("route_path").value,
            self.get_parameter("route_metadata_path").value)
        classifier = StartBranchClassifier(
            segments["START_A"], segments["START_B"])
        self.core = StartDoubleValidator(
            self.get_parameter("user_branch").value, classifier,
            self.get_parameter("classification_timeout_s").value)
        self.pose = None
        self.pose_at = None
        self.tracking = False
        self.tracking_at = None
        self.logged = False
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/vslam/pose",
            self._pose, 10)
        self.create_subscription(
            Bool, "/depth_slam/vslam/tracking_valid",
            self._tracking, 10)
        self.stop_pub = self.create_publisher(
            Bool, "/depth_slam/route/start_validation_stop", 10)
        self.result_pub = self.create_publisher(
            String, "/depth_slam/route/start_validation_result", 10)
        self.create_timer(0.05, self._tick)

    def _pose(self, message):
        if str(message.header.frame_id).lstrip("/") != "map":
            return
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        values = (float(p.x), float(p.y), yaw_from_quaternion(q))
        if not all(math.isfinite(value) for value in values):
            return
        self.pose = Pose2D(values[0], values[1], values[2], 0.0)
        self.pose_at = time.monotonic()

    def _tracking(self, message):
        self.tracking = bool(message.data)
        self.tracking_at = time.monotonic()

    def _tick(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("pose_timeout_s").value)
        fresh = (
            self.pose_at is not None and self.tracking_at is not None and
            now - self.pose_at <= timeout and
            now - self.tracking_at <= timeout)
        decision = self.core.update(
            self.pose if fresh else None, self.tracking and fresh, now)
        self.stop_pub.publish(Bool(data=decision.stop))
        payload = {
            "final": decision.final, "passed": decision.passed,
            "stop": decision.stop,
            "vslam_branch": decision.detected_branch,
            "user_branch": self.core.user_branch,
            "reason": decision.reason,
        }
        self.result_pub.publish(String(data=json.dumps(
            payload, separators=(",", ":"))))
        if decision.final and not self.logged:
            verdict = "COMPLETE" if decision.passed else "FAIL"
            self.get_logger().info(
                f"[SEGMENT 1] {verdict} - {decision.reason}")
            self.logged = True


def main(args=None):
    rclpy.init(args=args)
    node = StartValidationNode()
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
