"""Convert the canonical real /odom stream into the map-pose contract."""

import json
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .localization_core import VslamRecoveryGate
from .ros_helpers import quaternion_from_yaw, yaw_from_quaternion


class OdomLocalizationNode(Node):
    def __init__(self):
        super().__init__("odom_localization")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("vslam_evidence_timeout_s", 0.5)
        self.declare_parameter("position_jump_threshold_m", 0.75)
        self.declare_parameter("yaw_jump_threshold_deg", 25.0)
        self.declare_parameter("vslam_recovery_s", 1.0)
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.last_received = None
        self.visual_consistent = False
        self.visual_received = None
        self.last_gate_state = "RUNNING_ODOM_ONLY"
        self.last_logged_gate_state = None
        self.gate = VslamRecoveryGate(
            self.get_parameter("position_jump_threshold_m").value,
            self.get_parameter("yaw_jump_threshold_deg").value,
            self.get_parameter("vslam_recovery_s").value)
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
        self.pub_watchdog = self.create_publisher(
            String, "/depth_slam/localization/watchdog", 10)
        self.create_subscription(Odometry, "/odom", self._odom, 20)
        self.create_subscription(
            Bool, "/depth_slam/vslam/visual_consistent", self._visual, 10)
        self.create_timer(0.1, self._health)

    def _odom(self, message):
        now = time.monotonic()
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
        candidate = None
        if source != target:
            try:
                tf = self.buffer.lookup_transform(
                    target, source, rclpy.time.Time())
            except TransformException:
                tf = None
            if tf is not None:
                translation = tf.transform.translation
                candidate = (float(translation.x), float(translation.y),
                             yaw_from_quaternion(tf.transform.rotation))
        elif source == target:
            candidate = (0.0, 0.0, 0.0)
        evidence_fresh = (
            self.visual_received is not None and
            now-self.visual_received <= float(self.get_parameter(
                "vslam_evidence_timeout_s").value))
        decision = self.gate.update(
            candidate, visual_consistent=self.visual_consistent,
            evidence_fresh=evidence_fresh, now=now)
        tx, ty, tf_yaw = decision.transform
        c, s = math.cos(tf_yaw), math.sin(tf_yaw)
        x, y = tx+c*x-s*y, ty+s*x+c*y
        yaw = math.atan2(math.sin(tf_yaw+yaw), math.cos(tf_yaw+yaw))
        output = PoseWithCovarianceStamped()
        output.header = message.header
        output.header.frame_id = target
        output.pose = message.pose
        output.pose.pose.position.x = x
        output.pose.pose.position.y = y
        output.pose.pose.orientation = quaternion_from_yaw(yaw)
        self.pub_pose.publish(output)
        self.last_received = now
        self.last_gate_state = decision.state
        if decision.state != self.last_logged_gate_state:
            if decision.state in ("WARNING", "RECOVERING"):
                self.get_logger().warning("[VSLAM] "+decision.state)
            else:
                self.get_logger().info("[VSLAM] "+decision.state)
            self.last_logged_gate_state = decision.state
        self._publish_health(True, decision)

    def _visual(self, message):
        self.visual_consistent = bool(message.data)
        self.visual_received = time.monotonic()

    def _publish_health(self, valid, decision=None):
        stop = bool(decision and decision.stop)
        state = "STALE" if not valid else (
            decision.state if stop else "TRACKING")
        self.pub_state.publish(String(data=state))
        confidence = 0.0 if not valid else (1.0 if decision and
                                            decision.use_vslam else 0.6)
        self.pub_confidence.publish(Float32(data=confidence))
        self.pub_jump.publish(Bool(data=stop))
        self.pub_tracking.publish(Bool(data=bool(valid and not stop)))
        now = time.monotonic()
        self.pub_watchdog.publish(String(data=json.dumps({
            "runtime_state": "FAIL" if not valid else self.last_gate_state,
            "odom": "OK" if valid else "STALE",
            "odom_age_s": None if self.last_received is None else
            max(0.0, now-self.last_received),
            "vslam": ("OK" if decision and decision.use_vslam else
                       "DEGRADED_ODOM_ONLY"),
            "visual_evidence_age_s": None if self.visual_received is None else
            max(0.0, now-self.visual_received),
        }, separators=(",", ":"))))

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
