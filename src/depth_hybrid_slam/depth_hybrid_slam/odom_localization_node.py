"""Convert the canonical real /odom stream into the map-pose contract."""

import json
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from tf2_ros import (
    Buffer, TransformBroadcaster, TransformException, TransformListener)

from .localization_core import (
    VslamRecoveryGate, align_odom_pose_to_route_entry, odom_only_map_edge)
from .parking_planner_core import map_to_odom_from_base_poses
from .ros_helpers import quaternion_from_yaw, yaw_from_quaternion


class OdomLocalizationNode(Node):
    def __init__(self):
        super().__init__("odom_localization")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("enable_vslam", True)
        self.declare_parameter("enable_parking_slam", False)
        self.declare_parameter("vslam_map_frame", "map")
        self.declare_parameter("centralize_vslam_map_tf", False)
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("vslam_evidence_timeout_s", 0.5)
        self.declare_parameter("position_jump_threshold_m", 0.75)
        self.declare_parameter("yaw_jump_threshold_deg", 25.0)
        self.declare_parameter("vslam_recovery_s", 1.0)
        self.declare_parameter("odom_route_entry_x_m", 0.0)
        self.declare_parameter("odom_route_entry_y_m", 0.0)
        self.declare_parameter("odom_route_entry_yaw_rad", 0.0)
        self.vslam_enabled = bool(self.get_parameter("enable_vslam").value)
        self.parking_slam_enabled = bool(
            self.get_parameter("enable_parking_slam").value)
        self.parking_slam_active = False
        self.vslam_map_frame = str(
            self.get_parameter("vslam_map_frame").value).lstrip("/")
        self.centralize_vslam_map_tf = bool(
            self.get_parameter("centralize_vslam_map_tf").value)
        self.vslam_pose = None
        self.vslam_pose_at = None
        self.buffer = (Buffer(cache_time=Duration(seconds=10.0))
                       if self.vslam_enabled or self.parking_slam_enabled
                       else None)
        self.listener = (TransformListener(self.buffer, self)
                         if self.buffer is not None else None)
        self.odom_map_transform = (
            TransformBroadcaster(self)
            if not self.vslam_enabled or self.centralize_vslam_map_tf else
            None)
        self.odom_only_alignment = None
        self.last_received = None
        self.visual_consistent = False
        self.visual_received = None
        self.last_gate_state = "RUNNING_ODOM_ONLY"
        self.last_logged_gate_state = None
        self.gate = (VslamRecoveryGate(
            self.get_parameter("position_jump_threshold_m").value,
            self.get_parameter("yaw_jump_threshold_deg").value,
            self.get_parameter("vslam_recovery_s").value)
            if self.vslam_enabled else None)
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
        if self.centralize_vslam_map_tf:
            self.create_subscription(
                PoseWithCovarianceStamped, "/rtabmap/localization_pose",
                self._vslam_pose, 10)
        self.create_subscription(
            Bool, "/depth_slam/parking/slam_active",
            lambda message: setattr(
                self, "parking_slam_active", bool(message.data)), 10)
        if self.vslam_enabled:
            self.create_subscription(
                Bool, "/depth_slam/vslam/visual_consistent", self._visual, 10)
            self.get_logger().info("[LOCALIZATION] ODOM+VSLAM")
        else:
            self.get_logger().info(
                "[LOCALIZATION] ODOM_ONLY - VSLAM DISABLED")
        self.create_timer(0.1, self._health)

    def _odom(self, message):
        now = time.monotonic()
        source = str(message.header.frame_id).lstrip("/")
        child = str(message.child_frame_id).lstrip("/")
        target = str(self.get_parameter("map_frame").value).lstrip("/")
        lookup_target = (
            target if self.parking_slam_active else
            self.vslam_map_frame if self.vslam_enabled else target)
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
        if (self.centralize_vslam_map_tf and self.vslam_enabled and
                not self.parking_slam_active):
            if (self.vslam_pose is not None and
                    self.vslam_pose_at is not None and
                    now-self.vslam_pose_at <= float(self.get_parameter(
                        "vslam_evidence_timeout_s").value)):
                mx, my, myaw = self.vslam_pose
                candidate = map_to_odom_from_base_poses(
                    (mx, my, myaw), (x, y, yaw))
        elif ((self.vslam_enabled or self.parking_slam_active) and
                source != lookup_target):
            try:
                tf = self.buffer.lookup_transform(
                    lookup_target, source, rclpy.time.Time())
            except TransformException:
                tf = None
            if tf is not None:
                translation = tf.transform.translation
                candidate = (float(translation.x), float(translation.y),
                             yaw_from_quaternion(tf.transform.rotation))
        elif ((self.vslam_enabled or self.parking_slam_active) and
              source == lookup_target):
            candidate = (0.0, 0.0, 0.0)
        if self.vslam_enabled:
            evidence_fresh = (
                self.visual_received is not None and
                now-self.visual_received <= float(self.get_parameter(
                    "vslam_evidence_timeout_s").value))
            decision = self.gate.update(
                candidate, visual_consistent=self.visual_consistent,
                evidence_fresh=evidence_fresh, now=now)
            tx, ty, tf_yaw = decision.transform
            self.last_gate_state = decision.state
            if (self.centralize_vslam_map_tf and
                    not self.parking_slam_active and candidate is not None):
                transform = TransformStamped()
                transform.header.stamp = message.header.stamp
                transform.header.frame_id = target
                transform.child_frame_id = source
                transform.transform.translation.x = tx
                transform.transform.translation.y = ty
                transform.transform.rotation = quaternion_from_yaw(tf_yaw)
                self.odom_map_transform.sendTransform(transform)
        else:
            decision = None
            map_edge = odom_only_map_edge(
                self.vslam_enabled or self.parking_slam_active,
                source, target)
            if self.parking_slam_active and candidate is not None:
                tx, ty, tf_yaw = candidate
                map_edge = None
            if map_edge is not None:
                if self.odom_only_alignment is None:
                    self.odom_only_alignment = align_odom_pose_to_route_entry(
                        x, y, yaw,
                        self.get_parameter("odom_route_entry_x_m").value,
                        self.get_parameter("odom_route_entry_y_m").value,
                        self.get_parameter("odom_route_entry_yaw_rad").value)
                    self.get_logger().info(
                        "[LOCALIZATION] ODOM_ONLY route entry aligned: "
                        f"map<-{source} x={self.odom_only_alignment[0]:.3f} "
                        f"y={self.odom_only_alignment[1]:.3f} "
                        f"yaw={math.degrees(self.odom_only_alignment[2]):.2f}deg")
                tx, ty, tf_yaw = self.odom_only_alignment
                # ODOM_ONLY uses the MCU's measured odom->base_link TF and
                # owns only map->odom. VSLAM owns that edge when enabled.
                transform = TransformStamped()
                transform.header.stamp = message.header.stamp
                transform.header.frame_id, transform.child_frame_id = map_edge
                transform.transform.translation.x = tx
                transform.transform.translation.y = ty
                transform.transform.rotation = quaternion_from_yaw(tf_yaw)
                self.odom_map_transform.sendTransform(transform)
            elif not self.parking_slam_active or candidate is None:
                tx, ty, tf_yaw = 0.0, 0.0, 0.0
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
        if decision is not None and decision.state != self.last_logged_gate_state:
            if decision.state in ("WARNING", "RECOVERING"):
                self.get_logger().warning("[VSLAM] "+decision.state)
            else:
                self.get_logger().info("[VSLAM] "+decision.state)
            self.last_logged_gate_state = decision.state
        self._publish_health(True, decision)

    def _visual(self, message):
        self.visual_consistent = bool(message.data)
        self.visual_received = time.monotonic()

    def _vslam_pose(self, message):
        if str(message.header.frame_id).lstrip("/") != self.vslam_map_frame:
            return
        pose = message.pose.pose
        self.vslam_pose = (
            float(pose.position.x), float(pose.position.y),
            yaw_from_quaternion(pose.orientation))
        self.vslam_pose_at = time.monotonic()

    def _publish_health(self, valid, decision=None):
        stop = bool(self.vslam_enabled and decision and decision.stop)
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
            "localization_mode": (
                "PARKING_SLAM" if self.parking_slam_active else
                "VSLAM" if self.vslam_enabled else "ODOM_ONLY"),
            "odom": "OK" if valid else "STALE",
            "odom_age_s": None if self.last_received is None else
            max(0.0, now-self.last_received),
            "vslam": ("DISABLED" if not self.vslam_enabled else
                       "OK" if decision and decision.use_vslam else
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
