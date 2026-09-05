"""Publish high-rate map-frame localization from RTAB-Map correction + cuVSLAM."""

import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rtabmap_msgs.msg import Info
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .localization_core import LocalizationFusion
from .ros_helpers import (pose2d_from_odometry, quaternion_from_yaw,
                          safe_shutdown, stamp_seconds, yaw_from_quaternion)


def apply_route_policy(state, localization_mode, require_route_match,
                       map_route_match):
    """Keep map-only localization independent from route verification."""
    if localization_mode and require_route_match and not map_route_match:
        return "MAP_MISMATCH"
    return state


class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_fusion")
        for name, default in (("map_frame", "map"), ("odom_frame", "odom"),
                              ("map_id", "UNSELECTED"), ("route_id", "UNSELECTED"),
                              ("safe_to_apply_large_correction", False),
                              ("localization_mode", False),
                              ("map_route_match", False),
                              ("require_route_match", True)):
            self.declare_parameter(name, default)
        self.core = LocalizationFusion()
        self.tracking = False
        self.last_odom_message = None
        self.last_info_receipt = None
        self.match_receipt = None
        self.match_confidence = 0.0
        self.consistent_corrections = 0
        self.previous_correction = None
        self.relocalized_confirmed = False
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/depth_slam/localization/pose", 10)
        self.pub_odom = self.create_publisher(Odometry, "/depth_slam/localization/odometry", 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/localization/state", 10)
        self.pub_conf = self.create_publisher(Float32, "/depth_slam/localization/confidence", 10)
        self.pub_relocalized = self.create_publisher(Bool, "/depth_slam/localization/relocalized", 10)
        self.pub_map = self.create_publisher(String, "/depth_slam/localization/map_id", 10)
        self.pub_route = self.create_publisher(String, "/depth_slam/localization/route_id", 10)
        self.pub_jump = self.create_publisher(Bool, "/depth_slam/localization/pose_jump", 10)
        self.create_subscription(Bool, "/depth_slam/cuvslam/tracking_valid",
                                 lambda m: setattr(self, "tracking", bool(m.data)), 10)
        self.create_subscription(Odometry, "/depth_slam/cuvslam/odometry", self.on_odom, 10)
        self.create_subscription(Info, "/rtabmap/info", self.on_info, 10)
        self.create_timer(0.2, self.update_correction)

    def on_info(self, message):
        self.last_info_receipt = time.monotonic()
        statistics = dict(zip(message.stats_keys, message.stats_values))
        inliers = max((float(value) for key, value in statistics.items()
                       if "inlier" in key.lower()), default=0.0)
        posterior = max((float(value) for key, value in
                         zip(message.posterior_keys, message.posterior_values)
                         if int(key) > 0), default=0.0)
        likelihood = max((float(value) for key, value in
                          zip(message.likelihood_keys, message.likelihood_values)
                          if int(key) > 0), default=0.0)
        matched = (message.loop_closure_id > 0 or
                   message.proximity_detection_id > 0)
        if matched and inliers >= 15.0 and max(posterior, likelihood) >= 0.05:
            self.match_receipt = time.monotonic()
            self.match_confidence = min(1.0, inliers/30.0,
                                        max(posterior, likelihood))
            self.consistent_corrections = 0

    def update_correction(self):
        try:
            transform = self.buffer.lookup_transform(
                str(self.get_parameter("map_frame").value),
                str(self.get_parameter("odom_frame").value), rclpy.time.Time())
        except TransformException:
            return
        t, q = transform.transform.translation, transform.transform.rotation
        correction = (t.x, t.y, yaw_from_quaternion(q))
        accepted = self.core.set_correction(correction,
                                            stamp_seconds(transform.header.stamp),
                                            bool(self.get_parameter(
                                                "safe_to_apply_large_correction").value))
        if accepted and self.match_receipt is not None and time.monotonic()-self.match_receipt <= 3.0:
            if self.previous_correction is not None:
                delta = math.hypot(correction[0]-self.previous_correction[0],
                                   correction[1]-self.previous_correction[1])
                yaw = abs(math.atan2(math.sin(correction[2]-self.previous_correction[2]),
                                     math.cos(correction[2]-self.previous_correction[2])))
                self.consistent_corrections = self.consistent_corrections+1 \
                    if delta <= 0.15 and yaw <= math.radians(5.0) else 0
            self.previous_correction = correction
            if self.consistent_corrections >= 5:
                self.relocalized_confirmed = True

    def on_odom(self, message):
        fused, state = self.core.fuse(pose2d_from_odometry(message), self.tracking)
        localization_mode = bool(self.get_parameter("localization_mode").value)
        info_fresh = (self.last_info_receipt is not None and
                      time.monotonic()-self.last_info_receipt <= 1.0)
        if localization_mode and not self.relocalized_confirmed:
            state = "NOT_LOCALIZED"
        elif fused is None:
            state = "STOP_REQUIRED" if self.core.pose_jump else "LOST"
        elif self.core.pending is not None:
            state = "DEGRADED"
        elif not info_fresh:
            state = "DEGRADED"
        else:
            state = "RELOCALIZED" if localization_mode else "TRACKING"
        map_route_match = bool(self.get_parameter("map_route_match").value)
        require_route_match = bool(self.get_parameter("require_route_match").value)
        state = apply_route_policy(state, localization_mode,
                                   require_route_match, map_route_match)
        self.pub_state.publish(String(data=state))
        self.pub_relocalized.publish(Bool(data=self.relocalized_confirmed))
        self.pub_jump.publish(Bool(data=self.core.pose_jump))
        self.pub_map.publish(String(data=str(self.get_parameter("map_id").value)))
        self.pub_route.publish(String(data=str(self.get_parameter("route_id").value)))
        confidence = (self.match_confidence if state == "RELOCALIZED" and info_fresh
                      else (0.75 if state == "TRACKING" and info_fresh else 0.0))
        self.pub_conf.publish(Float32(data=confidence))
        if fused is None:
            return
        output = Odometry()
        output.header = message.header
        output.header.frame_id = str(self.get_parameter("map_frame").value)
        output.child_frame_id = message.child_frame_id
        output.pose = message.pose
        output.pose.pose.position.x = fused.x
        output.pose.pose.position.y = fused.y
        output.pose.pose.orientation = quaternion_from_yaw(fused.yaw)
        output.twist = message.twist
        self.pub_odom.publish(output)
        pose = PoseWithCovarianceStamped(header=output.header, pose=output.pose)
        self.pub_pose.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
