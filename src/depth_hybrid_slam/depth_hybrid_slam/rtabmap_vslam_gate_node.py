"""Expose map pose as VSLAM evidence only after an RTAB-Map visual match."""

import json
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rtabmap_msgs.msg import Info
from std_msgs.msg import Bool, Float32, String


def visual_match_evidence(message, minimum_inliers=15.0,
                          minimum_probability=0.05):
    """Score existing RTAB-Map localization evidence; never compare pixels."""
    statistics = dict(zip(message.stats_keys, message.stats_values))
    inliers = max((float(value) for key, value in statistics.items()
                   if "inlier" in key.lower()), default=0.0)
    posterior = max((float(value) for key, value in
                     zip(message.posterior_keys, message.posterior_values)
                     if int(key) > 0), default=0.0)
    likelihood = max((float(value) for key, value in
                      zip(message.likelihood_keys, message.likelihood_values)
                      if int(key) > 0), default=0.0)
    matched = (int(message.loop_closure_id) > 0 or
               int(message.proximity_detection_id) > 0)
    probability = max(posterior, likelihood)
    valid = (matched and inliers >= float(minimum_inliers) and
             probability >= float(minimum_probability))
    confidence = min(1.0, min(inliers/max(float(minimum_inliers), 1.0),
                              probability/max(float(minimum_probability),
                                              1.0e-6))) if valid else 0.0
    return valid, confidence, inliers, probability, matched


class RtabmapVslamGateNode(Node):
    def __init__(self):
        super().__init__("rtabmap_vslam_gate")
        self.declare_parameter("match_timeout_s", 0.5)
        self.declare_parameter("minimum_inliers", 15.0)
        self.declare_parameter("minimum_probability", 0.05)
        self.last_visual_match = None
        self.last_info = None
        self.confidence = 0.0
        self.evidence = {}
        self.latest_pose = None
        self.pose_at = None
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/depth_slam/vslam/pose", 10)
        self.valid_pub = self.create_publisher(
            Bool, "/depth_slam/vslam/tracking_valid", 10)
        self.visual_pub = self.create_publisher(
            Bool, "/depth_slam/vslam/visual_consistent", 10)
        self.confidence_pub = self.create_publisher(
            Float32, "/depth_slam/vslam/confidence", 10)
        self.evidence_pub = self.create_publisher(
            String, "/depth_slam/vslam/evidence", 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self._pose, 10)
        self.create_subscription(Info, "/rtabmap/info", self._info, 10)
        self.create_timer(0.1, self._tick)

    def _pose(self, message):
        if str(message.header.frame_id).lstrip("/") != "map":
            return
        self.latest_pose = message
        self.pose_at = time.monotonic()

    def _info(self, message):
        valid, confidence, inliers, probability, matched = visual_match_evidence(
            message, self.get_parameter("minimum_inliers").value,
            self.get_parameter("minimum_probability").value)
        self.last_info = time.monotonic()
        self.confidence = confidence
        self.evidence = {"inliers": inliers, "probability": probability,
                         "matched": matched, "visual_consistent": valid}
        if valid:
            self.last_visual_match = time.monotonic()

    def _tick(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("match_timeout_s").value)
        valid = (
            bool(self.evidence.get("visual_consistent", False)) and
            self.last_visual_match is not None and
            now - self.last_visual_match <= timeout and
            self.last_info is not None and now-self.last_info <= timeout and
            self.pose_at is not None and now - self.pose_at <= 0.5 and
            self.latest_pose is not None)
        self.valid_pub.publish(Bool(data=valid))
        self.visual_pub.publish(Bool(data=valid))
        self.confidence_pub.publish(Float32(data=(self.confidence if valid
                                                  else 0.0)))
        evidence = dict(self.evidence)
        evidence.update({
            "fresh": bool(valid),
            "age_s": None if self.last_info is None else now-self.last_info})
        self.evidence_pub.publish(String(data=json.dumps(
            evidence, separators=(",", ":"))))
        if valid:
            self.pose_pub.publish(self.latest_pose)


def main(args=None):
    rclpy.init(args=args)
    node = RtabmapVslamGateNode()
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
