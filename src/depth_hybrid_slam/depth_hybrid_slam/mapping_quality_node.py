"""Measure D456/SLAM mapping quality without altering sensor data or the DB."""

import json
import os
from pathlib import Path
import time

import cv2
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rtabmap_msgs.msg import Info
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, UInt32
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from .mapping_quality_core import MappingQuality
from .ros_helpers import safe_shutdown, status, yaw_from_quaternion


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def image_quality(message):
    channels = 1 if message.encoding in ("mono8", "8UC1") else 3
    array = np.frombuffer(message.data, dtype=np.uint8)
    row_width = int(message.step)
    if array.size < row_width * int(message.height):
        raise ValueError("short RGB image buffer")
    rows = array[:row_width * int(message.height)].reshape(int(message.height), row_width)
    pixels = rows[:, :int(message.width) * channels]
    if channels == 1:
        gray = pixels.reshape(int(message.height), int(message.width))
    else:
        color = pixels.reshape(int(message.height), int(message.width), channels)
        conversion = cv2.COLOR_RGB2GRAY if message.encoding.startswith("rgb") else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(color, conversion)
    sample = gray[::2, ::2]
    return (float(cv2.Laplacian(sample, cv2.CV_64F).var()),
            float(np.count_nonzero(sample >= 250) / sample.size),
            float(np.count_nonzero(sample <= 5) / sample.size))


def depth_valid_ratio(message):
    if message.encoding in ("16UC1", "mono16"):
        dtype = np.uint16
    elif message.encoding == "32FC1":
        dtype = np.float32
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    row_values = int(message.step) // np.dtype(dtype).itemsize
    array = np.frombuffer(message.data, dtype=dtype)
    if array.size < row_values * int(message.height):
        raise ValueError("short depth image buffer")
    depth = array[:row_values * int(message.height)].reshape(int(message.height), row_values)
    depth = depth[:, :int(message.width)]
    valid = np.isfinite(depth) & (depth > 0)
    return float(np.count_nonzero(valid) / valid.size)


class MappingQualityNode(Node):
    def __init__(self):
        super().__init__("mapping_quality_monitor")
        self.declare_parameter("map_path", "")
        self.declare_parameter("quality_config", "")
        self.declare_parameter("report_path", "")
        map_path = str(self.get_parameter("map_path").value)
        config_path = str(self.get_parameter("quality_config").value)
        if not map_path or not config_path:
            raise ValueError("map_path and quality_config are required")
        with open(config_path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        self.quality = MappingQuality(document["mapping_quality"])
        self.map_path = Path(map_path).expanduser().resolve()
        self.report_path = str(self.get_parameter("report_path").value)
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.pub_ready = self.create_publisher(Bool, "/depth_slam/mapping/ready", 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/mapping/state", 10)
        self.pub_record = self.create_publisher(
            Bool, "/depth_slam/mapping/recording_allowed", 10)
        self.pub_record_reason = self.create_publisher(
            String, "/depth_slam/mapping/recording_hold_reason", 10)
        self.pub_diagnostics = self.create_publisher(
            DiagnosticArray, "/depth_slam/mapping/diagnostics", 10)
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.on_rgb, 10)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                 self.on_depth, 10)
        self.create_subscription(Odometry, "/depth_slam/cuvslam/odometry", self.on_odom, 10)
        self.create_subscription(Bool, "/depth_slam/cuvslam/tracking_valid",
                                 self.on_tracking, 10)
        self.create_subscription(UInt32, "/depth_slam/cuvslam/reset_count", self.on_reset, 10)
        self.create_subscription(Info, "/rtabmap/info", self.on_rtabmap, 10)
        self.create_subscription(Bool, "/depth_slam/route/sample_valid",
                                 lambda message: self.quality.observe_route_sample(
                                     message.data), 10)
        self.create_timer(0.2, self.publish_state)
        self.last_state = "INITIALIZING"
        self.last_reasons = ["WAITING_FOR_INPUTS"]

    def on_rgb(self, message):
        self.quality.observe_stream("rgb", stamp_ns(message))
        try:
            blur, overexposed, underexposed = image_quality(message)
            self.quality.metrics["blur_score"].observe(blur)
            self.quality.metrics["overexposed_ratio"].observe(overexposed)
            self.quality.metrics["underexposed_ratio"].observe(underexposed)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_depth(self, message):
        self.quality.observe_stream("depth", stamp_ns(message))
        try:
            self.quality.metrics["depth_valid_ratio"].observe(depth_valid_ratio(message))
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_odom(self, message):
        position = message.pose.pose.position
        self.quality.observe_pose(
            stamp_ns(message), position.x, position.y,
            yaw_from_quaternion(message.pose.pose.orientation))

    def on_tracking(self, message):
        self.quality.observe_tracking(message.data, time.monotonic())

    def on_reset(self, message):
        self.quality.observe_reset(message.data)

    def on_rtabmap(self, message):
        latency = None
        inliers = None
        for key, value in zip(message.stats_keys, message.stats_values):
            lowered = key.lower()
            if "timing" in lowered and "total" in lowered:
                latency = float(value) if "ms" in lowered else float(value) * 1000.0
            if "inlier" in lowered:
                inliers = max(float(value), inliers or 0.0)
        self.quality.observe_rtabmap(latency, inliers, message.loop_closure_id,
                                     message.ref_id,
                                     message.proximity_detection_id)

    def tf_available(self):
        try:
            self.buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return True
        except TransformException:
            return False

    def publish_state(self):
        publisher_count = len(self.get_publishers_info_by_topic(
            "/depth_slam/cuvslam/odometry"))
        now = time.monotonic()
        try:
            correction = self.buffer.lookup_transform("map", "odom", rclpy.time.Time())
            value = correction.transform
            self.quality.observe_map_correction(
                value.translation.x, value.translation.y,
                yaw_from_quaternion(value.rotation), now)
        except TransformException:
            pass
        ready, state_name, reasons = self.quality.state(
            now, publisher_count, self.tf_available(),
            self.map_path.is_file())
        recording_allowed, recording_reasons = self.quality.recording_status(now)
        self.last_state, self.last_reasons = state_name, reasons
        self.pub_ready.publish(Bool(data=ready))
        self.pub_state.publish(String(data=state_name + (
            ":" + ",".join(reasons) if reasons else "")))
        self.pub_record.publish(Bool(data=recording_allowed))
        self.pub_record_reason.publish(String(data=",".join(recording_reasons)))
        report = self.quality.report()
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        diagnostics.status = [status(
            "depth_slam/mapping_quality",
            DiagnosticStatus.OK if ready else DiagnosticStatus.WARN,
            state_name,
            (("reasons", ",".join(reasons)),
             ("rgb_fps", report["streams"]["rgb"]["current_fps"]),
             ("depth_fps", report["streams"]["depth"]["current_fps"]),
             ("odometry_fps", report["streams"]["odometry"]["current_fps"]),
             ("blur_score", report["metrics"]["blur_score"]["latest"]),
             ("depth_valid_ratio", report["metrics"]["depth_valid_ratio"]["latest"]),
             ("overexposed_ratio", report["metrics"]["overexposed_ratio"]["latest"]),
             ("underexposed_ratio", report["metrics"]["underexposed_ratio"]["latest"]),
             ("rtabmap_inliers", report["metrics"]["rtabmap_inliers"]["latest"]),
             ("tracking_true_percent", report["tracking_true_percent"]),
             ("reset_increase", report["reset_increase"]),
             ("recording_allowed", recording_allowed),
             ("recording_hold_reason", ",".join(recording_reasons)),
             ("loop_closure_count", report["loop_closure_count"]),
             ("revisit_match_count", report["revisit_match_count"]),
             ("maximum_pose_correction_m", report["maximum_pose_correction_m"]),
             ("uncalibrated_metrics", ",".join(report["uncalibrated_metrics"])),
             ("odometry_publishers", publisher_count)))]
        self.pub_diagnostics.publish(diagnostics)

    def write_report(self):
        if not self.report_path:
            return
        path = Path(self.report_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        values = self.quality.report()
        publisher_count = len(self.get_publishers_info_by_topic(
            "/depth_slam/cuvslam/odometry"))
        verdict, state, reasons = self.quality.verdict(
            time.monotonic(), publisher_count, self.tf_available(),
            self.map_path.is_file())
        values.update({
            "map_path": str(self.map_path),
            "final_state": state,
            "final_reasons": reasons,
            "quality_verdict": verdict,
            "thresholds": self.quality.thresholds,
        })
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def destroy_node(self):
        self.write_report()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MappingQualityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
