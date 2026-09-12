"""Command-gated synthetic odometry for the integrated CSV/Nav2 test stack."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path as FilePath

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Int32, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .hybrid_core import commanded_synthetic_speed


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def map_path_to_local(poses, start_index):
    if start_index < 0 or start_index >= len(poses):
        raise ValueError(
            f"start_index {start_index} is outside Path with {len(poses)} poses")
    origin = poses[start_index].pose
    x0, y0 = float(origin.position.x), float(origin.position.y)
    yaw0 = yaw_from_quaternion(origin.orientation)
    c, s = math.cos(yaw0), math.sin(yaw0)
    local = []
    for stamped in poses:
        pose = stamped.pose
        dx = float(pose.position.x) - x0
        dy = float(pose.position.y) - y0
        local.append((
            c * dx + s * dy,
            -s * dx + c * dy,
            normalize_angle(yaw_from_quaternion(pose.orientation) - yaw0),
        ))
    return local, (x0, y0, yaw0)


def interpolate_pose(a, b, fraction):
    fraction = max(0.0, min(1.0, float(fraction)))
    return (
        a[0] + fraction * (b[0] - a[0]),
        a[1] + fraction * (b[1] - a[1]),
        normalize_angle(a[2] + fraction * normalize_angle(b[2] - a[2])),
    )


class ControlledSyntheticOdom(Node):
    def __init__(self):
        super().__init__("controlled_synthetic_route_odom")
        self.declare_parameter("start_index", 0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("travel_speed_mps", 0.25)
        self.declare_parameter("path_topic", "/route_compare/csv_path")
        self.declare_parameter("log_path", "/tmp/nav2_csv_route_synthetic.csv")
        self.start_index = int(self.get_parameter("start_index").value)
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.travel_speed_mps = float(
            self.get_parameter("travel_speed_mps").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.rate_hz <= 0.0 or self.travel_speed_mps <= 0.0:
            raise ValueError("synthetic rate and speed must be positive")

        self.local_poses = []
        self.geometry_signature = None
        self.segment_index = self.start_index
        self.segment_progress_m = 0.0
        self.route_drive = 0.0
        self.latest_status = {}
        self.finished = False

        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, self.path_topic, self.path, latched)
        self.create_subscription(Float32, "/route_drive", self.drive, 20)
        self.create_subscription(
            String, "/route_compare/status", self.status, 20)
        self.odom_pub = self.create_publisher(Odometry, "/rviz_check/odom", 20)
        self.index_pub = self.create_publisher(
            Int32, "/route_debug/synthetic_index", 20)
        self.alignment_pub = self.create_publisher(
            TransformStamped, "/route_debug/map_odom_alignment", latched)
        self.static_tf = StaticTransformBroadcaster(self)
        self.dynamic_tf = TransformBroadcaster(self)

        log_path = FilePath(str(self.get_parameter("log_path").value))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("w", newline="", encoding="utf-8")
        self.log_writer = csv.DictWriter(self.log_stream, fieldnames=[
            "time", "synthetic_index", "synthetic_x", "synthetic_y",
            "target_index", "source",
            "current_csv_index", "current_segment_index", "current_mode",
            "current_segment", "current_event", "current_drive_level",
            "direction", "speed_reason", "stop_state",
            "stop_source", "stop_distance_m", "route_drive", "route_wheel",
            "cross_track_error_m",
        ])
        self.log_writer.writeheader()
        self.log_stream.flush()
        self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            f"controlled synthetic odom: path={self.path_topic}, "
            f"start_index={self.start_index}, rate={self.rate_hz:.1f}Hz, "
            f"visual_speed_limit={self.travel_speed_mps:.2f}m/s")

    def path(self, message):
        if message.header.frame_id not in ("", "map"):
            self.get_logger().error(
                f"synthetic Path frame {message.header.frame_id!r} is not map")
            return
        signature = tuple((
            float(item.pose.position.x), float(item.pose.position.y),
            yaw_from_quaternion(item.pose.orientation)) for item in message.poses)
        if not signature or signature == self.geometry_signature:
            return
        try:
            local, origin = map_path_to_local(message.poses, self.start_index)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        self.local_poses = local
        self.geometry_signature = signature
        self.segment_index = self.start_index
        self.segment_progress_m = 0.0
        self.finished = False
        self.publish_alignment(origin)
        self.get_logger().info(
            f"controlled synthetic route ready: {len(local)} poses, "
            f"starting at {self.start_index}")

    def publish_alignment(self, origin):
        x, y, yaw = origin
        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.child_frame_id = "rviz_odom"
        message.transform.translation.x = x
        message.transform.translation.y = y
        message.transform.rotation.z = math.sin(yaw / 2.0)
        message.transform.rotation.w = math.cos(yaw / 2.0)
        self.static_tf.sendTransform(message)
        self.alignment_pub.publish(message)

    def drive(self, message):
        self.route_drive = float(message.data)

    def status(self, message):
        try:
            self.latest_status = json.loads(message.data)
        except json.JSONDecodeError:
            self.latest_status = {}

    def current_pose(self):
        if self.segment_index >= len(self.local_poses) - 1:
            return self.local_poses[-1]
        a = self.local_poses[self.segment_index]
        b = self.local_poses[self.segment_index + 1]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        fraction = 1.0 if length <= 1e-9 else self.segment_progress_m / length
        return interpolate_pose(a, b, fraction)

    def advance(self):
        speed = commanded_synthetic_speed(
            self.route_drive, self.travel_speed_mps)
        distance = speed / self.rate_hz
        while distance > 0.0 and self.segment_index < len(self.local_poses) - 1:
            a = self.local_poses[self.segment_index]
            b = self.local_poses[self.segment_index + 1]
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            remaining = max(0.0, length - self.segment_progress_m)
            if distance < remaining:
                self.segment_progress_m += distance
                return
            distance -= remaining
            self.segment_index += 1
            self.segment_progress_m = 0.0
        if self.segment_index >= len(self.local_poses) - 1:
            self.finished = True

    def tick(self):
        if not self.local_poses:
            return
        x, y, yaw = self.current_pose()
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "rviz_odom"
        odom.child_frame_id = "rviz_base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.odom_pub.publish(odom)
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = "rviz_base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation = odom.pose.pose.orientation
        self.dynamic_tf.sendTransform(transform)
        self.index_pub.publish(Int32(data=int(self.segment_index)))
        self.write_log()
        self.advance()

    def write_log(self):
        status = self.latest_status
        x, y, _ = self.current_pose()
        row = {
            "time": f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
            "synthetic_index": self.segment_index,
            "synthetic_x": f"{x:.9f}",
            "synthetic_y": f"{y:.9f}",
            "target_index": status.get("csv_target", ""),
            "source": status.get("source", ""),
            "current_csv_index": status.get("current_csv_index", ""),
            "current_segment_index": status.get("current_segment_index", ""),
            "current_mode": status.get("current_mode", ""),
            "current_segment": status.get("current_segment", ""),
            "current_event": status.get("current_event", ""),
            "current_drive_level": status.get("current_drive_level", ""),
            "direction": status.get("direction", ""),
            "speed_reason": status.get("speed_reason", ""),
            "stop_state": status.get("stop_state", ""),
            "stop_source": status.get("stop_source", ""),
            "stop_distance_m": status.get("stop_distance_m", ""),
            "route_drive": f"{self.route_drive:.2f}",
            "route_wheel": status.get("final_wheel", ""),
            "cross_track_error_m": status.get("cross_track_error_m", ""),
        }
        self.log_writer.writerow(row)
        self.log_stream.flush()

    def destroy_node(self):
        if not self.log_stream.closed:
            self.log_stream.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlledSyntheticOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
