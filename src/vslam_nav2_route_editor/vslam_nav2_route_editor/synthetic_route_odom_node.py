"""Replay a map-frame Path as relative rviz_odom for sensor-free testing."""
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
from std_msgs.msg import Int32, String
from tf2_ros import TransformBroadcaster


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def map_path_to_local(poses, start_index: int):
    """Express Path poses in an odom frame whose start pose is identity."""
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
        dx, dy = float(pose.position.x) - x0, float(pose.position.y) - y0
        local.append((
            c * dx + s * dy,
            -s * dx + c * dy,
            normalize_angle(yaw_from_quaternion(pose.orientation) - yaw0),
        ))
    return local


def interpolate_pose(a, b, fraction: float):
    fraction = max(0.0, min(1.0, float(fraction)))
    yaw_delta = normalize_angle(b[2] - a[2])
    return (
        a[0] + fraction * (b[0] - a[0]),
        a[1] + fraction * (b[1] - a[1]),
        normalize_angle(a[2] + fraction * yaw_delta),
    )


class SyntheticRouteOdom(Node):
    def __init__(self):
        super().__init__("synthetic_route_odom")
        self.declare_parameter("start_index", 0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("travel_speed_mps", 1.0)
        self.declare_parameter("stop_hold_s", 0.5)
        self.declare_parameter(
            "log_path", "/tmp/vslam_nav2_route_editor_synthetic.csv")
        self.start_index = int(self.get_parameter("start_index").value)
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.travel_speed_mps = float(
            self.get_parameter("travel_speed_mps").value)
        self.stop_hold_s = float(self.get_parameter("stop_hold_s").value)
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.travel_speed_mps <= 0.0:
            raise ValueError("travel_speed_mps must be positive")

        self.local_poses = []
        self.geometry_signature = None
        self.segment_index = self.start_index
        self.segment_progress_m = 0.0
        self.stop_hold_remaining_s = 0.0
        self.events = {}
        self.last_logged_key = None
        # Publish the selected start pose until map->rviz_odom has been
        # computed from that exact zero pose.  This removes startup offset
        # caused by DDS discovery taking one or more 20 Hz ticks.
        self.alignment_ready = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/nav2_route/path", self.path, qos)
        self.create_subscription(String, "/nav2_route/metadata", self.metadata, qos)
        self.create_subscription(
            TransformStamped, "/route_debug/map_odom_alignment",
            self.alignment, qos)
        self.create_subscription(String, "/route_debug/status", self.status, 20)
        self.odom_publisher = self.create_publisher(
            Odometry, "/rviz_check/odom", 20)
        self.index_publisher = self.create_publisher(
            Int32, "/route_debug/synthetic_index", 20)
        self.tf_broadcaster = TransformBroadcaster(self)

        log_path = FilePath(str(self.get_parameter("log_path").value)).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log_path.open("w", newline="", encoding="utf-8")
        self.log_writer = csv.DictWriter(self.log_stream, fieldnames=[
            "time", "current_index", "target_index", "direction", "event",
            "route_drive", "route_wheel", "cross_track_error",
        ])
        self.log_writer.writeheader(); self.log_stream.flush()
        self.log_path = log_path
        self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info(
            f"synthetic odom enabled: start_index={self.start_index}, "
            f"rate={self.rate_hz:.1f}Hz, speed={self.travel_speed_mps:.2f}m/s, "
            f"log={self.log_path}")

    def path(self, message: Path):
        if message.header.frame_id not in ("", "map"):
            self.get_logger().error(
                f"synthetic Path rejected frame {message.header.frame_id!r}; expected map")
            return
        signature = tuple(
            (float(p.pose.position.x), float(p.pose.position.y),
             yaw_from_quaternion(p.pose.orientation)) for p in message.poses)
        if not signature or signature == self.geometry_signature:
            return
        try:
            self.local_poses = map_path_to_local(message.poses, self.start_index)
        except ValueError as error:
            self.get_logger().error(str(error)); return
        self.geometry_signature = signature
        self.segment_index = self.start_index
        self.segment_progress_m = 0.0
        self.stop_hold_remaining_s = 0.0
        self.get_logger().info(
            f"synthetic route ready: {len(self.local_poses)} poses, "
            f"starting at index {self.start_index}")

    def metadata(self, message: String):
        try:
            points = json.loads(message.data).get("points", [])
            self.events = {
                int(point["index"]): str(point.get("event", "NONE")).upper()
                for point in points}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"synthetic metadata rejected: {error}")

    def alignment(self, message: TransformStamped):
        if (message.header.frame_id == "map" and
                message.child_frame_id == "rviz_odom" and
                not self.alignment_ready):
            self.alignment_ready = True
            self.get_logger().info(
                "map->rviz_odom alignment received; synthetic replay moving")

    def current_pose(self):
        if self.segment_index >= len(self.local_poses) - 1:
            return self.local_poses[-1]
        a = self.local_poses[self.segment_index]
        b = self.local_poses[self.segment_index + 1]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        fraction = 1.0 if length <= 1e-9 else self.segment_progress_m / length
        return interpolate_pose(a, b, fraction)

    def advance(self):
        if self.stop_hold_remaining_s > 0.0:
            self.stop_hold_remaining_s = max(
                0.0, self.stop_hold_remaining_s - 1.0 / self.rate_hz)
            return
        distance = self.travel_speed_mps / self.rate_hz
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
            if self.events.get(self.segment_index) == "STOP":
                self.stop_hold_remaining_s = self.stop_hold_s
                return

    def tick(self):
        if not self.local_poses:
            return
        x, y, yaw = self.current_pose()
        stamp = self.get_clock().now().to_msg()
        odom = Odometry(); odom.header.stamp = stamp
        odom.header.frame_id = "rviz_odom"; odom.child_frame_id = "rviz_base_link"
        odom.pose.pose.position.x = x; odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.odom_publisher.publish(odom)
        transform = TransformStamped(); transform.header.stamp = stamp
        transform.header.frame_id = "rviz_odom"
        transform.child_frame_id = "rviz_base_link"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(transform)
        self.index_publisher.publish(Int32(data=int(self.segment_index)))
        if self.alignment_ready:
            self.advance()

    def status(self, message: String):
        try:
            status = json.loads(message.data)
            row = {
                "time": f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
                "current_index": int(status["nearest_index"]),
                "target_index": int(status["target_index"]),
                "direction": str(status.get("direction", "F")),
                "event": str(status.get("event", "NONE")),
                "route_drive": f"{float(status['drive']):.2f}",
                "route_wheel": int(status["wheel_deg"]),
                "cross_track_error": f"{float(status['cross_track_error_m']):.6f}",
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        self.log_writer.writerow(row); self.log_stream.flush()
        key = (row["current_index"], row["direction"], row["event"],
               row["route_drive"], row["route_wheel"])
        if key != self.last_logged_key:
            self.get_logger().info(
                f"index={row['current_index']} {row['direction']} {row['event']} "
                f"drive={row['route_drive']} wheel={row['route_wheel']:+d}")
            self.last_logged_key = key

    def destroy_node(self):
        if not self.log_stream.closed:
            self.log_stream.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args); node = SyntheticRouteOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
