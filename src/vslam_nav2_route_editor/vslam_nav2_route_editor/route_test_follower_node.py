"""Follow the published map Path and emit only isolated route commands."""
from __future__ import annotations

import json
import math

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from .map_odom_alignment_node import yaw_from_quaternion
from .route_follower_core import (
    Pose2D, RouteFollower, transform_pose,
)
from .route_model import RoutePoint


def quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class RouteTestFollowerNode(Node):
    def __init__(self):
        super().__init__("route_follower")
        self.declare_parameter("lookahead_m", 1.0)
        self.declare_parameter("stop_capture_m", 0.35)
        self.declare_parameter("start_index", 0)
        self.start_index = int(self.get_parameter("start_index").value)
        self.follower = RouteFollower(
            lookahead_m=float(self.get_parameter("lookahead_m").value),
            stop_capture_m=float(self.get_parameter("stop_capture_m").value),
            start_index=self.start_index)
        self.points = []
        self.path_message = None
        self.metadata_by_index = {}
        self.alignment = None
        self.pending_odom = None
        self.trajectory = Path(); self.trajectory.header.frame_id = "map"
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/nav2_route/path", self.route_path, qos)
        # nav_msgs/Path intentionally has no direction/event fields. Metadata
        # augments those attributes only; all route geometry comes from Path.
        self.create_subscription(String, "/nav2_route/metadata", self.metadata, qos)
        self.create_subscription(
            TransformStamped, "/route_debug/map_odom_alignment",
            self.alignment_message, qos)
        self.create_subscription(Odometry, "/rviz_check/odom", self.odom, 20)
        self.drive_pub = self.create_publisher(Float32, "/route_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/route_wheel", 10)
        self.index_pub = self.create_publisher(Int32, "/route_debug/target_index", 10)
        self.error_pub = self.create_publisher(
            Float32, "/route_debug/cross_track_error_m", 10)
        self.status_pub = self.create_publisher(String, "/route_debug/status", 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, "/route_debug/current_pose", 10)
        self.path_pub = self.create_publisher(Path, "/route_debug/odom_path", qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/route_debug/markers", qos)
        self.vehicle_marker_pub = self.create_publisher(
            Marker, "/route_debug/vehicle_marker", qos)

    def route_path(self, message: Path):
        if message.header.frame_id not in ("", "map"):
            self.get_logger().error(
                f"route rejected frame {message.header.frame_id!r}; expected map")
            self.path_message = None
            self.points = []
            return
        self.path_message = message
        self.rebuild_points()

    def metadata(self, message: String):
        try:
            data = json.loads(message.data)
            self.metadata_by_index = {
                int(point["index"]): point for point in data.get("points", [])}
            self.rebuild_points()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"route metadata rejected: {error}")

    def rebuild_points(self):
        if self.path_message is None:
            return
        points = []
        for index, stamped in enumerate(self.path_message.poses):
            pose = stamped.pose
            metadata = self.metadata_by_index.get(index, {})
            direction = str(metadata.get("direction", "F")).upper()
            event = str(metadata.get("event", "NONE")).upper()
            if direction not in {"F", "R"}:
                direction = "F"
            if event not in {"NONE", "STOP", "ACCEL"}:
                event = "NONE"
            points.append(RoutePoint(
                index=index,
                segment_id=int(metadata.get("segment_id", 1)),
                x=float(pose.position.x), y=float(pose.position.y),
                yaw_deg=math.degrees(yaw_from_quaternion(pose.orientation)),
                direction=direction, speed=float(metadata.get("speed", 2.0)),
                required_steering_deg=float(
                    metadata.get("required_steering_deg", 0.0)),
                event=event))
        geometry_changed = (
            len(points) != len(self.points) or
            any(a.x != b.x or a.y != b.y
                for a, b in zip(points, self.points)))
        self.points = points
        if geometry_changed:
            self.follower.last_nearest = None
    def alignment_message(self, message):
        transform = message.transform
        self.alignment = Pose2D(
            transform.translation.x, transform.translation.y,
            yaw_from_quaternion(transform.rotation))
        self.get_logger().info(
            "using authoritative map->rviz_odom route-test alignment")

    def odom(self, message):
        pose = message.pose.pose
        odom_pose = Pose2D(pose.position.x, pose.position.y,
                           yaw_from_quaternion(pose.orientation))
        self.pending_odom = odom_pose
        if not self.points:
            return
        if self.alignment is None:
            return
        map_pose = transform_pose(self.alignment, odom_pose)
        command = self.follower.command(self.points, map_pose)
        near_point = self.points[command.nearest_index]
        stamp = message.header.stamp
        current = PoseStamped(); current.header.frame_id = "map"; current.header.stamp = stamp
        current.pose.position.x = map_pose.x; current.pose.position.y = map_pose.y
        qx, qy, qz, qw = quaternion(map_pose.yaw)
        current.pose.orientation.x = qx; current.pose.orientation.y = qy
        current.pose.orientation.z = qz; current.pose.orientation.w = qw
        self.pose_pub.publish(current)
        self.trajectory.header.stamp = stamp
        if not self.trajectory.poses or math.hypot(
                self.trajectory.poses[-1].pose.position.x - map_pose.x,
                self.trajectory.poses[-1].pose.position.y - map_pose.y) >= 0.02:
            self.trajectory.poses.append(current)
        self.path_pub.publish(self.trajectory)
        self.drive_pub.publish(Float32(data=float(command.drive)))
        wheel_output = max(-22, min(22, int(command.wheel_deg)))
        self.wheel_pub.publish(Int32(data=wheel_output))
        self.index_pub.publish(Int32(data=int(command.target_index)))
        self.error_pub.publish(Float32(data=float(command.cross_track_error_m)))
        self.status_pub.publish(String(data=json.dumps({
            "frame_id": "map", "status": command.status,
            "start_index": self.start_index,
            "nearest_index": command.nearest_index,
            "target_index": command.target_index,
            "direction": near_point.direction,
            "event": near_point.event,
            "drive": command.drive, "wheel_deg": wheel_output,
            "wheel_computed_deg": command.wheel_deg,
            "cross_track_error_m": command.cross_track_error_m,
            "steering_saturated": command.saturated,
            "control_topics": "/route_drive and /route_wheel only; no MCU publisher",
        }, separators=(",", ":"))))
        self.publish_markers(current, command.target_index)

    def publish_markers(self, current, target_index):
        stamp = current.header.stamp
        vehicle = Marker(); vehicle.header.frame_id = "map"; vehicle.header.stamp = stamp
        vehicle.ns = "route_vehicle"; vehicle.id = 0
        vehicle.type = Marker.ARROW; vehicle.action = Marker.ADD
        # Keep the marker pose exactly equal to rviz_base_link in map.  The
        # larger arrow remains visible in the initial whole-course RViz view.
        vehicle.pose = current.pose; vehicle.scale.x = 2.0
        vehicle.scale.y = 0.8; vehicle.scale.z = 0.45
        vehicle.color.r = 1.0; vehicle.color.g = 0.15
        vehicle.color.b = 0.05; vehicle.color.a = 1.0
        target = Marker(); target.header = vehicle.header
        target.ns = "route_target"; target.id = 0
        target.type = Marker.SPHERE; target.action = Marker.ADD
        target.pose.position.x = self.points[target_index].x
        target.pose.position.y = self.points[target_index].y
        target.pose.position.z = 0.2; target.pose.orientation.w = 1.0
        target.scale.x = target.scale.y = target.scale.z = 0.45
        target.color.r = 1.0; target.color.g = 0.2
        target.color.b = 0.1; target.color.a = 1.0
        self.vehicle_marker_pub.publish(vehicle)
        self.marker_pub.publish(MarkerArray(markers=[target]))


def main(args=None):
    rclpy.init(args=args); node = RouteTestFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
