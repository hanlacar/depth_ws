"""ROS 2 node for Nav2-first geometry with segmented CSV fallback."""
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

from .hybrid_core import BranchSelector, HybridFollower, Pose2D, special_section
from .route_network import RouteNetwork, RoutePoint, transform_points


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def transform_pose(transform, pose):
    c, s = math.cos(transform.yaw), math.sin(transform.yaw)
    return Pose2D(
        transform.x + c * pose.x - s * pose.y,
        transform.y + s * pose.x + c * pose.y,
        transform.yaw + pose.yaw)


class HybridRouteFollowerNode(Node):
    def __init__(self):
        super().__init__("nav2_csv_route_follower")
        defaults = {
            "csv_path": "/home/qor/depth_ws/routes/network/route_network_segmented.csv",
            "yaml_path": "/home/qor/depth_ws/routes/network/route_network_segmented.yaml",
            # Scale is fixed at 1.000. This robust SE(2) was fitted between
            # AAA_BASE and the v10 map START_A geometry and remains tunable.
            "csv_to_map_x": -1.84276105,
            "csv_to_map_y": -1.48548570,
            "csv_to_map_yaw_deg": -153.98206441,
            "start_index": 0,
            "lookahead_m": 1.0,
            "nav_to_csv_position_m": 1.5,
            "csv_to_nav_position_m": 0.75,
            "nav_to_csv_heading_deg": 20.0,
            "csv_to_nav_heading_deg": 10.0,
            "nav_to_csv_samples": 10,
            "csv_to_nav_samples": 20,
            "stop_hold_seconds": 1.0,
            "stop_trigger_distance_m": 0.30,
            "stop_approach_distance_m": 1.0,
            "stop_merge_distance_m": 0.75,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.network = RouteNetwork.load(
            self.get_parameter("csv_path").value,
            self.get_parameter("yaml_path").value)
        self.branch = BranchSelector()
        self.csv_points = []
        self.rebuild_csv_route()
        params = {name: self.get_parameter(name).value for name in (
            "start_index", "lookahead_m", "nav_to_csv_position_m",
            "csv_to_nav_position_m", "nav_to_csv_heading_deg",
            "csv_to_nav_heading_deg", "nav_to_csv_samples",
            "csv_to_nav_samples", "stop_hold_seconds",
            "stop_trigger_distance_m", "stop_approach_distance_m",
            "stop_merge_distance_m")}
        self.follower = HybridFollower(**params)
        self.nav_points = []
        self.nav_path_message = None
        self.nav_stop_indices = set()
        self.alignment = None
        self.last_source = None
        self.last_active_published = None
        self.last_speed_log_key = None
        self.odom_trajectory = Path()
        self.odom_trajectory.header.frame_id = "map"

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, "/nav2_route/path", self.nav_path, latched)
        self.create_subscription(
            String, "/nav2_route/metadata", self.nav_metadata, latched)
        self.create_subscription(
            TransformStamped, "/route_debug/map_odom_alignment",
            self.map_odom_alignment, latched)
        self.create_subscription(Odometry, "/rviz_check/odom", self.odom, 20)
        self.create_subscription(
            String, "/route_branch/decision", self.branch_decision, 10)

        self.drive_pub = self.create_publisher(Float32, "/route_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/route_wheel", 10)
        self.csv_path_pub = self.create_publisher(
            Path, "/route_compare/csv_path", latched)
        self.active_path_pub = self.create_publisher(
            Path, "/route_compare/active_path", latched)
        self.odom_path_pub = self.create_publisher(
            Path, "/route_compare/odom_path", latched)
        self.status_pub = self.create_publisher(
            String, "/route_compare/status", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/route_compare/markers", latched)
        self.csv_marker_pub = self.create_publisher(
            MarkerArray, "/route_compare/csv_markers", latched)
        self.csv_index_pub = self.create_publisher(
            Int32, "/route_compare/csv_index", 10)
        self.csv_segment_pub = self.create_publisher(
            String, "/route_compare/csv_segment", 10)
        self.csv_event_pub = self.create_publisher(
            String, "/route_compare/csv_event", 10)
        self.csv_drive_level_pub = self.create_publisher(
            Float32, "/route_compare/csv_drive_level", 10)
        self.publish_csv_path()
        self.publish_csv_markers()
        self.get_logger().info(
            f"loaded segmented route: {self.network.analysis()['waypoint_count']} "
            "rows; default branches START=A,T=A,PARALLEL=A,END=A; "
            "command outputs are /route_drive and /route_wheel only")

    def rebuild_csv_route(self):
        selected = self.network.assemble(
            self.branch.values["START"], self.branch.values["T"],
            self.branch.values["PARALLEL"], self.branch.values["END"])
        self.csv_points = transform_points(
            selected,
            float(self.get_parameter("csv_to_map_x").value),
            float(self.get_parameter("csv_to_map_y").value),
            float(self.get_parameter("csv_to_map_yaw_deg").value))

    def route_path_message(self, points):
        message = Path(); message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        for point in points:
            pose = PoseStamped(); pose.header = message.header
            pose.pose.position.x = point.x; pose.pose.position.y = point.y
            pose.pose.orientation.z = math.sin(point.yaw / 2.0)
            pose.pose.orientation.w = math.cos(point.yaw / 2.0)
            message.poses.append(pose)
        return message

    def publish_csv_path(self):
        if hasattr(self, "csv_path_pub"):
            self.csv_path_pub.publish(self.route_path_message(self.csv_points))

    def publish_csv_markers(self):
        if not hasattr(self, "csv_marker_pub"):
            return
        stamp = self.get_clock().now().to_msg()
        markers = []
        clear = Marker()
        clear.header.frame_id = "map"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.append(clear)

        def add_marker(namespace, ident, kind, point, color, scale, label):
            item = Marker()
            item.header.frame_id = "map"
            item.header.stamp = stamp
            item.ns = namespace
            item.id = ident
            item.type = kind
            item.action = Marker.ADD
            item.pose.position.x = point.x
            item.pose.position.y = point.y
            item.pose.position.z = 0.18
            item.pose.orientation.w = 1.0
            item.scale.x = scale
            item.scale.y = scale
            item.scale.z = 0.12 if kind == Marker.CYLINDER else scale
            item.color.r, item.color.g, item.color.b = color
            item.color.a = 1.0
            markers.append(item)
            text = Marker()
            text.header = item.header
            text.ns = namespace + "_label"
            text.id = ident
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = point.x
            text.pose.position.y = point.y
            text.pose.position.z = 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.42
            text.color.r, text.color.g, text.color.b = color
            text.color.a = 1.0
            text.text = label
            markers.append(text)

        stop_id = 0
        direction_id = 0
        for point in self.csv_points:
            if point.event == "STOP_LINE":
                add_marker(
                    "csv_stop_line", stop_id, Marker.CYLINDER, point,
                    (1.0, 0.05, 0.05), 0.65,
                    f"STOP_LINE {point.route_index}\n{point.segment_id}:{point.point_index}")
                stop_id += 1
        for before, after in zip(self.csv_points, self.csv_points[1:]):
            if before.direction != after.direction:
                add_marker(
                    "csv_direction_change", direction_id, Marker.CUBE, after,
                    (0.9, 0.0, 1.0), 0.55,
                    f"{before.direction}->{after.direction} {after.route_index}\n"
                    f"{after.segment_id}:{after.point_index}")
                direction_id += 1
        self.csv_marker_pub.publish(MarkerArray(markers=markers))

    def nav_path(self, message):
        if message.header.frame_id not in ("", "map"):
            self.get_logger().error(
                f"Nav2 Path frame {message.header.frame_id!r} is not map")
            return
        self.nav_path_message = message
        self.rebuild_nav_points()
        if self.last_active_published is None:
            self.active_path_pub.publish(message)
            self.last_active_published = "NAV2"

    def nav_metadata(self, message):
        try:
            data = json.loads(message.data)
            self.nav_stop_indices = {
                int(point["index"]) for point in data.get("points", [])
                if str(point.get("event", "NONE")).upper() in
                {"STOP", "STOP_LINE"}}
            self.rebuild_nav_points()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Nav2 metadata rejected: {error}")

    def rebuild_nav_points(self):
        if self.nav_path_message is None:
            return
        points = []
        for index, stamped in enumerate(self.nav_path_message.poses):
            pose = stamped.pose
            points.append(RoutePoint(
                index, "NAV2", index, float(pose.position.x),
                float(pose.position.y), yaw_from_quaternion(pose.orientation),
                "F", 0, 2.0,
                "STOP_LINE" if index in self.nav_stop_indices else "NONE"))
        self.nav_points = points

    def map_odom_alignment(self, message):
        if message.header.frame_id != "map" or message.child_frame_id != "rviz_odom":
            return
        transform = message.transform
        self.alignment = Pose2D(
            transform.translation.x, transform.translation.y,
            yaw_from_quaternion(transform.rotation))

    def branch_decision(self, message):
        try:
            data = json.loads(message.data)
            stage, value = data.get("stage", ""), data.get("value", "")
        except json.JSONDecodeError:
            stage, value = "START", message.data
        stage_key = str(stage).strip().upper()
        if self.branch.request_or_default(stage_key, value):
            # A parking branch decision is immutable for that manoeuvre.  With
            # no decision message the A default is latched on parking entry.
            if stage_key in {"T", "PARALLEL"}:
                self.branch.latch(stage_key)
            self.rebuild_csv_route()
            self.follower.csv_tracker.last = None
            self.follower.csv_tracker.active_range = None
            self.follower.csv_tracker.ranges_signature = None
            self.follower.events_signature = None
            self.publish_csv_path()
            self.publish_csv_markers()
            self.get_logger().info(
                f"branch {stage_key}={self.branch.values[stage_key]} (latched)"
                if stage_key in {"T", "PARALLEL"} else
                f"branch {stage_key}={self.branch.values[stage_key]}")
        else:
            self.get_logger().warning(
                f"ignored invalid or latched branch decision: {message.data!r}")

    def odom(self, message):
        if not self.nav_points or not self.csv_points or self.alignment is None:
            return
        raw = message.pose.pose
        pose = transform_pose(self.alignment, Pose2D(
            raw.position.x, raw.position.y, yaw_from_quaternion(raw.orientation)))
        now = self.get_clock().now().nanoseconds / 1e9
        try:
            result = self.follower.command(
                now, pose, self.nav_points, self.csv_points,
                tuple(sorted(self.nav_stop_indices)))
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        csv_point = self.csv_points[result.csv_nearest]
        self.latch_current_branch(csv_point.segment_id)
        self.drive_pub.publish(Float32(data=float(result.drive)))
        self.wheel_pub.publish(Int32(data=int(result.wheel)))
        chosen = self.nav_points if result.source == "NAV2" else self.csv_points
        if result.source != self.last_active_published:
            self.active_path_pub.publish(self.route_path_message(chosen))
            self.last_active_published = result.source
        self.csv_index_pub.publish(Int32(data=int(result.csv_nearest)))
        self.csv_segment_pub.publish(String(data=csv_point.segment_id))
        self.csv_event_pub.publish(String(data=csv_point.event))
        self.csv_drive_level_pub.publish(Float32(data=float(csv_point.drive_level)))
        current = PoseStamped(); current.header.frame_id = "map"
        current.header.stamp = message.header.stamp
        current.pose.position.x = pose.x; current.pose.position.y = pose.y
        current.pose.orientation.z = math.sin(pose.yaw / 2.0)
        current.pose.orientation.w = math.cos(pose.yaw / 2.0)
        if (not self.odom_trajectory.poses or math.hypot(
                self.odom_trajectory.poses[-1].pose.position.x - pose.x,
                self.odom_trajectory.poses[-1].pose.position.y - pose.y) >= 0.05):
            self.odom_trajectory.poses.append(current)
        self.odom_trajectory.header.stamp = message.header.stamp
        self.odom_path_pub.publish(self.odom_trajectory)
        status = {
            "source": result.source,
            "segment": csv_point.segment_id,
            "current_segment": csv_point.segment_id,
            "current_mode": csv_point.mode,
            "current_event": csv_point.event,
            "current_csv_index": result.csv_nearest,
            "current_segment_index": csv_point.point_index,
            "current_drive_level": csv_point.drive_level,
            "section": special_section(csv_point),
            "active_branch": self.active_branch(
                self.csv_points[result.csv_nearest].segment_id),
            "branch": dict(self.branch.values),
            "csv_speed": result.csv_speed,
            "speed_reason": result.speed_reason,
            "final_drive": result.drive,
            "final_wheel": result.wheel,
            "path_distance_difference_m": result.path_distance_difference,
            "heading_difference_deg": result.heading_difference_deg,
            "stop_state": result.stop_state,
            "stop_source": result.stop_source,
            "stop_distance_m": result.stop_distance_m,
            "direction": result.direction,
            "nav_nearest": result.nav_nearest,
            "csv_nearest": result.csv_nearest,
            "nav_target": result.nav_target,
            "csv_target": result.csv_target,
            "cross_track_error_m": result.cross_track_error,
        }
        self.status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))
        self.publish_markers(pose, result, status)
        if result.source != self.last_source:
            self.get_logger().info(
                f"CURRENT_SOURCE={result.source}, deviation="
                f"{result.path_distance_difference:.3f}m/"
                f"{result.heading_difference_deg:.2f}deg")
            self.last_source = result.source
        speed_log_key = (csv_point.segment_id, csv_point.drive_level,
                         result.speed_reason, result.drive)
        if speed_log_key != self.last_speed_log_key:
            self.get_logger().info(
                f"CSV_SPEED index={result.csv_nearest} "
                f"segment={csv_point.segment_id}:{csv_point.point_index} "
                f"mode={csv_point.mode} drive_level={csv_point.drive_level:.0f} "
                f"wheel={result.wheel} final_drive={result.drive:.2f} "
                f"reason={result.speed_reason}")
            self.last_speed_log_key = speed_log_key

    def latch_current_branch(self, segment):
        if segment.startswith("START_"):
            self.branch.latch("START")
        elif segment.startswith("T_"):
            self.branch.latch("T")
        elif segment.startswith("V_"):
            self.branch.latch("PARALLEL")
        elif segment.startswith("END_"):
            self.branch.latch("END")

    def active_branch(self, segment):
        if segment.startswith("T_"):
            return self.branch.values["T"]
        if segment.startswith("V_"):
            return self.branch.values["PARALLEL"]
        if segment.startswith("START_"):
            return self.branch.values["START"]
        if segment.startswith("END_"):
            return self.branch.values["END"]
        return "-"

    def publish_markers(self, pose, result, status):
        stamp = self.get_clock().now().to_msg()
        markers = []
        def marker(ns, ident, marker_type, x, y, z, r, g, b, scale):
            item = Marker(); item.header.frame_id = "map"; item.header.stamp = stamp
            item.ns = ns; item.id = ident; item.type = marker_type
            item.action = Marker.ADD; item.pose.position.x = x
            item.pose.position.y = y; item.pose.position.z = z
            item.pose.orientation.w = 1.0
            item.scale.x = item.scale.y = item.scale.z = scale
            item.color.r = r; item.color.g = g; item.color.b = b; item.color.a = 1.0
            return item
        nav = self.nav_points[result.nav_target]
        csv = self.csv_points[result.csv_target]
        markers.append(marker("nav_target", 0, Marker.SPHERE,
                              nav.x, nav.y, 0.25, 0.1, 0.3, 1.0, 0.55))
        markers.append(marker("csv_target", 0, Marker.CUBE,
                              csv.x, csv.y, 0.25, 1.0, 0.55, 0.0, 0.55))
        vehicle = marker("vehicle", 0, Marker.ARROW,
                         pose.x, pose.y, 0.35, 0.1, 1.0, 0.2, 1.0)
        vehicle.scale.x = 1.8; vehicle.scale.y = 0.65; vehicle.scale.z = 0.35
        vehicle.pose.orientation.z = math.sin(pose.yaw / 2.0)
        vehicle.pose.orientation.w = math.cos(pose.yaw / 2.0)
        markers.append(vehicle)
        for ident, event in enumerate(self.follower.events):
            color = (1.0, 0.0, 0.0) if "STOP" in event.source else (0.8, 0.0, 1.0)
            markers.append(marker("stops", ident, Marker.CYLINDER,
                                  event.x, event.y, 0.15, *color, 0.45))
        text = marker("status", 0, Marker.TEXT_VIEW_FACING,
                      pose.x, pose.y, 3.0, 1.0, 1.0, 1.0, 0.8)
        text.text = (
            f"SOURCE={status['source']}\n"
            f"MODE={status['current_mode']}  SECTION={status['section']}\n"
            f"SEGMENT={status['current_segment']}  "
            f"BRANCH={status['active_branch']}\n"
            f"CSV index={status['current_csv_index']} "
            f"local={status['current_segment_index']} "
            f"event={status['current_event']}\n"
            f"direction={status['direction']}  "
            f"drive_level={status['current_drive_level']:.0f}\n"
            f"drive={status['final_drive']:.2f} "
            f"wheel={status['final_wheel']}\n"
            f"speed reason={status['speed_reason']}\n"
            f"difference={status['path_distance_difference_m']:.2f}m/"
            f"{status['heading_difference_deg']:.1f}deg  "
            f"stop={status['stop_state']}/{status['stop_source']}")
        markers.append(text)
        self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = HybridRouteFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
