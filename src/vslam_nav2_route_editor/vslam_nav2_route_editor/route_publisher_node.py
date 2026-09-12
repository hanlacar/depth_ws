"""Publish and safely edit a CSV route without connecting to vehicle control."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path as PathMessage
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .route_model import (
    ROUTE_NAMES, RoutePoint, VehiclePolicy, apply_vehicle_policy, load_route,
    nearest_route_point, save_route, set_direction_range, set_event_point,
    set_stop_point, validate_route,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion(yaw_deg):
    yaw = math.radians(yaw_deg)
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class RoutePublisher(Node):
    def __init__(self):
        super().__init__("route_editor")
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_name", "START_A")
        self.declare_parameter("source_db", "")
        self.declare_parameter("allow_clicked_point_append", True)
        self.declare_parameter("click_plane_enabled", True)
        self.declare_parameter("click_plane_min_x", -13.55)
        self.declare_parameter("click_plane_max_x", 73.15)
        self.declare_parameter("click_plane_min_y", -6.10)
        self.declare_parameter("click_plane_max_y", 86.50)
        self.declare_parameter("click_plane_z", 0.0)
        self.declare_parameter("selection_max_distance_m", 0.5)
        self.declare_parameter("show_all_indices", False)
        self.route_path = Path(str(self.get_parameter("route_path").value)).expanduser().resolve()
        self.route_name = str(self.get_parameter("route_name").value)
        self.source_db = Path(str(self.get_parameter("source_db").value)).expanduser().resolve()
        self.click_plane_enabled = bool(self.get_parameter("click_plane_enabled").value)
        self.click_plane_min_x = float(self.get_parameter("click_plane_min_x").value)
        self.click_plane_max_x = float(self.get_parameter("click_plane_max_x").value)
        self.click_plane_min_y = float(self.get_parameter("click_plane_min_y").value)
        self.click_plane_max_y = float(self.get_parameter("click_plane_max_y").value)
        self.click_plane_z = float(self.get_parameter("click_plane_z").value)
        self.selection_max_distance_m = float(
            self.get_parameter("selection_max_distance_m").value)
        self.show_all_indices = bool(self.get_parameter("show_all_indices").value)
        if self.route_name not in ROUTE_NAMES:
            raise ValueError(f"unsupported route_name: {self.route_name}")
        self.policy = VehiclePolicy()
        if self.route_path.is_file():
            self.points = load_route(self.route_path)
            self.get_logger().info(
                f"loaded {len(self.points)} points from {self.route_path}")
        else:
            self.points = []
            self.get_logger().info(
                f"route file not found; starting empty without creating {self.route_path}")
        self.dirty = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_publisher = self.create_publisher(PathMessage, "/nav2_route/path", qos)
        self.metadata_publisher = self.create_publisher(String, "/nav2_route/metadata", qos)
        self.marker_publisher = self.create_publisher(MarkerArray, "/nav2_route/markers", qos)
        self.create_subscription(String, "/nav2_route/command", self.command, 10)
        self.create_subscription(PointStamped, "/nav2_route/select_point",
                                 self.select_message, 10)
        if bool(self.get_parameter("allow_clicked_point_append").value):
            self.create_subscription(PointStamped, "/clicked_point", self.clicked, 10)
        self.create_service(Trigger, "/nav2_route/save", self.save_service)
        self.create_service(Trigger, "/nav2_route/new", self.new_service)
        self.create_service(Trigger, "/nav2_route/clear", self.new_service)
        self.create_service(Trigger, "/nav2_route/reload", self.reload_service)
        self.create_service(Trigger, "/nav2_route/delete_last", self.delete_last_service)
        self.previous_waypoint_marker_count = 0
        self.clear_markers_on_next_publish = True
        self.last_command = None
        self.selected_index = None
        self.selected_distance_m = None
        self.create_timer(1.0, self.publish)
        self.publish()

    def source_hash(self):
        return sha256(self.source_db) if self.source_db.is_file() else ""

    def persist(self):
        return save_route(self.route_path, self.points, self.route_name, self.policy,
                          str(self.source_db), self.source_hash())

    def clicked(self, message: PointStamped):
        if message.header.frame_id and message.header.frame_id != "map":
            self.get_logger().error("clicked point rejected: frame_id must be map")
            return
        yaw = self.points[-1].yaw_deg if self.points else 0.0
        segment = self.points[-1].segment_id if self.points else 1
        self.points.append(RoutePoint(len(self.points), segment,
                                      message.point.x, message.point.y, yaw, "F"))
        apply_vehicle_policy(self.points, self.policy, update_forward_yaw=True)
        self.dirty = True
        self.publish()
        self.get_logger().info(
            f"added map waypoint {len(self.points)-1}: "
            f"x={message.point.x:.3f}, y={message.point.y:.3f}")

    def select_message(self, message: PointStamped):
        if message.header.frame_id and message.header.frame_id != "map":
            self.get_logger().error("selection rejected: frame_id must be map")
            return
        try:
            self.select_nearest(message.point.x, message.point.y)
            self.publish()
        except ValueError as error:
            self.get_logger().warn(f"selection rejected: {error}")

    def select_nearest(self, x: float, y: float):
        index, distance = nearest_route_point(
            self.points, float(x), float(y), self.selection_max_distance_m)
        self.selected_index = index
        self.selected_distance_m = distance
        point = self.points[index]
        self.get_logger().info(
            f"selected index={index}, distance={distance:.3f}m, "
            f"x={point.x:.3f}, y={point.y:.3f}")
        return point, distance

    def save_service(self, _request, response):
        validation = validate_route(self.points, self.policy)
        if not validation["valid"]:
            response.success = False
            response.message = f"route validation failed: {json.dumps(validation)}"
            return response
        metadata = self.persist(); self.dirty = False; self.publish()
        response.success = True
        response.message = (f"saved route {self.route_name}: "
                            f"{metadata['point_count']} points to {self.route_path}")
        return response

    def new_service(self, _request, response):
        self.points = []
        self.selected_index = None
        self.selected_distance_m = None
        self.dirty = True
        self.publish()
        response.success = True
        response.message = (f"new empty route {self.route_name}; "
                            "existing files were not changed")
        return response

    def reload_service(self, _request, response):
        try:
            self.points = load_route(self.route_path) if self.route_path.is_file() else []
            self.selected_index = None
            self.selected_distance_m = None
            self.dirty = False
            self.publish()
            response.success = True
            response.message = f"reloaded {len(self.points)} points"
        except Exception as error:  # service must report malformed external edits
            response.success = False; response.message = str(error)
        return response

    def delete_last_service(self, _request, response):
        if not self.points:
            response.success = False; response.message = "route is empty"; return response
        self.points.pop(); apply_vehicle_policy(self.points, self.policy, update_forward_yaw=True)
        if self.selected_index is not None and self.selected_index >= len(self.points):
            self.selected_index = None
            self.selected_distance_m = None
        self.dirty = True
        self.publish(); response.success = True
        response.message = "deleted final waypoint"; return response

    def command(self, message: String):
        result = {"request_id": "", "command": "", "success": False, "message": ""}
        try:
            request = json.loads(message.data)
            result["request_id"] = str(request.get("request_id", ""))
            result["command"] = str(request.get("command", ""))
            if not result["request_id"]:
                raise ValueError("request_id is required")
            if request.get("route_name") != self.route_name:
                raise ValueError(
                    f"running route is {self.route_name}, not {request.get('route_name')}")
            if result["command"] == "set-direction":
                self.points = set_direction_range(
                    self.points, int(request["start"]), int(request["end"]),
                    str(request["direction"]), self.policy)
                result["message"] = (
                    f"set indices {request['start']}..{request['end']} "
                    f"to {str(request['direction']).upper()}")
            elif result["command"] == "set-stop":
                self.points = set_stop_point(
                    self.points, int(request["index"]), self.policy)
                result["message"] = f"set index {request['index']} event to STOP"
            elif result["command"] in {"mark-stop", "mark-accel", "clear-event"}:
                if self.selected_index is None:
                    raise ValueError("no waypoint is selected")
                event = {"mark-stop": "STOP", "mark-accel": "ACCEL",
                         "clear-event": "NONE"}[result["command"]]
                self.points = set_event_point(
                    self.points, self.selected_index, event, self.policy)
                result["message"] = (
                    f"set selected index {self.selected_index} event to {event}")
            elif result["command"] == "select-nearest":
                point, distance = self.select_nearest(
                    float(request["x"]), float(request["y"]))
                result["message"] = (
                    f"selected index={point.index}, distance={distance:.3f}m, "
                    f"x={point.x:.3f}, y={point.y:.3f}")
            elif result["command"] == "show-indices":
                self.show_all_indices = bool(request["enabled"])
                result["message"] = (
                    f"all waypoint indices {'shown' if self.show_all_indices else 'hidden'}")
            else:
                raise ValueError(f"unsupported live command: {result['command']}")
            if result["command"] not in {"select-nearest", "show-indices"}:
                self.dirty = True
            result["success"] = True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            result["message"] = str(error)
        self.last_command = result
        self.publish()
        if result["success"]:
            self.get_logger().info(result["message"])
        else:
            self.get_logger().error(result["message"])

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        path = PathMessage(); path.header.frame_id = "map"; path.header.stamp = stamp
        validation = validate_route(self.points, self.policy)
        invalid = set(validation["invalid_steering_indices"])
        markers = MarkerArray()
        # Clear markers once at node startup to remove transient data left by a
        # previous process. Repeating DELETEALL every second made RViz visibly
        # erase and redraw an otherwise unchanged route.
        if self.clear_markers_on_next_publish:
            clear = Marker(); clear.header = path.header
            clear.action = Marker.DELETEALL
            markers.markers.append(clear)
            self.clear_markers_on_next_publish = False
        if self.click_plane_enabled:
            plane = Marker(); plane.header = path.header
            plane.ns = "route_click_surface"; plane.id = 0
            plane.type = Marker.CUBE; plane.action = Marker.ADD
            plane.pose.position.x = (self.click_plane_min_x + self.click_plane_max_x) / 2.0
            plane.pose.position.y = (self.click_plane_min_y + self.click_plane_max_y) / 2.0
            plane.pose.position.z = self.click_plane_z - 0.01
            plane.pose.orientation.w = 1.0
            plane.scale.x = self.click_plane_max_x - self.click_plane_min_x
            plane.scale.y = self.click_plane_max_y - self.click_plane_min_y
            plane.scale.z = 0.01
            # The marker remains selectable in RViz's picking pass while this
            # very low alpha keeps the VSLAM cloud visually unobstructed.
            plane.color.r = 0.0; plane.color.g = 0.4; plane.color.b = 1.0
            plane.color.a = 0.003
            markers.markers.append(plane)
        line = Marker(); line.header = path.header; line.ns = "route"; line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD if self.points else Marker.DELETE
        line.scale.x = 0.08
        line.color.r = 0.1; line.color.g = 0.9; line.color.b = 0.2; line.color.a = 0.9
        for point in self.points:
            pose = PoseStamped(); pose.header = path.header
            pose.pose.position.x = point.x; pose.pose.position.y = point.y
            qx, qy, qz, qw = quaternion(point.yaw_deg)
            pose.pose.orientation.x = qx; pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz; pose.pose.orientation.w = qw
            path.poses.append(pose); line.points.append(pose.pose.position)
            marker = Marker(); marker.header = path.header; marker.ns = "waypoints"
            marker.id = point.index + 1; marker.action = Marker.ADD
            marker.pose = pose.pose
            if point.event == "STOP":
                marker.type = Marker.CYLINDER
                marker.scale.x = marker.scale.y = 0.34
                marker.scale.z = 0.12
            elif point.event == "ACCEL":
                marker.type = Marker.CUBE
                marker.scale.x = marker.scale.y = marker.scale.z = 0.32
            elif point.direction == "R":
                marker.type = Marker.CUBE
                marker.scale.x = marker.scale.y = marker.scale.z = 0.28
            else:
                marker.type = Marker.SPHERE
                marker.scale.x = marker.scale.y = marker.scale.z = 0.24
            marker.color.a = 1.0
            if point.index in invalid:
                marker.color.r = 1.0
            elif point.event == "STOP":
                marker.color.r = 1.0; marker.color.g = 0.65
            elif point.event == "ACCEL":
                marker.color.r = 1.0; marker.color.b = 1.0
            elif point.direction == "R":
                marker.color.b = 1.0
            else:
                marker.color.g = 1.0
            markers.markers.append(marker)
            if self.show_all_indices:
                label = Marker(); label.header = path.header
                label.ns = "waypoint_indices"; label.id = point.index + 1
                label.type = Marker.TEXT_VIEW_FACING; label.action = Marker.ADD
                label.pose.position.x = point.x; label.pose.position.y = point.y
                label.pose.position.z = 0.45; label.pose.orientation.w = 1.0
                label.scale.z = 0.28; label.color.r = label.color.g = label.color.b = 1.0
                label.color.a = 0.9; label.text = str(point.index)
                markers.markers.append(label)
        markers.markers.append(line)
        if not self.show_all_indices:
            for marker_id in range(1, max(len(self.points),
                                          self.previous_waypoint_marker_count) + 1):
                stale_label = Marker(); stale_label.header = path.header
                stale_label.ns = "waypoint_indices"; stale_label.id = marker_id
                stale_label.action = Marker.DELETE
                markers.markers.append(stale_label)
        for namespace, marker_id in (("selected_waypoint", 0), ("selected_index", 0)):
            selected = Marker(); selected.header = path.header
            selected.ns = namespace; selected.id = marker_id
            if self.selected_index is None or self.selected_index >= len(self.points):
                selected.action = Marker.DELETE
            else:
                point = self.points[self.selected_index]
                selected.action = Marker.ADD
                selected.pose.position.x = point.x; selected.pose.position.y = point.y
                selected.pose.orientation.w = 1.0
                if namespace == "selected_waypoint":
                    selected.type = Marker.SPHERE; selected.pose.position.z = 0.18
                    selected.scale.x = selected.scale.y = selected.scale.z = 0.58
                    selected.color.r = 1.0; selected.color.g = 1.0
                    selected.color.a = 0.75
                else:
                    selected.type = Marker.TEXT_VIEW_FACING
                    selected.pose.position.z = 0.75; selected.scale.z = 0.55
                    selected.color.r = selected.color.g = selected.color.b = 1.0
                    selected.color.a = 1.0
                    selected.text = f"index {point.index}  {point.event}"
            markers.markers.append(selected)
        for marker_id in range(len(self.points) + 1,
                               self.previous_waypoint_marker_count + 1):
            stale = Marker(); stale.header = path.header; stale.ns = "waypoints"
            stale.id = marker_id; stale.action = Marker.DELETE
            markers.markers.append(stale)
        self.previous_waypoint_marker_count = len(self.points)
        metadata = String(); metadata.data = json.dumps({
            "frame_id": "map", "route_name": self.route_name,
            "route_path": str(self.route_path),
            "route_file_exists": self.route_path.is_file(),
            "dirty": self.dirty, "point_count": len(self.points),
            "last_command": self.last_command,
            "selected": (None if self.selected_index is None else {
                "index": self.selected_index,
                "distance_m": self.selected_distance_m,
                "x": self.points[self.selected_index].x,
                "y": self.points[self.selected_index].y,
                "direction": self.points[self.selected_index].direction,
                "event": self.points[self.selected_index].event,
            }),
            "selection_max_distance_m": self.selection_max_distance_m,
            "show_all_indices": self.show_all_indices,
            "click_plane": {
                "enabled": self.click_plane_enabled,
                "frame_id": "map", "z": self.click_plane_z,
                "min_x": self.click_plane_min_x,
                "max_x": self.click_plane_max_x,
                "min_y": self.click_plane_min_y,
                "max_y": self.click_plane_max_y,
            },
            "validation": validation,
            "points": [{"index": p.index, "segment_id": p.segment_id,
                        "x": p.x, "y": p.y, "yaw_deg": p.yaw_deg,
                        "direction": p.direction, "speed": p.speed,
                        "required_steering_deg": p.required_steering_deg,
                        "event": p.event}
                       for p in self.points],
        }, separators=(",", ":"))
        self.path_publisher.publish(path); self.marker_publisher.publish(markers)
        self.metadata_publisher.publish(metadata)


def main(args=None):
    rclpy.init(args=args); node = RoutePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
