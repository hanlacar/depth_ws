"""Thirty-hertz dry-run route follower. MCU control is impossible by default."""

import csv
import hashlib
import math
from pathlib import Path as FilePath
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .models import ControllerResult, Pose2D, RoutePoint
from .rejoin_core import GridMap, RejoinPlanner, RejoinStateMachine
from .ros_helpers import quaternion_from_yaw, safe_shutdown, stamp_seconds, status, yaw_from_quaternion
from .route_follower_core import RouteFollower


def load_route(path):
    points = []
    with open(path, newline="", encoding="utf-8") as stream:
        for position, row in enumerate(csv.DictReader(stream)):
            points.append(RoutePoint(
                int(row.get("route_index", row.get("index", position))),
                float(row.get("map_x_m", row.get("x"))),
                float(row.get("map_y_m", row.get("y"))),
                float(row.get("yaw_rad", row.get("yaw", 0.0))),
                int(row.get("direction") or 1),
                float(row.get("drive_level", 1.0)), row.get("mission_marker", ""),
                row.get("section_id", "")))
    return points


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_route_binding(route_path, map_path, metadata_path=""):
    """Verify the exact finalized map/route pair before any motion is allowed."""
    route, database = FilePath(route_path), FilePath(map_path)
    candidates = ([FilePath(metadata_path)] if metadata_path else []) + [
        route.parent/"metadata.yaml", route.parent/"route_metadata.yaml"]
    metadata = next((item for item in candidates if item.is_file()), None)
    if not route.is_file() or not database.is_file() or metadata is None:
        return False, "MAP_ROUTE_FILES_MISSING"
    with open(metadata, encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not values.get("finalized", True):
        return False, "MAP_ROUTE_NOT_FINALIZED"
    route_hashes = {str(values.get(name, "")) for name in (
        "route_sha256", "route_csv_sha256", "final_route_csv_sha256")}
    if _sha256(route) not in route_hashes:
        return False, "ROUTE_CHECKSUM_MISMATCH"
    if _sha256(database) != str(values.get("rtabmap_db_sha256", "")):
        return False, "MAP_CHECKSUM_MISMATCH"
    return True, "VERIFIED"


class RouteFollowerNode(Node):
    def __init__(self):
        super().__init__("route_follower")
        for name, default in (("route_path", ""), ("enable_control", False),
                              ("dry_run", True), ("user_approved", False),
                              ("pose_timeout_s", 0.15), ("map_path", ""),
                              ("route_metadata_path", ""),
                              ("wheelbase_m", 0.73),
                              ("max_steering_deg", 22.0),
                              ("normal_corridor_m", 1.0),
                              ("rejoin_start_distance_m", 1.0),
                              ("rejoin_complete_distance_m", 0.25),
                              ("rejoin_complete_heading_deg", 10.0),
                              ("minimum_turning_radius_m", 1.8068134),
                              ("candidate_search_distance_m", 15.0),
                              ("candidate_spacing_m", 0.5),
                              ("vehicle_length_m", 1.40),
                              ("vehicle_width_m", 0.80),
                              ("footprint_safety_margin_m", 0.15),
                              ("allow_unknown", False),
                              ("rejoin_speed", 0.25),
                              ("heading_weight", 0.75),
                              ("max_index_backtrack", 3),
                              ("localization_stability_s", 2.0),
                              ("rejoin_verify_duration_s", 1.0),
                              ("occupancy_grid_timeout_s", 1.0),
                              ("reverse_rejoin_allowed", False)):
            self.declare_parameter(name, default)
        path = str(self.get_parameter("route_path").value)
        self.route = load_route(path) if path else []
        wheelbase = float(self.get_parameter("wheelbase_m").value)
        steering = float(self.get_parameter("max_steering_deg").value)
        configured_radius = float(self.get_parameter("minimum_turning_radius_m").value)
        physical_radius = wheelbase/math.tan(math.radians(steering))
        if configured_radius+1.0e-3 < physical_radius:
            raise ValueError("minimum_turning_radius_m is below T870 geometry")
        common = {
            "wheelbase": wheelbase, "max_steering_deg": steering,
            "heading_weight": float(self.get_parameter("heading_weight").value),
            "max_index_backtrack": int(self.get_parameter(
                "max_index_backtrack").value),
        }
        self.core = RouteFollower(
            corridor_m=float(self.get_parameter("normal_corridor_m").value), **common)
        self.rejoin_follower = RouteFollower(corridor_m=0.75, **common)
        self.rejoin_planner = RejoinPlanner(
            minimum_turning_radius_m=configured_radius,
            candidate_search_distance_m=float(self.get_parameter(
                "candidate_search_distance_m").value),
            candidate_spacing_m=float(self.get_parameter("candidate_spacing_m").value),
            heading_weight=float(self.get_parameter("heading_weight").value),
            vehicle_length_m=float(self.get_parameter("vehicle_length_m").value),
            vehicle_width_m=float(self.get_parameter("vehicle_width_m").value),
            safety_margin_m=float(self.get_parameter("footprint_safety_margin_m").value),
            allow_unknown=bool(self.get_parameter("allow_unknown").value))
        map_path = str(self.get_parameter("map_path").value)
        metadata_path = str(self.get_parameter("route_metadata_path").value)
        self.map_route_verified, self.binding_reason = verify_route_binding(
            path, map_path, metadata_path) if path and map_path else (
                False, "MAP_ROUTE_FILES_MISSING")
        self.pose = None
        self.pose_receipt = None
        self.safety_ready = False
        self.mission_stop = True
        self.mission_speed_limit = 0.0
        self.localization_state = "INITIALIZING"
        self.localization_stable_since = None
        self.grid = None
        self.grid_receipt = None
        self.machine = RejoinStateMachine(
            float(self.get_parameter("rejoin_start_distance_m").value),
            float(self.get_parameter("rejoin_complete_distance_m").value),
            float(self.get_parameter("rejoin_complete_heading_deg").value),
            float(self.get_parameter("rejoin_verify_duration_s").value))
        self.navigation_state = self.machine.state
        self.initial_search = True
        self.rejoin_plan = None
        self.rejoin_route = []
        self.pub_drive = self.create_publisher(Float32, "/slam_drive", 10)
        self.pub_wheel = self.create_publisher(Int32, "/slam_wheel", 10)
        self.pub_preview_drive = self.create_publisher(
            Float32, "/depth_slam/dry_run/drive", 10)
        self.pub_preview_wheel = self.create_publisher(
            Int32, "/depth_slam/dry_run/wheel", 10)
        self.pub_path = self.create_publisher(Path, "/depth_slam/route/reference_path", 10)
        self.pub_target = self.create_publisher(PointStamped, "/depth_slam/route/target_point", 10)
        self.pub_cross = self.create_publisher(Float32, "/depth_slam/route/cross_track_error", 10)
        self.pub_heading = self.create_publisher(Float32, "/depth_slam/route/heading_error", 10)
        self.pub_progress = self.create_publisher(Float32, "/depth_slam/route/progress", 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/route/controller_state", 10)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, "/depth_slam/route/controller_diagnostics", 10)
        self.pub_rejoin_path = self.create_publisher(
            Path, "/depth_slam/rejoin/path", 10)
        self.pub_candidates = self.create_publisher(
            MarkerArray, "/depth_slam/rejoin/candidates", 10)
        self.pub_rejoin_state = self.create_publisher(
            String, "/depth_slam/rejoin/state", 10)
        self.pub_binding = self.create_publisher(
            Bool, "/depth_slam/route/map_route_verified", 10)
        self.pub_within_map = self.create_publisher(
            Bool, "/depth_slam/route/within_map", 10)
        self.create_subscription(PoseWithCovarianceStamped, "/depth_slam/localization/pose", self.on_pose, 10)
        self.create_subscription(String, "/depth_slam/safety/state",
                                 lambda m: setattr(self, "safety_ready", m.data == "READY"), 10)
        self.create_subscription(Bool, "/depth_slam/mission/stop_required",
                                 lambda m: setattr(self, "mission_stop", bool(m.data)), 10)
        self.create_subscription(Float32, "/depth_slam/mission/speed_limit",
                                 lambda m: setattr(self, "mission_speed_limit", max(0.0, float(m.data))), 10)
        self.create_subscription(String, "/depth_slam/localization/state",
                                 self.on_localization_state, 10)
        self.create_subscription(OccupancyGrid, "/map", self.on_grid, 10)
        self.create_timer(1.0/30.0, self.tick)
        self.publish_reference_path()

    def publish_reference_path(self):
        message = Path()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        for point in self.route:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = point.x, point.y
            pose.pose.orientation = quaternion_from_yaw(point.yaw)
            message.poses.append(pose)
        self.pub_path.publish(message)

    def on_pose(self, message):
        p = message.pose.pose.position
        self.pose = Pose2D(p.x, p.y, yaw_from_quaternion(message.pose.pose.orientation),
                           stamp_seconds(message.header.stamp))
        self.pose_receipt = time.monotonic()

    def on_localization_state(self, message):
        state = str(message.data).split(":", 1)[0]
        if state in ("TRACKING", "RELOCALIZED"):
            if self.localization_state not in ("TRACKING", "RELOCALIZED"):
                self.localization_stable_since = time.monotonic()
        else:
            self.localization_stable_since = None
        self.localization_state = state

    def on_grid(self, message):
        origin = message.info.origin
        self.grid = GridMap(
            message.info.width, message.info.height, message.info.resolution,
            origin.position.x, origin.position.y,
            yaw_from_quaternion(origin.orientation), tuple(message.data),
            allow_unknown=bool(self.get_parameter("allow_unknown").value))
        self.grid_receipt = time.monotonic()

    def localization_ready(self):
        return (self.localization_state in ("TRACKING", "RELOCALIZED") and
                self.localization_stable_since is not None and
                time.monotonic()-self.localization_stable_since >= float(
                    self.get_parameter("localization_stability_s").value))

    def grid_ready(self):
        return (self.grid is not None and self.grid_receipt is not None and
                time.monotonic()-self.grid_receipt <= float(
                    self.get_parameter("occupancy_grid_timeout_s").value))

    def publish_rejoin_visuals(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        for item in self.rejoin_route:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = item.x, item.y
            pose.pose.orientation = quaternion_from_yaw(item.yaw)
            path.poses.append(pose)
        self.pub_rejoin_path.publish(path)
        markers = MarkerArray()
        candidate_indexes = ([] if self.pose is None else [
            index for _, index in self.rejoin_planner.candidates(
                self.pose, self.route, max(0, self.core.last_index or 0))])
        selected_index = self.rejoin_plan.route_index if self.rejoin_plan else -1
        for marker_id, route_index in enumerate(candidate_indexes):
            marker = Marker()
            marker.header = path.header
            marker.ns, marker.id = "rejoin_candidates", marker_id
            marker.type, marker.action = Marker.SPHERE, Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.35
            marker.color.a = 1.0
            marker.color.g = 1.0 if route_index == selected_index else 0.25
            marker.color.b = 0.0 if route_index == selected_index else 1.0
            candidate = self.route[route_index]
            marker.pose.position.x, marker.pose.position.y = candidate.x, candidate.y
            markers.markers.append(marker)
        self.pub_candidates.publish(markers)

    def tick(self):
        stale = (self.pose is None or self.pose_receipt is None or
                 time.monotonic()-self.pose_receipt >
                 float(self.get_parameter("pose_timeout_s").value))
        channel_enabled = (bool(self.get_parameter("enable_control").value) and
                           not bool(self.get_parameter("dry_run").value) and
                           bool(self.get_parameter("user_approved").value))
        pose = self.pose or Pose2D(0.0, 0.0, 0.0, 0.0)
        localized = self.localization_ready()
        if stale or not localized:
            original = ControllerResult(
                0.0, 0.0, self.core.last_index or 0, 0, 0.0, 0.0, 0.0,
                True, "STALE_POSE" if stale else "LOCALIZATION_NOT_STABLE")
        else:
            original = self.core.compute(
                pose, self.route, allow_motion=True,
                global_search=self.initial_search)
        if self.initial_search and not stale and localized:
            self.initial_search = False
        if stale:
            reason = "STALE_POSE"
            result = original
        elif not localized:
            reason = "LOCALIZATION_NOT_STABLE"
            result = original
        elif not self.map_route_verified:
            reason = self.binding_reason
            result = original
        else:
            if self.navigation_state == "FOLLOW_ROUTE":
                self.navigation_state = self.machine.observe_route(
                    original.cross_track_error)
            elif self.navigation_state == "ROUTE_DEVIATION_STOP":
                self.navigation_state = self.machine.vehicle_stopped()
            if self.navigation_state == "PLAN_REJOIN":
                if self.grid_ready():
                    self.rejoin_plan = self.rejoin_planner.plan(
                        pose, self.route, max(0, original.nearest_index),
                        self.grid,
                        bool(self.get_parameter("reverse_rejoin_allowed").value))
                    if self.rejoin_plan.feasible:
                        self.rejoin_route = [RoutePoint(
                            index, item.x, item.y, item.yaw, item.direction,
                            float(self.get_parameter("rejoin_speed").value))
                            for index, item in enumerate(self.rejoin_plan.path)]
                        self.rejoin_follower.reset_progress()
                        self.core.last_index = self.rejoin_plan.route_index
                    self.navigation_state = self.machine.plan_completed(
                        self.rejoin_plan.feasible)
            if self.navigation_state == "FOLLOW_REJOIN_PATH":
                result = self.rejoin_follower.compute(
                    pose, self.rejoin_route, allow_motion=True)
                remaining = self.rejoin_route[max(0, result.nearest_index):]
                if (not self.grid_ready() or not all(self.grid.footprint_free(
                        item, self.rejoin_planner.length,
                        self.rejoin_planner.width, self.rejoin_planner.margin)
                        for item in remaining)):
                    self.navigation_state = self.machine.path_blocked()
                elif result.reason == "ROUTE_COMPLETE":
                    self.navigation_state = self.machine.path_completed()
            elif self.navigation_state == "VERIFY_REJOIN":
                result = self.core.compute(pose, self.route, allow_motion=False)
                self.navigation_state = self.machine.verify(
                    result.cross_track_error, result.heading_error,
                    time.monotonic())
                if self.navigation_state == "FOLLOW_ROUTE":
                    self.rejoin_route = []
            else:
                result = original
            reason = self.navigation_state if self.navigation_state != "FOLLOW_ROUTE" else result.reason
        allow = (not stale and localized and self.map_route_verified and
                 self.safety_ready and not self.mission_stop and channel_enabled and
                 self.navigation_state in ("FOLLOW_ROUTE", "FOLLOW_REJOIN_PATH"))
        requested_drive = math.copysign(
            min(abs(result.drive), self.mission_speed_limit), result.drive)
        preview_drive = 0.0 if (stale or not localized or not self.map_route_verified or
                                result.stop_required or self.mission_stop or
                                self.navigation_state not in (
                                    "FOLLOW_ROUTE", "FOLLOW_REJOIN_PATH")) else requested_drive
        self.pub_preview_drive.publish(Float32(data=preview_drive))
        self.pub_preview_wheel.publish(Int32(data=int(round(result.steering_deg))))
        # Actual command topics remain completely silent unless all independent
        # control gates are true. Default launches therefore emit zero vehicle commands.
        if channel_enabled:
            self.pub_drive.publish(Float32(data=float(requested_drive if allow else 0.0)))
            self.pub_wheel.publish(Int32(data=int(round(result.steering_deg if allow else 0.0))))
        self.pub_cross.publish(Float32(data=float(result.cross_track_error)))
        self.pub_heading.publish(Float32(data=float(result.heading_error)))
        self.pub_progress.publish(Float32(data=float(result.progress)))
        self.pub_state.publish(String(data=reason))
        self.pub_rejoin_state.publish(String(data=self.navigation_state))
        self.pub_binding.publish(Bool(data=self.map_route_verified))
        within_map = bool(self.grid and self.pose and self.grid.footprint_free(
            self.pose, float(self.get_parameter("vehicle_length_m").value),
            float(self.get_parameter("vehicle_width_m").value),
            float(self.get_parameter("footprint_safety_margin_m").value)))
        self.pub_within_map.publish(Bool(data=within_map))
        self.publish_rejoin_visuals()
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        diagnostic.status = [status(
            "depth_slam/route_follower",
            DiagnosticStatus.WARN if result.stop_required or stale else DiagnosticStatus.OK,
            reason,
            (("cross_track_error_m", result.cross_track_error),
             ("heading_error_rad", result.heading_error),
             ("steering_deg", result.steering_deg),
             ("navigation_state", self.navigation_state),
             ("map_route_verified", self.map_route_verified),
             ("minimum_turning_radius_m", self.rejoin_planner.radius),
             ("rejoin_candidates_checked", self.rejoin_plan.candidates_checked
              if self.rejoin_plan else 0),
             ("rejoin_collision_free", self.rejoin_plan.collision_free
              if self.rejoin_plan else False),
             ("rejoin_selection_reason", self.rejoin_plan.reason
              if self.rejoin_plan else "NOT_PLANNED"),
             ("rejoin_selected_route_index", self.rejoin_plan.route_index
              if self.rejoin_plan else -1),
             ("rejoin_maximum_curvature", self.rejoin_plan.maximum_curvature
              if self.rejoin_plan else 0.0),
             ("actual_command_published", allow)))]
        self.pub_diag.publish(diagnostic)
        active_route = self.rejoin_route if self.navigation_state == "FOLLOW_REJOIN_PATH" else self.route
        if active_route:
            point = active_route[min(result.target_index, len(active_route)-1)]
            target = PointStamped()
            target.header.frame_id = "map"
            target.header.stamp = self.get_clock().now().to_msg()
            target.point.x, target.point.y = point.x, point.y
            self.pub_target.publish(target)


def main(args=None):
    rclpy.init(args=args)
    node = RouteFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
