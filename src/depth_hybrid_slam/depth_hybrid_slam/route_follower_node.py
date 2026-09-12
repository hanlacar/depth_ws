"""Thirty-hertz dry-run route follower. MCU control is impossible by default."""

import csv
from dataclasses import replace
import math
from pathlib import Path as FilePath
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from .models import ControllerResult, Pose2D, RoutePoint
from .mode_completion import RouteModeCompletionTracker
from .prehardware_core import StopWaypointMachine
from .rejoin_core import GridMap, RejoinPlanner, RejoinStateMachine
from .ros_helpers import (
    quaternion_from_yaw,
    safe_shutdown,
    stamp_seconds,
    status,
    yaw_from_quaternion,
)
from .route_follower_core import RouteFollower
from .csv_only_branching import load_csv_only_route_case, remap_case_progress
from .route_io import (
    DEFAULT_BRANCH, is_segmented_columns, load_segmented_route,
    verify_route_binding,
)


def load_route(path):
    """Load the legacy finalized map-route format."""
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


def route_is_segmented(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return is_segmented_columns(next(csv.reader(stream), ()))


class RouteFollowerNode(Node):
    def __init__(self):
        super().__init__("route_follower")
        for name, default in (("route_path", ""), ("enable_control", False),
                              ("dry_run", True), ("user_approved", False),
                              ("pose_timeout_s", 0.15), ("map_path", ""),
                              ("route_metadata_path", ""),
                              ("piecewise_preview_path", ""),
                              ("piecewise_preview_metadata_path", ""),
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
                              ("max_index_backtrack", 0),
                              ("localization_stability_s", 2.0),
                              ("rejoin_verify_duration_s", 1.0),
                              ("occupancy_grid_timeout_s", 1.0),
                              ("reverse_rejoin_allowed", False),
                              ("start_mode", 1), ("end_mode", 11),
                              ("direction_stop_trigger_distance_m", 1.0),
                              ("prehardware_test_override_alignment", False),
                              ("prehardware_csv_only_case_selection", False),
                              ("controller_hz", 30.0)):
            self.declare_parameter(name, default)
        path = str(self.get_parameter("route_path").value)
        metadata_path = str(self.get_parameter("route_metadata_path").value)
        self.route_path = path
        self.route_metadata_path = metadata_path
        self.active_branch = DEFAULT_BRANCH
        self.route_info = None
        if path and route_is_segmented(path):
            self.route_info = load_segmented_route(
                path, metadata_path, branch=self.active_branch)
            self.route = self._select_mode_range(self.route_info.points)
        else:
            self.route = load_route(path) if path else []
        preview_path = str(self.get_parameter("piecewise_preview_path").value)
        preview_metadata = str(self.get_parameter(
            "piecewise_preview_metadata_path").value)
        self.piecewise_preview = None
        if (preview_path and FilePath(preview_path).is_file() and
                route_is_segmented(preview_path)):
            self.piecewise_preview = load_segmented_route(
                preview_path, preview_metadata)
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
        self.map_route_verified, self.binding_reason = verify_route_binding(
            path, map_path, metadata_path) if path and map_path else (
                False, "MAP_ROUTE_FILES_MISSING")
        self.test_alignment_override = bool(self.get_parameter(
            "prehardware_test_override_alignment").value)
        self.csv_only_case_selection = bool(self.get_parameter(
            "prehardware_csv_only_case_selection").value)
        if self.csv_only_case_selection:
            if not self.test_alignment_override or self.route_info is None:
                raise ValueError(
                    "CSV-only case selection requires the explicit prehardware "
                    "override and a segmented route")
            self.active_case = "AAAA"
            self.route = self._select_mode_range(load_csv_only_route_case(
                self.route_path, self.route_metadata_path, self.active_case))
        else:
            self.active_case = ""
        self.control_route_available = (
            self.map_route_verified or
            (self.test_alignment_override and FilePath(path).is_file() and
             FilePath(map_path).is_file()))
        if self.test_alignment_override:
            self.get_logger().warn(
                "PREHARDWARE TEST ONLY: using unvalidated CSV-map transform; "
                "production map_route_verified remains false")
        if self.route_info is not None:
            info = self.route_info
            self.get_logger().info(f"[ROUTE] source CSV: {path}")
            self.get_logger().info(f"[ROUTE] raw rows: {info.raw_row_count}")
            self.get_logger().info(f"[ROUTE] default branch: {DEFAULT_BRANCH}")
            self.get_logger().info(f"[ROUTE] active rows: {info.active_row_count}")
            self.get_logger().info(
                f"[ROUTE] excluded B rows: {info.excluded_b_row_count} "
                f"segments={','.join(info.excluded_b_segments)}")
            self.get_logger().info(
                f"[ROUTE] segments: {' -> '.join(info.segment_order)}")
            self.get_logger().info(
                f"[ROUTE] start segment: {info.segment_order[0]}")
            self.get_logger().info(
                f"[ROUTE] end segment: {info.segment_order[-1]}")
            self.get_logger().info(
                f"[ROUTE] source frame: {info.source_frame}")
            transform = info.coordinate_transform
            self.get_logger().info(
                "[ROUTE] csv_to_map: "
                f"x={transform['x_m']:.8f}m y={transform['y_m']:.8f}m "
                f"yaw={transform['yaw_deg']:.8f}deg scale={transform['scale']:.3f}")
            self.get_logger().info(f"[ROUTE] frame: {info.frame_id}")
            self.get_logger().info(
                "[ROUTE] continuity: "
                f"max_step={info.maximum_step_m:.3f}m "
                f"max_connection={info.maximum_connection_m:.3f}m")
        log = self.get_logger().info if self.map_route_verified else self.get_logger().error
        log("[ROUTE] map-route verification: " +
            ("PASS" if self.map_route_verified else
             f"FAIL ({self.binding_reason})"))
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
        self.stop_waypoint = StopWaypointMachine(3.0)
        self.mode_completion = (
            RouteModeCompletionTracker(self.route) if self.route else None)
        self.pub_drive = self.create_publisher(
            Float32, "/depth_slam/follower/candidate_drive", 10)
        self.pub_wheel = self.create_publisher(
            Int32, "/depth_slam/follower/candidate_wheel", 10)
        self.pub_stop = self.create_publisher(
            Bool, "/depth_slam/follower/candidate_stop", 10)
        self.pub_mode = self.create_publisher(String, "/drive_mode", 10)
        self.pub_active_index = self.create_publisher(
            Int32, "/depth_slam/route/active_index", 10)
        self.pub_active_branch = self.create_publisher(
            String, "/depth_slam/route/active_branch", 10)
        self.pub_active_case = self.create_publisher(
            String, "/depth_slam/route/active_case", 10)
        self.pub_active_segment = self.create_publisher(
            String, "/depth_slam/route/active_segment", 10)
        self.pub_mode_status = self.create_publisher(
            String, "/depth_slam/route/mode_status", 10)
        self.pub_stop_state = self.create_publisher(
            String, "/depth_slam/route/stop_waypoint_state", 10)
        self.pub_stop_key = self.create_publisher(
            String, "/depth_slam/route/stop_waypoint_key", 10)
        self.pub_preview_drive = self.create_publisher(
            Float32, "/depth_slam/dry_run/drive", 10)
        self.pub_preview_wheel = self.create_publisher(
            Int32, "/depth_slam/dry_run/wheel", 10)
        latched_path = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_path = self.create_publisher(
            Path, "/depth_slam/route/reference_path", latched_path)
        self.pub_raw_path = self.create_publisher(
            Path, "/depth_slam/route/raw_csv_path", latched_path)
        self.pub_piecewise_path = self.create_publisher(
            Path, "/depth_slam/route/piecewise_path", latched_path)
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
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/depth_slam/localization/pose",
            self.on_pose,
            10,
        )
        self.create_subscription(String, "/depth_slam/safety/state",
                                 lambda m: setattr(self, "safety_ready", m.data == "READY"), 10)
        self.create_subscription(Bool, "/depth_slam/mission/stop_required",
                                 lambda m: setattr(self, "mission_stop", bool(m.data)), 10)
        self.create_subscription(
            Float32,
            "/depth_slam/mission/speed_limit",
            lambda m: setattr(
                self, "mission_speed_limit", max(0.0, float(m.data))
            ),
            10,
        )
        self.create_subscription(String, "/depth_slam/localization/state",
                                 self.on_localization_state, 10)
        self.create_subscription(String, "/depth_slam/route/selected_branch",
                                 self.on_selected_branch, 10)
        self.create_subscription(String, "/depth_slam/route/selected_case",
                                 self.on_selected_case, 10)
        self.create_subscription(OccupancyGrid, "/map", self.on_grid, 10)
        controller_hz = float(self.get_parameter("controller_hz").value)
        if not 1.0 <= controller_hz <= 250.0:
            raise ValueError("controller_hz must be in [1, 250]")
        self.create_timer(1.0/controller_hz, self.tick)
        self.publish_reference_path()

    def _select_mode_range(self, points):
        start = int(self.get_parameter("start_mode").value)
        end = int(self.get_parameter("end_mode").value)
        if not 1 <= start <= end <= 11:
            raise ValueError("start_mode/end_mode must satisfy 1 <= start <= end <= 11")
        selected = [point for point in points if start <= int(point.mode) <= end]
        if not selected:
            raise ValueError("selected mode range contains no route points")
        return [replace(point, index=index)
                for index, point in enumerate(selected)]

    @staticmethod
    def _equivalent_segment(segment_id, branch):
        mappings = {
            "START_A": f"START_{branch}", "START_B": f"START_{branch}",
            "T_A": f"T_{branch}", "T_B": f"T_{branch}",
            "V_A": f"V_{branch}", "V_B": f"V_{branch}",
            "END_AA": f"END_A{branch}", "END_AB": f"END_A{branch}",
        }
        return mappings.get(segment_id, segment_id)

    def on_selected_branch(self, message):
        if self.csv_only_case_selection:
            return
        branch = str(message.data).strip().upper()
        if branch not in ("A", "B") or branch == self.active_branch:
            return
        if self.route_info is None:
            return
        previous = None
        if self.route and self.core.last_index is not None:
            previous = self.route[min(self.core.last_index, len(self.route)-1)]
        new_info = load_segmented_route(
            self.route_path, self.route_metadata_path, branch=branch)
        new_route = self._select_mode_range(new_info.points)
        new_index = None
        if previous is not None:
            segment = self._equivalent_segment(previous.segment_id, branch)
            candidates = [point.index for point in new_route
                          if point.segment_id == segment and
                          point.point_index >= previous.point_index]
            if candidates:
                new_index = candidates[0]
        self.route_info = new_info
        self.route = new_route
        self.active_branch = branch
        self.core.reset_progress()
        self.rejoin_follower.reset_progress()
        self.core.last_index = new_index
        self.initial_search = new_index is None
        self.rejoin_route = []
        self.rejoin_plan = None
        self.navigation_state = "FOLLOW_ROUTE"
        self.stop_waypoint.reset()
        if self.mode_completion is not None:
            self.mode_completion.bind_route(self.route, new_index)
        self.publish_reference_path()

    def on_selected_case(self, message):
        if not self.csv_only_case_selection:
            return
        route_case = str(message.data).strip().upper()
        if (len(route_case) != 4 or any(value not in "AB" for value in route_case)
                or route_case == self.active_case):
            return
        previous = None
        if self.route and self.core.last_index is not None:
            previous = self.route[min(self.core.last_index, len(self.route)-1)]
        new_route = self._select_mode_range(load_csv_only_route_case(
            self.route_path, self.route_metadata_path, route_case))
        new_index = None
        if previous is not None:
            # T/V late decisions move only onto actual waypoints from the
            # equivalent branch.  No averaged or offset connector is made.
            new_index = remap_case_progress(previous, new_route, route_case)
        preserve_transition_stop = bool(
            self.stop_waypoint.active_key and previous is not None and
            previous.segment_id.startswith(("T_", "V_")))
        self.route = new_route
        self.active_case = route_case
        self.active_branch = route_case[0]
        self.core.reset_progress()
        self.rejoin_follower.reset_progress()
        self.core.last_index = new_index
        self.initial_search = new_index is None
        self.rejoin_route = []
        self.rejoin_plan = None
        self.navigation_state = "FOLLOW_ROUTE"
        if not preserve_transition_stop:
            self.stop_waypoint.reset()
        if self.mode_completion is not None:
            self.mode_completion.bind_route(self.route, new_index)
        self.publish_reference_path()

    def _stop_line_ahead(self, nearest, pose, trigger_distance=0.50):
        direction_trigger = float(self.get_parameter(
            "direction_stop_trigger_distance_m").value)
        if direction_trigger < trigger_distance:
            direction_trigger = trigger_distance
        distance = 0.0
        last = self.route[nearest]
        for point in self.route[nearest:min(len(self.route), nearest+80)]:
            if point is not last:
                distance += math.hypot(point.x-last.x, point.y-last.y)
            last = point
            required_direction_stop = (
                point.index+1 < len(self.route) and
                point.direction != self.route[point.index+1].direction)
            limit = direction_trigger if required_direction_stop else trigger_distance
            if (point.event == "STOP_LINE" or required_direction_stop) and (
                    distance <= limit or
                    math.hypot(point.x-pose.x, point.y-pose.y) <=
                    limit):
                return f"{point.segment_id}:{point.point_index}", True
            if distance > direction_trigger:
                break
        return "", False

    def _next_stop_index(self):
        first = max(0, self.core.last_index or 0)
        for index in range(first, len(self.route)):
            point = self.route[index]
            key = f"{point.segment_id}:{point.point_index}"
            direction_change = (index+1 < len(self.route) and
                                point.direction != self.route[index+1].direction)
            if ((point.event == "STOP_LINE" or direction_change) and
                    key not in self.stop_waypoint.completed):
                return index
        return None

    def publish_reference_path(self):
        message = Path()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        for point in self.route:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = point.x, point.y
            pose.pose.position.z = 0.25
            pose.pose.orientation = quaternion_from_yaw(point.yaw)
            message.poses.append(pose)
        self.pub_path.publish(message)
        if self.route_info is not None:
            # RViz has no physical TF between an unaligned GPS-local frame and
            # map.  Publish the untouched numeric ENU coordinates in map axes
            # solely as a clearly named before/after diagnostic overlay.
            raw = Path()
            raw.header = message.header
            for point in self.route_info.source_points:
                pose = PoseStamped()
                pose.header = raw.header
                pose.pose.position.x, pose.pose.position.y = point.x, point.y
                pose.pose.position.z = 0.25
                pose.pose.orientation = quaternion_from_yaw(point.yaw)
                raw.poses.append(pose)
            self.pub_raw_path.publish(raw)
        if self.piecewise_preview is not None:
            preview = Path()
            preview.header = message.header
            for point in self.piecewise_preview.points:
                pose = PoseStamped()
                pose.header = preview.header
                pose.pose.position.x, pose.pose.position.y = point.x, point.y
                pose.pose.position.z = 0.35
                pose.pose.orientation = quaternion_from_yaw(point.yaw)
                preview.poses.append(pose)
            self.pub_piecewise_path.publish(preview)

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
                global_search=self.initial_search,
                progress_ceiling=self._next_stop_index())
        if self.initial_search and not stale and localized:
            self.initial_search = False
        if stale:
            reason = "STALE_POSE"
            result = original
        elif not localized:
            reason = "LOCALIZATION_NOT_STABLE"
            result = original
        elif not self.control_route_available:
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
            reason = (
                self.navigation_state
                if self.navigation_state != "FOLLOW_ROUTE"
                else result.reason
            )
        waypoint_key, waypoint_reached = (self._stop_line_ahead(
            max(0, original.nearest_index), pose) if self.route and
            not stale and localized else ("", False))
        waypoint_decision = self.stop_waypoint.update(
            waypoint_key, waypoint_reached, self.mission_stop,
            time.monotonic())
        transition_release = False
        if waypoint_decision.state == "RELEASED" and waypoint_key:
            for index, point in enumerate(self.route[:-1]):
                if (f"{point.segment_id}:{point.point_index}" == waypoint_key and
                        point.direction != self.route[index+1].direction):
                    # One stopped output cycle separates longitudinal signs.
                    # Start the next compute inside the new direction block so
                    # overlapping parking geometry cannot select the old leg.
                    self.core.last_index = index+1
                    self.core.last_steering = 0.0
                    transition_release = True
                    break
        allow = (not stale and localized and self.control_route_available and
                 self.safety_ready and not self.mission_stop and
                 not waypoint_decision.stop and not transition_release and
                 not result.stop_required and
                 channel_enabled and
                 self.navigation_state in ("FOLLOW_ROUTE", "FOLLOW_REJOIN_PATH"))
        requested_drive = math.copysign(
            min(abs(result.drive), self.mission_speed_limit), result.drive)
        preview_drive = 0.0 if (stale or not localized or not self.control_route_available or
                                result.stop_required or self.mission_stop or
                                waypoint_decision.stop or transition_release or
                                self.navigation_state not in (
                                    "FOLLOW_ROUTE", "FOLLOW_REJOIN_PATH")) else requested_drive
        self.pub_preview_drive.publish(Float32(data=preview_drive))
        self.pub_preview_wheel.publish(Int32(data=int(round(result.steering_deg))))
        # The follower owns candidate topics only. BehaviorSelector is the sole
        # owner of /slam_* and applies road/localization/branch gates.
        if channel_enabled:
            self.pub_drive.publish(Float32(data=float(requested_drive if allow else 0.0)))
            self.pub_wheel.publish(Int32(data=int(round(result.steering_deg if allow else 0.0))))
            self.pub_stop.publish(Bool(data=not allow))
        if self.route and not stale and localized:
            mode_index = min(max(0, original.nearest_index), len(self.route)-1)
            self.pub_mode.publish(String(data=str(self.route[mode_index].mode)))
            self.pub_active_index.publish(Int32(data=mode_index))
            self.pub_active_segment.publish(String(
                data=self.route[mode_index].segment_id))
        if self.route and self.mode_completion is not None:
            mode_index = min(max(0, original.nearest_index), len(self.route)-1)
            normal_reason = original.reason in (
                "OK", "CURVATURE_SLOWDOWN", "ROUTE_COMPLETE")
            failure_reason = ""
            if stale:
                failure_reason = "STALE_POSE"
            elif not localized:
                failure_reason = "LOCALIZATION_NOT_STABLE"
            elif not self.control_route_available:
                failure_reason = self.binding_reason
            elif not self.safety_ready:
                failure_reason = "SAFETY_NOT_READY"
            elif self.navigation_state != "FOLLOW_ROUTE":
                failure_reason = self.navigation_state
            elif not normal_reason:
                failure_reason = original.reason
            completion_healthy = (
                channel_enabled and normal_reason and not failure_reason)
            completion_stopped = (
                self.mission_stop or waypoint_decision.stop or
                transition_release)
            for event in self.mode_completion.observe(
                    mode_index, original.progress,
                    healthy=completion_healthy,
                    stopped=completion_stopped,
                    failure_reason=failure_reason,
                    controller_reason=original.reason):
                self.get_logger().info(event)
            self.pub_mode_status.publish(String(
                data=self.mode_completion.status_json()))
        self.pub_active_branch.publish(String(data=self.active_branch))
        if self.csv_only_case_selection:
            self.pub_active_case.publish(String(data=self.active_case))
        self.pub_stop_state.publish(String(data=waypoint_decision.state))
        self.pub_stop_key.publish(String(data=waypoint_key))
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
             ("prehardware_test_override_alignment", self.test_alignment_override),
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
             ("active_branch", self.active_branch),
             ("stop_waypoint_state", waypoint_decision.state),
             ("stop_waypoint_elapsed_s", waypoint_decision.elapsed_s),
             ("direction_transition_release_stop", transition_release),
             ("actual_command_published", allow)))]
        self.pub_diag.publish(diagnostic)
        active_route = (
            self.rejoin_route
            if self.navigation_state == "FOLLOW_REJOIN_PATH"
            else self.route
        )
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
        try:
            node.destroy_node()
            safe_shutdown()
        except KeyboardInterrupt:
            pass
