"""ROS wiring for LiDAR temporary paths, parking and mode-specific gates."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from nav2_msgs.action import ComputePathThroughPoses
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .lidar_local_planner import (
    audit_path, LocalPlan, plan_parking, plan_route_detour,
    swept_footprint_clear)
from .lidar_mission_core import (
    ManeuverDecision, Mode11ExitGate, Mode5Avoidance, Mode5ObstacleLatch,
    Mode9Emergency, ParkingManeuver, StationaryConfirmation,
    mode5_planning_distance_ready, parking_reverse_phase,
    select_parking_fallback)
from .lidar_path_tracker import LocalPathTracker, TrackCommand
from .parking_planner_core import (
    assess_parking_path, fresh_explicit_b, parking_csv_fallback_segment,
    parking_slot_observation_state,
    ParkingRuntimeCoordinator, ParkingVehicleGeometry,
    select_explicit_b_slot, validate_nav2_parking_path)
from .route_io import load_segmented_route


class ManeuverManagerNode(Node):
    def __init__(self):
        super().__init__("depth_maneuver_manager")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("prehardware_case_commands", False)
        # False uses the commissioned front-LiDAR slot evidence and recorded
        # CSV path. Explicit B selects B; every other observation defaults A.
        self.declare_parameter("rear_lidar_enabled", True)
        self.declare_parameter("enable_parking_slam", False)
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        for name, default in (
                ("wheelbase_m", 0.73),
                ("planner_max_steering_deg", 20.0),
                ("vehicle_width_m", 0.80),
                ("vehicle_length_m", 1.40),
                ("parking_vehicle_width_m", 0.78),
                ("parking_front_overhang_m", 0.47),
                ("parking_rear_overhang_m", 0.20),
                ("parking_wheel_track_m", 0.62),
                ("parking_wheel_length_m", 0.20),
                ("parking_wheel_width_m", 0.08),
                ("obstacle_margin_m", 0.15),
                ("detour_length_m", 1.5),
                ("detour_maximum_ahead_m", 8.0),
                ("planner_maximum_replans", 48),
                ("obstacle_confirmation_s", 2.0),
                ("minimum_planning_lidar_distance_m", 1.0),
                ("stopped_confirmation_s", 0.30),
                ("stopped_linear_speed_mps", 0.03),
                ("stopped_angular_speed_rps", 0.03),
                ("odom_timeout_s", 0.50),
                ("parking_slot_timeout_s", 0.50),
                ("parking_map_wait_timeout_s", 3.0)):
            self.declare_parameter(name, default)
        self.mode = -1
        self.avoidance = self.hard = self.rear_hard = False
        self.a_free = self.b_free = False
        self.slots_fresh = False
        self.slots_at = None
        self.perception_fresh = False
        self.perception_at = None
        self.planner_state = "IDLE"
        self.path_valid = self.path_complete = self.rejoin_valid = False
        self.obstacle_y = 0.0
        self.obstacles = ()
        self.obstacle_lidar_distance = None
        self.curbs = ()
        self.pose = None
        self.odom_linear = self.odom_angular = None
        self.odom_at = None
        self.map_pose = None
        self.active_index = 0
        self.active_segment = ""
        self.csv_drive = 0.0
        self.case_choices = {"T": "pending", "V": "pending"}
        self.last_state = None
        self.last_branch = None
        self.last_plan_log = None
        self.plan_audit = {}
        self.active_branch = "A"
        self.route = ()
        self.routes_by_branch = {}
        self.mode5_plan = None
        self.mode5_obstacles_map = ()
        self.mode5_curbs_map = ()
        self.mode5_obstacle_lidar_distance = None
        self._load_route()
        self.mode5 = Mode5Avoidance()
        self.mode5_obstacle = Mode5ObstacleLatch(
            self.get_parameter("obstacle_confirmation_s").value)
        self.mode5_stationary = StationaryConfirmation(
            self.get_parameter("stopped_confirmation_s").value,
            self.get_parameter("stopped_linear_speed_mps").value,
            self.get_parameter("stopped_angular_speed_rps").value)
        self.mode9 = Mode9Emergency()
        self.mode11 = Mode11ExitGate()
        self.parking = {7: ParkingManeuver(7), 10: ParkingManeuver(10)}
        # Once LiDAR takes exclusive ownership the CSV candidate is
        # intentionally zeroed by the suspended route follower.  Keep the
        # already-audited activation plan/branch latched until handoff; using
        # the live CSV drive as an eligibility gate after activation would
        # discard the plan and falsely report PATH_ABORT.
        self.parking_plans = {7: None, 10: None}
        self.parking_branches = {7: "", 10: ""}
        self.parking_slam_enabled = bool(
            self.get_parameter("enable_parking_slam").value)
        self.parking_runtime = ParkingRuntimeCoordinator(
            self.get_parameter("parking_map_wait_timeout_s").value)
        self.parking_slam_active = False
        self.parking_map_ready = False
        self.parking_assessments = {"A": None, "B": None}
        self.parking_selected_source = ""
        self.parking_failed_reason = ""
        self.parking_slot_state = {"A": "UNKNOWN", "B": "UNKNOWN"}
        self.parking_grid_at = None
        self.parking_grid_unknown_ratio = 1.0
        self.parking_map_obstacles = ()
        self.nav2_client = ActionClient(
            self, ComputePathThroughPoses, "/compute_path_through_poses")
        self.nav2_pending = False
        self.nav2_attempted = []
        self.nav2_selected_slot = ""
        self.nav2_plan = None
        self.nav2_anchor = None
        self.nav2_failure = ""
        self.nav2_planning_state = "IDLE"
        self.nav2_validation_reason = "NOT_PLANNED"
        self.parking_mode_entered_at = None
        self.nav2_requested_at = None
        self.nav2_last_attempt_at = None
        self.nav2_goal_handle = None
        self.nav2_generation = 0
        self.declare_parameter("nav2_plan_timeout_s", 8.0)
        self.declare_parameter("nav2_replan_interval_s", 0.5)
        self.declare_parameter("nav2_minimum_path_points", 3)
        self.declare_parameter("nav2_maximum_connection_distance_m", 0.75)
        self.declare_parameter("nav2_maximum_connection_heading_deg", 45.0)
        self.stop_waypoint_state = "IDLE"
        self.stop_waypoint_key = ""
        self.path_owner = "CSV"
        self.owner_handoff_active = False
        self.owner_handoff_elapsed_s = 0.0
        self.tracker = LocalPathTracker()
        self.tracker_key = ""
        self.previous_mode = -1
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(Bool, "/depth_slam/lidar/avoidance_required",
                                 lambda m: setattr(self, "avoidance", bool(m.data)), 10)
        self.create_subscription(Bool, "/depth_slam/lidar/hard_emergency",
                                 lambda m: setattr(self, "hard", bool(m.data)), 10)
        self.create_subscription(
            Bool, "/depth_slam/lidar/rear_hard_emergency",
            lambda m: setattr(self, "rear_hard", bool(m.data)), 10)
        self.create_subscription(String, "/depth_slam/lidar/parking_slots",
                                 self._slots, 10)
        self.create_subscription(String, "/depth_slam/lidar/perception",
                                 self._perception, 10)
        self.create_subscription(String, "/depth_slam/lidar/planner_state",
                                 lambda m: setattr(self, "planner_state", m.data), 10)
        self.create_subscription(Bool, "/depth_slam/lidar/path_complete",
                                 lambda m: setattr(self, "path_complete", bool(m.data)), 10)
        self.create_subscription(Bool, "/depth_slam/lidar/parking_path_valid",
                                 lambda m: setattr(self, "path_valid", bool(m.data)), 10)
        self.create_subscription(Bool, "/depth_slam/lidar/csv_rejoin_valid",
                                 lambda m: setattr(self, "rejoin_valid", bool(m.data)), 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self._map_pose, 10)
        self.create_subscription(
            Int32, "/depth_slam/route/active_index",
            lambda m: setattr(self, "active_index", int(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/active_segment",
            lambda m: setattr(self, "active_segment", str(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/stop_waypoint_state",
            lambda m: setattr(self, "stop_waypoint_state", str(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/stop_waypoint_key",
            lambda m: setattr(self, "stop_waypoint_key", str(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/selected_branch",
            self._selected_branch, 10)
        self.create_subscription(
            Float32, "/depth_slam/follower/candidate_drive",
            lambda m: setattr(self, "csv_drive", float(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/case_state", self._case_state, 10)
        self.create_subscription(
            Bool, "/depth_slam/parking/slam_active",
            lambda m: setattr(self, "parking_slam_active", bool(m.data)), 10)
        self.create_subscription(
            Bool, "/depth_slam/parking/map_ready",
            lambda m: setattr(self, "parking_map_ready", bool(m.data)), 10)
        self.create_subscription(
            OccupancyGrid, "/map", self._parking_map, 10)
        self.create_subscription(
            String, "/depth_slam/path_owner",
            lambda m: setattr(self, "path_owner", str(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/command_arbiter/diagnostics",
            self._arbiter_diagnostics, 10)
        self.create_subscription(String, "/camera/exit_branch_signal",
                                 self._exit_signal, 10)
        self.create_subscription(String, "/depth_slam/mode11/signal_event",
                                 self._exit_signal_event, 10)
        self.drive_pub = self.create_publisher(
            Float32, "/depth_slam/lidar/candidate_drive", 10)
        self.wheel_pub = self.create_publisher(
            Int32, "/depth_slam/lidar/candidate_wheel", 10)
        self.valid_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/candidate_valid", 10)
        self.hold_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/hold", 10)
        self.branch_pub = self.create_publisher(
            String, "/depth_slam/route/branch_command", 10)
        self.path_pub = self.create_publisher(
            Path, "/depth_slam/lidar/local_path", 10)
        self.rejoin_target_pub = self.create_publisher(
            Int32, "/depth_slam/lidar/planned_rejoin_index", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/lidar/maneuver", 10)
        self.event_pub = self.create_publisher(
            String, "/depth_slam/lidar/maneuver_event", 10)
        self.parking_diag_pub = self.create_publisher(
            String, "/depth_slam/parking/planner_diagnostics", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    def _mode(self, message):
        try:
            self.mode = int(str(message.data).strip())
        except ValueError:
            self.mode = -1

    def _slots(self, message):
        try:
            value = json.loads(message.data)
            self.a_free = bool(value.get("a_free", False))
            self.b_free = bool(value.get("b_free", False))
            self.slots_fresh = bool(value.get("fresh", False))
            self.slots_at = time.monotonic()
        except (AttributeError, TypeError, json.JSONDecodeError):
            self.a_free = self.b_free = False
            self.slots_fresh = False
            self.slots_at = None

    def _arbiter_diagnostics(self, message):
        try:
            values = json.loads(message.data)
            self.owner_handoff_active = bool(values.get(
                "owner_handoff_active", False))
            self.owner_handoff_elapsed_s = float(values.get(
                "owner_handoff_elapsed_s", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            self.owner_handoff_active = False
            self.owner_handoff_elapsed_s = 0.0

    def _parking_reverse_decision_point(self):
        """True at the first forward-to-reverse STOP of Mode 7/10."""
        if self.mode not in (7, 10) or not self.route:
            return False
        key = str(self.stop_waypoint_key)
        if self.stop_waypoint_state != "IDLE" and key:
            for index, point in enumerate(self.route[:-1]):
                following = self.route[index+1]
                if (int(point.mode) == self.mode and point.direction > 0 and
                        following.direction < 0 and
                        f"{point.segment_id}:{point.point_index}" == key):
                    return True
        # This is only a sampled-event fallback. Normally the STOP key above
        # commits and freezes SLAM before the route enters the reverse point.
        index = min(max(0, int(self.active_index)), len(self.route)-1)
        point = self.route[index]
        return bool(int(point.mode) == self.mode and point.direction < 0 and
                    point.segment_id in (
                        f"{'T' if self.mode == 7 else 'V'}_A",
                        f"{'T' if self.mode == 7 else 'V'}_B"))

    def _perception(self, message):
        try:
            value = json.loads(message.data)
            self.obstacle_y = float(value.get("obstacle_y_m", 0.0))
            raw_distance = value.get("avoidance_nearest_lidar_m")
            self.obstacle_lidar_distance = (
                None if raw_distance is None else float(raw_distance))
            self.obstacles = tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("obstacles", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2)
            self.curbs = tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("curbs", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2)
            self.perception_fresh = bool(value.get("fresh", False))
            self.perception_at = time.monotonic()
        except (TypeError, ValueError, json.JSONDecodeError):
            self.obstacle_y = 0.0
            self.obstacle_lidar_distance = None
            self.obstacles = ()
            self.curbs = ()
            self.perception_fresh = False

    def _case_state(self, message):
        try:
            selected = json.loads(message.data).get("selected", {})
            for choice in ("T", "V"):
                value = str(selected.get(choice, "pending")).upper()
                self.case_choices[choice] = (
                    value if value in ("A", "B") else "pending")
        except (TypeError, ValueError, json.JSONDecodeError):
            self.case_choices = {"T": "pending", "V": "pending"}

    def _odom(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0*(orientation.w*orientation.z+orientation.x*orientation.y),
            1.0-2.0*(orientation.y**2+orientation.z**2))
        self.pose = (float(position.x), float(position.y), yaw)
        self.odom_linear = float(message.twist.twist.linear.x)
        self.odom_angular = float(message.twist.twist.angular.z)
        self.odom_at = time.monotonic()

    def _map_pose(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0*(orientation.w*orientation.z+orientation.x*orientation.y),
            1.0-2.0*(orientation.y**2+orientation.z**2))
        self.map_pose = (float(position.x), float(position.y), yaw)

    def _load_route(self):
        path = str(self.get_parameter("route_path").value)
        metadata = str(self.get_parameter("route_metadata_path").value)
        if not path:
            self.route = ()
            self.routes_by_branch = {}
            return
        self.routes_by_branch = {
            branch: load_segmented_route(
                path, metadata, branch=branch).points
            for branch in ("A", "B")}
        self.route = self.routes_by_branch[self.active_branch]

    def _parking_geometry(self):
        return ParkingVehicleGeometry(
            wheelbase_m=float(self.get_parameter("wheelbase_m").value),
            vehicle_length_m=float(
                self.get_parameter("vehicle_length_m").value),
            vehicle_width_m=float(
                self.get_parameter("parking_vehicle_width_m").value),
            front_overhang_m=float(
                self.get_parameter("parking_front_overhang_m").value),
            rear_overhang_m=float(
                self.get_parameter("parking_rear_overhang_m").value),
            wheel_track_m=float(
                self.get_parameter("parking_wheel_track_m").value),
            wheel_length_m=float(
                self.get_parameter("parking_wheel_length_m").value),
            wheel_width_m=float(
                self.get_parameter("parking_wheel_width_m").value),
            safety_margin_m=float(
                self.get_parameter("obstacle_margin_m").value),
            planner_max_steering_deg=float(
                self.get_parameter("planner_max_steering_deg").value))

    def _parking_map(self, message):
        """Cache occupied parking-map cells in the current vehicle frame."""
        if not self.parking_slam_enabled or self.mode not in (7, 10):
            return
        total = len(message.data)
        if total <= 0 or self.map_pose is None:
            self.parking_grid_at = None
            self.parking_map_obstacles = ()
            self.parking_grid_unknown_ratio = 1.0
            return
        self.parking_grid_unknown_ratio = sum(
            1 for value in message.data if int(value) < 0)/total
        info = message.info
        resolution = float(info.resolution)
        width = int(info.width)
        origin = info.origin
        oq = origin.orientation
        origin_yaw = math.atan2(
            2.0*(oq.w*oq.z+oq.x*oq.y),
            1.0-2.0*(oq.y*oq.y+oq.z*oq.z))
        oc, os = math.cos(origin_yaw), math.sin(origin_yaw)
        px, py, pyaw = self.map_pose
        pc, ps = math.cos(pyaw), math.sin(pyaw)
        # One 10 cm sample per occupied patch is enough for a conservative
        # footprint sweep and keeps the 20 Hz maneuver loop bounded.
        stride = max(1, int(round(0.10/max(resolution, 1.0e-6))))
        occupied = []
        for index, value in enumerate(message.data):
            if int(value) < 50:
                continue
            row, column = divmod(index, width)
            if row % stride or column % stride:
                continue
            gx0 = (column+0.5)*resolution
            gy0 = (row+0.5)*resolution
            gx = float(origin.position.x)+oc*gx0-os*gy0
            gy = float(origin.position.y)+os*gx0+oc*gy0
            dx, dy = gx-px, gy-py
            lx, ly = pc*dx+ps*dy, -ps*dx+pc*dy
            if lx*lx+ly*ly <= 144.0:
                occupied.append((lx, ly))
                if len(occupied) >= 5000:
                    break
        self.parking_map_obstacles = tuple(occupied)
        self.parking_grid_at = time.monotonic()

    def _parking_path_assessment(self, branch, objects=None):
        if self.map_pose is None:
            return None
        prefix = "T" if self.mode == 7 else "V"
        segment = prefix+"_"+str(branch)
        points = tuple(point for point in self.routes_by_branch.get(branch, ())
                       if point.segment_id == segment)
        if not points:
            return None
        px, py, pyaw = self.map_pose
        c, s = math.cos(pyaw), math.sin(pyaw)
        local = tuple((c*(point.x-px)+s*(point.y-py),
                       -s*(point.x-px)+c*(point.y-py),
                       math.atan2(math.sin(point.yaw-pyaw),
                                  math.cos(point.yaw-pyaw)))
                      for point in points)
        evidence = self.obstacles+self.curbs if objects is None else objects
        return assess_parking_path(local, evidence, self._parking_geometry())

    def _parking_path_safe(self, branch):
        assessment = self._parking_path_assessment(branch)
        return None if assessment is None else assessment.safe

    @staticmethod
    def _pose_message(x, y, yaw, stamp):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = stamp
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(float(yaw)/2.0)
        pose.pose.orientation.w = math.cos(float(yaw)/2.0)
        return pose

    def _request_nav2_plan(self, slot):
        """Request an Ackermann/Reeds-Shepp path through parking waypoints."""
        slot = str(slot).strip().upper()
        self.nav2_attempted.append(slot)
        self.nav2_selected_slot = slot
        self.nav2_plan = None
        self.nav2_failure = ""
        self.nav2_validation_reason = "PLANNING"
        self.nav2_last_attempt_at = time.monotonic()
        self.nav2_generation += 1
        generation = self.nav2_generation
        seed = plan_parking(
            self.mode, slot,
            wheelbase_m=float(self.get_parameter("wheelbase_m").value),
            planner_max_steering_deg=float(self.get_parameter(
                "planner_max_steering_deg").value))
        if not seed.valid or self.map_pose is None:
            self.nav2_failure = "INVALID_ACKERMANN_SEED"
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return False
        if not self.nav2_client.server_is_ready():
            self.nav2_failure = "NAV2_ACTION_UNAVAILABLE"
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return False
        px, py, pyaw = self.map_pose
        self.nav2_anchor = self.map_pose
        c, s = math.cos(pyaw), math.sin(pyaw)
        stamp = self.get_clock().now().to_msg()
        # Preserve the commissioned reverse parking shape while allowing
        # Smac Hybrid-A* to alter it around occupied map cells.
        sample_step = max(1, len(seed.points)//8)
        sampled = list(seed.points[sample_step::sample_step])
        if not sampled or sampled[-1] != seed.points[-1]:
            sampled.append(seed.points[-1])
        goal = ComputePathThroughPoses.Goal()
        goal.goals = [self._pose_message(
            px+c*x-s*y, py+s*x+c*y, pyaw+yaw, stamp)
            for x, y, yaw in sampled]
        goal.start = self._pose_message(px, py, pyaw, stamp)
        goal.use_start = True
        goal.planner_id = "GridBased"
        self.nav2_pending = True
        self.nav2_planning_state = "PENDING"
        self.nav2_requested_at = time.monotonic()
        future = self.nav2_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result: self._nav2_goal_response(
                result, generation, slot))
        return True

    def _nav2_goal_response(self, future, generation, slot):
        if generation != self.nav2_generation:
            return
        try:
            handle = future.result()
        except Exception as error:
            self.nav2_pending = False
            self.nav2_requested_at = None
            self.nav2_failure = "NAV2_GOAL_ERROR:"+type(error).__name__
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return
        if not handle.accepted:
            self.nav2_pending = False
            self.nav2_requested_at = None
            self.nav2_failure = "NAV2_GOAL_REJECTED"
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return
        self.nav2_goal_handle = handle
        result = handle.get_result_async()
        result.add_done_callback(
            lambda value: self._nav2_result(value, generation, slot))

    def _nav2_result(self, future, generation, slot):
        if generation != self.nav2_generation:
            return
        self.nav2_pending = False
        self.nav2_requested_at = None
        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as error:
            self.nav2_goal_handle = None
            self.nav2_failure = "NAV2_RESULT_ERROR:"+type(error).__name__
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return
        if (int(result.error_code) != ComputePathThroughPoses.Result.NONE or
                not result.path.poses or self.nav2_anchor is None):
            self.nav2_failure = (
                "NAV2_PLAN_FAILED:"+str(int(result.error_code)))
            self.nav2_goal_handle = None
            self.nav2_planning_state = "FAILED"
            self.nav2_validation_reason = self.nav2_failure
            return
        px, py, pyaw = self.nav2_anchor
        c, s = math.cos(pyaw), math.sin(pyaw)
        points = []
        for stamped in result.path.poses:
            position = stamped.pose.position
            orientation = stamped.pose.orientation
            yaw = math.atan2(
                2.0*(orientation.w*orientation.z+
                     orientation.x*orientation.y),
                1.0-2.0*(orientation.y**2+orientation.z**2))
            dx, dy = float(position.x)-px, float(position.y)-py
            points.append((c*dx+s*dy, -s*dx+c*dy,
                           math.atan2(math.sin(yaw-pyaw),
                                      math.cos(yaw-pyaw))))
        audit = audit_path(points, float(
            self.get_parameter("wheelbase_m").value))
        footprint = assess_parking_path(
            points,
            self.parking_map_obstacles or self.obstacles+self.curbs,
            self._parking_geometry())
        maximum = float(self.get_parameter(
            "planner_max_steering_deg").value)
        validation = validate_nav2_parking_path(
            points, planning_success=True, audit_feasible=audit.feasible,
            max_required_steering_deg=audit.max_required_steering_deg,
            footprint=footprint, selected_slot=slot,
            target_slot=self.nav2_selected_slot,
            steering_limit_deg=maximum,
            minimum_points=int(self.get_parameter(
                "nav2_minimum_path_points").value),
            maximum_connection_distance_m=float(self.get_parameter(
                "nav2_maximum_connection_distance_m").value),
            maximum_connection_heading_deg=float(self.get_parameter(
                "nav2_maximum_connection_heading_deg").value))
        valid = validation.valid
        self.nav2_plan = LocalPlan(
            valid, "NAV2_READY" if valid else "NAV2_PATH_REJECTED",
            tuple(points), audit.max_required_steering_deg,
            audit.max_curvature, audit.min_turn_radius_m, 0,
            "" if valid else validation.reason,
            audit.max_required_steering_deg, -1)
        self.nav2_validation_reason = validation.reason
        self.nav2_planning_state = "SUCCESS" if valid else "FAILED"
        if not valid:
            self.nav2_failure = self.nav2_plan.replan_reason
        self.nav2_goal_handle = None

    def _expire_nav2_request(self):
        if self.nav2_goal_handle is not None:
            self.nav2_goal_handle.cancel_goal_async()
        self.nav2_generation += 1
        self.nav2_pending = False
        self.nav2_requested_at = None
        self.nav2_goal_handle = None
        self.nav2_plan = None
        self.nav2_failure = "NAV2_PLAN_TIMEOUT"
        self.nav2_planning_state = "FAILED"
        self.nav2_validation_reason = self.nav2_failure

    def _selected_branch(self, message):
        branch = str(message.data).strip().upper()
        if branch not in ("A", "B") or branch == self.active_branch:
            return
        self.active_branch = branch
        self._load_route()
        self.mode5_plan = None

    @staticmethod
    def _local_to_fixed(points, pose):
        px, py, yaw = pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return tuple((px+cosine*x-sine*y, py+sine*x+cosine*y)
                     for x, y in points)

    @staticmethod
    def _fixed_to_local(points, pose):
        px, py, yaw = pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return tuple((cosine*(x-px)+sine*(y-py),
                      -sine*(x-px)+cosine*(y-py))
                     for x, y in points)

    def _snapshot_mode5_scene(self):
        if self.map_pose is None or not self.obstacles:
            return
        self.mode5_obstacles_map = self._local_to_fixed(
            self.obstacles, self.map_pose)
        self.mode5_curbs_map = self._local_to_fixed(
            self.curbs, self.map_pose)
        if self.obstacle_lidar_distance is not None:
            self.mode5_obstacle_lidar_distance = \
                self.obstacle_lidar_distance

    def _mode5_scene_at_stop(self):
        obstacles = (self._fixed_to_local(
            self.mode5_obstacles_map, self.map_pose)
            if self.map_pose is not None and self.mode5_obstacles_map else
            self.obstacles)
        curbs = (self._fixed_to_local(self.mode5_curbs_map, self.map_pose)
                 if self.map_pose is not None and self.mode5_curbs_map else
                 self.curbs)
        obstacle_y = (sum(y for _, y in obstacles)/len(obstacles)
                      if obstacles else self.obstacle_y)
        left = min((y for _, y in curbs if y > 0.0), default=None)
        right = max((y for _, y in curbs if y < 0.0), default=None)
        return obstacles, obstacle_y, left, right, curbs

    def _exit_signal(self, message):
        if self.mode == 11:
            self.mode11.observe(message.data, time.monotonic())

    def _exit_signal_event(self, message):
        if self.mode != 11:
            return
        try:
            value = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if str(value.get("state", "")) not in ("LATCHED", "DEFAULTED"):
            return
        route = str(value.get("route", "")).upper()
        reason = str(value.get("reason", "CAMERA_WEIGHTED"))
        self.mode11.commit_external(route, "CAMERA_WEIGHTED:"+reason)

    def _publish_plan(self, plan):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        # The tracker has already anchored the local plan at the measured
        # odom pose. Publish those immutable points so RViz does not drag the
        # path along with a moving base_link.
        fixed_points = (self.tracker.points if self.tracker.points else
                        plan.points)
        for x, y, yaw in fixed_points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z = math.sin(yaw/2.0)
            pose.pose.orientation.w = math.cos(yaw/2.0)
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _track(self, key, plan, drive):
        if not plan or not plan.valid or self.pose is None:
            return TrackCommand(False, False, 0.0, 0, 0)
        if key != self.tracker_key:
            self.tracker.clear()
            self.tracker.set_plan(plan.points, self.pose, drive)
            self.tracker_key = key
            self._publish_plan(plan)
            self.get_logger().info(f"[LIDAR PATH] ACTIVE mode={self.mode}")
        return self.tracker.update(self.pose)

    def _record_plan(self, plan):
        if plan is None:
            self.plan_audit = {}
            self.last_plan_log = None
            return
        self.plan_audit = {
            "path_point_count": plan.path_point_count,
            "max_required_steering_deg": plan.max_steering_deg,
            "max_curvature": plan.max_curvature,
            "min_turn_radius_m": plan.min_turn_radius_m,
            "feasible": plan.valid,
            "replans": plan.replans,
            "replan_reason": plan.replan_reason,
            "csv_rejoin_index": plan.route_index,
        }
        signature = (
            self.mode, plan.state, round(plan.max_steering_deg, 3),
            plan.replans, plan.valid)
        if signature == self.last_plan_log:
            return
        if plan.replans:
            self.get_logger().info(
                "[LIDAR PATH] REPLAN "
                f"reason={plan.replan_reason} "
                f"candidate={plan.initial_steering_deg:.1f}deg")
        prefix = "PARKING PATH" if self.mode in (7, 10) else "LIDAR PATH"
        self.get_logger().info(
            f"[{prefix}] {'FEASIBLE' if plan.valid else 'INFEASIBLE'} "
            f"mode={self.mode} path_point_count={plan.path_point_count} "
            f"max_steer={plan.max_steering_deg:.1f}deg "
            f"max_curvature={plan.max_curvature:.3f} "
            f"min_turn_radius_m={plan.min_turn_radius_m:.3f}")
        self.last_plan_log = signature

    def _transition_mode(self):
        if self.mode == self.previous_mode:
            return
        # Branch names may legitimately repeat across T, V and END.  The
        # duplicate-publish cache is mode-local; otherwise V:B followed by
        # Mode11 END:B would suppress the required exit command.
        self.last_branch = None
        if self.mode != 5:
            self.mode5.reset()
            self.mode5_plan = None
            self.mode5_obstacle.reset()
            self.mode5_stationary.reset()
            self.mode5_obstacles_map = ()
            self.mode5_curbs_map = ()
            self.mode5_obstacle_lidar_distance = None
        if self.mode != 9:
            self.mode9.reset()
        if self.mode == 11:
            self.mode11.reset()
            self.mode11.enter(time.monotonic())
        self.tracker.clear()
        self.tracker_key = ""
        self.last_plan_log = None
        self.plan_audit = {}
        for mode, parking in self.parking.items():
            if self.mode != mode:
                parking.reset()
                self.parking_plans[mode] = None
                self.parking_branches[mode] = ""
        self.parking_mode_entered_at = (
            time.monotonic() if self.mode in (7, 10) else None)
        self.parking_assessments = {"A": None, "B": None}
        self.parking_selected_source = ""
        self.parking_failed_reason = ""
        self.parking_slot_state = {"A": "UNKNOWN", "B": "UNKNOWN"}
        self.parking_runtime.reset(self.mode)
        self.nav2_pending = False
        self.nav2_attempted = []
        self.nav2_selected_slot = ""
        self.nav2_plan = None
        self.nav2_anchor = None
        self.nav2_failure = ""
        self.nav2_planning_state = "IDLE"
        self.nav2_validation_reason = "NOT_PLANNED"
        self.nav2_requested_at = None
        self.nav2_last_attempt_at = None
        if self.nav2_goal_handle is not None:
            self.nav2_goal_handle.cancel_goal_async()
        self.nav2_goal_handle = None
        self.nav2_generation += 1
        self.stop_waypoint_state = "IDLE"
        self.stop_waypoint_key = ""
        self.previous_mode = self.mode

    def _decision(self):
        self._transition_mode()
        if self.mode == 5:
            now = time.monotonic()
            qualified = self.mode5_obstacle.update(
                self.avoidance, self.mode, now)
            stop_request = self.avoidance or qualified
            # Keep refreshing the fixed-frame snapshot until braking has
            # completed. If the object then disappears, its last confirmed
            # position still remains available to the stopped planner.
            if self.avoidance and self.mode5_plan is None:
                self._snapshot_mode5_scene()
            if self.mode5.state == "CSV_TRACKING":
                self.mode5_stationary.reset()
                decision = self.mode5.update(
                    stop_request, self.hard, self.planner_state, False,
                    False, False, 0)
                if decision.state == "STOP_FOR_PLANNING":
                    self.tracker.clear()
                    self.tracker_key = ""
                return decision

            if self.mode5.state == "STOP_FOR_PLANNING" and \
                    self.mode5_plan is None:
                odom_fresh = (
                    self.odom_at is not None and
                    now-self.odom_at <= float(self.get_parameter(
                        "odom_timeout_s").value))
                stopped = self.mode5_stationary.update(
                    self.odom_linear, self.odom_angular, odom_fresh, now)
                distance_ready = mode5_planning_distance_ready(
                    self.mode5_obstacle_lidar_distance,
                    self.get_parameter(
                        "minimum_planning_lidar_distance_m").value)
                if (qualified and stopped and distance_ready and
                        self.map_pose is not None and self.route and
                        0 <= self.active_index < len(self.route)):
                    obstacles, obstacle_y, left, right, curbs = \
                        self._mode5_scene_at_stop()
                    self.mode5_plan = plan_route_detour(
                        self.map_pose, self.route, self.active_index,
                        obstacle_y,
                        minimum_ahead_m=float(self.get_parameter(
                            "detour_length_m").value),
                        maximum_ahead_m=float(self.get_parameter(
                            "detour_maximum_ahead_m").value),
                        wheelbase_m=float(self.get_parameter(
                            "wheelbase_m").value),
                        planner_max_steering_deg=float(self.get_parameter(
                            "planner_max_steering_deg").value),
                        obstacles=obstacles,
                        vehicle_width_m=float(self.get_parameter(
                            "vehicle_width_m").value),
                        obstacle_margin_m=float(self.get_parameter(
                            "obstacle_margin_m").value),
                        vehicle_length_m=float(self.get_parameter(
                            "vehicle_length_m").value),
                        maximum_replans=int(self.get_parameter(
                            "planner_maximum_replans").value),
                        left_boundary_m=left, right_boundary_m=right,
                        curbs=curbs)
                # Even when synchronous generation finishes on this tick,
                # keep publishing zero. Tracking can start on the next tick.
                self._record_plan(self.mode5_plan)
                if self.mode5_plan is None:
                    return self.mode5.update(
                        stop_request, self.hard, self.planner_state, False,
                        False, False, 0)
                if not self.mode5_plan.valid:
                    return self.mode5.update(
                        stop_request, self.hard, "NO_FEASIBLE_DETOUR", False,
                        False, False, 0)
                return self.mode5.update(
                    stop_request, self.hard, self.planner_state, False,
                    False, False, 0)

            plan = self.mode5_plan
            self._record_plan(plan)
            track = self._track(
                "MODE5:"+str(plan.route_index) if plan else "MODE5",
                plan, 1.0)
            path_valid = bool(track.valid)
            state = self.planner_state
            if self.avoidance and plan is not None and not plan.valid:
                state = "NO_FEASIBLE_DETOUR"
            elif (self.tracker_key and not track.valid and
                  self.mode5.state == "LIDAR_PATH_TRACKING"):
                state = "PATH_ABORT"
            decision = self.mode5.update(
                stop_request, self.hard, state, path_valid,
                track.complete or self.path_complete, self.rejoin_valid,
                track.wheel)
            if decision.state == "CSV_TRACKING":
                self.mode5_plan = None
                self.tracker.clear()
                self.tracker_key = ""
                self.mode5_obstacle.reset()
                self.mode5_stationary.reset()
                self.mode5_obstacles_map = ()
                self.mode5_curbs_map = ()
                self.mode5_obstacle_lidar_distance = None
            return decision
        if self.mode in (7, 10):
            prefix = "T" if self.mode == 7 else "V"
            now = time.monotonic()
            perception_fresh = (
                self.perception_fresh and self.perception_at is not None and
                now-self.perception_at <= 0.5)
            if not perception_fresh:
                self.parking_failed_reason = "FRONT_LIDAR_STALE"
                return ManeuverDecision(
                    prefix+"_FRONT_LIDAR_STALE", "SAFETY", True,
                    branch="A")
            slot_timeout = float(self.get_parameter(
                "parking_slot_timeout_s").value)
            explicit_b = fresh_explicit_b(
                self.b_free, self.slots_fresh, self.slots_at, now,
                slot_timeout)
            self.parking_assessments = {
                branch: self._parking_path_assessment(branch)
                for branch in ("A", "B")}
            if not self.parking_slam_enabled:
                selected = select_explicit_b_slot(explicit_b, "CSV")
                selected_slot = selected.slot
                if (bool(self.get_parameter(
                        "prehardware_case_commands").value) and
                        self.case_choices[prefix] == "B"):
                    selected_slot = "B"
                self.parking_selected_source = selected.reason
                self.parking_failed_reason = ""
                return ManeuverDecision(
                    prefix+"_CSV_FALLBACK", "CSV", False,
                    branch=selected_slot)
            wait_elapsed = (0.0 if self.parking_mode_entered_at is None else
                            now-self.parking_mode_entered_at)
            map_waiting = (not self.parking_slam_active or
                           not self.parking_map_ready)
            parking = self.parking[self.mode]
            slam_evidence = bool(
                not map_waiting and self.parking_grid_at is not None and
                now-self.parking_grid_at <= 3.0 and
                self.parking_grid_unknown_ratio <= 0.70)
            if slam_evidence:
                self.parking_assessments = {
                    branch: self._parking_path_assessment(
                        branch, self.parking_map_obstacles)
                    for branch in ("A", "B")}
                states = {
                    branch: ("UNKNOWN" if assessment is None else
                             "FREE" if assessment.safe else "OCCUPIED")
                    for branch, assessment in self.parking_assessments.items()}
            else:
                states = {"A": "UNKNOWN", "B": "UNKNOWN"}
            self.parking_slot_state = dict(states)

            # Observe B before considering a completed A plan. A may only be
            # committed after the observation window or at the reverse point.
            current_valid = bool(
                self.nav2_plan is not None and self.nav2_plan.valid)
            runtime = self.parking_runtime.update(
                mode=self.mode, explicit_b=explicit_b,
                observation_elapsed_s=wait_elapsed,
                nav2_planning=self.nav2_pending,
                nav2_slot=self.nav2_selected_slot,
                nav2_path_valid=current_valid,
                reverse_decision_point=False)
            candidate_slot = self.parking_runtime.candidate_slot
            if (not runtime.branch_locked and self.nav2_selected_slot and
                    self.nav2_selected_slot != candidate_slot):
                if self.nav2_goal_handle is not None:
                    self.nav2_goal_handle.cancel_goal_async()
                self.nav2_generation += 1
                self.nav2_pending = False
                self.nav2_requested_at = None
                self.nav2_goal_handle = None
                self.nav2_plan = None
                self.nav2_failure = "SLOT_CANDIDATE_CHANGED"
                self.nav2_validation_reason = self.nav2_failure
                self.nav2_planning_state = "IDLE"
            if (self.nav2_pending and self.nav2_requested_at is not None and
                    now-self.nav2_requested_at >= float(self.get_parameter(
                        "nav2_plan_timeout_s").value)):
                self._expire_nav2_request()
            plan_matches = bool(
                self.nav2_plan is not None and self.nav2_plan.valid and
                self.nav2_selected_slot == candidate_slot)
            retry_due = (
                self.nav2_last_attempt_at is None or
                now-self.nav2_last_attempt_at >= float(self.get_parameter(
                    "nav2_replan_interval_s").value))
            if (not runtime.branch_locked and not map_waiting and
                    not self.nav2_pending and not plan_matches and retry_due):
                self._request_nav2_plan(candidate_slot)

            reverse_decision = self._parking_reverse_decision_point()
            current_valid = bool(
                self.nav2_plan is not None and self.nav2_plan.valid and
                self.nav2_selected_slot ==
                self.parking_runtime.candidate_slot)
            runtime = self.parking_runtime.update(
                mode=self.mode, explicit_b=explicit_b,
                observation_elapsed_s=wait_elapsed,
                nav2_planning=self.nav2_pending,
                nav2_slot=self.nav2_selected_slot,
                nav2_path_valid=current_valid,
                reverse_decision_point=reverse_decision,
                reverse_started=(parking.state == "LIDAR_PATH_TRACKING"))
            branch = runtime.selected_slot or "A"
            self.parking_selected_source = runtime.selection_source

            if not runtime.branch_locked:
                self.parking_failed_reason = (
                    "MAP_NOT_READY" if map_waiting else
                    self.nav2_failure)
                # Nav2 planning and map observation are advisory during the
                # approach. CSV remains the command owner and keeps moving.
                return ManeuverDecision(
                    prefix+"_"+runtime.state, "CSV", False,
                    branch=branch)

            self.parking_branches[self.mode] = branch
            if runtime.nav2_path_ready:
                self.parking_plans[self.mode] = self.nav2_plan
                self.parking_failed_reason = ""
            else:
                if self.nav2_goal_handle is not None:
                    self.nav2_goal_handle.cancel_goal_async()
                self.nav2_generation += 1
                self.nav2_pending = False
                self.nav2_requested_at = None
                self.nav2_goal_handle = None
                self.parking_failed_reason = (
                    self.nav2_failure or self.nav2_validation_reason or
                    "NAV2_NOT_READY_AT_REVERSE_DECISION")

            # branch_locked is the freeze request consumed by the lifecycle
            # manager. Keep the car stopped until slam_toolbox has relinquished
            # map->odom; cached map/slot/path data remains immutable.
            if self.parking_slam_active:
                owner = "LIDAR" if runtime.nav2_path_ready else "CSV"
                return ManeuverDecision(
                    prefix+"_SLAM_FREEZE", owner, True, branch=branch)

            if runtime.csv_fallback:
                # The route follower performs the existing 3 s longitudinal
                # direction-change hold at T_A/T_B or V_A/V_B.
                parking_csv_fallback_segment(self.mode, branch)
                return ManeuverDecision(
                    prefix+"_CSV_FALLBACK", "CSV", False, branch=branch)

            plan = self.parking_plans[self.mode]
            self._record_plan(plan)
            tracking_active = parking.state == "LIDAR_PATH_TRACKING"
            track = (self._track(prefix+branch, plan, -1.0)
                     if tracking_active else
                     TrackCommand(False, False, 0.0, 0, 0))
            state = self.planner_state
            if plan is not None and not plan.valid:
                state = "PLANNER_GIVE_UP"
            elif (self.tracker_key and not track.valid and
                  parking.state == "LIDAR_PATH_TRACKING"):
                state = "PATH_ABORT"
            path_available = bool(plan and plan.valid)
            # A close object in front is behind the direction of travel while
            # the vehicle reverses into a parking slot. Rear evidence, slot
            # freshness and every non-parking safety gate remain active.
            front_hard = self.hard and not parking_reverse_phase(
                parking.state, path_available)
            return parking.update(
                branch, path_available,
                track.complete or self.path_complete,
                state,
                front_hard or self.rear_hard,
                track.drive, track.wheel, self.rejoin_valid,
                time.monotonic())
        if self.mode == 9:
            return self.mode9.update(self.hard)
        if self.mode == 11:
            return self.mode11.evaluate(time.monotonic())
        return None

    def _tick(self):
        decision = self._decision()
        planned_rejoin_index = (
            self.mode5_plan.route_index
            if self.mode == 5 and self.mode5_plan is not None and
            self.mode5_plan.valid else -1)
        self.rejoin_target_pub.publish(Int32(
            data=int(planned_rejoin_index)))
        if decision is None:
            drive, wheel, valid, hold, state, branch = 0.0, 0, False, False, "CSV_ONLY", ""
        else:
            drive, wheel = decision.drive, int(decision.wheel)
            if abs(wheel) > 22:
                drive, wheel = 0.0, 0
                decision = None
                state = "INVALID_STEERING_COMMAND"
            # Valid denotes exclusive candidate ownership, including a
            # zero-speed planning/rejoin hold.  The separate hold topic makes
            # the arbiter stop, while the follower remains suspended until
            # LiDAR explicitly hands control back to CSV.
            if decision is None:
                valid, hold, branch = False, True, ""
            else:
                valid = decision.owner == "LIDAR"
                hold, state, branch = (
                    decision.stop, decision.state, decision.branch)
        self.drive_pub.publish(Float32(data=float(drive)))
        self.wheel_pub.publish(Int32(data=int(wheel)))
        self.valid_pub.publish(Bool(data=valid))
        self.hold_pub.publish(Bool(data=hold))
        if branch and branch != self.last_branch:
            command = branch
            if bool(self.get_parameter(
                    "prehardware_case_commands").value):
                choice = ("T" if self.mode == 7 else
                          "V" if self.mode == 10 else "END")
                command = f"{choice}:{branch}"
                # The isolated four-choice selector already defaults A and
                # accepts only an in-window B override.
                if branch == "A":
                    command = ""
            if command:
                self.branch_pub.publish(String(data=command))
            self.last_branch = branch
        if state != self.last_state:
            prefix = ("[MODE9] " if self.mode == 9 else
                      "[PARKING] " if self.mode in (7, 10) else
                      "[MODE11] " if self.mode == 11 else "[LIDAR] ")
            self.get_logger().info(prefix+state)
            event = {"mode": self.mode, "state": state, "branch": branch}
            if self.mode == 5 and self.last_state == "CSV_REJOIN" and \
                    state == "CSV_TRACKING":
                event["event"] = "AVOIDANCE_REJOINED"
            elif self.mode in (7, 10) and state in ("T_COMPLETE", "V_COMPLETE"):
                event["event"] = "PARKING_CSV_REJOINED"
                event["source"] = "LIDAR"
            elif self.mode in (7, 10) and state.endswith("CSV_FALLBACK"):
                event["event"] = "PARKING_CSV_FALLBACK"
                event["source"] = "CSV_FALLBACK"
            elif self.mode == 9 and state == "ACCEL_TRACKING" and \
                    self.last_state == "MODE9_OBSTACLE_WAIT":
                event["event"] = "MODE9_ACCEL_RESUME"
            elif self.mode == 11 and state.startswith("MODE11_COMMIT_"):
                event["event"] = "MODE11_BRANCH_COMMITTED"
                event["source"] = self.mode11.commit_source
            if "event" in event:
                self.event_pub.publish(String(data=json.dumps(
                    event, separators=(",", ":"))))
            self.last_state = state
        self.diag_pub.publish(String(data=json.dumps({
            "mode": self.mode, "state": state, "drive": drive,
            "wheel": wheel, "valid": valid, "hold": hold,
            "branch": branch,
            "obstacle_visible": self.avoidance,
            "obstacle_confirmed_2s": self.mode5_obstacle.latched,
            "planning_lidar_distance_m": self.mode5_obstacle_lidar_distance,
            "planning_distance_ready": mode5_planning_distance_ready(
                self.mode5_obstacle_lidar_distance,
                self.get_parameter(
                    "minimum_planning_lidar_distance_m").value),
            "obstacle_seen_elapsed_s": (
                0.0 if self.mode5_obstacle.first_seen_at is None else
                max(0.0, time.monotonic()-
                    self.mode5_obstacle.first_seen_at)),
            "vehicle_stationary_confirmed": (
                self.mode5_stationary.since is not None and
                time.monotonic()-self.mode5_stationary.since >=
                self.mode5_stationary.duration_s),
            "path_audit": self.plan_audit,
            "avoidance_path_locked": self.mode5.path_locked,
            "avoidance_path_lock_state": (
                "AVOIDANCE_PATH_LOCKED" if self.mode5.path_locked else
                "UNLOCKED"),
        }, separators=(",", ":"))))
        if self.mode in (7, 10) or self.parking_slam_enabled:
            assessments = self.parking_assessments
            parking = self.parking.get(self.mode)
            reverse = bool(parking and drive < 0.0 and
                           parking.state == "LIDAR_PATH_TRACKING")
            selected_slot = (
                self.parking_branches.get(self.mode, "") or
                self.parking_runtime.selected_slot or
                self.nav2_selected_slot)
            observation_state = parking_slot_observation_state(
                self.b_free, self.slots_fresh, self.slots_at,
                time.monotonic(), self.get_parameter(
                    "parking_slot_timeout_s").value)
            self.parking_diag_pub.publish(String(data=json.dumps({
                "mode": self.mode,
                "parking_slam_enabled": self.parking_slam_enabled,
                "parking_slam_state": (
                    self.parking_runtime.state if self.parking_slam_enabled
                    else "DISABLED"),
                "slam_active": self.parking_slam_active,
                "slam_stop_reason": (
                    "BRANCH_LOCKED_BEFORE_REVERSE" if
                    self.parking_runtime.branch_locked else ""),
                "parking_map_ready": self.parking_map_ready,
                "parking_slot_state": self.parking_slot_state,
                "slot_observation_state": observation_state,
                "explicit_b_received": (
                    self.parking_runtime.explicit_b_received),
                "selected_slot": selected_slot,
                "parking_selected_slot": selected_slot,
                "selection_source": self.parking_selected_source,
                "parking_selected_source": self.parking_selected_source,
                "path_a_safe": (None if assessments["A"] is None else
                                assessments["A"].safe),
                "path_b_safe": (None if assessments["B"] is None else
                                assessments["B"].safe),
                "path_a_clearance": (
                    None if assessments["A"] is None else
                    assessments["A"].minimum_clearance_m),
                "path_b_clearance": (
                    None if assessments["B"] is None else
                    assessments["B"].minimum_clearance_m),
                "nav2_planning": self.nav2_pending,
                "nav2_path_ready": self.parking_runtime.nav2_committed,
                "nav2_plan_valid": bool(
                    self.nav2_plan is not None and self.nav2_plan.valid),
                "nav2_planning_state": self.nav2_planning_state,
                "nav2_failure_reason": (
                    self.parking_failed_reason or self.nav2_failure),
                "nav2_plan_failed_reason": self.parking_failed_reason,
                "nav2_attempted_slots": self.nav2_attempted,
                "reverse_decision_point_reached": (
                    self.parking_runtime.reverse_decision_point_reached),
                "slot_latched": self.parking_runtime.branch_locked,
                "branch_locked": self.parking_runtime.branch_locked,
                "owner": self.path_owner,
                "parking_owner": self.path_owner,
                "owner_handoff_active": self.owner_handoff_active,
                "owner_handoff_elapsed_s": self.owner_handoff_elapsed_s,
                "csv_fallback": self.parking_runtime.csv_fallback,
                "fallback_branch": (
                    selected_slot if self.parking_runtime.csv_fallback else ""),
                "drive": drive,
                "wheel": wheel,
                "parking_reverse": reverse,
                "front_hard_raw": self.hard,
                "front_hard_ignored_reverse": bool(self.hard and reverse),
                "rear_collision_protection": False,
                "parking_rejoin_state": (
                    parking.state if parking is not None else "INACTIVE"),
            }, separators=(",", ":"), sort_keys=True)))


def main(args=None):
    rclpy.init(args=args)
    node = ManeuverManagerNode()
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
