"""ROS wiring for LiDAR temporary paths, parking and mode-specific gates."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .lidar_local_planner import plan_parking, plan_route_detour
from .lidar_mission_core import (
    ManeuverDecision, Mode11ExitGate, Mode5Avoidance, Mode5ObstacleLatch,
    Mode9Emergency, ParkingManeuver, StationaryConfirmation,
    mode5_planning_distance_ready)
from .lidar_path_tracker import LocalPathTracker, TrackCommand
from .route_io import load_segmented_route


class ManeuverManagerNode(Node):
    def __init__(self):
        super().__init__("depth_maneuver_manager")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("prehardware_case_commands", False)
        # False selects the commissioned front-LiDAR slot evidence and the
        # recorded CSV parking path. It must still wait for an actual A/B slot
        # result; it never silently defaults A without LiDAR evidence.
        self.declare_parameter("rear_lidar_enabled", True)
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        for name, default in (
                ("wheelbase_m", 0.73),
                ("planner_max_steering_deg", 20.0),
                ("vehicle_width_m", 0.80),
                ("obstacle_margin_m", 0.15),
                ("detour_length_m", 1.5),
                ("detour_maximum_ahead_m", 8.0),
                ("detour_lateral_m", 0.65),
                ("planner_maximum_replans", 48),
                ("obstacle_confirmation_s", 2.0),
                ("minimum_planning_lidar_distance_m", 1.0),
                ("stopped_confirmation_s", 0.30),
                ("stopped_linear_speed_mps", 0.03),
                ("stopped_angular_speed_rps", 0.03),
                ("odom_timeout_s", 0.50)):
            self.declare_parameter(name, default)
        self.mode = -1
        self.avoidance = self.hard = self.rear_hard = False
        self.a_free = self.b_free = False
        self.slots_fresh = False
        self.planner_state = "IDLE"
        self.path_valid = self.path_complete = self.rejoin_valid = False
        self.steering = 0.0
        self.obstacle_y = 0.0
        self.obstacles = ()
        self.obstacle_lidar_distance = None
        self.curbs = ()
        self.left_curb = self.right_curb = None
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
            String, "/depth_slam/route/selected_branch",
            self._selected_branch, 10)
        self.create_subscription(
            Float32, "/depth_slam/follower/candidate_drive",
            lambda m: setattr(self, "csv_drive", float(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/case_state", self._case_state, 10)
        self.create_subscription(Float32, "/mcu/steer_deg",
                                 lambda m: setattr(self, "steering", float(m.data)), 10)
        self.create_subscription(String, "/camera/exit_branch_signal",
                                 self._exit_signal, 10)
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
        except (TypeError, json.JSONDecodeError):
            self.a_free = self.b_free = False
            self.slots_fresh = False

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
            self.left_curb = min((y for _, y in self.curbs if y > 0.0),
                                 default=None)
            self.right_curb = max((y for _, y in self.curbs if y < 0.0),
                                  default=None)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.obstacle_y = 0.0
            self.obstacle_lidar_distance = None
            self.obstacles = ()
            self.curbs = ()
            self.left_curb = self.right_curb = None

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
            return
        self.route = load_segmented_route(
            path, metadata, branch=self.active_branch).points

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
        return obstacles, obstacle_y, left, right

    def _exit_signal(self, message):
        if self.mode == 11:
            self.mode11.observe(message.data, time.monotonic())

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
                    obstacles, obstacle_y, left, right = \
                        self._mode5_scene_at_stop()
                    self.mode5_plan = plan_route_detour(
                        self.map_pose, self.route, self.active_index,
                        obstacle_y,
                        minimum_ahead_m=float(self.get_parameter(
                            "detour_length_m").value),
                        maximum_ahead_m=float(self.get_parameter(
                            "detour_maximum_ahead_m").value),
                        lateral_m=float(self.get_parameter(
                            "detour_lateral_m").value),
                        wheelbase_m=float(self.get_parameter(
                            "wheelbase_m").value),
                        planner_max_steering_deg=float(self.get_parameter(
                            "planner_max_steering_deg").value),
                        obstacles=obstacles,
                        vehicle_width_m=float(self.get_parameter(
                            "vehicle_width_m").value),
                        obstacle_margin_m=float(self.get_parameter(
                            "obstacle_margin_m").value),
                        maximum_replans=int(self.get_parameter(
                            "planner_maximum_replans").value),
                        left_boundary_m=left, right_boundary_m=right)
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
            if not bool(self.get_parameter("rear_lidar_enabled").value):
                if not self.slots_fresh or not (self.a_free or self.b_free):
                    return ManeuverDecision(
                        prefix+"_WAIT_LIDAR_SLOT", "LIDAR", True)
                branch = "A" if self.a_free else "B"
                return ManeuverDecision(
                    prefix+"_CSV_FALLBACK", "CSV", False,
                    branch=branch)
            parking = self.parking[self.mode]
            branch = (self.parking_branches[self.mode] or
                      ("A" if self.a_free or not self.b_free else "B"))
            # T_A/T_B begin with a five-point forward approach and their local
            # endpoint rejoins the reverse block, so wait for CSV reverse
            # lineage. V_A/V_B intentionally start and rejoin on forward
            # lineage around the existing p105 commit while the temporary
            # maneuver itself reverses under exclusive LiDAR ownership.
            expected_csv_direction = -1.0 if self.mode == 7 else 1.0
            eligible = (parking.state == "CSV_APPROACH" and
                        self.active_segment == prefix+"_"+branch and
                        self.csv_drive*expected_csv_direction > 0.0)
            if (eligible and bool(self.get_parameter(
                    "prehardware_case_commands").value)):
                # The isolated test selector deliberately keeps the A route
                # visible while its late T/V choice is still pending.  Do not
                # let the temporary LiDAR path pre-empt the CSV point at which
                # that choice is committed.
                eligible = eligible and self.case_choices[prefix] == branch
            if eligible and self.parking_plans[self.mode] is None:
                self.parking_branches[self.mode] = branch
                self.parking_plans[self.mode] = plan_parking(
                    self.mode, branch,
                    wheelbase_m=float(self.get_parameter("wheelbase_m").value),
                    planner_max_steering_deg=float(self.get_parameter(
                        "planner_max_steering_deg").value))
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
            return parking.update(
                branch, bool(plan and plan.valid) or self.path_valid,
                track.complete or self.path_complete,
                state,
                self.hard or self.rear_hard or not self.slots_fresh,
                track.drive, track.wheel, self.rejoin_valid,
                time.monotonic())
        if self.mode == 9:
            return self.mode9.update(self.hard, self.steering,
                                     self.rejoin_valid, True)
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
                    self.last_state == "EMERGENCY_STOP":
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
        }, separators=(",", ":"))))


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
