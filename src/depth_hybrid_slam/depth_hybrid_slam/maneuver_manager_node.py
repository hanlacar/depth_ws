"""ROS wiring for LiDAR temporary paths, parking and mode-specific gates."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .lidar_local_planner import plan_detour, plan_parking
from .lidar_mission_core import (
    ManeuverDecision, Mode11ExitGate, Mode5Avoidance, Mode9Emergency,
    ParkingManeuver)
from .lidar_path_tracker import LocalPathTracker, TrackCommand


class ManeuverManagerNode(Node):
    def __init__(self):
        super().__init__("depth_maneuver_manager")
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("prehardware_case_commands", False)
        # Production keeps rear safety enabled.  A front-only commissioning
        # run must opt out explicitly so missing rear parking-slot evidence
        # falls back to the recorded CSV maneuver instead of hard-stopping.
        self.declare_parameter("rear_lidar_enabled", True)
        for name, default in (
                ("wheelbase_m", 0.73),
                ("planner_max_steering_deg", 21.0),
                ("vehicle_width_m", 0.80),
                ("obstacle_margin_m", 0.15),
                ("detour_length_m", 5.5),
                ("detour_lateral_m", 0.65),
                ("planner_maximum_replans", 24)):
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
        self.left_curb = self.right_curb = None
        self.pose = None
        self.active_segment = ""
        self.csv_drive = 0.0
        self.case_choices = {"T": "pending", "V": "pending"}
        self.last_state = None
        self.last_branch = None
        self.last_plan_log = None
        self.plan_audit = {}
        self.mode5 = Mode5Avoidance()
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
            String, "/depth_slam/route/active_segment",
            lambda m: setattr(self, "active_segment", str(m.data)), 10)
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
            self.obstacles = tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("obstacles", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2)
            curbs = tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("curbs", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2)
            self.left_curb = min((y for _, y in curbs if y > 0.0),
                                 default=None)
            self.right_curb = max((y for _, y in curbs if y < 0.0),
                                  default=None)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.obstacle_y = 0.0
            self.obstacles = ()
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

    def _exit_signal(self, message):
        if self.mode == 11:
            self.mode11.observe(message.data, time.monotonic())

    def _publish_plan(self, plan):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        for x, y, yaw in plan.points:
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
            plan = (plan_detour(
                self.obstacle_y,
                length_m=float(self.get_parameter("detour_length_m").value),
                lateral_m=float(self.get_parameter("detour_lateral_m").value),
                wheelbase_m=float(self.get_parameter("wheelbase_m").value),
                planner_max_steering_deg=float(self.get_parameter(
                    "planner_max_steering_deg").value),
                obstacles=self.obstacles,
                vehicle_width_m=float(self.get_parameter(
                    "vehicle_width_m").value),
                obstacle_margin_m=float(self.get_parameter(
                    "obstacle_margin_m").value),
                maximum_replans=int(self.get_parameter(
                    "planner_maximum_replans").value),
                left_boundary_m=self.left_curb,
                right_boundary_m=self.right_curb)
                    if self.avoidance else None)
            self._record_plan(plan)
            track = self._track("MODE5", plan, 1.0)
            path_valid = bool(track.valid)
            state = self.planner_state
            if self.avoidance and plan is not None and not plan.valid:
                state = "NO_FEASIBLE_DETOUR"
            elif (self.tracker_key and not track.valid and
                  self.mode5.state == "LIDAR_PATH_TRACKING"):
                state = "PATH_ABORT"
            return self.mode5.update(
                self.avoidance, self.hard, state, path_valid,
                track.complete or self.path_complete, self.rejoin_valid,
                track.wheel)
        if self.mode in (7, 10):
            prefix = "T" if self.mode == 7 else "V"
            if not bool(self.get_parameter("rear_lidar_enabled").value):
                return ManeuverDecision(
                    prefix+"_CSV_FALLBACK", "CSV", False,
                    branch=(self.case_choices[prefix]
                            if self.case_choices[prefix] in ("A", "B")
                            else "A"))
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
                    self.last_state == "CSV_REJOIN":
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
