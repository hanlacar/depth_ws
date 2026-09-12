"""Observe and grade one live full-network CSV-only closed-loop run."""

import json
import math
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .csv_only_branching import load_csv_only_route_case
from .ros_helpers import safe_shutdown
from .stop_editor_network import CHOICE_ORDER, CHOICE_SEGMENTS


class CsvOnlyClosedLoopProbe(Node):
    def __init__(self):
        super().__init__("depth_csv_only_closed_loop_probe")
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        self.declare_parameter("expected_case", "AAAA")
        self.declare_parameter("timeout_s", 300.0)
        self.expected_case = str(
            self.get_parameter("expected_case").value).strip().upper()
        if len(self.expected_case) != 4 or any(
                value not in "AB" for value in self.expected_case):
            raise ValueError("expected_case must be exactly four A/B letters")
        self.route = load_csv_only_route_case(
            str(self.get_parameter("route_path").value),
            str(self.get_parameter("route_metadata_path").value),
            self.expected_case)
        self.expected_stops = sum(
            point.event == "STOP_LINE" for point in self.route)
        self.expected_direction_changes = sum(
            self.route[index].direction != self.route[index+1].direction
            for index in range(len(self.route)-1))
        self.expected_segments = {
            CHOICE_SEGMENTS[choice][branch]
            for choice, branch in zip(CHOICE_ORDER, self.expected_case)}
        self.opposite_segments = {
            CHOICE_SEGMENTS[choice]["B" if branch == "A" else "A"]
            for choice, branch in zip(CHOICE_ORDER, self.expected_case)}
        # T intentionally keeps its established late-decision contract: the
        # vehicle enters the short T_A pending approach before a stop-time B
        # commit remaps onto T_B.  Mode 10 no longer has that exception;
        # seeing V_A in a V:B run remains a hard failure.
        self.permitted_pending_segments = (
            {"T_A"} if self.expected_case[1] == "B" else set())
        self.requested = {
            choice: branch for choice, branch in
            zip(CHOICE_ORDER[1:], self.expected_case[1:]) if branch == "B"}
        self.selected = {choice: "pending" for choice in CHOICE_ORDER}
        self.parking_lifecycle = {"T": "INACTIVE", "V": "INACTIVE"}
        self.lifecycle_history = {"T": [], "V": []}
        self.case_events = set()
        self.active_cases = set()
        self.started = time.monotonic()
        self.done = False
        self.failure = ""
        self.report = {}
        self.first_pose = None
        self.last_pose = None
        self.first_encoder = None
        self.last_encoder = None
        self.drive_values = set()
        self.wheel_min = 0
        self.wheel_max = 0
        self.segments = set()
        self.controller_states = set()
        self.has_moved = False
        self.stop_active = False
        self.stop_started = None
        self.stop_after_motion = False
        self.stop_holds = []
        self.last_motion_sign = 0
        self.direction_holds = []

        self.command_publisher = self.create_publisher(
            String, "/depth_slam/route/branch_command", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Int32, "/mcu/encoder", self.on_encoder, 10)
        self.create_subscription(
            Float32, "/depth_slam/follower/candidate_drive", self.on_drive, 10)
        self.create_subscription(
            Int32, "/depth_slam/follower/candidate_wheel", self.on_wheel, 10)
        self.create_subscription(
            Bool, "/depth_slam/follower/candidate_stop", self.on_stop, 10)
        self.create_subscription(
            String, "/depth_slam/route/active_segment",
            lambda message: self.segments.add(str(message.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/active_case",
            lambda message: self.active_cases.add(str(message.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/case_state", self.on_case_state, 10)
        self.create_subscription(
            String, "/depth_slam/route/controller_state", self.on_state, 10)
        self.create_timer(0.2, self.check)

    def on_odom(self, message):
        pose = message.pose.pose.position
        xy = (float(pose.x), float(pose.y))
        if self.first_pose is None:
            self.first_pose = xy
        self.last_pose = xy

    def on_encoder(self, message):
        value = int(message.data)
        if self.first_encoder is None:
            self.first_encoder = value
        self.last_encoder = value

    def current_stop_duration(self):
        if not self.stop_active or self.stop_started is None:
            return 0.0
        return time.monotonic()-self.stop_started

    def on_drive(self, message):
        drive = int(round(float(message.data)))
        self.drive_values.add(drive)
        sign = 1 if drive > 0 else -1 if drive < 0 else 0
        if sign:
            if self.last_motion_sign and sign != self.last_motion_sign:
                hold = (self.current_stop_duration() if self.stop_active else
                        (self.stop_holds[-1] if self.stop_holds else 0.0))
                self.direction_holds.append(hold)
            self.last_motion_sign = sign
            self.has_moved = True

    def on_wheel(self, message):
        value = int(message.data)
        self.wheel_min = min(self.wheel_min, value)
        self.wheel_max = max(self.wheel_max, value)

    def on_stop(self, message):
        now = time.monotonic()
        value = bool(message.data)
        if value and not self.stop_active:
            self.stop_active = True
            self.stop_started = now
            self.stop_after_motion = self.has_moved
        elif not value and self.stop_active:
            if self.stop_after_motion and self.stop_started is not None:
                self.stop_holds.append(now-self.stop_started)
            self.stop_active = False
            self.stop_started = None
            self.stop_after_motion = False

    def on_case_state(self, message):
        try:
            state = json.loads(message.data)
            self.selected.update(state.get("selected", {}))
            self.parking_lifecycle.update(
                state.get("parking_lifecycle", {}))
            for choice in ("T", "V"):
                lifecycle = self.parking_lifecycle[choice]
                history = self.lifecycle_history[choice]
                if not history or history[-1] != lifecycle:
                    history.append(lifecycle)
            event = state.get("last_event", "")
            if event:
                self.case_events.add(event)
        except (TypeError, ValueError):
            self.failure = "invalid case_state JSON"
            self.done = True

    def on_state(self, message):
        state = str(message.data)
        self.controller_states.add(state)
        if state == "ROUTE_COMPLETE":
            self.grade()

    def publish_requests(self):
        for choice, branch in self.requested.items():
            in_window = (choice == "END" or
                         self.parking_lifecycle.get(choice) in (
                             "PENDING", "B_REQUESTED"))
            if self.selected.get(choice) == "pending" and in_window:
                self.command_publisher.publish(String(data=f"{choice}:{branch}"))

    def grade(self):
        if self.done:
            return
        if self.stop_active and self.stop_after_motion:
            self.stop_holds.append(self.current_stop_duration())
        odom_change = (0.0 if self.first_pose is None or self.last_pose is None else
                       math.hypot(self.last_pose[0]-self.first_pose[0],
                                  self.last_pose[1]-self.first_pose[1]))
        encoder_change = (0 if self.first_encoder is None or self.last_encoder is None
                          else self.last_encoder-self.first_encoder)
        waypoint_holds = [value for value in self.stop_holds if value >= 1.0]
        final_selected = "".join(self.selected[choice] for choice in CHOICE_ORDER)
        pending_hits = self.permitted_pending_segments & self.segments
        wrong_hits = ((self.opposite_segments & self.segments) -
                      self.permitted_pending_segments)
        report = {
            "result": "PASS", "expected_case": self.expected_case,
            "selected_case": final_selected,
            "selection_state": dict(self.selected),
            "parking_lifecycle_history": self.lifecycle_history,
            "case_events": sorted(self.case_events),
            "active_cases": sorted(self.active_cases),
            "segments_hit": sorted(self.segments),
            "selected_exclusive_hit": sorted(
                self.expected_segments & self.segments),
            "opposite_exclusive_hit": sorted(wrong_hits),
            "permitted_pending_hit": sorted(pending_hits),
            "drive_values": sorted(self.drive_values),
            "wheel_min": self.wheel_min, "wheel_max": self.wheel_max,
            "odom_net_change_m": odom_change,
            "encoder_change": encoder_change,
            "stop_count": len(waypoint_holds),
            "expected_stop_count": self.expected_stops,
            "stop_holds_s": waypoint_holds,
            "minimum_stop_s": min(waypoint_holds) if waypoint_holds else 0.0,
            "separator_stop_count": len(self.stop_holds)-len(waypoint_holds),
            "direction_transition_holds_s": self.direction_holds,
            "controller_state": "ROUTE_COMPLETE",
        }
        checks = (
            final_selected == self.expected_case,
            self.expected_case in self.active_cases,
            self.expected_segments <= self.segments,
            not wrong_hits,
            odom_change > 1.0,
            encoder_change > 100,
            any(value > 0 for value in self.drive_values),
            any(value < 0 for value in self.drive_values),
            self.wheel_min < 0 < self.wheel_max,
            len(waypoint_holds) == self.expected_stops,
            bool(waypoint_holds) and min(waypoint_holds) >= 2.90,
            len(self.direction_holds) == self.expected_direction_changes,
            all(value >= 2.90 for value in self.direction_holds),
        )
        if not all(checks):
            report["result"] = "FAIL"
            self.failure = "CSV-only closed-loop branch contract failed"
        self.report = report
        self.done = True

    def check(self):
        if self.done:
            return
        self.publish_requests()
        if time.monotonic()-self.started >= float(
                self.get_parameter("timeout_s").value):
            self.failure = "CSV-only closed-loop timed out before ROUTE_COMPLETE"
            self.report = {
                "result": "FAIL", "reason": self.failure,
                "expected_case": self.expected_case,
                "selection_state": dict(self.selected),
                "active_cases": sorted(self.active_cases),
                "segments_hit": sorted(self.segments),
                "drive_values": sorted(self.drive_values),
                "encoder": self.last_encoder,
                "states": sorted(self.controller_states),
            }
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = CsvOnlyClosedLoopProbe()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        print(json.dumps(node.report, indent=2, sort_keys=True))
        failure = node.failure
    finally:
        try:
            node.destroy_node()
            safe_shutdown()
        except KeyboardInterrupt:
            pass
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
