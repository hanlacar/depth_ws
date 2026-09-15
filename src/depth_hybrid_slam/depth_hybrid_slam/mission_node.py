"""Adapter from verified camera_ws topics to the depth_slam mission contract."""

import json
import math
import time
from collections import deque

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from .intersection_route import progress_from_json
from .mission_completion import MissionCompletionTracker
from .mission_core import MissionMachine
from .models import MissionDecision, MissionInputs
from .ros_helpers import safe_shutdown, status
from .segment_result import SegmentResultLatch


class MissionNode(Node):
    def __init__(self):
        super().__init__("depth_mission_manager")
        for name, default in (("ramp_sections", [""]),
                              ("acceleration_sections", [""])):
            self.declare_parameter(name, default)
        for name, default in (("traffic_timeout_s", 0.3),
                              ("min_traffic_confidence", 0.65),
                              ("direct_signal_timeout_s", 0.5),
                              ("unknown_hold_s", 3.0),
                              ("minimum_intersection_stop_s", 3.0),
                              ("intersection_commit_margin_m", 0.35),
                              ("imu_timeout_s", 0.5),
                              ("imu_average_window_s", 0.5),
                              ("slope_threshold_deg", 4.5)):
            self.declare_parameter(name, default)
        self.declare_parameter("intersection_modes", [4, 6, 8])

        def values(name):
            return [str(v) for v in self.get_parameter(name).value if str(v)]
        self.core = MissionMachine(values("ramp_sections"),
                                   values("acceleration_sections"),
                                   traffic_timeout_s=float(self.get_parameter(
                                       "traffic_timeout_s").value),
                                   min_traffic_confidence=float(
                                       self.get_parameter(
                                           "min_traffic_confidence").value),
                                   unknown_hold_s=float(self.get_parameter(
                                       "unknown_hold_s").value),
                                   minimum_intersection_stop_s=float(
                                       self.get_parameter(
                                           "minimum_intersection_stop_s").value),
                                   intersection_commit_margin_m=float(
                                       self.get_parameter(
                                           "intersection_commit_margin_m").value),
                                   intersection_modes=tuple(
                                       self.get_parameter(
                                           "intersection_modes").value))
        self.value = MissionInputs(now=time.monotonic())
        self.camera_stop_line = False
        self.csv_stop_line_active = False
        self.traffic_received = None
        self.traffic_diagnostics_received = None
        self.direct_signals = {
            "FUSED": ("UNKNOWN", None),
            "YOLO": ("UNKNOWN", None),
            "RGB": ("UNKNOWN", None),
        }
        self.last_logged_state = None
        self.last_logged_event = ""
        self.completion = MissionCompletionTracker(
            slope_threshold_deg=self.get_parameter(
                "slope_threshold_deg").value)
        self.segment_results = SegmentResultLatch()
        self.segment3 = {
            "camera_valid": False, "camera_violation": False,
            "lidar_hard": False}
        self.segment9 = {"nominal_3": False, "violation": ""}
        self.latest_lidar_distance = None
        self.final_drive = 0.0
        self.imu_pitch = 0.0
        self.imu_valid = False
        self.imu_pitch_at = self.imu_valid_at = None
        self.imu_samples = deque()
        self.entered_segments = set()
        self.maneuver_state = ""
        self.create_subscription(String, "/camera/traffic_light_fused/state", self.on_traffic, 10)
        self.create_subscription(
            String, "/camera_traffic_light",
            lambda message: self.on_direct_signal("YOLO", message.data), 10)
        self.create_subscription(
            String, "/camera/traffic_light_rgb/state",
            lambda message: self.on_direct_signal("RGB", message.data), 10)
        self.create_subscription(String, "/camera/traffic_light_fused/aspect",
                                 lambda m: setattr(self.value, "traffic_aspect", m.data.upper()), 10)
        self.create_subscription(Float32, "/camera/traffic_light_fused/confidence",
                                 lambda m: setattr(self.value, "traffic_confidence", float(m.data)), 10)
        self.create_subscription(
            String, "/camera/traffic_light_fused/diagnostics",
            self.on_traffic_diagnostics, 10)
        self.create_subscription(Bool, "/camera/mission/stop_line_detected",
                                 lambda m: setattr(self, "camera_stop_line", bool(m.data)), 10)
        self.create_subscription(Float32, "/camera/mission/stop_line_distance_m",
                                 lambda m: setattr(self.value, "stop_line_distance_m", float(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/stop_waypoint_state",
            self.on_csv_stop_state, 10)
        self.create_subscription(
            String, "/depth_slam/route/intersection_progress",
            lambda m: setattr(
                self.value, "intersection_progress",
                progress_from_json(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/sign_detected",
                                 lambda m: setattr(self.value, "sign_detected", bool(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/uphill_detected",
                                 lambda m: setattr(self.value, "uphill_detected", bool(m.data)), 10)
        self.create_subscription(String, "/camera/mission/diagnostics",
                                 self.on_camera_diagnostics, 10)
        self.create_subscription(Float32, "/mcu/steer_deg",
                                 lambda m: setattr(self.value, "steering_deg", float(m.data)), 10)
        self.create_subscription(String, "/camera/mission/section",
                                 lambda m: setattr(self.value, "section_id", str(m.data)), 10)
        self.create_subscription(String, "/drive_mode", self.on_mode, 10)
        self.create_subscription(Float32, "/cmd_drive",
                                 self.on_final_drive, 10)
        self.create_subscription(Float32, "/imu/pitch_deg",
                                 self.on_imu_pitch, 10)
        self.create_subscription(Bool, "/imu/valid",
                                 self.on_imu_valid, 10)
        self.create_subscription(String, "/depth_slam/route/mode_status",
                                 lambda m: self.completion.observe_route_status(m.data), 10)
        self.create_subscription(String, "/depth_slam/lidar/safety_event",
                                 self.on_lidar_safety, 10)
        self.create_subscription(String, "/depth_slam/camera/csv_validation",
                                 self.on_csv_validation, 10)
        self.create_subscription(String, "/depth_slam/lidar/maneuver_event",
                                 lambda m: self.completion.observe_maneuver(m.data), 10)
        self.create_subscription(String, "/depth_slam/lidar/maneuver",
                                 self.on_maneuver, 10)
        self.create_subscription(String, "/depth_slam/route/case_state",
                                 self.on_case_state, 10)
        self.pub_state = self.create_publisher(String, "/depth_slam/mission/state", 10)
        self.pub_section = self.create_publisher(String, "/depth_slam/mission/section", 10)
        self.pub_stop = self.create_publisher(Bool, "/depth_slam/mission/stop_required", 10)
        self.pub_reason = self.create_publisher(String, "/depth_slam/mission/stop_reason", 10)
        self.pub_speed = self.create_publisher(Float32, "/depth_slam/mission/speed_limit", 10)
        self.pub_permission = self.create_publisher(String, "/depth_slam/mission/traffic_permission", 10)
        self.pub_traffic_stop_allowed = self.create_publisher(
            Bool, "/depth_slam/mission/traffic_stop_allowed", 10)
        self.pub_intersection_committed = self.create_publisher(
            Bool, "/depth_slam/mission/intersection_committed", 10)
        self.pub_traffic_gate_state = self.create_publisher(
            String, "/depth_slam/mission/traffic_gate_state", 10)
        self.pub_traffic_gate_reason = self.create_publisher(
            String, "/depth_slam/mission/traffic_gate_reason", 10)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, "/depth_slam/mission/diagnostics", 10)
        self.pub_event = self.create_publisher(
            String, "/depth_slam/mission/event", 10)
        self.pub_mode_status = self.create_publisher(
            String, "/depth_slam/mission/mode_status", 10)
        self.create_timer(1.0/30.0, self.tick)

    def _report_segment(self, mode, passed, reason):
        line = self.segment_results.report(mode, passed, reason)
        if line is not None:
            self.get_logger().info(line)

    def on_mode(self, message):
        try:
            mode = int(str(message.data).strip())
        except ValueError:
            return
        previous = self.completion.mode
        if previous == 3 and mode != 3:
            passed = (self.segment3["camera_valid"] and
                      not self.segment3["camera_violation"] and
                      not self.segment3["lidar_hard"])
            if passed:
                reason = "CSV inside road, no lane overlap, LiDAR >0.5m"
            elif self.segment3["lidar_hard"]:
                reason = "LiDAR obstacle detected within 0.5m"
            elif self.segment3["camera_violation"]:
                reason = "CSV road/lane validation violated"
            else:
                reason = "no valid road/lane evidence"
            self._report_segment(3, passed, reason)
        if previous == 9 and mode != 9:
            passed = self.segment9["nominal_3"] and not self.segment9["violation"]
            self._report_segment(
                9, passed, self.segment9["violation"] or
                "drive=3 nominal and LiDAR speed limits respected")
        self.completion.set_mode(mode)
        if mode not in self.entered_segments:
            self.entered_segments.add(mode)
            self.get_logger().info(f"[SEGMENT {mode}] ENTER")

    def on_imu_pitch(self, message):
        now = time.monotonic()
        self.imu_pitch = float(message.data)
        self.imu_pitch_at = now
        self.imu_samples.append((now, self.imu_pitch))

    def on_imu_valid(self, message):
        self.imu_valid = bool(message.data)
        self.imu_valid_at = time.monotonic()

    def _imu_average(self, now):
        window = float(self.get_parameter("imu_average_window_s").value)
        while self.imu_samples and now-self.imu_samples[0][0] > window:
            self.imu_samples.popleft()
        timeout = float(self.get_parameter("imu_timeout_s").value)
        valid = (self.imu_valid and self.imu_pitch_at is not None and
                 self.imu_valid_at is not None and
                 now-self.imu_pitch_at <= timeout and
                 now-self.imu_valid_at <= timeout and bool(self.imu_samples))
        average = (sum(value for _, value in self.imu_samples) /
                   len(self.imu_samples)) if valid else 0.0
        return average, valid

    def on_csv_validation(self, message):
        if self.completion.mode != 3:
            return
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not bool(value.get("fresh", False)):
            return
        state = str(value.get("state", "UNKNOWN")).upper()
        valid = (state == "TRUE" and bool(value.get("road_valid", False)))
        self.segment3["camera_valid"] |= valid
        # UNKNOWN is explicitly advisory and keeps CSV. Only the temporally
        # confirmed wheel/footprint collision state is a segment violation.
        if state == "FAIL":
            self.segment3["camera_violation"] = True
            self._report_segment(3, False, "CSV road/lane validation violated")

    def on_lidar_safety(self, message):
        self.completion.observe_lidar_safety(message.data)
        try:
            value = json.loads(message.data)
            distance = value.get("distance_m")
            self.latest_lidar_distance = (
                None if distance is None else float(distance))
            mode = int(value.get("mode", self.completion.mode or -1))
            if mode == 3 and bool(value.get("hard_stop", False)):
                self.segment3["lidar_hard"] = True
                self._report_segment(
                    3, False, "LiDAR obstacle detected within 0.5m")
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def on_final_drive(self, message):
        self.final_drive = float(message.data)
        if self.completion.mode != 9:
            return
        distance = self.latest_lidar_distance
        if abs(self.final_drive-3.0) <= 1.0e-6:
            self.segment9["nominal_3"] = True
        if distance is None:
            return
        if distance <= 1.0 and abs(self.final_drive) > 1.0e-6:
            self.segment9["violation"] = "drive nonzero with obstacle <=1.0m"
        elif 1.0 < distance <= 1.5 and self.final_drive > 1.0+1.0e-6:
            self.segment9["violation"] = "drive >1 with obstacle <=1.5m"

    def on_traffic(self, message):
        self.value.traffic_state = message.data.upper()
        self.traffic_received = time.monotonic()
        self.on_direct_signal("FUSED", message.data)

    def on_direct_signal(self, source, state):
        value = str(state).strip().upper()
        if value in ("R", "RED", "RED_X", "Y", "YELLOW"):
            value = "R"
        elif value in ("G", "GREEN", "GREEN_CIRCLE", "GREEN_DOWN",
                       "GREEN_LEFT", "GREEN_OTHER"):
            value = "G"
        else:
            value = "UNKNOWN"
        self.direct_signals[str(source)] = (value, time.monotonic())

    def on_csv_stop_state(self, message):
        self.csv_stop_line_active = str(message.data) in (
            "MINIMUM_3S_HOLD", "WAIT_TRAFFIC_RELEASE")
        self.value.csv_stop_line_active = self.csv_stop_line_active

    def on_traffic_diagnostics(self, message):
        try:
            values = json.loads(message.data)
            self.value.traffic_red_present = bool(
                values.get("valid_red_present", False))
            self.value.traffic_green_present = bool(
                values.get("valid_green_present", False))
            self.traffic_diagnostics_received = time.monotonic()
        except (TypeError, ValueError, json.JSONDecodeError):
            self.value.traffic_red_present = False
            self.value.traffic_green_present = False

    def on_camera_diagnostics(self, message):
        try:
            values = json.loads(message.data)
            pitch = values.get("imu_relative_uphill_deg", values.get("imu_pitch_deg"))
            if pitch is not None and math.isfinite(float(pitch)):
                self.value.pitch_deg = float(pitch)
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def on_maneuver(self, message):
        try:
            value = json.loads(message.data)
            self.maneuver_state = str(value.get("state", ""))
            self.completion.observe_maneuver(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.maneuver_state = ""

    def on_case_state(self, message):
        try:
            lifecycle = json.loads(message.data).get("parking_lifecycle", {})
            if str(lifecycle.get("T", "")) == "COMPLETE":
                self.completion.observe_csv_parking_complete(7)
            if str(lifecycle.get("V", "")) == "COMPLETE":
                self.completion.observe_csv_parking_complete(10)
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def tick(self):
        now = time.monotonic()
        self.value.now = now
        self.value.stop_line_detected = (
            self.camera_stop_line or self.csv_stop_line_active)
        self.value.traffic_age = float("inf") if self.traffic_received is None else now-self.traffic_received
        self.value.traffic_diagnostics_age = (
            float("inf") if self.traffic_diagnostics_received is None else
            now-self.traffic_diagnostics_received)
        timeout = float(self.get_parameter(
            "direct_signal_timeout_s").value)
        direct = tuple(
            state for state, received in self.direct_signals.values()
            if received is not None and 0.0 <= now-received <= timeout)
        # Any current R input wins over G/UNKNOWN. Once every R source has
        # actually changed to UNKNOWN/G (or gone stale), normal G or the
        # three-second UNKNOWN release policy can proceed.
        self.value.traffic_red_override = "R" in direct
        self.value.traffic_green_override = (
            not self.value.traffic_red_override and "G" in direct)
        decision = self.core.update(self.value)
        traffic = self.core.last_traffic_decision
        if traffic is not None and traffic.active:
            if decision.state != self.last_logged_state:
                suffix = ""
                progress = self.value.intersection_progress
                if progress is not None and getattr(progress, "mode", 0):
                    suffix = f" mode={progress.mode}"
                self.get_logger().info(
                    "[INTERSECTION] "+decision.state+suffix)
            if traffic.event and traffic.event != self.last_logged_event:
                for event in traffic.event.split("; "):
                    self.get_logger().info("[INTERSECTION] "+event)
                    self.completion.observe_intersection_event(event)
            self.last_logged_event = traffic.event
        elif decision.state != self.last_logged_state:
            self.get_logger().info("[MISSION] "+decision.state)
            self.last_logged_event = ""
        if decision.state != self.last_logged_state:
            self.last_logged_state = decision.state
        imu_average, imu_fresh = self._imu_average(now)
        self.completion.tick(
            now, self.final_drive,
            stop_waypoint_active=self.csv_stop_line_active,
            pitch_deg=imu_average, pitch_valid=imu_fresh,
            maneuver_state=self.maneuver_state)
        if (self.completion.force_mode2_stop() or
                self.completion.force_mode11_stop()):
            reason = ("MODE2_4S_STOP" if self.completion.force_mode2_stop()
                      else "MODE11_5S_STOP")
            decision = MissionDecision(
                reason, True, reason, 0.0,
                decision.traffic_permission, decision.traffic_stop_allowed,
                decision.intersection_committed)
        for event in self.completion.drain_events():
            event_name = str(event.get("event", ""))
            event_mode = int(event.get("mode", -1))
            if event_name == "MODE2_STOP_EVALUATED":
                pitch = event.get("pitch_deg")
                display = float(pitch) if pitch is not None else float("nan")
                threshold = float(event.get("threshold_deg", 4.5))
                self.get_logger().info(
                    f"[SEGMENT 2] STOP - IMU_AVG={display:.2f}deg "
                    f"THRESHOLD={threshold:.2f}deg")
            elif event_name == "MODE2_SLOPE_INVALID":
                self._report_segment(2, False, "")
            elif event_name == "MODE5_MISSION_FAILED":
                self._report_segment(
                    5, False, "successful avoidance count=" +
                    str(event.get("avoidance_rejoined_count", 0)))
            elif event_name == "MISSION_MODE_COMPLETE":
                if event_mode == 2:
                    self._report_segment(2, True, "")
                elif event_mode in (4, 6, 8):
                    self._report_segment(
                        event_mode, True,
                        "minimum 3s stop and traffic gate released")
                elif event_mode == 5:
                    self._report_segment(
                        5, True, "successful avoidance count=" +
                        str(self.completion.mode5_rejoins))
                elif event_mode in (7, 10):
                    self._report_segment(
                        event_mode, True,
                        "LiDAR slot selected and " +
                        self.completion.parking_source[event_mode] +
                        " rejoined CSV")
                elif event_mode == 11:
                    self._report_segment(
                        11, True,
                        f"branch={self.completion.mode11_branch} source="
                        f"{self.completion.mode11_source} after 5s stop")
            payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
            self.pub_event.publish(String(data=payload))
            self.get_logger().info("[MISSION] "+payload)
        self.pub_mode_status.publish(String(data=json.dumps(
            self.completion.status(), sort_keys=True, separators=(",", ":"))))
        self.pub_state.publish(String(data=decision.state))
        self.pub_section.publish(String(data=self.value.section_id))
        self.pub_stop.publish(Bool(data=decision.stop_required))
        self.pub_reason.publish(String(data=decision.stop_reason))
        self.pub_speed.publish(Float32(data=float(decision.speed_limit)))
        self.pub_permission.publish(String(data=decision.traffic_permission))
        self.pub_traffic_stop_allowed.publish(Bool(
            data=decision.traffic_stop_allowed))
        self.pub_intersection_committed.publish(Bool(
            data=decision.intersection_committed))
        self.pub_traffic_gate_state.publish(String(data=(
            traffic.state if traffic is not None and traffic.active else
            "INACTIVE_MODE_GATE")))
        self.pub_traffic_gate_reason.publish(String(data=(
            traffic.event if traffic is not None and traffic.active else
            "TRAFFIC_CONTROL_NOT_APPLICABLE")))
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        diag.status = [status(
            "depth_slam/mission",
            DiagnosticStatus.WARN if decision.stop_required else DiagnosticStatus.OK,
            decision.state,
            (("section", self.value.section_id),
             ("stop_reason", decision.stop_reason),
             ("traffic_age_s", self.value.traffic_age),
             ("direct_red_override", self.value.traffic_red_override),
             ("direct_green_override", self.value.traffic_green_override),
             ("traffic_stop_allowed", decision.traffic_stop_allowed),
             ("intersection_committed", decision.intersection_committed),
             ("pitch_deg", self.value.pitch_deg),
             ("imu_average_deg", imu_average),
             ("imu_fresh", imu_fresh),
             ("uphill_detected", self.value.uphill_detected)))]
        self.pub_diag.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
