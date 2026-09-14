"""Adapter from verified camera_ws topics to the depth_slam mission contract."""

import json
import math
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

from .intersection_route import progress_from_json
from .mission_completion import MissionCompletionTracker
from .mission_core import MissionMachine
from .models import MissionDecision, MissionInputs
from .ros_helpers import safe_shutdown, status


class MissionNode(Node):
    def __init__(self):
        super().__init__("depth_mission_manager")
        for name, default in (("ramp_sections", [""]),
                              ("acceleration_sections", [""]),
                              ("finish_sections", [""])):
            self.declare_parameter(name, default)
        for name, default in (("traffic_timeout_s", 0.3),
                              ("min_traffic_confidence", 0.65),
                              ("direct_signal_timeout_s", 0.5),
                              ("unknown_hold_s", 3.0),
                              ("intersection_commit_margin_m", 0.35)):
            self.declare_parameter(name, default)
        self.declare_parameter("intersection_modes", [4, 6, 8])

        def values(name):
            return [str(v) for v in self.get_parameter(name).value if str(v)]
        self.core = MissionMachine(values("ramp_sections"),
                                   values("acceleration_sections"),
                                   values("finish_sections"),
                                   traffic_timeout_s=float(self.get_parameter(
                                       "traffic_timeout_s").value),
                                   min_traffic_confidence=float(
                                       self.get_parameter(
                                           "min_traffic_confidence").value),
                                   unknown_hold_s=float(self.get_parameter(
                                       "unknown_hold_s").value),
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
        self.completion = MissionCompletionTracker()
        self.final_drive = 0.0
        self.imu_pitch = 0.0
        self.imu_valid = False
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
        self.create_subscription(String, "/drive_mode",
                                 lambda m: self.completion.set_mode(m.data), 10)
        self.create_subscription(Float32, "/cmd_drive",
                                 lambda m: setattr(self, "final_drive", float(m.data)), 10)
        self.create_subscription(Float32, "/imu/pitch_deg",
                                 lambda m: setattr(self, "imu_pitch", float(m.data)), 10)
        self.create_subscription(Bool, "/imu/valid",
                                 lambda m: setattr(self, "imu_valid", bool(m.data)), 10)
        self.create_subscription(String, "/depth_slam/route/mode_status",
                                 lambda m: self.completion.observe_route_status(m.data), 10)
        self.create_subscription(String, "/depth_slam/lidar/safety_event",
                                 lambda m: self.completion.observe_lidar_safety(m.data), 10)
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
        self.completion.tick(
            now, self.final_drive,
            stop_waypoint_active=self.csv_stop_line_active,
            pitch_deg=self.imu_pitch, pitch_valid=self.imu_valid,
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
             (("section", self.value.section_id), ("stop_reason", decision.stop_reason),
             ("traffic_age_s", self.value.traffic_age),
             ("direct_red_override", self.value.traffic_red_override),
             ("direct_green_override", self.value.traffic_green_override),
             ("traffic_stop_allowed", decision.traffic_stop_allowed),
             ("intersection_committed", decision.intersection_committed),
             ("pitch_deg", self.value.pitch_deg),
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
