#!/usr/bin/env python3
"""Publish section-aware mission decisions without owning any drive topic."""

import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import (Bool, Float32, Float32MultiArray, Int32, String)
from std_srvs.srv import Trigger

from .camera_mission_perception_node import image_array
from .mission_decision_core import (
    MissionDecisionConfig, MissionDecisionMachine, MissionInputs, SECTIONS,
    map_route_section)


MCU_FEEDBACK_CONTRACT = {
    "/mcu/encoder": {"type": "std_msgs/msg/Int32", "unit": "count",
                     "sign": "signed_cumulative_v37", "nominal_hz": 5.0},
    "/mcu/distance_m": {"type": "std_msgs/msg/Float32", "unit": "m",
                        "sign": "nonnegative_cumulative_wheel_roll",
                        "nominal_hz": 5.0},
    "/mcu/speed_mps": {"type": "std_msgs/msg/Float32", "unit": "m/s",
                       "sign": "signed_motion_direction", "nominal_hz": 5.0},
    "/mcu/speed_valid": {"type": "std_msgs/msg/Bool",
                         "meaning": "counts_per_meter_configured",
                         "nominal_hz": 5.0},
}


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def vehicle_steering_from_bev(required_steering_deg):
    """Convert BEV left-positive geometry to vehicle right-positive degrees."""
    value = float(required_steering_deg)
    return -value if math.isfinite(value) else math.nan


class CameraMissionDecisionNode(Node):
    def __init__(self):
        super().__init__("camera_mission_decision_node")
        defaults = {
            "mode_topic": "/mcu/current_mode",
            "section_topic": "/camera/mission/section",
            "input_timeout_sec": 0.50, "planner_timeout_sec": 0.30,
            "publish_rate_hz": 20.0, "slope_stop_duration_sec": 4.0,
            "slope_near_crossing_m": 0.25,
            "slope_line_spacing_min_m": 0.40,
            "slope_line_spacing_max_m": 8.0,
            "uphill_arm_pitch_deg": 15.0,
            "uphill_off_deg": 12.0,
            "uphill_min_duration_sec": 0.25,
            "slope_second_line_slow_distance_m": 1.50,
            "slope_second_line_stop_distance_m": 0.55,
            "slope_second_line_stop_tolerance_m": 0.10,
            "slope_first_to_second_min_travel_m": 0.50,
            "slope_first_to_second_max_travel_m": 8.0,
            "slope_stationary_speed_threshold_mps": 0.03,
            "slope_stationary_confirm_sec": 0.40,
            "slope_second_line_lost_timeout_sec": 0.50,
            "intersection_stop_trigger_distance_m": 1.50,
            "intersection_crossing_margin_m": 0.0,
            "intersection_detection_range_m": 2.0,
            "intersection_target_stop_margin_m": 1.0,
            "total_control_latency_sec": 0.20,
            "calibrated_deceleration_mps2": 0.0,
            "stop_distance_safety_margin_m": 0.10,
            "minimum_line_clearance_m": 0.20,
            "green_release_frames": 3.0,
            "maximum_steering_deg": 22.0,
            "straight_enter_deg": 1.0, "straight_exit_deg": 1.5,
            "turn_enter_deg": 2.0, "straight_min_duration_sec": 0.50,
            "turn_min_duration_sec": 0.25,
            "exit_signal_min_confidence": 0.60,
            "exit_green_down_confirm_frames": 3.0,
            "exit_signal_timeout_sec": 0.50,
            "mode_11_exit_signal_enabled": False,
            "debug_overlay_enabled": False,
            "debug_overlay_rate_hz": 5.0,
            "overlay_source_topic": "/camera/mission/debug_overlay",
            "encoder_topic": "/mcu/encoder",
            "speed_topic": "/mcu/speed_mps",
            "distance_topic": "/mcu/distance_m",
            "speed_valid_topic": "/mcu/speed_valid",
            "traffic_state_topic": "/camera/traffic_light_fused/state",
            "traffic_aspect_topic": "/camera/traffic_light_fused/aspect",
            "traffic_aspect_confidence_topic": "/camera/traffic_light_fused/confidence",
            "traffic_fusion_diagnostics_topic": "/camera/traffic_light_fused/diagnostics",
            "candidate_drive_topic": "/camera/candidate/path/drive",
            "candidate_wheel_topic": "/camera/candidate/path/wheel",
            "encoder_timeout_sec": 0.50,
            "speed_timeout_sec": 0.50,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        fields = MissionDecisionConfig.__dataclass_fields__
        self.machine = MissionDecisionMachine(MissionDecisionConfig(**{
            name: float(self.get_parameter(name).value) for name in fields
        }))
        self.timeout = float(self.p("input_timeout_sec"))
        self.planner_timeout = float(self.p("planner_timeout_sec"))
        self.encoder_timeout = float(self.p("encoder_timeout_sec"))
        self.speed_timeout = float(self.p("speed_timeout_sec"))
        self.values = {
            "section": "NORMAL_1", "stop_detected": False,
            "stop_count": 0, "stop_distances": (),
            "traffic": "UNKNOWN", "sign": False, "uphill": False,
            "planner_valid": False, "planner_state": "INVALID",
            "planner_drive": 0.0, "steering": math.nan,
            "wheel_steering": math.nan, "traffic_sequence": 0,
            "stop_tf_valid": False, "imu_pitch": math.nan,
            "encoder_count": None, "speed": math.nan,
            "speed_valid_flag": False, "distance": math.nan,
            "imu_valid": False, "traffic_aspect": "UNKNOWN",
            "traffic_aspect_confidence": 0.0, "aspect_sequence": 0,
            "rgb_green_down_verified": False,
            "last_fused_stamp": None,
            "route_mode_source": "NONE",
        }
        self.receipts = {}
        self.source_stamp = None
        self.source_stamps = {}
        self.timestamp_rewind = False
        self.last_decision = None
        self.latest_image = None
        self.last_overlay_at = float("-inf")
        self.debug_enabled = bool(self.p("debug_overlay_enabled"))

        self.active_pub = self.create_publisher(
            Bool, "/camera/mission/drive_override_active", 10)
        self.override_pub = self.create_publisher(
            Float32, "/camera/mission/drive_override", 10)
        self.state_pub = self.create_publisher(
            String, "/camera/mission/decision_state", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/mission/decision_diagnostics", 10)
        self.overlay_pub = self.create_publisher(
            Image, "/camera/mission/overlay", 1)

        self.create_subscription(String, self.p("mode_topic"),
                                 lambda m: self._route_mode(m, "MCU"), 10)
        self.create_subscription(String, self.p("section_topic"),
                                 lambda m: self._route_mode(m, "LEGACY_SECTION"), 10)
        self.create_subscription(Bool, "/camera/mission/stop_line_detected",
                                 lambda m: self._set("stop_detected", m.data), 10)
        self.create_subscription(Int32, "/camera/mission/stop_line_count",
                                 lambda m: self._set("stop_count", m.data), 10)
        self.create_subscription(
            Float32MultiArray, "/camera/mission/stop_line_distances_m",
            lambda m: self._set("stop_distances", tuple(m.data)), 10)
        self.create_subscription(String, self.p("traffic_state_topic"),
                                 lambda m: self._set(
                                     "fused_state_topic_value", m.data), 10)
        self.create_subscription(Bool, "/camera/mission/sign_detected",
                                 lambda m: self._set("sign", m.data), 10)
        self.create_subscription(Bool, "/camera/mission/uphill_detected",
                                 lambda m: self._set("uphill", m.data), 10)
        self.create_subscription(String, "/camera/mission/diagnostics",
                                 self._mission_diagnostics, 10)
        self.create_subscription(Bool, "/camera/bev/valid",
                                 lambda m: self._set("planner_valid", m.data), 10)
        self.create_subscription(String, "/camera/bev/diagnostics",
                                 self._planner_diagnostics, 10)
        self.create_subscription(Float32, self.p("candidate_drive_topic"),
                                 lambda m: self._set("planner_drive", m.data), 10)
        self.create_subscription(Int32, self.p("candidate_wheel_topic"),
                                 lambda m: self._set("wheel_steering", m.data), 10)
        self.create_subscription(Int32, self.p("encoder_topic"),
                                 lambda m: self._set("encoder_count", m.data), 10)
        self.create_subscription(Float32, self.p("speed_topic"),
                                 lambda m: self._set("speed", m.data), 10)
        self.create_subscription(Float32, self.p("distance_topic"),
                                 lambda m: self._set("distance", m.data), 10)
        self.create_subscription(Bool, self.p("speed_valid_topic"),
                                 lambda m: self._set("speed_valid_flag", m.data), 10)
        self.create_subscription(String, self.p("traffic_aspect_topic"),
                                 lambda m: self._set(
                                     "fused_aspect_topic_value", m.data), 10)
        self.create_subscription(
            Float32, self.p("traffic_aspect_confidence_topic"),
            lambda m: self._set(
                "fused_confidence_topic_value", m.data), 10)
        self.create_subscription(
            String, self.p("traffic_fusion_diagnostics_topic"),
            self._fusion_diagnostics, 10)
        if self.debug_enabled:
            self.create_subscription(Image, self.p("overlay_source_topic"),
                                     self._image, qos_profile_sensor_data)
        self.create_service(Trigger, "/camera/mission/reset_decision", self._reset)
        self.create_timer(1.0/float(self.p("publish_rate_hz")), self._tick)

    def p(self, name):
        return self.get_parameter(name).value

    def _set(self, key, value):
        self.values[key] = value
        self.receipts[key] = time.monotonic()

    def _route_mode(self, message, source):
        self._set("section", message.data)
        self.values["route_mode_source"] = source

    def _traffic(self, message):
        self.values["traffic"] = str(message.data).upper()
        self.values["traffic_sequence"] += 1
        self.receipts["traffic"] = time.monotonic()

    def _aspect(self, message):
        self.values["traffic_aspect"] = str(message.data).upper()
        self.values["aspect_sequence"] += 1
        self.receipts["traffic_aspect"] = time.monotonic()

    def _fusion_diagnostics(self, message):
        try:
            data = json.loads(message.data)
            state = str(data.get("fused_state", "UNKNOWN")).upper()
            aspect = str(data.get("fused_aspect", "UNKNOWN")).upper()
            confidence = float(data.get("fused_confidence", math.nan))
            fused_stamp = data.get("stamp")
            if (state not in ("R", "G", "UNKNOWN") or
                    aspect not in ("RED", "RED_X", "YELLOW",
                                   "GREEN_CIRCLE", "GREEN_LEFT", "GREEN_DOWN",
                                   "GREEN_OTHER", "UNKNOWN") or
                    not math.isfinite(confidence)):
                raise ValueError("invalid fused traffic diagnostics")
            now = time.monotonic()
            self.values["traffic"] = state
            self.values["traffic_aspect"] = aspect
            self.values["traffic_aspect_confidence"] = confidence
            if fused_stamp is not None and fused_stamp != self.values[
                    "last_fused_stamp"]:
                self.values["aspect_sequence"] += 1
                self.values["last_fused_stamp"] = fused_stamp
            self.values["rgb_green_down_verified"] = bool(
                data.get("rgb_green_down_verified", False))
            for key in ("traffic", "traffic_aspect",
                        "traffic_aspect_confidence", "traffic_fusion_diag"):
                self.receipts[key] = now
        except (TypeError, ValueError):
            self.values["rgb_green_down_verified"] = False

    def _record_source_stamp(self, source, stamp):
        if stamp is None or not math.isfinite(stamp):
            return
        previous = self.source_stamps.get(source)
        if previous is not None and stamp < previous:
            self.timestamp_rewind = True
        self.source_stamps[source] = stamp
        # Different pipeline workers can legitimately report slightly different
        # source stamps. Rewind detection must therefore be per input stream.
        self.source_stamp = max(self.source_stamps.values())

    def _mission_diagnostics(self, message):
        self.receipts["mission_diag"] = time.monotonic()
        try:
            data = json.loads(message.data)
            self._record_source_stamp("mission", data.get("input_timestamp"))
            self.values["stop_tf_valid"] = bool(
                data.get("stop_line_tf_valid", False))
            self.values["imu_valid"] = bool(data.get("imu_valid", False))
            pitch = data.get("imu_pitch_deg")
            self.values["imu_pitch"] = (float(pitch) if pitch is not None
                                        else math.nan)
        except (TypeError, ValueError):
            pass

    def _planner_diagnostics(self, message):
        self.receipts["planner_diag"] = time.monotonic()
        try:
            data = json.loads(message.data)
            self.values["planner_state"] = str(
                data.get("state", data.get("planner_state", "INVALID")))
            steering = data.get("required_steering_deg")
            # Direct BEV geometry is left-positive. The final vehicle command
            # contract is right-positive, so convert exactly once here.
            self.values["steering"] = (vehicle_steering_from_bev(steering)
                                        if steering is not None else math.nan)
            stamp_ns = data.get("source_stamp_ns", data.get("stamp_ns"))
            if stamp_ns is not None:
                self._record_source_stamp("planner", float(stamp_ns)*1.0e-9)
        except (TypeError, ValueError):
            self.values["planner_state"] = "INVALID"
            self.values["steering"] = math.nan

    def _fresh(self, keys, now, timeout):
        return all(key in self.receipts and now-self.receipts[key] <= timeout
                   for key in keys)

    def _section_fresh(self, now):
        if "section" not in self.receipts:
            return False
        # Section is a latched route-mode selection, not a streaming sensor.
        common = ("planner_valid", "planner_diag", "planner_drive")
        if not self._fresh(common, now, self.planner_timeout):
            return False
        _route_mode, section = map_route_section(self.values["section"])
        if section is None:
            return False
        if section == "SLOPE":
            keys = ("stop_distances", "uphill", "mission_diag")
        elif section in ("INTERSECTION", "INTERSECTION_4", "INTERSECTION_6"):
            keys = ("stop_distances", "traffic", "mission_diag")
        elif section == "ACCELERATION":
            keys = ("sign", "mission_diag")
        elif section in ("LEFT_SIGNAL_MONITOR", "EXIT_SIGNAL", "NORMAL_11"):
            keys = ()
        else:
            keys = ("mission_diag",)
        return self._fresh(keys, now, self.timeout)

    def _tick(self):
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds*1.0e-9
        raw_section = str(self.values["section"]).upper()
        route_mode, section = map_route_section(raw_section)
        section = section if section is not None else raw_section
        source = self.source_stamp if self.source_stamp is not None else ros_now
        if self.timestamp_rewind:
            source = ((self.machine.last_stamp-1.0) if
                      self.machine.last_stamp is not None else source)
            self.timestamp_rewind = False
        steering = float(self.values["steering"])
        if not math.isfinite(steering):
            steering = float(self.values["wheel_steering"])
        encoder_valid = self._fresh(("encoder_count",), now,
                                    self.encoder_timeout)
        speed_valid = (bool(self.values["speed_valid_flag"]) and
                       self._fresh(("speed", "speed_valid_flag"), now,
                                   self.speed_timeout))
        distance_valid = (bool(self.values["speed_valid_flag"]) and
                          self._fresh(("distance", "speed_valid_flag"), now,
                                      self.encoder_timeout))
        decision = self.machine.update(MissionInputs(
            now=now, stamp=source, section=section,
            route_mode=route_mode,
            input_fresh=self._section_fresh(now),
            planner_valid=bool(self.values["planner_valid"]),
            planner_state=str(self.values["planner_state"]),
            planner_drive=float(self.values["planner_drive"]),
            stop_line_detected=bool(self.values["stop_detected"]),
            stop_line_distances_m=tuple(self.values["stop_distances"]),
            traffic_light=str(self.values["traffic"]),
            traffic_sample_id=int(self.values["traffic_sequence"]),
            sign_detected=bool(self.values["sign"]),
            uphill_detected=bool(self.values["uphill"]),
            imu_pitch_deg=float(self.values["imu_pitch"]),
            imu_valid=bool(self.values["imu_valid"]),
            stop_line_tf_valid=bool(self.values["stop_tf_valid"]),
            encoder_valid=encoder_valid,
            encoder_count=self.values["encoder_count"],
            speed_valid=speed_valid,
            current_speed_mps=float(self.values["speed"]),
            distance_valid=distance_valid,
            distance_m=float(self.values["distance"]),
            required_steering_deg=steering,
            traffic_aspect=str(self.values["traffic_aspect"]),
            traffic_aspect_confidence=float(
                self.values["traffic_aspect_confidence"]),
            traffic_aspect_fresh=self._fresh(
                ("traffic_aspect", "traffic_aspect_confidence",
                 "traffic_fusion_diag"), now,
                float(self.p("exit_signal_timeout_sec"))),
            traffic_aspect_sample_id=int(self.values["aspect_sequence"]),
            rgb_green_down_verified=bool(
                self.values["rgb_green_down_verified"])))
        self.last_decision = decision
        decision.diagnostics["mcu_feedback_contract"] = MCU_FEEDBACK_CONTRACT
        decision.diagnostics["route_mode_source"] = self.values[
            "route_mode_source"]
        decision.diagnostics["mcu_feedback_fresh"] = {
            "/mcu/encoder": encoder_valid,
            "/mcu/distance_m": distance_valid,
            "/mcu/speed_mps": speed_valid,
            "/mcu/speed_valid": self._fresh(
                ("speed_valid_flag",), now, self.speed_timeout),
        }
        last_aspect = self.receipts.get("traffic_aspect")
        decision.diagnostics["traffic_aspect_age_sec"] = (
            None if last_aspect is None else max(0.0, now-last_aspect))
        decision.diagnostics["odom_contract"] = {
            "topic": "/odom", "mcu_owned": False,
            "mcu_reference_topic": "/mcu/odom",
            "reason": "T870_MCU effective YAML remaps bridge odometry",
        }
        self.active_pub.publish(Bool(data=decision.override_active))
        self.override_pub.publish(Float32(data=decision.drive_override))
        self.state_pub.publish(String(data=decision.state))
        self.diag_pub.publish(String(data=json.dumps(
            decision.diagnostics, separators=(",", ":"), allow_nan=False)))
        self._publish_overlay(now)

    def _reset(self, request, response):
        del request
        _route_mode, section = map_route_section(self.values["section"])
        section = section or "NORMAL_1"
        self.machine.reset(section if section in SECTIONS else "NORMAL_1")
        self.timestamp_rewind = False
        self.source_stamps.clear()
        self.source_stamp = None
        response.success = True
        response.message = f"mission decision reset for {self.machine.section}"
        return response

    def _image(self, message):
        try:
            image = image_array(message)
            if message.encoding.upper() == "RGB8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.latest_image = (message.header, image)
        except ValueError:
            self.latest_image = None

    def _publish_overlay(self, now):
        if (not self.debug_enabled or self.latest_image is None or
                self.last_decision is None or
                now-self.last_overlay_at < 1.0/float(self.p("debug_overlay_rate_hz"))):
            return
        header, canvas = self.latest_image[0], self.latest_image[1].copy()
        d = self.last_decision.diagnostics
        lines = [
            f"SECTION {d['section']}  STATE {d['decision_state']}",
            f"LINES {d['stop_line_distances_m']} first={d['first_line_crossed']} "
            f"between={d['between_lines']} timer={d['stop_elapsed_sec']:.1f}s",
            f"TL={d['traffic_light']} stop={d['intersection_stop_required']} "
            f"late={d['late_red_action']} sign={d['sign_detected']}",
            f"v={d['current_speed_mps']} req={d['required_stop_m']} "
            f"avail={d['available_to_target_m']}",
            f"shape={d['path_shape']} {d['turn_direction']} "
            f"steer={d['required_steering_deg']}",
            f"override={d['override_active']}:{d['drive_override']:.1f} "
            f"planner={self.values['planner_drive']:.1f} "
            f"effective={d['effective_drive_if_connected']:.1f}",
            f"fresh={d['input_fresh']} blocked={d['safety_blocked']} "
            f"reason={d['failure_reason']}",
        ]
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1]-1, 168), (0, 0, 0), -1)
        for index, text in enumerate(lines):
            cv2.putText(canvas, text, (8, 20+22*index),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1,
                        cv2.LINE_AA)
        message = Image(); message.header = header
        message.height, message.width = canvas.shape[:2]
        message.encoding = "bgr8"; message.is_bigendian = False
        message.step = message.width*3; message.data = canvas.tobytes()
        self.overlay_pub.publish(message)
        self.last_overlay_at = now


def main(args=None):
    rclpy.init(args=args)
    node = CameraMissionDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
