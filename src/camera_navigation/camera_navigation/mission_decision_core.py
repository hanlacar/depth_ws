"""ROS-independent section-aware advisory mission decision state machine."""

from dataclasses import dataclass
import math


ROUTE_MODE_SECTIONS = {
    1: "NORMAL_1", 2: "SLOPE", 3: "D_COURSE",
    4: "INTERSECTION_4", 5: "S_COURSE", 6: "INTERSECTION_6",
    7: "T_PARK", 8: "NORMAL_8", 9: "ACCELERATION",
    10: "PARALLEL_PARK", 11: "NORMAL_11",
}
SECTIONS = frozenset(ROUTE_MODE_SECTIONS.values()) | frozenset((
    "NORMAL", "INTERSECTION", "LEFT_SIGNAL_MONITOR", "EXIT_SIGNAL"))
DRIVE_STAGES = frozenset((0.0, 1.0, 2.0, 3.0))
CONTROL_DRIVE_STAGES = frozenset((-1.0, 0.0, 1.0, 2.0, 3.0))


@dataclass(frozen=True)
class MissionDecisionConfig:
    slope_stop_duration_sec: float = 4.0
    slope_near_crossing_m: float = 0.25
    slope_line_spacing_min_m: float = 0.40
    slope_line_spacing_max_m: float = 8.0
    uphill_arm_pitch_deg: float = 15.0
    uphill_off_deg: float = 12.0
    uphill_min_duration_sec: float = 0.25
    slope_second_line_slow_distance_m: float = 1.50
    # Front-axle distance. Conservative initial value; calibrate on the car.
    slope_second_line_stop_distance_m: float = 0.55
    slope_second_line_stop_tolerance_m: float = 0.10
    slope_first_to_second_min_travel_m: float = 0.50
    slope_first_to_second_max_travel_m: float = 8.0
    slope_stationary_speed_threshold_mps: float = 0.03
    slope_stationary_confirm_sec: float = 0.40
    slope_second_line_lost_timeout_sec: float = 0.50
    intersection_stop_trigger_distance_m: float = 1.5
    intersection_crossing_margin_m: float = 0.0
    intersection_detection_range_m: float = 2.0
    intersection_target_stop_margin_m: float = 1.0
    total_control_latency_sec: float = 0.20
    # Zero is deliberately uncommissioned. COMMIT_TO_CROSS is impossible
    # until a measured worst-case value is explicitly supplied.
    calibrated_deceleration_mps2: float = 0.0
    stop_distance_safety_margin_m: float = 0.10
    minimum_line_clearance_m: float = 0.20
    green_release_frames: float = 3.0
    maximum_steering_deg: float = 22.0
    straight_enter_deg: float = 1.0
    straight_exit_deg: float = 1.5
    turn_enter_deg: float = 2.0
    straight_min_duration_sec: float = 0.50
    turn_min_duration_sec: float = 0.25
    exit_signal_min_confidence: float = 0.60
    exit_green_down_confirm_frames: float = 3.0
    exit_signal_timeout_sec: float = 0.50
    mode_11_exit_signal_enabled: bool = False

    def validate(self):
        values = tuple(value for value in vars(self).values()
                       if not isinstance(value, bool))
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("mission thresholds must be finite and nonnegative")
        if not (self.straight_enter_deg <= self.straight_exit_deg <
                self.turn_enter_deg):
            raise ValueError("straight/turn hysteresis is invalid")
        if not self.slope_line_spacing_min_m < self.slope_line_spacing_max_m:
            raise ValueError("slope line spacing bounds are invalid")
        if not (self.intersection_crossing_margin_m <=
                self.minimum_line_clearance_m <
                self.intersection_target_stop_margin_m <
                self.intersection_detection_range_m):
            raise ValueError("intersection distance thresholds are invalid")
        if self.green_release_frames < 1 or not float(
                self.green_release_frames).is_integer():
            raise ValueError("green release frames must be a positive integer")
        if self.exit_green_down_confirm_frames < 1 or not float(
                self.exit_green_down_confirm_frames).is_integer():
            raise ValueError("exit confirmation frames must be a positive integer")
        if not (self.slope_second_line_stop_distance_m <
                self.slope_second_line_slow_distance_m):
            raise ValueError("second-line stop distance must be below slow distance")


@dataclass(frozen=True)
class MissionInputs:
    now: float
    stamp: float
    section: str = "NORMAL"
    input_fresh: bool = False
    planner_valid: bool = False
    planner_state: str = "INVALID"
    planner_drive: float = 0.0
    stop_line_detected: bool = False
    stop_line_distances_m: tuple = ()
    traffic_light: str = "UNKNOWN"
    sign_detected: bool = False
    uphill_detected: bool = False
    required_steering_deg: float = math.nan
    imu_pitch_deg: float = math.nan
    stop_line_tf_valid: bool = False
    encoder_valid: bool = False
    encoder_count: int | None = None
    speed_valid: bool = False
    current_speed_mps: float = math.nan
    distance_valid: bool = False
    distance_m: float = math.nan
    traffic_sample_id: int | None = None
    route_mode: int | None = None
    imu_valid: bool = False
    traffic_aspect: str = "UNKNOWN"
    traffic_aspect_confidence: float = 0.0
    traffic_aspect_fresh: bool = False
    traffic_aspect_sample_id: int | None = None
    rgb_green_down_verified: bool = False


def map_route_section(value):
    """Map the deployed String route-mode contract without inventing a topic."""
    text = str(value).strip().upper()
    try:
        mode = int(text)
    except ValueError:
        return (None, text if text in SECTIONS else None)
    return (mode, ROUTE_MODE_SECTIONS.get(mode))


@dataclass(frozen=True)
class MissionDecision:
    section: str
    state: str
    override_active: bool
    drive_override: float
    effective_drive: float
    safety_blocked: bool
    failure_reason: str
    diagnostics: dict


class MissionDecisionMachine:
    def __init__(self, config=MissionDecisionConfig()):
        config.validate()
        self.config = config
        self.section = "NORMAL"
        self.route_mode = None
        self.state = "NORMAL_IDLE"
        self.last_stamp = None
        self.first_line_crossed = False
        self.first_line_last_distance = None
        self.line_spacing_m = None
        self.between_lines = False
        self.slope_stop_started = None
        self.slope_stop_completed = False
        self.slope_armed = False
        self.uphill_candidate_since = None
        self.uphill_confirmed_after_first_line = False
        self.first_line_odometer_m = None
        self.first_to_second_travel_m = math.nan
        self.second_line_tracked = False
        self.second_line_distance_m = math.nan
        self.second_line_last_seen_at = None
        self.second_line_last_odometer_m = None
        self.second_line_previous_detection_m = None
        self.stationary_since = None
        self.vehicle_stationary = False
        self.hold_started = None
        self.intersection_line_crossed = False
        self.red_stop_latched = False
        self.commit_to_cross_latched = False
        self.line_crossed_latched = False
        self.green_release_latched = False
        self.green_release_count = 0
        self.last_traffic_sample_id = None
        self.commit_line_distance_m = None
        self.commit_odometer_m = None
        self.late_red_action = "NONE"
        self.available_to_target_m = math.nan
        self.required_stop_m = math.nan
        self.can_stop_at_target = False
        self.can_stop_before_line = False
        self.mission_failure_reason = ""
        self.intersection_distance_m = math.nan
        self.intersection_proceed_permitted = False
        self.intersection_permission_source = "NONE"
        self.acceleration_armed = False
        self.high_speed_completed = False
        self.turn_detected = False
        self.post_turn_cruise = False
        self.path_shape = "UNKNOWN"
        self.shape_candidate = None
        self.shape_since = None
        self.turn_direction = "NONE"
        self.left_signal_confirmed = False
        self.exit_release_latched = False
        self.exit_green_count = 0
        self.last_aspect_sample_id = None

    def reset(self, section="NORMAL"):
        self.__init__(self.config)
        self.section = section if section in SECTIONS else "NORMAL"
        self.state = f"{self.section}_IDLE"

    def _section_reset(self, section):
        last_stamp = self.last_stamp
        self.reset(section)
        self.last_stamp = last_stamp

    def update(self, inputs):
        route_mode, mapped = map_route_section(inputs.section)
        if mapped is None:
            return self._safe(inputs, "SECTION_INVALID")
        if mapped != inputs.section or route_mode is not None:
            from dataclasses import replace
            inputs = replace(inputs, section=mapped,
                             route_mode=route_mode if route_mode is not None
                             else inputs.route_mode)
        if not all(math.isfinite(v) for v in (inputs.now, inputs.stamp)):
            return self._safe(inputs, "TIMESTAMP_INVALID")
        if self.last_stamp is not None and inputs.stamp < self.last_stamp:
            self.reset(inputs.section)
            self.last_stamp = inputs.stamp
            return self._safe(inputs, "TIMESTAMP_REWIND")
        self.last_stamp = inputs.stamp
        route_changed = (inputs.route_mode != self.route_mode and
                         (inputs.route_mode is not None or
                          self.route_mode is not None))
        if inputs.section != self.section or route_changed:
            self._section_reset(inputs.section)
        self.route_mode = inputs.route_mode
        if not inputs.input_fresh:
            # A stale sample may not preserve one-shot or persistence latches.
            # Keep only the active section and timestamp so recovery starts from
            # an explicit, safe initial state.
            self._section_reset(inputs.section)
            self.state = "SAFE_STOP_INPUT_TIMEOUT"
            return self._safe(inputs, "INPUT_TIMEOUT")
        if float(inputs.planner_drive) not in CONTROL_DRIVE_STAGES:
            self.state = "SAFE_STOP_DRIVE_STAGE_INVALID"
            return self._safe(inputs, "DRIVE_STAGE_INVALID")
        if (math.isfinite(inputs.required_steering_deg) and
                abs(inputs.required_steering_deg) >
                self.config.maximum_steering_deg):
            self.state = "SAFE_STOP_STEERING_RANGE_INVALID"
            return self._safe(inputs, "STEERING_RANGE_INVALID")

        self.mission_failure_reason = ""
        if self.section == "SLOPE":
            active, drive = self._slope(inputs)
        elif self.section in ("INTERSECTION", "INTERSECTION_4",
                              "INTERSECTION_6"):
            active, drive = self._intersection(inputs)
        elif self.section == "ACCELERATION":
            active, drive = self._acceleration(inputs)
        elif self.section == "LEFT_SIGNAL_MONITOR":
            active, drive = self._left_signal(inputs)
        elif self.section == "EXIT_SIGNAL" or (
                self.section == "NORMAL_11" and
                self.config.mode_11_exit_signal_enabled):
            active, drive = self._exit_signal(inputs)
        elif self.section == "NORMAL_11":
            self.state = f"MODE_11_DIAGNOSTIC_{inputs.traffic_aspect}"
            active, drive = False, 0.0
        else:
            self.state = f"{self.section}_IDLE"
            active, drive = False, 0.0
        mission_requested = active
        requested_drive = drive
        if not inputs.planner_valid or inputs.planner_state in (
                "INVALID", "INPUT_TIMEOUT", "CALIBRATION_INVALID"):
            mission_state = self.state
            self.state = "SAFE_STOP_PLANNER_INVALID"
            result = self._safe(inputs, "PLANNER_INVALID", mission_requested,
                                requested_drive)
            result.diagnostics["mission_state_before_safety"] = mission_state
            return result
        effective = requested_drive if mission_requested else inputs.planner_drive
        if self.section not in ("NORMAL", "NORMAL_1", "NORMAL_8", "NORMAL_11") and effective < 0.0:
            self.state = "SAFE_STOP_REVERSE_FORBIDDEN"
            return self._safe(inputs, "REVERSE_FORBIDDEN", mission_requested,
                              requested_drive)
        return self._decision(inputs, active, drive, effective,
                              bool(self.mission_failure_reason),
                              self.mission_failure_reason)

    def _safe(self, inputs, reason, mission_requested=False,
              requested_drive=0.0):
        return self._decision(inputs, True, 0.0, 0.0, True, reason,
                              mission_requested, requested_drive)

    def _decision(self, inputs, active, drive, effective, blocked, reason,
                  mission_requested=None, requested_drive=None):
        drive = float(drive)
        if drive not in DRIVE_STAGES:
            active, drive, effective, blocked, reason = \
                True, 0.0, 0.0, True, "DRIVE_STAGE_INVALID"
        distances = tuple(float(v) for v in inputs.stop_line_distances_m
                          if math.isfinite(float(v)))
        second = self._second_line_ahead(distances)
        elapsed = (0.0 if self.slope_stop_started is None else
                   max(0.0, inputs.now-self.slope_stop_started))
        diagnostics = {
            "section": self.section, "mapped_section": self.section,
            "route_mode": inputs.route_mode,
            "decision_state": self.state, "stop_line_phase": self.state,
            "override_active": bool(active), "drive_override": drive,
            "effective_drive_if_connected": float(effective),
            "stop_line_detected": inputs.stop_line_detected,
            "stop_line_count": len(distances),
            "stop_line_distances_m": list(distances),
            "target_stop_line_distance_m": self._target_distance(distances),
            "first_line_crossed": self.first_line_crossed,
            "second_line_ahead": second,
            "between_lines": self.between_lines,
            "stop_elapsed_sec": elapsed,
            "slope_stop_completed": self.slope_stop_completed,
            "slope_armed": self.slope_armed,
            "uphill_confirmed_after_first_line": self.uphill_confirmed_after_first_line,
            "second_line_tracked": self.second_line_tracked,
            "second_line_distance_m": (self.second_line_distance_m if
                math.isfinite(self.second_line_distance_m) else None),
            "first_to_second_travel_m": (self.first_to_second_travel_m if
                math.isfinite(self.first_to_second_travel_m) else None),
            "vehicle_stationary": self.vehicle_stationary,
            "stationary_duration_sec": (0.0 if self.stationary_since is None
                else max(0.0, inputs.now-self.stationary_since)),
            "slope_hold_elapsed_sec": (0.0 if self.hold_started is None
                else max(0.0, inputs.now-self.hold_started)),
            "slope_hold_remaining_sec": (self.config.slope_stop_duration_sec
                if self.hold_started is None else max(0.0,
                self.config.slope_stop_duration_sec-(inputs.now-self.hold_started))),
            "traffic_light": inputs.traffic_light,
            "intersection_line_crossed": self.intersection_line_crossed,
            "red_stop_latched": self.red_stop_latched,
            "commit_to_cross_latched": self.commit_to_cross_latched,
            "line_crossed_latched": self.line_crossed_latched,
            "green_release_latched": self.green_release_latched,
            "encoder_valid": bool(inputs.encoder_valid),
            "encoder_count": inputs.encoder_count,
            "speed_valid": bool(inputs.speed_valid),
            "current_speed_mps": (float(inputs.current_speed_mps) if
                math.isfinite(inputs.current_speed_mps) else None),
            "distance_valid": bool(inputs.distance_valid),
            "distance_m": (float(inputs.distance_m) if
                math.isfinite(inputs.distance_m) else None),
            "stop_line_distance_m": (self.intersection_distance_m if
                self.section in ("INTERSECTION", "INTERSECTION_4",
                                 "INTERSECTION_6") and
                math.isfinite(self.intersection_distance_m) else
                self._target_distance(distances)),
            "stop_line_tf_valid": bool(inputs.stop_line_tf_valid),
            "target_stop_margin_m": self.config.intersection_target_stop_margin_m,
            "available_to_target_m": (self.available_to_target_m if
                math.isfinite(self.available_to_target_m) else None),
            "required_stop_m": (self.required_stop_m if
                math.isfinite(self.required_stop_m) else None),
            "control_latency_sec": self.config.total_control_latency_sec,
            "calibrated_deceleration_mps2": (
                self.config.calibrated_deceleration_mps2),
            "can_stop_at_target": self.can_stop_at_target,
            "can_stop_before_line": self.can_stop_before_line,
            "late_red_action": self.late_red_action,
            "intersection_stop_required": (
                self.section in ("INTERSECTION", "INTERSECTION_4",
                                 "INTERSECTION_6") and
                bool(active) and drive == 0.0),
            "intersection_proceed_permitted": self.intersection_proceed_permitted,
            "intersection_permission_source": self.intersection_permission_source,
            "sign_detected": inputs.sign_detected,
            "acceleration_armed": self.acceleration_armed,
            "acceleration_sign_latched": self.acceleration_armed,
            "acceleration_straight_confirmed": self.high_speed_completed,
            "traffic_aspect": inputs.traffic_aspect,
            "traffic_aspect_confidence": (float(inputs.traffic_aspect_confidence)
                if math.isfinite(inputs.traffic_aspect_confidence) else None),
            "traffic_aspect_age_sec": None,
            "left_signal_confirmed": self.left_signal_confirmed,
            "exit_signal_state": self.state if self.section == "EXIT_SIGNAL" else "INACTIVE",
            "exit_release_latched": self.exit_release_latched,
            "rgb_green_down_verified": bool(inputs.rgb_green_down_verified),
            "required_steering_deg": (inputs.required_steering_deg
                                       if math.isfinite(inputs.required_steering_deg)
                                       else None),
            "steering_deg": (inputs.required_steering_deg
                              if math.isfinite(inputs.required_steering_deg)
                              else None),
            "imu_pitch_deg": (inputs.imu_pitch_deg if
                              math.isfinite(inputs.imu_pitch_deg) else None),
            "uphill_detected": bool(inputs.uphill_detected),
            "path_shape": self.path_shape,
            "turn_direction": self.turn_direction,
            "high_speed_completed": self.high_speed_completed,
            "turn_detected": self.turn_detected,
            "post_turn_cruise": self.post_turn_cruise,
            "planner_valid": inputs.planner_valid,
            "input_fresh": inputs.input_fresh,
            "mission_override_requested": (bool(active) if
                mission_requested is None else bool(mission_requested)),
            "mission_drive_requested": (drive if requested_drive is None
                                         else float(requested_drive)),
            "safety_blocked": blocked, "failure_reason": reason,
        }
        return MissionDecision(self.section, self.state, bool(active), drive,
                               float(effective), blocked, reason, diagnostics)

    @staticmethod
    def _target_distance(distances):
        positives = [v for v in distances if v > 0.0]
        return min(positives) if positives else (
            min(distances, key=abs) if distances else None)

    def _second_line_ahead(self, distances):
        positives = sorted(v for v in distances if v > 0.0)
        if self.first_line_crossed:
            return positives[0] if positives else None
        return positives[1] if len(positives) > 1 else None

    def _slope(self, inputs):
        distances = sorted((float(v) for v in inputs.stop_line_distances_m
                            if math.isfinite(float(v))), key=abs)
        if self.slope_stop_completed:
            self.state = "SLOPE_COMPLETE"
            return False, 0.0
        nearest = min(distances, key=abs) if distances else None
        if not self.first_line_crossed:
            if nearest is not None and nearest > 0.0:
                self.state = "APPROACH_FIRST_LINE"
                self.first_line_last_distance = nearest
            elif (nearest is not None and nearest <= 0.0 and
                  self.first_line_last_distance is not None and
                  0.0 < self.first_line_last_distance <=
                  self.config.slope_near_crossing_m):
                self.first_line_crossed = True
                self.first_line_odometer_m = (float(inputs.distance_m) if
                    inputs.distance_valid and math.isfinite(inputs.distance_m)
                    else None)
                self.state = "FIRST_LINE_CROSSED"
            else:
                self.state = "WAIT_FIRST_LINE"
            return False, 0.0

        # Uphill must be established after the first physical line crossing.
        uphill_gate = (inputs.imu_valid and inputs.uphill_detected and
                       math.isfinite(inputs.imu_pitch_deg) and
                       inputs.imu_pitch_deg >= self.config.uphill_arm_pitch_deg)
        if not self.uphill_confirmed_after_first_line:
            if uphill_gate:
                if self.uphill_candidate_since is None:
                    self.uphill_candidate_since = inputs.now
                if inputs.now-self.uphill_candidate_since >= \
                        self.config.uphill_min_duration_sec:
                    self.uphill_confirmed_after_first_line = True
                    self.slope_armed = True
                    self.state = "UPHILL_CONFIRMED"
                    return False, 0.0
            else:
                self.uphill_candidate_since = None
            if not self.uphill_confirmed_after_first_line:
                self.state = "FIRST_LINE_CROSSED"
                return False, 0.0

        # Nose-up pitch is part of the stop condition, not merely an arming
        # edge. A later level/downhill sample must never preserve a stop.
        if not uphill_gate:
            self.state = "BETWEEN_LINES_WAIT_UPHILL"
            return False, 0.0

        # The aligned-depth, exact-stamp camera->front_axle measurement is the
        # primary stop-line distance. Wheel odometry may slip on the ramp and
        # is therefore never required to identify the second line.
        travel = self.first_to_second_travel_m
        travel_valid = math.isfinite(travel)
        second = min((v for v in distances if v > 0.0), default=None)
        if second is not None and inputs.stop_line_tf_valid:
            decreasing = (self.second_line_previous_detection_m is not None and
                          second < self.second_line_previous_detection_m-1.0e-3)
            eligible = decreasing and self.first_line_crossed
            self.second_line_previous_detection_m = second
            if eligible or self.second_line_tracked:
                self.second_line_tracked = True
                self.second_line_distance_m = second
                self.second_line_last_seen_at = inputs.now
                self.second_line_last_odometer_m = None
                self.between_lines = True

        if not self.second_line_tracked:
            self.state = "APPROACH_SECOND_LINE"
            return False, 0.0
        feedback_valid = (inputs.stop_line_tf_valid and inputs.imu_valid and
                          inputs.speed_valid and
                          math.isfinite(inputs.current_speed_mps))
        line_fresh = (self.second_line_last_seen_at is not None and
                      inputs.now-self.second_line_last_seen_at <=
                      self.config.slope_second_line_lost_timeout_sec)
        if not feedback_valid or not line_fresh:
            self.state = "STOP_SECOND_LINE"
            self.mission_failure_reason = "SLOPE_FEEDBACK_STALE"
            return True, 0.0
        distance = self.second_line_distance_m
        stop_limit = (self.config.slope_second_line_stop_distance_m+
                      self.config.slope_second_line_stop_tolerance_m)
        if distance > self.config.slope_second_line_slow_distance_m:
            self.state = "APPROACH_SECOND_LINE"
            return False, 0.0
        if distance > stop_limit:
            self.state = "DECELERATE_SECOND_LINE"
            return True, 1.0

        self.state = "STOP_SECOND_LINE"
        stationary = abs(inputs.current_speed_mps) <= \
            self.config.slope_stationary_speed_threshold_mps
        if not stationary:
            self.stationary_since = None
            self.hold_started = None
            self.vehicle_stationary = False
            return True, 0.0
        if self.stationary_since is None:
            self.stationary_since = inputs.now
        stationary_elapsed = inputs.now-self.stationary_since
        if stationary_elapsed < self.config.slope_stationary_confirm_sec:
            self.state = "CONFIRM_STATIONARY"
            return True, 0.0
        self.vehicle_stationary = True
        if self.hold_started is None:
            self.hold_started = inputs.now
            self.slope_stop_started = inputs.now
        if inputs.now-self.hold_started < self.config.slope_stop_duration_sec:
            self.state = "SLOPE_HOLD"
            return True, 0.0
        self.slope_stop_completed = True
        self.state = "SLOPE_COMPLETE"
        return False, 0.0

    def _new_aspect_sample(self, inputs):
        new = (inputs.traffic_aspect_sample_id is None or
               inputs.traffic_aspect_sample_id != self.last_aspect_sample_id)
        if new:
            self.last_aspect_sample_id = inputs.traffic_aspect_sample_id
        return new

    def _left_signal(self, inputs):
        if (not self.left_signal_confirmed and self._new_aspect_sample(inputs) and
                inputs.traffic_aspect_fresh and
                inputs.traffic_aspect == "GREEN_LEFT" and
                math.isfinite(inputs.traffic_aspect_confidence) and
                inputs.traffic_aspect_confidence >=
                self.config.exit_signal_min_confidence):
            self.left_signal_confirmed = True
        self.state = ("LEFT_SIGNAL_CONFIRMED" if self.left_signal_confirmed
                      else "LEFT_SIGNAL_MONITOR")
        return False, 0.0

    def _exit_signal(self, inputs):
        if self.exit_release_latched:
            self.state = "EXIT_RELEASED"
            return False, 0.0
        good = (inputs.traffic_aspect_fresh and
                inputs.traffic_aspect == "GREEN_DOWN" and
                inputs.rgb_green_down_verified and
                math.isfinite(inputs.traffic_aspect_confidence) and
                inputs.traffic_aspect_confidence >=
                self.config.exit_signal_min_confidence)
        if self._new_aspect_sample(inputs):
            self.exit_green_count = self.exit_green_count+1 if good else 0
        if good and self.exit_green_count >= int(
                self.config.exit_green_down_confirm_frames):
            self.exit_release_latched = True
            self.state = "EXIT_RELEASED"
            return False, 0.0
        self.state = ("EXIT_GREEN_DOWN_CONFIRMING" if good else
                      "EXIT_RED_X_STOP" if inputs.traffic_aspect == "RED_X" else
                      "EXIT_WAIT_SIGNAL")
        return True, 0.0

    def _intersection(self, inputs):
        # Direction is inferred from the same stable steering/path contract
        # used elsewhere. LEFT is permissive only for an established left
        # turn; G remains permissive for every intersection direction.
        self._shape(inputs.required_steering_deg, inputs.now)
        left_permitted = (inputs.traffic_light == "LEFT" and
                          self.path_shape == "TURN_LEFT")
        proceed_signal = inputs.traffic_light == "G" or left_permitted
        self.intersection_proceed_permitted = proceed_signal
        self.intersection_permission_source = (
            "G" if inputs.traffic_light == "G" else
            "RED_LEFT" if left_permitted else "NONE")
        distances = [float(v) for v in inputs.stop_line_distances_m
                     if math.isfinite(float(v))]
        target = self._target_distance(distances)
        if (target is None and self.commit_to_cross_latched and
                inputs.distance_valid and math.isfinite(inputs.distance_m) and
                self.commit_odometer_m is not None):
            travelled = max(0.0, float(inputs.distance_m)-self.commit_odometer_m)
            target = self.commit_line_distance_m-travelled
        self.intersection_distance_m = (float(target) if target is not None
                                        else math.nan)
        crossing_candidate = (min(distances, key=abs) if distances else None)
        if crossing_candidate is None:
            crossing_candidate = target
        newly_crossed = False
        if (not self.intersection_line_crossed and
                crossing_candidate is not None and
                crossing_candidate <= self.config.intersection_crossing_margin_m):
            self.intersection_line_crossed = True
            self.line_crossed_latched = True
            newly_crossed = True
        if self.intersection_line_crossed:
            # Expose the crossing edge once, then a distinct terminal state.
            # Both states deliberately relinquish the override so a later red
            # cannot create an abrupt stop inside the intersection.
            self.state = "LINE_CROSSED" if newly_crossed else \
                "INTERSECTION_COMPLETE"
            return False, 0.0

        if self.commit_to_cross_latched:
            self.state = "LATE_RED_COMMIT_TO_CROSS"
            self.late_red_action = "COMMIT_TO_CROSS"
            return True, 1.0

        new_traffic_sample = (inputs.traffic_sample_id is None or
                              inputs.traffic_sample_id !=
                              self.last_traffic_sample_id)
        if new_traffic_sample:
            self.last_traffic_sample_id = inputs.traffic_sample_id
            if proceed_signal:
                self.green_release_count += 1
            else:
                self.green_release_count = 0
        if self.red_stop_latched:
            if self.green_release_count >= int(self.config.green_release_frames):
                self.red_stop_latched = False
                self.green_release_latched = True
                self.state = "GREEN_PROCEED"
                return False, 0.0
            self.state = ("LATE_RED_STOP_BEFORE_LINE" if
                          self.late_red_action == "STOP_BEFORE_LINE" else
                          "RED_STOPPED")
            return True, 0.0

        if target is None:
            # Traffic lights are advisory unless a stop line is present.
            self.state = "INTERSECTION_NO_STOP_LINE"
            return False, 0.0
        if not inputs.stop_line_tf_valid:
            self.state = "INTERSECTION_DISTANCE_UNSAFE"
            self.mission_failure_reason = "STOP_LINE_TF_INVALID"
            return True, 0.0
        if proceed_signal:
            self.state = "GREEN_PROCEED"
            return False, 0.0
        # With a real stop line ahead, red, unknown, and low-confidence fused
        # states are all an immediate camera STOP. The selector then makes the
        # zero command effective; no advisory-only result is possible.
        self.red_stop_latched = inputs.traffic_light == "R"
        self.state = ("RED_STOPPED" if inputs.traffic_light == "R" else
                      "UNKNOWN_SAFE_STOP")
        return True, 0.0
        if target > self.config.intersection_detection_range_m:
            self.state = "INTERSECTION_APPROACH"
            return False, 0.0
        if proceed_signal:
            self.state = "GREEN_PROCEED"
            return False, 0.0
        if inputs.traffic_light == "UNKNOWN":
            self.state = "INTERSECTION_SIGNAL_MONITOR"
            return (True, 1.0) if target > self.config.intersection_target_stop_margin_m \
                else (True, 0.0)

        # A vehicle already stationary at/inside the target can latch the red
        # stop directly. A moving vehicle must still pass through the dynamic
        # feasibility branches; otherwise a dangerously late red would be
        # mislabeled as a successful target stop.
        if (target <= self.config.intersection_target_stop_margin_m and
                self.late_red_action == "CAN_STOP_AT_TARGET"):
            self.red_stop_latched = True
            self.late_red_action = "STOPPED_AT_TARGET"
            self.state = "RED_STOPPED"
            return True, 0.0
        if (target > self.config.intersection_target_stop_margin_m and
                self.late_red_action == "CAN_STOP_AT_TARGET"):
            self.state = "RED_DECELERATE"
            return True, 1.0
        if (target <= self.config.intersection_target_stop_margin_m and
                inputs.speed_valid and math.isfinite(inputs.current_speed_mps) and
                abs(inputs.current_speed_mps) <= 1.0e-3):
            self.red_stop_latched = True
            self.late_red_action = "STOPPED_AT_TARGET"
            self.state = "RED_STOPPED"
            return True, 0.0
        feasibility_valid = (
            inputs.encoder_valid and inputs.speed_valid and
            inputs.distance_valid and math.isfinite(inputs.distance_m) and
            math.isfinite(inputs.current_speed_mps) and
            inputs.current_speed_mps >= 0.0 and
            self.config.calibrated_deceleration_mps2 > 0.0)
        if not feasibility_valid:
            self.state = "INTERSECTION_SIGNAL_MONITOR"
            self.late_red_action = "STOP_FEASIBILITY_UNKNOWN"
            self.mission_failure_reason = "STOP_FEASIBILITY_UNKNOWN"
            return True, 0.0
        speed = float(inputs.current_speed_mps)
        self.available_to_target_m = (
            target-self.config.intersection_target_stop_margin_m)
        self.required_stop_m = (
            speed*self.config.total_control_latency_sec+
            speed*speed/(2.0*self.config.calibrated_deceleration_mps2)+
            self.config.stop_distance_safety_margin_m)
        self.can_stop_at_target = self.available_to_target_m >= self.required_stop_m
        available_before_line = target-self.config.minimum_line_clearance_m
        self.can_stop_before_line = available_before_line >= self.required_stop_m
        if self.can_stop_at_target:
            self.state = "LATE_RED_CAN_STOP_AT_TARGET"
            self.late_red_action = "CAN_STOP_AT_TARGET"
            return True, 1.0
        if self.can_stop_before_line:
            self.red_stop_latched = True
            self.state = "LATE_RED_STOP_BEFORE_LINE"
            self.late_red_action = "STOP_BEFORE_LINE"
            return True, 0.0
        self.commit_to_cross_latched = True
        self.commit_line_distance_m = target
        self.commit_odometer_m = float(inputs.distance_m)
        self.state = "LATE_RED_COMMIT_TO_CROSS"
        self.late_red_action = "COMMIT_TO_CROSS"
        return True, 1.0

    def _shape(self, steering, now):
        if not math.isfinite(steering):
            return self.path_shape
        magnitude = abs(steering)
        if (self.path_shape == "STRAIGHT" and
                magnitude <= self.config.straight_exit_deg):
            candidate = "STRAIGHT"
        elif magnitude <= self.config.straight_enter_deg:
            candidate = "STRAIGHT"
        elif magnitude >= self.config.turn_enter_deg:
            candidate = "TURN_RIGHT" if steering > 0.0 else "TURN_LEFT"
        else:
            candidate = self.path_shape
        if candidate != self.shape_candidate:
            self.shape_candidate, self.shape_since = candidate, now
        duration = 0.0 if self.shape_since is None else now-self.shape_since
        required = (self.config.straight_min_duration_sec
                    if candidate == "STRAIGHT" else
                    self.config.turn_min_duration_sec)
        if duration >= required:
            self.path_shape = candidate
            if candidate.startswith("TURN_"):
                self.turn_direction = candidate.removeprefix("TURN_")
        return self.path_shape

    def _acceleration(self, inputs):
        if not self.acceleration_armed:
            self.state = "WAIT_SIGN"
            if not inputs.sign_detected:
                return True, min(2.0, max(0.0, float(inputs.planner_drive)))
            self.acceleration_armed = True
        steering = abs(float(inputs.required_steering_deg))
        if not math.isfinite(steering):
            self.state = "HIGH_ACCEL_INPUT_INVALID"
            return True, 0.0
        if steering > 5.0:
            self.turn_detected = True
        if self.turn_detected:
            self.state = "HIGH_ACCEL_LOCKED_OUT"
            return True, 2.0
        self.state = "HIGH_ACCEL_ALLOWED"
        return True, 3.0
