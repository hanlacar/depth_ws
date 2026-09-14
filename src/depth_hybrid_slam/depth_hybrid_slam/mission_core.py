"""Mission state machine consuming camera_ws results; no detector is duplicated."""

from .models import MissionDecision
from .traffic_gate import IntersectionProgress, IntersectionTrafficGate


class MissionMachine:
    def __init__(self, ramp_sections=(), acceleration_sections=(),
                 finish_sections=(), traffic_timeout_s=0.3,
                 min_traffic_confidence=0.65, ramp_pitch_deg=15.0,
                 ramp_duration_s=0.5, ramp_stop_s=4.0,
                 sign_confirmations=3, stop_distance_m=0.7,
                 slowdown_distance_m=2.5, unknown_hold_s=3.0,
                 intersection_commit_margin_m=0.35,
                 intersection_modes=(4, 6, 8)):
        self.ramp_sections = set(ramp_sections)
        self.acceleration_sections = set(acceleration_sections)
        self.finish_sections = set(finish_sections)
        self.traffic_timeout = float(traffic_timeout_s)
        self.min_traffic_confidence = float(min_traffic_confidence)
        self.ramp_pitch = float(ramp_pitch_deg)
        self.ramp_duration = float(ramp_duration_s)
        self.ramp_stop = float(ramp_stop_s)
        self.sign_confirmations = int(sign_confirmations)
        self.stop_distance = float(stop_distance_m)
        self.slowdown_distance = float(slowdown_distance_m)
        self.pitch_since = None
        self.ramp_stop_until = None
        self.ramp_latched = set()
        self.sign_count = 0
        self.accel_slow_latched = False
        self.previous_section = None
        self.traffic_gate = IntersectionTrafficGate(
            unknown_hold_s=unknown_hold_s,
            traffic_timeout_s=self.traffic_timeout,
            commit_margin_m=intersection_commit_margin_m,
            intersection_modes=intersection_modes)
        self.last_traffic_decision = None

    def update(self, value):
        section = str(value.section_id)
        if section != self.previous_section:
            self.pitch_since = None
            self.sign_count = 0
            self.accel_slow_latched = False
            self.previous_section = section

        traffic_fresh = (value.traffic_age <= self.traffic_timeout and
                         value.traffic_confidence >= self.min_traffic_confidence)
        aspect = str(value.traffic_aspect).upper()
        state = str(value.traffic_state).upper()

        gate_aspect = "UNKNOWN"
        diagnostic_fresh = (
            value.traffic_diagnostics_age <= self.traffic_timeout)
        red_present = diagnostic_fresh and value.traffic_red_present
        green_present = diagnostic_fresh and value.traffic_green_present
        if value.traffic_red_override or red_present:
            gate_aspect = "R"
        elif value.traffic_green_override:
            gate_aspect = "G"
        elif traffic_fresh:
            gate_aspect = ("G" if aspect == "GREEN_DOWN" else
                           "R" if aspect == "RED_X" or "YELLOW" in aspect else
                           state if state in ("R", "G") else "UNKNOWN")
        progress = (value.intersection_progress
                    if isinstance(value.intersection_progress,
                                  IntersectionProgress)
                    else IntersectionProgress())
        red_override = value.traffic_red_override or red_present
        if not red_override and traffic_fresh and green_present:
            if aspect == "GREEN_LEFT":
                gate_aspect = "GREEN_LEFT"
        elif (not red_override and traffic_fresh and
              aspect == "GREEN_LEFT"):
            gate_aspect = "GREEN_LEFT"
        gate_signal_age = (
            0.0 if red_override or value.traffic_green_override else
            value.traffic_age)
        traffic = self.traffic_gate.evaluate(
            progress, value.csv_stop_line_active, gate_aspect,
            gate_signal_age, value.now)
        self.last_traffic_decision = traffic
        if traffic.active:
            return MissionDecision(
                traffic.state, traffic.stop,
                "" if not traffic.stop else "TRAFFIC_"+traffic.state,
                0.0 if traffic.stop else 2.0,
                "STOP" if traffic.stop else "GO",
                traffic.traffic_stop_allowed, traffic.committed)

        if section in self.ramp_sections:
            if value.uphill_detected and value.pitch_deg >= self.ramp_pitch:
                if self.pitch_since is None:
                    self.pitch_since = value.now
            else:
                self.pitch_since = None
            confirmed = (self.pitch_since is not None and
                         value.now-self.pitch_since >= self.ramp_duration)
            if confirmed and section not in self.ramp_latched:
                self.ramp_latched.add(section)
                self.ramp_stop_until = value.now+self.ramp_stop
            if self.ramp_stop_until is not None and value.now < self.ramp_stop_until:
                return MissionDecision("RAMP_HOLD", True, "RAMP_4S_HOLD",
                                       0.0, "STOP")

        if section in self.acceleration_sections:
            self.sign_count = self.sign_count+1 if value.sign_detected else 0
            active = self.sign_count >= self.sign_confirmations
            if active and abs(value.steering_deg) > 5.0:
                self.accel_slow_latched = True
            if active:
                speed = 2.0 if self.accel_slow_latched else 3.0
                name = "ACCEL_LATCHED_SLOW" if self.accel_slow_latched else "ACCEL_FAST"
                return MissionDecision(name, False, "", speed, "GO")

        return MissionDecision("CRUISE", False, "", 2.0, "CONDITIONAL")
