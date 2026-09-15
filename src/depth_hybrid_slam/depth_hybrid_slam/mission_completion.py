"""Mission-specific completion layered above immutable route completion."""

import json


MISSION_MODES = frozenset((2, 4, 5, 6, 7, 8, 9, 10, 11))


class MissionCompletionTracker:
    def __init__(self, slope_threshold_deg=4.5,
                 slope_confirmation_s=0.5):
        self.slope_threshold_deg = float(slope_threshold_deg)
        self.slope_confirmation_s = float(slope_confirmation_s)
        if (self.slope_threshold_deg <= 0.0 or
                self.slope_confirmation_s <= 0.0):
            raise ValueError("slope thresholds must be positive")
        self.mode = None
        self.route_complete = set()
        self.invalid_modes = set()
        self.mode2_stop_started = None
        self.mode2_slope_started = None
        self.mode2_stop_4s_done = False
        self.mode2_slope_seen = False
        self.mode2_pitch_checked = False
        self.mode2_stop_pitch_deg = None
        self.intersection_exited = set()
        self.mode5_rejoins = 0
        self.slot_seen = {7: False, 10: False}
        self.parking_completed = {7: False, 10: False}
        self.parking_rejoined = {7: False, 10: False}
        self.parking_source = {7: "", 10: ""}
        self.mode9_hard_pending = False
        self.mode9_hard_distance = None
        self.mode9_emergency_applied = False
        self.mode11_stop_started = None
        self.mode11_stop_5s_done = False
        self.mode11_branch = ""
        self.mode11_source = ""
        self._events = []
        self._completed_latch = set()

    def _event(self, event, mode=None, **values):
        payload = {"event": str(event), "mode": int(self.mode if mode is None else mode)}
        payload.update(values)
        self._events.append(payload)

    def set_mode(self, mode):
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            return
        previous = self.mode
        if previous == 5 and mode != 5 and self.mode5_rejoins < 2:
            self.invalid_modes.add(5)
            self._event("MODE5_MISSION_FAILED", 5,
                        avoidance_rejoined_count=self.mode5_rejoins)
        self.mode = mode

    def observe_route_status(self, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return
        completed = {int(mode) for mode in
                     value.get("route_complete_modes", ())}
        self.route_complete.update(completed)
        # In the one-front-LiDAR production graph, LiDAR selects a measured
        # A/B slot and the corresponding recorded CSV parking case executes.
        # Reaching the end of that selected case is the CSV rejoin proof.
        for mode in (7, 10):
            if (mode in completed and self.slot_seen[mode] and
                    self.parking_source[mode] == "CSV_FALLBACK"):
                self.parking_completed[mode] = True
                self.parking_rejoined[mode] = True

    def observe_intersection_event(self, event):
        text = str(event)
        if "EXITED mode=" not in text:
            return
        try:
            mode = int(text.split("EXITED mode=", 1)[1].split()[0])
        except (ValueError, IndexError):
            return
        if mode in (4, 6, 8):
            self.intersection_exited.add(mode)
            self._event("INTERSECTION_EXITED", mode)

    def observe_lidar_safety(self, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return
        mode = int(value.get("mode", self.mode or -1))
        if mode in (7, 10) and (value.get("slot_a") or value.get("slot_b")):
            if not self.slot_seen[mode]:
                slot = "A" if value.get("slot_a") else "B"
                self._event("PARKING_SLOT_SEEN", mode, slot=slot)
            self.slot_seen[mode] = True
        distance = value.get("distance_m")
        if (mode == 9 and value.get("hard_stop") and distance is not None and
                float(distance) <= 1.00):
            self.mode9_hard_pending = True
            self.mode9_hard_distance = float(distance)

    def observe_maneuver(self, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return
        mode = int(value.get("mode", self.mode or -1))
        event = str(value.get("event", ""))
        state = str(value.get("state", ""))
        if mode == 5 and event == "AVOIDANCE_REJOINED":
            self.mode5_rejoins += 1
            self._event("AVOIDANCE_REJOINED", 5,
                        count=self.mode5_rejoins)
        if mode in (7, 10):
            if event == "PARKING_CSV_FALLBACK" or state.endswith("CSV_FALLBACK"):
                self.parking_source[mode] = "CSV_FALLBACK"
                if str(value.get("branch", "")) in ("A", "B"):
                    self.slot_seen[mode] = True
            if event == "PARKING_CSV_REJOINED" or state in ("T_COMPLETE", "V_COMPLETE"):
                newly_completed = not self.parking_completed[mode]
                self.parking_completed[mode] = True
                self.parking_rejoined[mode] = True
                self.parking_source[mode] = str(value.get("source", "LIDAR"))
                if newly_completed:
                    self._event("PARKING_COMPLETE", mode,
                                source=self.parking_source[mode])
        if mode == 11 and (event == "MODE11_BRANCH_COMMITTED" or
                           state.startswith("MODE11_COMMIT_")):
            branch = str(value.get("branch", state[-1:]))
            if not self.mode11_branch:
                self.mode11_branch = branch
                self.mode11_source = str(value.get("source", "DEFAULT"))
                self._event("MODE11_BRANCH_COMMITTED", 11,
                            branch=self.mode11_branch,
                            source=self.mode11_source)

    def observe_csv_parking_complete(self, mode):
        mode = int(mode)
        if mode in (7, 10) and self.parking_source[mode] == "CSV_FALLBACK":
            newly_completed = not self.parking_completed[mode]
            self.parking_completed[mode] = True
            self.parking_rejoined[mode] = True
            if newly_completed:
                self._event("PARKING_COMPLETE", mode, source="CSV_FALLBACK")

    @staticmethod
    def _zero(command):
        return abs(float(command)) <= 1.0e-6

    def tick(self, now, cmd_drive, *, stop_waypoint_active=False,
             pitch_deg=0.0, pitch_valid=False, maneuver_state=""):
        now = float(now)
        if self.mode == 2:
            if ((stop_waypoint_active or self.mode2_stop_started is not None) and
                    self._zero(cmd_drive)):
                if self.mode2_stop_started is None:
                    self.mode2_stop_started = now
                    self._event("MODE2_STOP_BEGIN", 2)
                valid_pitch = (bool(pitch_valid) and
                               abs(float(pitch_deg)) >=
                               self.slope_threshold_deg)
                self.mode2_pitch_checked = True
                self.mode2_stop_pitch_deg = (
                    float(pitch_deg) if bool(pitch_valid) else None)
                if valid_pitch:
                    if self.mode2_slope_started is None:
                        self.mode2_slope_started = now
                    if (not self.mode2_slope_seen and
                            now-self.mode2_slope_started >=
                            self.slope_confirmation_s):
                        self.mode2_slope_seen = True
                        self._event(
                            "MODE2_SLOPE_VALID", 2,
                            pitch_deg=self.mode2_stop_pitch_deg,
                            pitch_valid=True)
                else:
                    self.mode2_slope_started = None
                if now-self.mode2_stop_started >= 4.0 and not self.mode2_stop_4s_done:
                    self.mode2_stop_4s_done = True
                    self._event("MODE2_STOP_4S_DONE", 2)
                    self.mode2_slope_seen = valid_pitch
                    self._event(
                        "MODE2_STOP_EVALUATED", 2,
                        pitch_deg=self.mode2_stop_pitch_deg,
                        pitch_valid=bool(pitch_valid),
                        threshold_deg=self.slope_threshold_deg,
                        passed=valid_pitch)
                    if not self.mode2_slope_seen:
                        self._event(
                            "MODE2_SLOPE_INVALID", 2,
                            pitch_deg=self.mode2_stop_pitch_deg,
                            pitch_valid=bool(pitch_valid))
            elif not self.mode2_stop_4s_done:
                self.mode2_stop_started = None
                self.mode2_slope_started = None
                self.mode2_slope_seen = False
        if self.mode == 9 and self.mode9_hard_pending and self._zero(cmd_drive):
            if not self.mode9_emergency_applied:
                self.mode9_emergency_applied = True
                self._event("MODE9_EMERGENCY_STOP_APPLIED", 9,
                            distance_m=self.mode9_hard_distance)
        if (self.mode == 11 and self._zero(cmd_drive) and
                (str(maneuver_state) == "MODE11_5S_HOLD" or
                 self.mode11_stop_started is not None)):
            if self.mode11_stop_started is None:
                self.mode11_stop_started = now
            if now-self.mode11_stop_started >= 5.0 and not self.mode11_stop_5s_done:
                self.mode11_stop_5s_done = True
                self._event("MODE11_STOP_5S_DONE", 11)
        self._emit_completions()

    def mission_complete(self, mode):
        mode = int(mode)
        if mode == 2:
            return self.mode2_stop_4s_done and self.mode2_slope_seen
        if mode in (4, 6, 8):
            return mode in self.intersection_exited
        if mode == 5:
            return self.mode5_rejoins >= 2 and mode not in self.invalid_modes
        if mode in (7, 10):
            parking_done = (self.parking_completed[mode] and
                            self.parking_rejoined[mode])
            return self.slot_seen[mode] and parking_done
        if mode == 9:
            return self.mode9_emergency_applied
        if mode == 11:
            return (self.mode11_stop_5s_done and
                    self.mode11_branch in ("A", "B"))
        return True

    def mode_complete(self, mode):
        return int(mode) in self.route_complete and self.mission_complete(mode)

    def _emit_completions(self):
        for mode in range(1, 12):
            if self.mode_complete(mode) and mode not in self._completed_latch:
                self._completed_latch.add(mode)
                self._event("MISSION_MODE_COMPLETE", mode)

    def force_mode2_stop(self):
        return (self.mode == 2 and self.mode2_stop_started is not None and
                not self.mode2_stop_4s_done)

    def force_mode11_stop(self):
        return (self.mode == 11 and self.mode11_stop_started is not None and
                not self.mode11_stop_5s_done)

    def drain_events(self):
        values, self._events = tuple(self._events), []
        return values

    def status(self):
        mission = [mode for mode in range(1, 12)
                   if self.mission_complete(mode)]
        completed = [mode for mode in range(1, 12)
                     if self.mode_complete(mode)]
        return {
            "mode": self.mode, "criterion": "ROUTE_AND_MISSION",
            "route_complete_modes": sorted(self.route_complete),
            "mission_complete_modes": mission,
            "completed_modes": completed,
            "route_complete": self.mode in self.route_complete if self.mode else False,
            "mission_complete": self.mission_complete(self.mode) if self.mode else False,
            "mode_complete": self.mode_complete(self.mode) if self.mode else False,
            "course_complete": completed == list(range(1, 12)),
            "invalid_modes": sorted(self.invalid_modes),
            "mode2_stop_4s_done": self.mode2_stop_4s_done,
            "mode2_slope_seen": self.mode2_slope_seen,
            "mode2_pitch_checked": self.mode2_pitch_checked,
            "mode2_stop_pitch_deg": self.mode2_stop_pitch_deg,
            "mode5_avoidance_rejoined_count": self.mode5_rejoins,
            "parking_slot_seen": self.slot_seen,
            "parking_source": self.parking_source,
            "mode9_emergency_seen": self.mode9_emergency_applied,
            "mode11_branch": self.mode11_branch,
            "mode11_source": self.mode11_source,
        }
