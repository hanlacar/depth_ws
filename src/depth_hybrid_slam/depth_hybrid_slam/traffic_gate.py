"""ROS-independent, route-progress-latched intersection traffic gate."""

from dataclasses import dataclass


APPROACH = "APPROACH"
STOP_LINE_HOLD = "STOP_LINE_HOLD"
RELEASE_PENDING = "RELEASE_PENDING"
INTERSECTION_COMMITTED = "INTERSECTION_COMMITTED"
INTERSECTION_EXITED = "INTERSECTION_EXITED"


@dataclass(frozen=True)
class IntersectionProgress:
    """Route-local evidence emitted by the bounded CSV follower."""

    valid: bool = False
    mode: int = 0
    segment_id: str = ""
    direction: int = 0
    route_index: int = -1
    progress_m: float = 0.0
    stop_segment_id: str = ""
    stop_direction: int = 0
    stop_route_index: int = -1
    stop_index: int = -1
    stop_progress_m: float = 0.0
    exit_route_index: int = -1
    exit_progress_m: float = 0.0


@dataclass(frozen=True)
class TrafficDecision:
    stop: bool
    state: str
    aspect: str
    event: str = ""
    active: bool = False
    traffic_stop_allowed: bool = True
    committed: bool = False


class IntersectionTrafficGate:
    """Gate Modes 4/6 until their CSV STOP_LINE is actually crossed.

    The commit boundary uses only the follower's bounded monotonic cursor and
    metric progress. Camera observations can release/re-hold the vehicle
    before that boundary, but can never clear the commit latch.
    """

    def __init__(self, unknown_hold_s=3.0, traffic_timeout_s=0.5,
                 minimum_stop_s=3.0,
                 commit_margin_m=0.35, intersection_modes=(4, 6, 8)):
        self.unknown_hold_s = float(unknown_hold_s)
        self.minimum_stop_s = float(minimum_stop_s)
        self.traffic_timeout_s = float(traffic_timeout_s)
        self.commit_margin_m = float(commit_margin_m)
        self.intersection_modes = frozenset(int(mode)
                                            for mode in intersection_modes)
        if (self.unknown_hold_s < 0.0 or self.minimum_stop_s < 0.0 or
                self.traffic_timeout_s <= 0.0 or
                self.commit_margin_m <= 0.0 or not self.intersection_modes):
            raise ValueError("invalid intersection traffic gate configuration")
        self.reset()

    def reset(self):
        self.state = APPROACH
        self.active_mode = None
        self.active_stop_route_index = None
        self.active_stop_index = None
        self.last_route_index = None
        self.high_water_progress_m = None
        self.unknown_since = None
        self.stop_started = None
        self.release_cause = None

    def _signal(self, aspect, signal_age_s, mode):
        value = str(aspect).strip().upper()
        fresh = 0.0 <= float(signal_age_s) <= self.traffic_timeout_s
        if not fresh:
            return "STALE", "TRAFFIC_STALE"
        if value in ("R+GREEN_LEFT", "Y+GREEN_LEFT"):
            return "R", "RED_PRESENT_HOLD"
        if value in ("R+G", "Y+G"):
            return "R", "RED_PRESENT_HOLD"
        if value == "GREEN_LEFT":
            if int(mode) == 8:
                return "G", "MODE8_GREEN_LEFT_RELEASE"
            return "UNKNOWN", ""
        if value in ("G", "GREEN_CIRCLE", "GREEN_DOWN", "GREEN_OTHER"):
            return "G", ""
        if value in ("R", "Y", "RED", "RED_X", "YELLOW"):
            return "R", ""
        return "UNKNOWN", ""

    def _decision(self, stop, aspect, event="", active=True):
        committed = self.state == INTERSECTION_COMMITTED
        return TrafficDecision(
            bool(stop), self.state, aspect, event, active,
            traffic_stop_allowed=not committed, committed=committed)

    def _activate(self, progress, now):
        self.active_mode = int(progress.mode)
        self.active_stop_route_index = int(progress.stop_route_index)
        self.active_stop_index = int(progress.stop_index)
        self.last_route_index = int(progress.route_index)
        self.high_water_progress_m = float(progress.progress_m)
        self.state = STOP_LINE_HOLD
        self.stop_started = float(now)
        self.unknown_since = None

    def _exited(self, progress):
        if self.active_mode is None:
            return False
        if not progress.valid or int(progress.mode) != self.active_mode:
            return True
        return (int(progress.route_index) > int(progress.exit_route_index) or
                float(progress.progress_m) >
                float(progress.exit_progress_m)+1.0e-6)

    def _monotonic_progress(self, progress):
        """Accept only same-lineage, non-backtracking follower evidence."""
        if (not progress.valid or int(progress.mode) != self.active_mode or
                int(progress.stop_route_index) !=
                self.active_stop_route_index or
                int(progress.stop_index) != self.active_stop_index or
                str(progress.segment_id) != str(progress.stop_segment_id) or
                int(progress.direction) != int(progress.stop_direction)):
            return False
        route_index = int(progress.route_index)
        metric = float(progress.progress_m)
        if self.last_route_index is not None and route_index < self.last_route_index:
            return False
        if (self.high_water_progress_m is not None and
                metric+1.0e-6 < self.high_water_progress_m):
            return False
        self.last_route_index = route_index
        self.high_water_progress_m = metric
        return True

    def evaluate(self, progress, stop_line_active, aspect, signal_age_s, now):
        """Advance one intersection using CSV progress and fresh R/G only."""
        now = float(now)
        signal, signal_event = self._signal(
            aspect, signal_age_s, progress.mode)

        if self.active_mode is not None and self._exited(progress):
            exited_mode = self.active_mode
            self.reset()
            self.state = INTERSECTION_EXITED
            return self._decision(
                False, signal, f"EXITED mode={exited_mode}", active=True)

        mode_supported = progress.valid and int(progress.mode) in \
            self.intersection_modes
        if self.active_mode is None:
            if self.state == INTERSECTION_EXITED:
                self.state = APPROACH
            if not mode_supported:
                return self._decision(False, signal, active=False)
            if bool(stop_line_active):
                self._activate(progress, now)
                event = f"STOP_LINE_HOLD traffic={signal}"
            else:
                return self._decision(False, signal, "", active=True)
        else:
            event = ""

        if self.state == STOP_LINE_HOLD:
            stopped_for = max(0.0, now-float(self.stop_started))
            if signal == "STALE":
                self.unknown_since = None
                return self._decision(True, signal, "TRAFFIC_STALE_HOLD")
            if signal == "R":
                self.unknown_since = None
                return self._decision(
                    True, signal, "; ".join(filter(None, (
                        event, "RED_HOLD"))))
            if signal == "G":
                self.unknown_since = None
                if stopped_for < self.minimum_stop_s:
                    return self._decision(
                        True, signal,
                        f"MINIMUM_STOP elapsed={stopped_for:.3f}s")
                self.state = RELEASE_PENDING
                self.release_cause = "GREEN"
                return self._decision(
                    False, signal, "; ".join(filter(None, (
                        signal_event, "GREEN_RELEASE", "RELEASE_PENDING"))))
            if self.unknown_since is None:
                self.unknown_since = now
                return self._decision(
                    True, signal, "UNKNOWN_HOLD_STARTED")
            elapsed = max(0.0, now-self.unknown_since)
            if (elapsed < self.unknown_hold_s or
                    stopped_for < self.minimum_stop_s):
                return self._decision(
                    True, signal,
                    f"UNKNOWN_HOLD elapsed={elapsed:.3f}s")
            self.state = RELEASE_PENDING
            self.release_cause = "UNKNOWN_TIMEOUT"
            return self._decision(
                False, signal,
                f"UNKNOWN_RELEASE_AFTER_{self.unknown_hold_s:.1f}S; "
                "RELEASE_PENDING")

        if self.state == RELEASE_PENDING:
            if self._monotonic_progress(progress):
                crossed = (
                    int(progress.route_index) > int(progress.stop_route_index)
                    and float(progress.progress_m) >
                    float(progress.stop_progress_m)+self.commit_margin_m)
                if crossed:
                    self.state = INTERSECTION_COMMITTED
                    event = (
                        "STOP_LINE_CROSSED; COMMITTED "
                        f"stop_index={progress.stop_index} "
                        f"route_index={progress.route_index} "
                        f"progress_m={progress.progress_m:.3f}")
                    if signal == "R":
                        event += ("; RED_IGNORED_AFTER_COMMIT; "
                                  "CSV_TRACKING CONTINUES")
                    return self._decision(
                        False, signal, event)
            if signal in ("R", "STALE"):
                self.state = STOP_LINE_HOLD
                self.unknown_since = None
                self.release_cause = None
                return self._decision(
                    True, signal, ("RED_REHOLD_BEFORE_LINE" if signal == "R"
                                   else "TRAFFIC_STALE_REHOLD_BEFORE_LINE"))
            if signal == "G":
                self.release_cause = "GREEN"
                return self._decision(False, signal)
            if self.release_cause == "UNKNOWN_TIMEOUT":
                return self._decision(
                    False, signal, "UNKNOWN_TIMEOUT_RELEASE_CONTINUES")
            self.state = STOP_LINE_HOLD
            self.unknown_since = now
            self.release_cause = None
            return self._decision(
                True, signal, "UNKNOWN_REHOLD_BEFORE_LINE")

        if self.state == INTERSECTION_COMMITTED:
            event = ("RED_IGNORED_AFTER_COMMIT; CSV_TRACKING CONTINUES"
                     if signal == "R" else "")
            return self._decision(False, signal, event)

        return self._decision(False, signal, active=mode_supported)
