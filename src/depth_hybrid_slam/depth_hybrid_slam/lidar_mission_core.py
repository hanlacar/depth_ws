"""Mode 5/7/9/10/11 state machines without ROS dependencies."""

from dataclasses import dataclass
import math

from .vehicle_kinematics import steering_from_curvature
import time


FAILURES = frozenset(("PLANNER_GIVE_UP", "NO_VALID_DETOUR",
                      "NO_FEASIBLE_DETOUR", "PATH_ABORT"))


@dataclass(frozen=True)
class ManeuverDecision:
    state: str
    owner: str
    stop: bool
    drive: float = 0.0
    wheel: int = 0
    branch: str = ""


class Mode5Avoidance:
    def __init__(self):
        self.state = "CSV_TRACKING"

    def reset(self):
        self.state = "CSV_TRACKING"

    def update(self, avoidance_required=False, hard_obstacle=False,
               planner_state="IDLE", path_valid=False,
               path_complete=False, rejoin_valid=False, local_wheel=0):
        failure = str(planner_state) in FAILURES
        if failure:
            if hard_obstacle or avoidance_required:
                self.state = "PLANNER_FAILED_HARD_STOP"
                return ManeuverDecision(self.state, "SAFETY", True)
            self.state = "CSV_TRACKING"
            return ManeuverDecision("PLANNER_FAILED_CSV_FALLBACK", "CSV", False)
        # Keep a failed hard stop fail-safe while a current hazard remains,
        # but do not latch a historical planner failure after the ROI clears.
        if self.state == "PLANNER_FAILED_HARD_STOP" and not hard_obstacle:
            self.state = ("STOP_FOR_PLANNING" if avoidance_required else
                          "CSV_TRACKING")
        if self.state == "CSV_TRACKING" and avoidance_required:
            self.state = "STOP_FOR_PLANNING"
        elif (self.state == "STOP_FOR_PLANNING" and
              not avoidance_required and not hard_obstacle):
            self.state = "CSV_TRACKING"
        elif self.state == "STOP_FOR_PLANNING" and path_valid:
            self.state = "LIDAR_PATH_TRACKING"
        elif self.state == "LIDAR_PATH_TRACKING" and path_complete:
            self.state = "CSV_REJOIN"
        elif self.state == "CSV_REJOIN" and rejoin_valid:
            self.state = "CSV_TRACKING"
        if self.state == "CSV_TRACKING":
            return ManeuverDecision(self.state, "CSV", False)
        if self.state == "STOP_FOR_PLANNING":
            return ManeuverDecision(self.state, "LIDAR", True)
        if self.state == "LIDAR_PATH_TRACKING":
            wheel = int(local_wheel)
            if abs(wheel) > 22:
                self.state = "PLANNER_FAILED_HARD_STOP"
                return ManeuverDecision(self.state, "SAFETY", True)
            return ManeuverDecision(self.state, "LIDAR", False, 1.0, wheel)
        return ManeuverDecision(self.state, "LIDAR", True)


class Mode5ObstacleLatch:
    """Require a continuous observation, then remember it through planning."""

    def __init__(self, confirmation_s=2.0):
        self.confirmation_s = float(confirmation_s)
        if self.confirmation_s < 2.0:
            raise ValueError("Mode 5 obstacle confirmation must be >= 2 s")
        self.first_seen_at = None
        self.latched = False

    def reset(self):
        self.first_seen_at = None
        self.latched = False

    def update(self, visible, mode=5, now=None):
        timestamp = time.monotonic() if now is None else float(now)
        if int(mode) != 5:
            self.reset()
            return False
        if self.latched:
            return True
        if not bool(visible):
            self.first_seen_at = None
            return False
        if self.first_seen_at is None:
            self.first_seen_at = timestamp
        if timestamp-self.first_seen_at >= self.confirmation_s:
            self.latched = True
        return self.latched


class StationaryConfirmation:
    """Confirm measured standstill before any path generation starts."""

    def __init__(self, duration_s=0.30, linear_limit_mps=0.03,
                 angular_limit_rps=0.03):
        self.duration_s = float(duration_s)
        self.linear_limit_mps = float(linear_limit_mps)
        self.angular_limit_rps = float(angular_limit_rps)
        if (self.duration_s <= 0.0 or self.linear_limit_mps < 0.0 or
                self.angular_limit_rps < 0.0):
            raise ValueError("invalid stationary confirmation policy")
        self.since = None

    def reset(self):
        self.since = None

    def update(self, linear_mps, angular_rps, fresh=True, now=None):
        timestamp = time.monotonic() if now is None else float(now)
        stopped = (
            bool(fresh) and linear_mps is not None and angular_rps is not None and
            abs(float(linear_mps)) <= self.linear_limit_mps and
            abs(float(angular_rps)) <= self.angular_limit_rps)
        if not stopped:
            self.since = None
            return False
        if self.since is None:
            self.since = timestamp
        return timestamp-self.since >= self.duration_s


def mode5_planning_distance_ready(distance_m, minimum_m=1.0):
    """Allow generation only before the front LiDAR gap falls below 1 m."""
    try:
        distance = float(distance_m)
        minimum = float(minimum_m)
    except (TypeError, ValueError):
        return False
    return math.isfinite(distance) and minimum >= 0.0 and distance >= minimum


def select_parking_branch(a_free, b_free):
    if bool(a_free):
        return "A"
    if bool(b_free):
        return "B"
    return ""


def parking_decision(mode, a_free, b_free, path_valid,
                     planner_state="IDLE", hard_obstacle=False):
    branch = select_parking_branch(a_free, b_free)
    prefix = "T" if int(mode) == 7 else "V"
    if not branch:
        return ManeuverDecision(prefix+"_WAIT_LIDAR_SLOT", "LIDAR", True)
    if hard_obstacle:
        return ManeuverDecision("PARKING_HARD_STOP", "SAFETY", True,
                                branch=branch)
    if str(planner_state) in FAILURES:
        return ManeuverDecision(prefix+"_CSV_FALLBACK", "CSV", False,
                                branch=branch)
    if path_valid:
        return ManeuverDecision(prefix+"_LIDAR_PATH", "LIDAR", False,
                                -1.0, 0, branch)
    return ManeuverDecision(prefix+"_SELECT_"+branch, "CSV", False,
                            branch=branch)


class Mode9Emergency:
    def __init__(self):
        self.state = "ACCEL_TRACKING"

    def reset(self):
        self.state = "ACCEL_TRACKING"

    def update(self, hard_obstacle):
        if hard_obstacle:
            self.state = "EMERGENCY_STOP"
            return ManeuverDecision(self.state, "SAFETY", True)
        # The LiDAR emergency input is already held until the 1.5 m corridor
        # has remained clear for one second.  Once that qualified input drops,
        # Mode 9 resumes its mandated fixed speed without an extra steering or
        # CSV-rejoin hold.
        if self.state != "ACCEL_TRACKING":
            self.state = "ACCEL_TRACKING"
        return ManeuverDecision(self.state, "CSV", False, 3.0)


class Mode9EmergencyLatch:
    """Release a confirmed Mode 9 stop only after continuously clear scans."""

    def __init__(self, clear_distance_m=1.5, clear_duration_s=1.0):
        self.clear_distance_m = float(clear_distance_m)
        self.clear_duration_s = float(clear_duration_s)
        if self.clear_distance_m < 1.0 or self.clear_duration_s <= 0.0:
            raise ValueError("invalid Mode 9 emergency clear policy")
        self.latched = False
        self.clear_since = None

    def reset(self):
        self.latched = False
        self.clear_since = None

    def update(self, confirmed_hard, nearest_m, scan_fresh, new_scan,
               now=None):
        timestamp = time.monotonic() if now is None else float(now)
        if bool(confirmed_hard) and bool(scan_fresh):
            self.latched = True
            self.clear_since = None
            return True
        if not self.latched:
            return False
        # A stale stream or a timer tick without a new scan is never evidence
        # that the obstacle was removed.
        if not bool(scan_fresh):
            self.clear_since = None
            return True
        if not bool(new_scan):
            return True
        distance = math.inf if nearest_m is None else float(nearest_m)
        clear = (not math.isfinite(distance) or
                 distance > self.clear_distance_m)
        if not clear:
            self.clear_since = None
            return True
        if self.clear_since is None:
            self.clear_since = timestamp
            return True
        if timestamp-self.clear_since >= self.clear_duration_s:
            self.reset()
        return self.latched


class SteeringSlowdownLatch:
    """Debounce measured steering before applying or clearing slowdown."""

    def __init__(self, threshold_deg=10.0, enter_duration_s=1.0,
                 exit_duration_s=1.0):
        self.threshold_deg = float(threshold_deg)
        self.enter_duration_s = float(enter_duration_s)
        self.exit_duration_s = float(exit_duration_s)
        if (self.threshold_deg <= 0.0 or self.enter_duration_s <= 0.0 or
                self.exit_duration_s <= 0.0):
            raise ValueError("invalid steering slowdown policy")
        self.active = False
        self.above_since = None
        self.below_since = None

    def reset(self):
        self.active = False
        self.above_since = None
        self.below_since = None

    def update(self, steering_deg, now=None):
        timestamp = time.monotonic() if now is None else float(now)
        # Missing/stale measured steering cannot establish either continuous
        # one-second interval.  Preserve an already active slowdown fail-safe.
        if steering_deg is None:
            self.above_since = None
            self.below_since = None
            return self.active
        above = abs(float(steering_deg)) >= self.threshold_deg
        if above:
            self.below_since = None
            if self.active:
                return True
            if self.above_since is None:
                self.above_since = timestamp
            if timestamp-self.above_since >= self.enter_duration_s:
                self.active = True
                self.above_since = None
            return self.active
        # Clear only below the threshold; exactly +/-10 remains slowdown
        # evidence as required by the production contract.
        self.above_since = None
        if not self.active:
            self.below_since = None
            return False
        if self.below_since is None:
            self.below_since = timestamp
        if timestamp-self.below_since >= self.exit_duration_s:
            self.active = False
            self.below_since = None
        return self.active


class ParkingManeuver:
    """Mode 7/10 selection, temporary tracking and CSV fallback."""

    def __init__(self, mode, direction_hold_s=3.0, rejoin_hold_s=3.0):
        self.mode = int(mode)
        self.direction_hold_s = max(3.0, float(direction_hold_s))
        self.rejoin_hold_s = max(3.0, float(rejoin_hold_s))
        self.state = "CSV_APPROACH"
        self.direction_hold_started_at = None
        self.rejoin_started_at = None

    def reset(self):
        self.state = "CSV_APPROACH"
        self.direction_hold_started_at = None
        self.rejoin_started_at = None

    def update(self, branch, path_valid=False, path_complete=False,
               planner_state="IDLE", hard_obstacle=False,
               drive=-1.0, wheel=0, rejoin_valid=False, now=None):
        timestamp = time.monotonic() if now is None else float(now)
        prefix = "T" if self.mode == 7 else "V"
        if hard_obstacle:
            return ManeuverDecision(prefix+"_HARD_STOP", "SAFETY", True,
                                    branch=branch)
        if str(planner_state) in FAILURES:
            self.state = "CSV_FALLBACK"
            return ManeuverDecision(prefix+"_CSV_FALLBACK", "CSV", False,
                                    branch=branch)
        if self.state == "CSV_APPROACH" and path_valid:
            # Acquire exclusive zero-speed ownership before fixing the local
            # path origin. This preserves the forward-to-reverse contract and
            # prevents one last CSV command from moving the vehicle after the
            # reverse plan has been anchored.
            self.state = "DIRECTION_CHANGE_HOLD"
            self.direction_hold_started_at = timestamp
        elif (self.state == "DIRECTION_CHANGE_HOLD" and path_valid and
              self.direction_hold_started_at is not None and
              timestamp-self.direction_hold_started_at >=
              self.direction_hold_s):
            self.state = "LIDAR_PATH_TRACKING"
        if self.state == "LIDAR_PATH_TRACKING" and path_complete:
            # Keep exclusive LiDAR ownership at zero speed until the strict
            # same-segment, forward-window validator confirms the CSV handoff.
            self.state = "CSV_REJOIN"
            self.rejoin_started_at = timestamp
        elif (self.state == "CSV_REJOIN" and rejoin_valid and
              self.rejoin_started_at is not None and
              timestamp-self.rejoin_started_at >= self.rejoin_hold_s):
            self.state = "COMPLETE"
        if self.state == "LIDAR_PATH_TRACKING":
            wheel = int(wheel)
            if abs(wheel) > 22:
                self.state = "CSV_FALLBACK"
                return ManeuverDecision(prefix+"_CSV_FALLBACK", "CSV", False,
                                        branch=branch)
            return ManeuverDecision(prefix+"_LIDAR_PATH", "LIDAR", False,
                                    drive, wheel, branch)
        if self.state == "DIRECTION_CHANGE_HOLD":
            return ManeuverDecision(prefix+"_DIRECTION_CHANGE_HOLD",
                                    "LIDAR", True, branch=branch)
        if self.state == "CSV_REJOIN":
            return ManeuverDecision(prefix+"_CSV_REJOIN", "LIDAR", True,
                                    branch=branch)
        return ManeuverDecision(prefix+"_"+self.state, "CSV", False,
                                branch=branch)


@dataclass(frozen=True)
class RejoinCandidate:
    segment: str
    index: int
    distance_m: float
    heading_error_deg: float
    required_steering_deg: float
    direction: int
    road_valid: bool = True


def bounded_rejoin(candidates, active_segment, current_index,
                   direction, forward_window=120, max_distance_m=0.8,
                   max_heading_deg=35.0):
    """Select only forward, same-segment/direction candidates."""
    valid = []
    for value in candidates:
        if value.segment != active_segment:
            continue
        if value.index < int(current_index):
            continue
        if value.index > int(current_index)+int(forward_window):
            continue
        if int(value.direction) != int(direction):
            continue
        if value.distance_m > max_distance_m:
            continue
        if abs(value.heading_error_deg) > max_heading_deg:
            continue
        if abs(value.required_steering_deg) > 22.0 or not value.road_valid:
            continue
        valid.append(value)
    return min(valid, key=lambda item: (item.distance_m, item.index),
               default=None)


def route_rejoin_candidates(route, active_segment, current_index, pose,
                            forward_window=120, wheelbase_m=0.73,
                            road_valid=True):
    """Build steering-aware candidates only from the current forward segment."""
    if float(wheelbase_m) != 0.73:
        raise ValueError("rejoin must use wheelbase 0.73 m")
    x, y, yaw = (float(value) for value in pose)
    start = max(0, int(current_index))
    stop = min(len(route), start+int(forward_window)+1)
    output = []
    for index in range(start, stop):
        point = route[index]
        if str(point.segment_id) != str(active_segment):
            continue
        # CSV yaw is the vehicle/body yaw for both longitudinal directions;
        # direction is validated independently below.  Adding pi here would
        # reject every legitimate reverse-to-reverse handoff.
        travel_yaw = float(point.yaw)
        heading = math.atan2(
            math.sin(travel_yaw-yaw), math.cos(travel_yaw-yaw))
        dx, dy = float(point.x)-x, float(point.y)-y
        distance = math.hypot(dx, dy)
        lateral = -math.sin(yaw)*dx+math.cos(yaw)*dy
        steering = steering_from_curvature(
            2.0*lateral/max(distance*distance, 1.0e-6), wheelbase_m)
        output.append(RejoinCandidate(
            str(point.segment_id), index, distance, math.degrees(heading),
            steering, int(point.direction), bool(road_valid)))
    return tuple(output)


class Mode11ExitGate:
    """Five-second A/B vote; default B and ignore late opposite signals."""

    def __init__(self, hold_s=5.0, stale_s=0.5, confirmations=60,
                 decision_ratio=0.75):
        self.hold_s = max(5.0, float(hold_s))
        self.stale_s = float(stale_s)
        self.confirmations = max(1, int(confirmations))
        self.decision_ratio = float(decision_ratio)
        self.started_at = None
        self.committed = None
        self.commit_source = ""
        self.signal_at = None
        self.votes = {"A": 0, "B": 0, "UNKNOWN": 0}

    def reset(self):
        self.__init__(self.hold_s, self.stale_s, self.confirmations,
                      self.decision_ratio)

    def enter(self, now):
        if self.started_at is None:
            self.started_at = float(now)

    def observe(self, signal, now):
        if self.committed is not None:
            return
        value = str(signal).strip().upper()
        value = value if value in ("1", "2", "A", "B") else "UNKNOWN"
        route = "A" if value in ("1", "A") else \
            "B" if value in ("2", "B") else "UNKNOWN"
        self.votes[route] += 1
        self.signal_at = float(now)

    def evaluate(self, now):
        self.enter(now)
        elapsed = float(now)-self.started_at
        if self.committed is not None:
            return ManeuverDecision("MODE11_COMMITTED", "CSV", False,
                                    branch=self.committed)
        if elapsed < self.hold_s:
            return ManeuverDecision("MODE11_5S_HOLD", "MISSION", True)
        fresh = (self.signal_at is not None and
                 float(now)-self.signal_at <= self.stale_s)
        valid = self.votes["A"]+self.votes["B"]
        winner = "B" if self.votes["B"] > self.votes["A"] else "A"
        confidence = self.votes[winner]/valid if valid else 0.0
        confirmed = (fresh and valid >= self.confirmations and
                     confidence >= self.decision_ratio)
        self.committed = winner if confirmed else "B"
        self.commit_source = "CAMERA" if confirmed else "DEFAULT"
        return ManeuverDecision("MODE11_COMMIT_"+self.committed,
                                "CSV", False, branch=self.committed)
