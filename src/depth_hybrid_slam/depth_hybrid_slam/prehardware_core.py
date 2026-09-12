"""ROS-independent pre-hardware autonomy policies and geometry."""

from dataclasses import dataclass
import math


VALID_BRANCHES = frozenset(("A", "B"))
VALID_DRIVE_STAGES = frozenset((-1.0, 0.0, 1.0, 2.0, 3.0))
ROAD_VALID = frozenset(("VALID_ROAD_AND_LANE", "VALID_ROAD_ONLY"))
ROAD_ADVISORY = frozenset(("CAMERA_UNAVAILABLE", "INVALID_GEOMETRY"))


@dataclass(frozen=True)
class BranchDecision:
    branch: str
    stop: bool
    state: str


class BranchSelector:
    """Default-A selector with a mode-11 decision hold and timeout."""

    def __init__(self, command_timeout_s=3.0, mode_11_wait_s=3.0):
        self.command_timeout_s = float(command_timeout_s)
        self.mode_11_wait_s = float(mode_11_wait_s)
        if self.command_timeout_s <= 0.0 or self.mode_11_wait_s <= 0.0:
            raise ValueError("branch timeouts must be positive")
        self.mode = None
        self.mode_11_entered_at = None
        self.command_branch = None
        self.command_at = None
        self.selected_branch = "A"

    def set_mode(self, mode, now):
        try:
            value = int(str(mode).strip())
        except ValueError:
            value = None
        if value == 11 and self.mode != 11:
            self.mode_11_entered_at = float(now)
            self.command_branch = None
            self.command_at = None
        elif value != 11:
            self.mode_11_entered_at = None
        self.mode = value

    def set_command(self, branch, now):
        value = str(branch).strip().upper()
        self.command_branch = value if value in VALID_BRANCHES else None
        self.command_at = float(now)

    def evaluate(self, now):
        now = float(now)
        fresh = (self.command_branch in VALID_BRANCHES and
                 self.command_at is not None and
                 0.0 <= now-self.command_at <= self.command_timeout_s)
        if self.mode == 11 and self.mode_11_entered_at is not None:
            if fresh and self.command_at >= self.mode_11_entered_at:
                self.selected_branch = self.command_branch
                return BranchDecision(
                    self.selected_branch, False, "MODE11_BRANCH_SELECTED")
            if now-self.mode_11_entered_at < self.mode_11_wait_s:
                return BranchDecision(
                    self.selected_branch, True, "MODE11_WAIT_BRANCH")
            self.selected_branch = "A"
            return BranchDecision("A", False, "MODE11_TIMEOUT_DEFAULT_A")
        if fresh:
            self.selected_branch = self.command_branch
            return BranchDecision(self.selected_branch, False, "BRANCH_SELECTED")
        self.selected_branch = "A"
        state = "INVALID_DEFAULT_A" if self.command_at is not None else \
            "NO_INPUT_DEFAULT_A"
        return BranchDecision("A", False, state)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    state: str
    elapsed_s: float


class StopWaypointMachine:
    """One-shot minimum hold for each CSV STOP_LINE waypoint."""

    def __init__(self, minimum_hold_s=3.0):
        self.minimum_hold_s = float(minimum_hold_s)
        if self.minimum_hold_s < 3.0:
            raise ValueError("STOP_LINE hold must be at least 3 seconds")
        self.active_key = None
        self.started_at = None
        self.completed = set()

    def reset(self):
        self.active_key = None
        self.started_at = None
        self.completed.clear()

    def update(self, waypoint_key, reached, traffic_stop, now):
        now = float(now)
        if reached and waypoint_key and waypoint_key not in self.completed and \
                self.active_key is None:
            self.active_key = str(waypoint_key)
            self.started_at = now
        if self.active_key is None:
            return StopDecision(False, "IDLE", 0.0)
        elapsed = max(0.0, now-float(self.started_at))
        if elapsed < self.minimum_hold_s:
            return StopDecision(True, "MINIMUM_3S_HOLD", elapsed)
        if bool(traffic_stop):
            return StopDecision(True, "WAIT_TRAFFIC_RELEASE", elapsed)
        self.completed.add(self.active_key)
        self.active_key = None
        self.started_at = None
        return StopDecision(False, "RELEASED", elapsed)


@dataclass(frozen=True)
class LocalPathResult:
    valid: bool
    state: str
    points: tuple
    route_index: int


class RouteLocalConnector:
    """Extract route progress forward without a global nearest-point search."""

    def __init__(self, forward_min_m=0.5, forward_max_m=4.0):
        self.forward_min_m = float(forward_min_m)
        self.forward_max_m = float(forward_max_m)
        if not 0.0 <= self.forward_min_m < self.forward_max_m:
            raise ValueError("invalid local path forward window")
        self.last_index = None

    def reset(self):
        self.last_index = None

    def extract(self, route, pose, route_index):
        if not route:
            return LocalPathResult(False, "PATH_UNAVAILABLE", (), -1)
        index = int(route_index)
        if index < 0 or index >= len(route):
            return LocalPathResult(False, "INDEX_INVALID", (), index)
        if self.last_index is not None and index < self.last_index:
            return LocalPathResult(False, "INDEX_BACKTRACK", (), index)
        self.last_index = index
        px, py, pyaw = (float(value) for value in pose)
        if not all(math.isfinite(value) for value in (px, py, pyaw)):
            return LocalPathResult(False, "POSE_INVALID", (), index)
        cosine, sine = math.cos(pyaw), math.sin(pyaw)
        distance = 0.0
        selected = []
        previous = route[index]
        for point in route[index:]:
            x, y, yaw = (float(value) for value in point[:3])
            if point is not previous:
                distance += math.hypot(x-float(previous[0]),
                                       y-float(previous[1]))
            previous = point
            if distance+1.0e-9 < self.forward_min_m:
                continue
            if distance-1.0e-9 > self.forward_max_m:
                break
            dx, dy = x-px, y-py
            local_x = cosine*dx+sine*dy
            local_y = -sine*dx+cosine*dy
            local_yaw = math.atan2(math.sin(yaw-pyaw), math.cos(yaw-pyaw))
            selected.append((local_x, local_y, local_yaw))
        if len(selected) < 2:
            return LocalPathResult(False, "INSUFFICIENT_FORWARD_PATH", (), index)
        return LocalPathResult(True, "READY", tuple(selected), index)


@dataclass(frozen=True)
class BehaviorDecision:
    drive: float
    wheel: int
    stop: bool
    state: str
    road_verified: bool


def select_behavior(candidate_drive, candidate_wheel, candidate_stop,
                    candidate_fresh, localization_state, localization_fresh,
                    connector_state, connector_fresh, validator_state,
                    validator_fresh, mission_stop=False, camera_stop=False,
                    branch_stop=False, lidar_stop=False):
    """Select the only internal /slam_* command, failing closed on path faults."""
    if not candidate_fresh:
        return BehaviorDecision(0.0, 0, True, "CANDIDATE_STALE", False)
    if (not localization_fresh or
            str(localization_state).split(":", 1)[0] not in
            ("TRACKING", "RELOCALIZED")):
        return BehaviorDecision(0.0, 0, True, "LOCALIZATION_UNAVAILABLE", False)
    if not connector_fresh or connector_state != "READY":
        return BehaviorDecision(0.0, 0, True, "LOCAL_PATH_UNAVAILABLE", False)
    if any((candidate_stop, mission_stop, camera_stop, branch_stop, lidar_stop)):
        return BehaviorDecision(0.0, 0, True, "UPSTREAM_STOP", False)
    try:
        drive = float(candidate_drive)
        wheel = int(candidate_wheel)
    except (TypeError, ValueError):
        return BehaviorDecision(0.0, 0, True, "COMMAND_INVALID", False)
    if drive not in VALID_DRIVE_STAGES or abs(wheel) > 22:
        return BehaviorDecision(0.0, 0, True, "COMMAND_INVALID", False)
    state = str(validator_state)
    if not validator_fresh:
        return BehaviorDecision(drive, wheel, False,
                                "CAMERA_VALIDATION_UNAVAILABLE", False)
    if state in ROAD_VALID:
        return BehaviorDecision(drive, wheel, False, state, True)
    if state == "DEGRADED_LANE_UNCERTAIN":
        limited = math.copysign(min(abs(drive), 1.0), drive)
        return BehaviorDecision(limited, wheel, False, state, True)
    if state in ROAD_ADVISORY:
        return BehaviorDecision(drive, wheel, False, state+"_CSV_FALLBACK", False)
    return BehaviorDecision(0.0, 0, True, "ROAD_VALIDATION_"+state, False)


@dataclass(frozen=True)
class LidarDecision:
    drive: float
    wheel_internal: int
    stop: bool
    avoidance_active: bool
    parking_active: bool
    front_obstacle_0_5m: bool
    gps_path_safe: bool
    state: str


class LidarPolicy:
    """Minimal mode 5/7/9/10 policy using base/sensor-forward points."""

    def __init__(self, half_width_m=0.60, general_distance_m=1.5,
                 mode9_distance_m=3.0, rear_stop_distance_m=0.5,
                 avoidance_wheel_deg=18, clear_samples=3):
        self.half_width = float(half_width_m)
        self.general_distance = float(general_distance_m)
        self.mode9_distance = float(mode9_distance_m)
        self.rear_stop_distance = float(rear_stop_distance_m)
        self.avoidance_wheel = int(avoidance_wheel_deg)
        self.clear_samples = int(clear_samples)
        if not 0 < self.avoidance_wheel <= 22 or self.clear_samples < 1:
            raise ValueError("invalid LiDAR policy configuration")
        self.avoiding = False
        self.last_wheel = 0
        self.clear_count = 0

    def _roi(self, points, distance):
        return tuple((float(x), float(y)) for x, y in (points or ())
                     if math.isfinite(float(x)) and math.isfinite(float(y))
                     and 0.0 < float(x) <= distance
                     and abs(float(y)) <= self.half_width)

    def evaluate(self, mode, front_points, rear_points,
                 front_fresh=True, rear_fresh=True):
        try:
            mode = int(str(mode).strip())
        except ValueError:
            mode = -1
        front = self._roi(front_points, self.general_distance) \
            if front_fresh else ()
        close = self._roi(front_points, 0.5) if front_fresh else ()
        front_3m = self._roi(front_points, self.mode9_distance) \
            if front_fresh else ()
        front_obstacle = bool(close) or not front_fresh
        gps_safe = front_fresh and not bool(front)

        if mode == 5:
            if not front_fresh:
                self.avoiding = True
                return LidarDecision(0.0, 0, True, True, False,
                                     True, False, "MODE5_SCAN_STALE")
            left = [point for point in front if point[1] >= 0.08]
            right = [point for point in front if point[1] <= -0.08]
            center = [point for point in front if abs(point[1]) < 0.08]
            blocked = bool(front)
            both = (bool(left) and bool(right)) or bool(center)
            if blocked:
                self.avoiding = True
                self.clear_count = 0
                if both:
                    self.last_wheel = 0
                    return LidarDecision(0.0, 0, True, True, False,
                                         front_obstacle, False,
                                         "MODE5_BOTH_BLOCKED")
                self.last_wheel = (-self.avoidance_wheel if left else
                                   self.avoidance_wheel)
                return LidarDecision(1.0, self.last_wheel, False, True, False,
                                     front_obstacle, False,
                                     "MODE5_AVOID_RIGHT" if left else
                                     "MODE5_AVOID_LEFT")
            if self.avoiding:
                self.clear_count += 1
                if self.clear_count < self.clear_samples:
                    return LidarDecision(1.0, self.last_wheel, False, True,
                                         False, False, True,
                                         "MODE5_CLEAR_CONFIRM")
            self.avoiding = False
            self.last_wheel = 0
            self.clear_count = 0
            return LidarDecision(0.0, 0, False, False, False,
                                 front_obstacle, True, "MODE5_CSV_REJOIN")

        self.avoiding = False
        self.clear_count = 0
        if mode in (7, 10):
            if not rear_fresh:
                return LidarDecision(0.0, 0, True, False, True,
                                     front_obstacle, gps_safe,
                                     "PARKING_REAR_SCAN_STALE")
            rear = self._roi(rear_points, 1.5)
            stop = any(math.hypot(x, y) <= self.rear_stop_distance
                       for x, y in rear)
            return LidarDecision(0.0 if stop else -1.0, 0, stop, False, True,
                                 front_obstacle, gps_safe,
                                 "PARKING_REAR_STOP" if stop else
                                 "PARKING_REVERSE")
        if mode == 9:
            if not front_fresh:
                return LidarDecision(0.0, 0, True, False, False,
                                     True, False, "MODE9_SCAN_STALE")
            stop = bool(front_3m)
            return LidarDecision(0.0 if stop else 3.0, 0, stop, False, False,
                                 front_obstacle, gps_safe,
                                 "MODE9_SUDDEN_OBSTACLE" if stop else
                                 "MODE9_CLEAR")
        return LidarDecision(0.0, 0, False, False, False,
                             front_obstacle, gps_safe,
                             "MONITORING" if front_fresh else "WAIT_SCAN")
