"""ROS-independent source selection, stopping and Ackermann control logic."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .route_network import RoutePoint, normalize_angle


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class ControlResult:
    source: str
    drive: float
    wheel: int
    nav_nearest: int
    csv_nearest: int
    nav_target: int
    csv_target: int
    path_distance_difference: float
    heading_difference_deg: float
    cross_track_error: float
    stop_state: str
    stop_source: str
    stop_distance_m: float | None
    direction: str
    csv_speed: float
    speed_reason: str
    steering_saturated: bool


class ProgressTracker:
    def __init__(self, start_index=0, back_window=10, forward_window=200):
        self.start_index = max(0, int(start_index))
        self.back_window = int(back_window)
        self.forward_window = int(forward_window)
        self.last = None

    def nearest(self, points, pose):
        if not points:
            raise ValueError("path is empty")
        if self.start_index >= len(points):
            raise ValueError("start_index outside path")
        if self.last is None:
            candidates = range(self.start_index, len(points))
        else:
            candidates = range(
                max(self.start_index, self.last - self.back_window),
                min(len(points), self.last + self.forward_window + 1))
        index = min(candidates, key=lambda i: math.hypot(
            points[i].x - pose.x, points[i].y - pose.y))
        previous = self.start_index if self.last is None else self.last
        self.last = max(previous, index)
        return self.last


class SegmentProgressTracker(ProgressTracker):
    """Monotonic nearest search constrained to the active CSV segment."""
    def __init__(self, start_index=0, back_window=10, forward_window=200,
                 transition_window=20):
        super().__init__(start_index, back_window, forward_window)
        self.transition_window = int(transition_window)
        self.ranges_signature = None
        self.ranges = []
        self.active_range = None

    def _build_ranges(self, points):
        # The node replaces the assembled route list when a branch changes.
        # Object identity therefore detects a rebuild without allocating a
        # 3,202-element tuple on every 20 Hz control cycle.
        signature = (id(points), len(points))
        if signature == self.ranges_signature:
            return
        self.ranges_signature = signature
        self.ranges = []
        start = 0
        for index in range(1, len(points) + 1):
            if index == len(points) or points[index].segment_id != points[start].segment_id:
                self.ranges.append((start, index - 1, points[start].segment_id))
                start = index
        self.active_range = None

    def _range_for_index(self, index):
        return next(i for i, (start, end, _) in enumerate(self.ranges)
                    if start <= index <= end)

    def nearest(self, points, pose):
        if not points:
            raise ValueError("path is empty")
        if self.start_index >= len(points):
            raise ValueError("start_index outside path")
        self._build_ranges(points)
        distance = lambda index: math.hypot(
            points[index].x - pose.x, points[index].y - pose.y)
        if self.last is None:
            index = min(range(self.start_index, len(points)), key=distance)
            self.last = index
            self.active_range = self._range_for_index(index)
            return index

        range_start, range_end, _ = self.ranges[self.active_range]
        candidates = range(
            max(range_start, self.last - self.back_window),
            min(range_end, self.last + self.forward_window) + 1)
        index = min(candidates, key=distance)
        index = max(self.last, index)

        # A segment transition is considered only near its physical end. This
        # prevents a nearby later pass from stealing progress at intersections.
        if (self.active_range + 1 < len(self.ranges) and
                index >= range_end - self.transition_window):
            next_start, next_end, _ = self.ranges[self.active_range + 1]
            next_candidates = range(
                next_start, min(next_end, next_start + self.transition_window) + 1)
            next_index = min(next_candidates, key=distance)
            if distance(next_index) <= distance(index):
                index = next_index
                self.active_range += 1
        self.last = max(self.last, index)
        return self.last


def lookahead_index(points, start, distance_m):
    distance = 0.0
    for index in range(start + 1, len(points)):
        distance += math.hypot(points[index].x - points[index - 1].x,
                               points[index].y - points[index - 1].y)
        if distance >= distance_m:
            return index
    return len(points) - 1


def path_heading(points, index):
    if len(points) < 2:
        return points[index].yaw
    if index + 1 < len(points):
        a, b = points[index], points[index + 1]
    else:
        a, b = points[index - 1], points[index]
    return math.atan2(b.y - a.y, b.x - a.x)


def project_path(points, nearest, pose, radius=3):
    """Closest local polyline projection and its geometric heading."""
    if len(points) == 1:
        return points[0].x, points[0].y, points[0].yaw
    first = max(0, nearest - radius)
    last = min(len(points) - 2, nearest + radius)
    candidates = []
    for index in range(first, last + 1):
        a, b = points[index], points[index + 1]
        dx, dy = b.x - a.x, b.y - a.y
        denominator = dx * dx + dy * dy
        fraction = 0.0 if denominator <= 1e-12 else max(0.0, min(
            1.0, ((pose.x - a.x) * dx + (pose.y - a.y) * dy) / denominator))
        x, y = a.x + fraction * dx, a.y + fraction * dy
        candidates.append((math.hypot(x - pose.x, y - pose.y),
                           x, y, math.atan2(dy, dx)))
    _, x, y, heading = min(candidates, key=lambda item: item[0])
    return x, y, heading


class DeviationHysteresis:
    def __init__(self, nav_to_csv_position_m=1.5,
                 csv_to_nav_position_m=0.75,
                 nav_to_csv_heading_deg=20.0,
                 csv_to_nav_heading_deg=10.0,
                 nav_to_csv_samples=10, csv_to_nav_samples=20):
        if csv_to_nav_position_m >= nav_to_csv_position_m:
            raise ValueError("CSV->Nav2 position threshold must be smaller")
        if csv_to_nav_heading_deg >= nav_to_csv_heading_deg:
            raise ValueError("CSV->Nav2 heading threshold must be smaller")
        self.enter_pos = float(nav_to_csv_position_m)
        self.exit_pos = float(csv_to_nav_position_m)
        self.enter_heading = float(nav_to_csv_heading_deg)
        self.exit_heading = float(csv_to_nav_heading_deg)
        self.enter_samples = int(nav_to_csv_samples)
        self.exit_samples = int(csv_to_nav_samples)
        self.source = "NAV2"
        self.bad_count = 0
        self.good_count = 0

    def update(self, position_m, heading_deg):
        if self.source == "NAV2":
            bad = position_m >= self.enter_pos or abs(heading_deg) >= self.enter_heading
            self.bad_count = self.bad_count + 1 if bad else 0
            self.good_count = 0
            if self.bad_count >= self.enter_samples:
                self.source = "CSV"
                self.bad_count = 0
        else:
            good = position_m <= self.exit_pos and abs(heading_deg) <= self.exit_heading
            self.good_count = self.good_count + 1 if good else 0
            self.bad_count = 0
            if self.good_count >= self.exit_samples:
                self.source = "NAV2"
                self.good_count = 0
        return self.source


@dataclass(frozen=True)
class StopEvent:
    key: str
    x: float
    y: float
    source: str
    csv_index: int | None = None
    nav_index: int | None = None


def merge_stop_events(csv_points, nav_points, nav_stop_indices,
                      merge_distance_m=0.75):
    events = []
    for point in csv_points:
        if point.event == "STOP_LINE":
            events.append(StopEvent(
                f"CSV_STOP:{point.route_index}", point.x, point.y,
                "CSV_STOP_LINE", csv_index=point.route_index))
    for before, after in zip(csv_points, csv_points[1:]):
        if before.direction != after.direction:
            direction_source = f"{before.direction}_TO_{after.direction}"
            # A direction change and STOP_LINE at the same physical location
            # are one stop. Do not merge two separate direction changes merely
            # because a compact manoeuvre puts them near each other.
            match = next((event for event in events
                          if event.source == "CSV_STOP_LINE" and math.hypot(
                              event.x - after.x,
                              event.y - after.y) <= merge_distance_m), None)
            if match is None:
                events.append(StopEvent(
                    f"DIRECTION:{after.route_index}", after.x, after.y,
                    direction_source, csv_index=after.route_index))
            else:
                events[events.index(match)] = StopEvent(
                    match.key, match.x, match.y,
                    match.source + "+" + direction_source,
                    after.route_index, match.nav_index)
    for index in sorted(set(nav_stop_indices)):
        if index < 0 or index >= len(nav_points):
            continue
        point = nav_points[index]
        match = next((event for event in events if math.hypot(
            event.x - point.x, event.y - point.y) <= merge_distance_m), None)
        if match is None:
            events.append(StopEvent(
                f"NAV_STOP:{index}", point.x, point.y, "NAV2_STOP",
                nav_index=index))
        else:
            events[events.index(match)] = StopEvent(
                match.key, match.x, match.y,
                match.source + "+NAV2_STOP", match.csv_index, index)
    return events


class StopController:
    RUNNING = "RUNNING"
    APPROACH_STOP = "APPROACH_STOP"
    HOLD_STOP = "HOLD_STOP"
    RELEASE = "RELEASE"

    def __init__(self, hold_seconds=1.0, trigger_distance_m=0.30,
                 approach_distance_m=1.0):
        self.hold_seconds = float(hold_seconds)
        self.trigger_distance_m = float(trigger_distance_m)
        self.approach_distance_m = max(
            float(approach_distance_m), self.trigger_distance_m)
        if self.hold_seconds <= 0.0:
            raise ValueError("stop hold_seconds must be positive")
        if self.trigger_distance_m <= 0.0:
            raise ValueError("stop trigger_distance_m must be positive")
        self.active = None
        self.active_until = None
        self.consumed = set()
        self.state = self.RUNNING
        self.source = "NONE"
        self.distance_m = None

    @staticmethod
    def _progress_close(event, csv_nearest, nav_nearest):
        csv_close = (event.csv_index is not None and
                     event.csv_index >= csv_nearest - 2 and
                     event.csv_index <= csv_nearest + 12)
        nav_close = (event.nav_index is not None and
                     event.nav_index >= nav_nearest - 1 and
                     event.nav_index <= nav_nearest + 5)
        return csv_close or nav_close

    def update(self, now, pose, events, csv_nearest, nav_nearest):
        if self.active is not None:
            # ROS clocks and a 20 Hz test clock are floating point values;
            # tolerate sub-nanosecond representation error at exactly 1.0 s.
            if now + 1e-9 < self.active_until:
                self.state = self.HOLD_STOP
                self.source = self.active.source
                self.distance_m = math.hypot(
                    self.active.x - pose.x, self.active.y - pose.y)
                return self.state
            self.consumed.add(self.active.key)
            self.active = None
            self.active_until = None
            self.state = self.RELEASE
            self.source = "NONE"
            self.distance_m = None
            return self.state
        candidates = []
        for event in events:
            if event.key in self.consumed:
                continue
            if not self._progress_close(event, csv_nearest, nav_nearest):
                continue
            distance = math.hypot(event.x - pose.x, event.y - pose.y)
            if distance <= self.approach_distance_m:
                candidates.append((distance, event))
        if candidates:
            distance, event = min(candidates, key=lambda item: item[0])
            self.source = event.source
            self.distance_m = distance
            if distance <= self.trigger_distance_m:
                self.active = event
                self.active_until = now + self.hold_seconds
                self.state = self.HOLD_STOP
            else:
                self.state = self.APPROACH_STOP
            return self.state
        self.state = self.RUNNING
        self.source = "NONE"
        self.distance_m = None
        return self.state


class BranchSelector:
    STAGES = ("START", "T", "PARALLEL", "END")

    def __init__(self):
        self.values = {stage: "A" for stage in self.STAGES}
        self.latched = {stage: False for stage in self.STAGES}

    def request(self, stage, value):
        stage = str(stage).strip().upper()
        value = str(value).strip().upper()
        if stage not in self.values or value not in {"A", "B"}:
            return False
        if self.latched[stage]:
            return False
        self.values[stage] = value
        return True

    def request_or_default(self, stage, value):
        # Missing, timeout, UNKNOWN and INVALID all deterministically select A.
        return self.request(stage, "B" if str(value).strip().upper() == "B" else "A")

    def latch(self, stage):
        stage = str(stage).strip().upper()
        if stage in self.latched:
            self.latched[stage] = True


def steering_command(pose, target, direction, wheelbase=0.73,
                     right_limit=-22.0, left_limit=22.0):
    motion_yaw = pose.yaw if direction == "F" else normalize_angle(pose.yaw + math.pi)
    bearing = math.atan2(target.y - pose.y, target.x - pose.x)
    alpha = normalize_angle(bearing - motion_yaw)
    distance = max(1e-6, math.hypot(target.x - pose.x, target.y - pose.y))
    raw = math.degrees(math.atan2(2.0 * wheelbase * math.sin(alpha), distance))
    if direction == "R":
        raw = -raw
    clamped = max(right_limit, min(left_limit, raw))
    return int(round(clamped)), abs(raw - clamped) > 1e-9


def select_drive(direction, csv_speed, wheel, stopping=False):
    return select_drive_with_reason(
        direction, csv_speed, wheel, stopping)[0]


def select_drive_with_reason(direction, csv_speed, wheel, stopping=False):
    if stopping:
        return 0.0, "STOP"
    if direction == "R":
        return -1.0, "REVERSE"
    drive = float(csv_speed)
    if abs(wheel) >= 10:
        limited = min(drive, 1.0)
        reason = ("STEERING_LIMIT_ABS_WHEEL_GE_10"
                  if limited < drive else "CSV_DRIVE_LEVEL")
        return limited, reason
    return drive, "CSV_DRIVE_LEVEL"


def commanded_synthetic_speed(route_drive, travel_speed_mps):
    """Map drive levels to slow visualization motion, never to physical m/s."""
    drive = float(route_drive)
    base = float(travel_speed_mps)
    if abs(drive) < 1e-9:
        return 0.0
    if drive < 0.0:
        return base
    return base * min(1.0, drive / 2.0)


SPECIAL_MODE_SECTIONS = {
    4: "INTERSECTION",
    6: "INTERSECTION",
    7: "T_PARK",
    9: "ACCELERATION",
    10: "PARALLEL_PARK",
}


def special_section(point):
    """Return the route contract's CSV-only section for this observation."""
    segment = point.segment_id.upper()
    if point.mode == 7 or segment in {"T_FOWORD", "T_A", "T_B"}:
        return "T_PARK"
    if point.mode == 10 or segment in {"V_A", "V_B"}:
        return "PARALLEL_PARK"
    return SPECIAL_MODE_SECTIONS.get(point.mode, "NORMAL")


def is_csv_only_point(point):
    return special_section(point) != "NORMAL"


def is_parking_point(point):
    return special_section(point) in {"T_PARK", "PARALLEL_PARK"}


class HybridFollower:
    def __init__(self, **parameters):
        self.lookahead_m = float(parameters.get("lookahead_m", 1.0))
        self.nav_tracker = ProgressTracker(parameters.get("start_index", 0))
        self.csv_tracker = SegmentProgressTracker(0)
        self.hysteresis = DeviationHysteresis(
            parameters.get("nav_to_csv_position_m", 1.5),
            parameters.get("csv_to_nav_position_m", 0.75),
            parameters.get("nav_to_csv_heading_deg", 20.0),
            parameters.get("csv_to_nav_heading_deg", 10.0),
            parameters.get("nav_to_csv_samples", 10),
            parameters.get("csv_to_nav_samples", 20))
        self.stop_controller = StopController(
            parameters.get("stop_hold_seconds", 1.0),
            parameters.get("stop_trigger_distance_m", 0.30),
            parameters.get("stop_approach_distance_m", 1.0))
        self.stop_merge_distance_m = float(
            parameters.get("stop_merge_distance_m", 0.75))
        self.events_signature = None
        self.events = []

    def command(self, now, pose, nav_points, csv_points, nav_stop_indices=()):
        nav_near = self.nav_tracker.nearest(nav_points, pose)
        csv_near = self.csv_tracker.nearest(csv_points, pose)
        nav_target = lookahead_index(nav_points, nav_near, self.lookahead_m)
        csv_target = lookahead_index(csv_points, csv_near, self.lookahead_m)
        nav_t, csv_t = nav_points[nav_target], csv_points[csv_target]
        nav_x, nav_y, nav_heading = project_path(
            nav_points, nav_near, pose)
        csv_x, csv_y, csv_heading = project_path(
            csv_points, csv_near, pose)
        difference = math.hypot(nav_x - csv_x, nav_y - csv_y)
        heading_difference = math.degrees(normalize_angle(
            nav_heading - csv_heading))
        fallback_source = self.hysteresis.update(difference, heading_difference)
        signature = (id(csv_points), id(nav_points), tuple(nav_stop_indices))
        if signature != self.events_signature:
            self.events = merge_stop_events(
                csv_points, nav_points, nav_stop_indices,
                self.stop_merge_distance_m)
            self.events_signature = signature
        stop_state = self.stop_controller.update(
            now, pose, self.events, csv_near, nav_near)
        csv_point = csv_points[csv_near]
        if is_csv_only_point(csv_point):
            source = "CSV_ONLY"
        elif fallback_source == "CSV":
            source = "CSV_FALLBACK"
        else:
            source = "NAV2"
        chosen_points = nav_points if source == "NAV2" else csv_points
        chosen_target = nav_target if source == "NAV2" else csv_target
        wheel, saturated = steering_command(
            pose, chosen_points[chosen_target], csv_point.direction)
        drive, speed_reason = select_drive_with_reason(
            csv_point.direction, csv_point.drive_level, wheel,
            stopping=stop_state == StopController.HOLD_STOP)
        if drive == 0.0:
            wheel = 0
        selected_near = nav_near if source == "NAV2" else csv_near
        selected_point = chosen_points[selected_near]
        cross_track = math.hypot(selected_point.x - pose.x,
                                 selected_point.y - pose.y)
        return ControlResult(
            source, drive, wheel, nav_near, csv_near, nav_target, csv_target,
            difference, abs(heading_difference), cross_track, stop_state,
            self.stop_controller.source, self.stop_controller.distance_m,
            csv_point.direction, csv_point.drive_level, speed_reason,
            saturated)
