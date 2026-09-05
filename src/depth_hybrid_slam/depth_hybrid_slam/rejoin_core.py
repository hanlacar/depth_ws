"""Forward Dubins rejoin planning and fail-closed occupancy-grid checks."""

from dataclasses import dataclass
import math

from .geometry import wrap_angle
from .models import PathPose, RejoinPlan


def _mod2pi(value):
    return value % (2.0*math.pi)


def _dubins_words(alpha, beta, distance):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha-beta)
    words = []

    def add(name, values):
        if values is not None and all(math.isfinite(v) and v >= 0.0 for v in values):
            words.append((sum(values), name, values))

    p2 = 2+distance*distance-2*cab+2*distance*(sa-sb)
    if p2 >= 0:
        tmp = math.atan2(cb-ca, distance+sa-sb)
        add("LSL", (_mod2pi(-alpha+tmp), math.sqrt(p2),
                    _mod2pi(beta-tmp)))
    p2 = 2+distance*distance-2*cab+2*distance*(sb-sa)
    if p2 >= 0:
        tmp = math.atan2(ca-cb, distance-sa+sb)
        add("RSR", (_mod2pi(alpha-tmp), math.sqrt(p2),
                    _mod2pi(-beta+tmp)))
    p2 = -2+distance*distance+2*cab+2*distance*(sa+sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca-cb, distance+sa+sb)-math.atan2(-2.0, p)
        add("LSR", (_mod2pi(-alpha+tmp), p, _mod2pi(-beta+tmp)))
    p2 = distance*distance-2+2*cab-2*distance*(sa+sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca+cb, distance-sa-sb)-math.atan2(2.0, p)
        add("RSL", (_mod2pi(alpha-tmp), p, _mod2pi(beta-tmp)))
    value = (6-distance*distance+2*cab+2*distance*(sa-sb))/8.0
    if abs(value) <= 1.0:
        p = _mod2pi(2*math.pi-math.acos(value))
        t = _mod2pi(alpha-math.atan2(ca-cb, distance-sa+sb)+p/2.0)
        add("RLR", (t, p, _mod2pi(alpha-beta-t+p)))
    value = (6-distance*distance+2*cab+2*distance*(-sa+sb))/8.0
    if abs(value) <= 1.0:
        p = _mod2pi(2*math.pi-math.acos(value))
        t = _mod2pi(-alpha-math.atan2(ca-cb, distance+sa-sb)+p/2.0)
        add("LRL", (t, p, _mod2pi(beta-alpha-t+p)))
    return sorted(words)


def dubins_path(start, goal, radius, step_m=0.10, direction=1):
    """Return the shortest sampled Dubins path; direction=-1 is reverse-only.

    Reverse-only is a valid no-cusp subset of the Reeds-Shepp family and is
    considered only on route sections explicitly marked for reverse travel.
    """
    radius, step_m = float(radius), float(step_m)
    if radius <= 0.0 or step_m <= 0.0:
        raise ValueError("radius and step must be positive")
    travel_start_yaw = start.yaw if direction > 0 else wrap_angle(start.yaw+math.pi)
    travel_goal_yaw = goal.yaw if direction > 0 else wrap_angle(goal.yaw+math.pi)
    dx, dy = goal.x-start.x, goal.y-start.y
    theta = math.atan2(dy, dx)
    normalized = math.hypot(dx, dy)/radius
    words = _dubins_words(_mod2pi(travel_start_yaw-theta),
                          _mod2pi(travel_goal_yaw-theta), normalized)
    if not words:
        return ()
    _, kinds, lengths = words[0]
    x, y, travel_yaw = start.x, start.y, travel_start_yaw
    result = [PathPose(x, y, start.yaw, direction)]
    for kind, normalized_length in zip(kinds, lengths):
        remaining = normalized_length*radius
        while remaining > 1.0e-9:
            distance = min(step_m, remaining)
            if kind == "S":
                x += distance*math.cos(travel_yaw)
                y += distance*math.sin(travel_yaw)
            else:
                curvature = (1.0/radius) * (1.0 if kind == "L" else -1.0)
                updated = travel_yaw+curvature*distance
                x += (math.sin(updated)-math.sin(travel_yaw))/curvature
                y += (-math.cos(updated)+math.cos(travel_yaw))/curvature
                travel_yaw = updated
            remaining -= distance
            body_yaw = travel_yaw if direction > 0 else wrap_angle(travel_yaw-math.pi)
            result.append(PathPose(x, y, wrap_angle(body_yaw), direction))
    result[-1] = PathPose(goal.x, goal.y, goal.yaw, direction)
    return tuple(result)


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: tuple
    occupied_threshold: int = 50
    allow_unknown: bool = False

    def value(self, x, y):
        dx, dy = x-self.origin_x, y-self.origin_y
        cosine, sine = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        local_x = cosine*dx+sine*dy
        local_y = -sine*dx+cosine*dy
        column, row = int(math.floor(local_x/self.resolution)), int(math.floor(local_y/self.resolution))
        if column < 0 or row < 0 or column >= self.width or row >= self.height:
            return None
        return int(self.data[row*self.width+column])

    def footprint_free(self, pose, length_m, width_m, margin_m):
        half_l = 0.5*float(length_m)+float(margin_m)
        half_w = 0.5*float(width_m)+float(margin_m)
        spacing = max(self.resolution*0.75, 0.025)
        longitudinal = _samples(-half_l, half_l, spacing)
        lateral = _samples(-half_w, half_w, spacing)
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        for forward in longitudinal:
            for side in lateral:
                value = self.value(pose.x+cosine*forward-sine*side,
                                   pose.y+sine*forward+cosine*side)
                if value is None or (value < 0 and not self.allow_unknown) or \
                        value >= self.occupied_threshold:
                    return False
        return True


def _samples(low, high, step):
    count = max(1, int(math.ceil((high-low)/step)))
    return [low+(high-low)*index/count for index in range(count+1)]


class RejoinPlanner:
    def __init__(self, minimum_turning_radius_m=1.8068134,
                 candidate_search_distance_m=15.0,
                 candidate_spacing_m=0.50, heading_weight=0.75,
                 candidate_heading_deg=80.0, sample_step_m=0.10,
                 vehicle_length_m=1.40, vehicle_width_m=0.80,
                 safety_margin_m=0.15, allow_unknown=False):
        self.radius = float(minimum_turning_radius_m)
        self.search_distance = float(candidate_search_distance_m)
        self.candidate_spacing = float(candidate_spacing_m)
        self.heading_weight = float(heading_weight)
        self.candidate_heading = math.radians(float(candidate_heading_deg))
        self.step = float(sample_step_m)
        self.length = float(vehicle_length_m)
        self.width = float(vehicle_width_m)
        self.margin = float(safety_margin_m)
        self.allow_unknown = bool(allow_unknown)
        if min(self.radius, self.search_distance, self.candidate_spacing,
               self.step, self.length, self.width) <= 0.0:
            raise ValueError("rejoin geometry parameters must be positive")

    def candidates(self, pose, route, start_index):
        output, last_distance = [], -float("inf")
        accumulated = 0.0
        for index in range(max(0, int(start_index)), len(route)):
            if index > 0:
                accumulated += math.hypot(route[index].x-route[index-1].x,
                                          route[index].y-route[index-1].y)
            if accumulated-last_distance < self.candidate_spacing:
                continue
            point = route[index]
            distance = math.hypot(point.x-pose.x, point.y-pose.y)
            if distance > self.search_distance:
                continue
            travel_yaw = point.yaw if point.direction >= 0 else wrap_angle(point.yaw+math.pi)
            heading = abs(wrap_angle(travel_yaw-pose.yaw))
            if heading > self.candidate_heading:
                continue
            output.append((distance+self.heading_weight*heading, index))
            last_distance = accumulated
        return sorted(output)

    def plan(self, pose, route, start_index, grid, reverse_allowed=False):
        if grid is None:
            return RejoinPlan(False, reason="OCCUPANCY_GRID_MISSING")
        checked = 0
        best = None
        for _, index in self.candidates(pose, route, start_index):
            point = route[index]
            directions = ([1] if point.direction >= 0 else
                          ([-1] if reverse_allowed else []))
            for direction in directions:
                checked += 1
                path = dubins_path(pose, point, self.radius, self.step, direction)
                if not path or not all(grid.footprint_free(
                        item, self.length, self.width, self.margin) for item in path):
                    continue
                length = sum(math.hypot(b.x-a.x, b.y-a.y)
                             for a, b in zip(path, path[1:]))
                score = length+self.heading_weight*abs(wrap_angle(point.yaw-pose.yaw))
                candidate = (score, index, path, length, direction)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            return RejoinPlan(False, candidates_checked=checked,
                              reason="REJOIN_NO_FEASIBLE_PATH")
        _, index, path, length, direction = best
        return RejoinPlan(True, path, index, length, 1.0/self.radius,
                          direction, "REJOIN_PATH_READY", checked, True)


class RejoinStateMachine:
    FOLLOW_ROUTE = "FOLLOW_ROUTE"
    ROUTE_DEVIATION_STOP = "ROUTE_DEVIATION_STOP"
    PLAN_REJOIN = "PLAN_REJOIN"
    FOLLOW_REJOIN_PATH = "FOLLOW_REJOIN_PATH"
    VERIFY_REJOIN = "VERIFY_REJOIN"
    REJOIN_NO_FEASIBLE_PATH = "REJOIN_NO_FEASIBLE_PATH"

    def __init__(self, start_distance_m=1.0, complete_distance_m=0.25,
                 complete_heading_deg=10.0, verify_duration_s=1.0):
        self.start_distance = float(start_distance_m)
        self.complete_distance = float(complete_distance_m)
        self.complete_heading = math.radians(float(complete_heading_deg))
        self.verify_duration = float(verify_duration_s)
        self.state = self.FOLLOW_ROUTE
        self.verify_since = None

    def observe_route(self, cross_track_m):
        if self.state == self.FOLLOW_ROUTE and abs(float(cross_track_m)) > self.start_distance:
            self.state = self.ROUTE_DEVIATION_STOP
        return self.state

    def vehicle_stopped(self):
        if self.state == self.ROUTE_DEVIATION_STOP:
            self.state = self.PLAN_REJOIN
        return self.state

    def plan_completed(self, feasible):
        if self.state == self.PLAN_REJOIN:
            self.state = (self.FOLLOW_REJOIN_PATH if feasible else
                          self.REJOIN_NO_FEASIBLE_PATH)
        return self.state

    def path_completed(self):
        if self.state == self.FOLLOW_REJOIN_PATH:
            self.state = self.VERIFY_REJOIN
            self.verify_since = None
        return self.state

    def path_blocked(self):
        if self.state == self.FOLLOW_REJOIN_PATH:
            self.state = self.REJOIN_NO_FEASIBLE_PATH
        return self.state

    def verify(self, cross_track_m, heading_error_rad, now):
        if self.state != self.VERIFY_REJOIN:
            return self.state
        if (abs(float(cross_track_m)) <= self.complete_distance and
                abs(float(heading_error_rad)) <= self.complete_heading):
            self.verify_since = float(now) if self.verify_since is None else self.verify_since
            if float(now)-self.verify_since >= self.verify_duration:
                self.state = self.FOLLOW_ROUTE
                self.verify_since = None
        else:
            self.verify_since = None
        return self.state
