"""Ackermann pure-pursuit route follower with T870 steering convention."""

import math

from .geometry import wrap_angle
from .models import ControllerResult


class RouteFollower:
    def __init__(self, wheelbase=0.73, max_steering_deg=22.0,
                 lookahead_m=0.7, corridor_m=1.0,
                 max_heading_deg=80.0, steering_rate_deg_s=90.0,
                 heading_weight=0.75, search_ahead_points=160,
                 search_behind_points=8, max_index_backtrack=0):
        self.wheelbase = float(wheelbase)
        self.max_steering = float(max_steering_deg)
        self.lookahead = float(lookahead_m)
        self.corridor = float(corridor_m)
        self.max_heading = math.radians(max_heading_deg)
        self.steering_rate = float(steering_rate_deg_s)
        self.heading_weight = float(heading_weight)
        self.search_ahead = int(search_ahead_points)
        self.search_behind = int(search_behind_points)
        self.max_backtrack = int(max_index_backtrack)
        self.last_steering = 0.0
        self.last_index = None

    @property
    def minimum_turning_radius(self):
        return self.wheelbase / math.tan(math.radians(self.max_steering))

    def reset_progress(self):
        self.last_index = None

    @staticmethod
    def _project_segment(pose, first, second):
        dx, dy = second.x-first.x, second.y-first.y
        squared = dx*dx+dy*dy
        if squared <= 1.0e-12:
            return first.x, first.y, 0.0, math.hypot(pose.x-first.x,
                                                     pose.y-first.y)
        t = max(0.0, min(1.0, ((pose.x-first.x)*dx+(pose.y-first.y)*dy)/squared))
        x, y = first.x+t*dx, first.y+t*dy
        distance = math.hypot(pose.x-x, pose.y-y)
        sign = 1.0 if dx*(pose.y-y)-dy*(pose.x-x) >= 0.0 else -1.0
        return x, y, t, sign*distance

    def _segment(self, pose, route, global_search=False, progress_ceiling=None):
        if len(route) == 1:
            return 0, 0.0, math.hypot(route[0].x-pose.x,
                                      route[0].y-pose.y)
        if global_search or self.last_index is None:
            stop = len(route)-1
            if progress_ceiling is not None:
                stop = min(stop, max(1, int(progress_ceiling)))
            indexes = range(stop)
        else:
            first = max(0, self.last_index-self.search_behind)
            last = min(len(route)-1, self.last_index+self.search_ahead+1)
            if progress_ceiling is not None:
                last = min(last, max(first+1, int(progress_ceiling)))
            indexes = range(first, last)
        scored = []
        for index in indexes:
            _, _, projection, lateral = self._project_segment(
                pose, route[index], route[index+1])
            # Route yaw is always the vehicle/body yaw, including reverse
            # waypoints.  This keeps branch selection and the later heading
            # safety gate on the same convention.
            heading = abs(wrap_angle(route[index].yaw-pose.yaw))
            # Heading is part of branch selection, not just a later stop gate.
            score = abs(lateral)+self.heading_weight*heading
            scored.append((score, heading, abs(lateral), index, projection, lateral))
        if not scored:
            return 0, 0.0, float("inf")
        direction_compatible = [item for item in scored
                                if item[1] <= self.max_heading]
        selected = min(direction_compatible or scored)
        index = selected[3]
        if self.last_index is not None and not global_search:
            index = max(index, self.last_index-self.max_backtrack)
            _, _, projection, lateral = self._project_segment(
                pose, route[index], route[index+1])
        else:
            projection, lateral = selected[4], selected[5]
        self.last_index = index
        return index, projection, lateral

    def compute(self, pose, route, dt=1.0/30.0, allow_motion=False,
                global_search=False, progress_ceiling=None):
        if not route:
            return ControllerResult(0.0, 0.0, 0, 0, 0.0, 0.0, 0.0,
                                    True, "EMPTY_ROUTE")
        nearest, projection, cross_track = self._segment(
            pose, route, global_search=global_search,
            progress_ceiling=progress_ceiling)
        target = nearest
        first_segment = route[min(nearest+1, len(route)-1)]
        segment_length = math.hypot(first_segment.x-route[nearest].x,
                                    first_segment.y-route[nearest].y)
        accumulated = max(0.0, (1.0-projection)*segment_length)
        if accumulated > 0.0:
            target = min(nearest+1, len(route)-1)
        direction = route[nearest].direction
        while (target + 1 < len(route) and accumulated < self.lookahead and
               route[target + 1].direction == direction and
               (progress_ceiling is None or target+1 <= int(progress_ceiling))):
            a, b = route[target], route[target+1]
            accumulated += math.hypot(b.x-a.x, b.y-a.y)
            target += 1
        point = route[target]
        # A reverse vehicle travels along body yaw + pi.  Pure pursuit is
        # evaluated on that motion axis, then its curvature sign is inverted
        # for negative longitudinal velocity.
        control_yaw = pose.yaw if direction > 0 else wrap_angle(pose.yaw+math.pi)
        alpha = wrap_angle(
            math.atan2(point.y-pose.y, point.x-pose.x)-control_yaw)
        heading_error = wrap_angle(route[nearest].yaw-pose.yaw)
        stop_reason = ""
        if abs(cross_track) > self.corridor:
            stop_reason = "CORRIDOR_VIOLATION"
        elif abs(heading_error) > self.max_heading:
            stop_reason = "HEADING_ERROR"
        elif nearest >= len(route)-2 and abs(cross_track) < 0.15 and projection > 0.9:
            stop_reason = "ROUTE_COMPLETE"
        elif str(point.mission_marker).upper() in ("STOP", "STOP_POINT") and \
                math.hypot(point.x-pose.x, point.y-pose.y) < 0.3:
            stop_reason = "ROUTE_STOP_POINT"
        # Stage-1 /slam_wheel contract is left-positive, right-negative.
        standard = math.degrees(math.atan2(
            2.0*self.wheelbase*math.sin(alpha), max(self.lookahead, 0.05)))
        if direction < 0:
            standard = -standard
        requested = max(-self.max_steering, min(self.max_steering, standard))
        curvature_limited = abs(standard) > self.max_steering
        slew = self.steering_rate * max(0.0, dt)
        steering = max(self.last_steering-slew,
                       min(self.last_steering+slew, requested))
        self.last_steering = steering
        stop = bool(stop_reason) or not allow_motion
        reason = stop_reason or ("CONTROL_NOT_APPROVED" if not allow_motion else
                                 ("CURVATURE_SLOWDOWN" if curvature_limited else "OK"))
        drive = 0.0 if stop else float(point.drive_level) * point.direction
        if curvature_limited and not stop:
            # The MCU contract accepts discrete stages only. Slow tight turns
            # to stage 1 without creating invalid fractional stages.
            drive = math.copysign(1.0, drive)
        return ControllerResult(
            drive, steering, nearest, target, cross_track, heading_error,
            (nearest+projection)/max(1, len(route)-1), stop, reason)
