"""Pure-pursuit geometry for a non-actuating map-route validation follower."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .route_model import RoutePoint, VehiclePolicy


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class FollowerCommand:
    drive: float
    wheel_deg: int
    nearest_index: int
    target_index: int
    cross_track_error_m: float
    saturated: bool
    status: str


def transform_pose(transform: Pose2D, pose: Pose2D) -> Pose2D:
    c, s = math.cos(transform.yaw), math.sin(transform.yaw)
    return Pose2D(transform.x + c * pose.x - s * pose.y,
                  transform.y + s * pose.x + c * pose.y,
                  normalize_angle(transform.yaw + pose.yaw))


def alignment_transform(route_start: RoutePoint, odom_initial: Pose2D) -> Pose2D:
    yaw = math.radians(route_start.yaw_deg) - odom_initial.yaw
    c, s = math.cos(yaw), math.sin(yaw)
    return Pose2D(route_start.x - (c * odom_initial.x - s * odom_initial.y),
                  route_start.y - (s * odom_initial.x + c * odom_initial.y), yaw)


class RouteFollower:
    def __init__(self, policy=VehiclePolicy(), lookahead_m=1.0,
                 stop_capture_m=0.35, start_index=0):
        self.policy = policy
        self.lookahead_m = float(lookahead_m)
        self.stop_capture_m = float(stop_capture_m)
        self.start_index = int(start_index)
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.last_nearest = None

    def nearest_index(self, points: list[RoutePoint], pose: Pose2D) -> int:
        if self.start_index >= len(points):
            raise ValueError(
                f"start_index {self.start_index} is outside route with "
                f"{len(points)} points")
        if self.last_nearest is None:
            candidates = range(self.start_index, len(points))
        else:
            candidates = range(max(self.start_index, self.last_nearest - 3),
                               min(len(points), self.last_nearest + 31))
        index = min(candidates, key=lambda i: math.hypot(
            points[i].x - pose.x, points[i].y - pose.y))
        previous = self.start_index if self.last_nearest is None else self.last_nearest
        self.last_nearest = max(index, previous)
        return self.last_nearest

    def target_index(self, points: list[RoutePoint], nearest: int) -> int:
        distance = 0.0
        for index in range(nearest + 1, len(points)):
            a, b = points[index - 1], points[index]
            distance += math.hypot(b.x - a.x, b.y - a.y)
            if distance >= self.lookahead_m or b.event == "STOP":
                return index
        return len(points) - 1

    def command(self, points: list[RoutePoint], pose: Pose2D) -> FollowerCommand:
        if not points:
            raise ValueError("route is empty")
        nearest = self.nearest_index(points, pose)
        target = self.target_index(points, nearest)
        near_point, target_point = points[nearest], points[target]
        cross_track = math.hypot(near_point.x - pose.x, near_point.y - pose.y)
        stop_index = next((i for i in range(nearest, target + 1)
                           if points[i].event == "STOP"), None)
        if stop_index is not None and math.hypot(
                points[stop_index].x - pose.x,
                points[stop_index].y - pose.y) <= self.stop_capture_m:
            return FollowerCommand(0.0, 0.0, nearest, stop_index,
                                   cross_track, False, "STOP")
        direction = near_point.direction
        motion_yaw = pose.yaw if direction == "F" else normalize_angle(pose.yaw + math.pi)
        bearing = math.atan2(target_point.y - pose.y, target_point.x - pose.x)
        alpha = normalize_angle(bearing - motion_yaw)
        distance = max(1e-6, math.hypot(target_point.x - pose.x,
                                       target_point.y - pose.y))
        raw = math.degrees(math.atan2(
            2.0 * self.policy.wheelbase_m * math.sin(alpha), distance))
        if direction == "R":
            raw = -raw
        clamped = max(self.policy.steering_right_deg,
                      min(self.policy.steering_left_deg, raw))
        saturated = abs(raw - clamped) > 1e-9
        # The route command contract is integer steering degrees. Quantize
        # before applying the 10-degree speed threshold so visible wheel and
        # speed outputs cannot disagree around 9.5 degrees.
        wheel = int(round(clamped))
        if direction == "R":
            drive = self.policy.reverse_speed
        elif abs(wheel) >= self.policy.turning_threshold_deg:
            drive = self.policy.turning_speed
        else:
            drive = self.policy.forward_speed
        return FollowerCommand(drive, wheel, nearest, target, cross_track,
                               saturated, "TRACKING")
