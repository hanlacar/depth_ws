"""Bounded odometry-feedback tracking for temporary LiDAR paths."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrackCommand:
    valid: bool
    complete: bool
    drive: float
    wheel: int
    target_index: int


class LocalPathTracker:
    def __init__(self, wheelbase_m=0.73, steering_limit_deg=22.0,
                 lookahead_m=0.60, completion_m=0.05,
                 reverse_completion_m=0.15,
                 maximum_deviation_m=0.80):
        if float(wheelbase_m) != 0.73 or float(steering_limit_deg) != 22.0:
            raise ValueError("tracker must use the commissioned vehicle geometry")
        self.wheelbase = float(wheelbase_m)
        self.limit = float(steering_limit_deg)
        self.lookahead = float(lookahead_m)
        self.completion = float(completion_m)
        self.reverse_completion = float(reverse_completion_m)
        self.maximum_deviation = float(maximum_deviation_m)
        self.points = ()
        self.cursor = 0
        self.drive = 0.0

    def clear(self):
        self.points = ()
        self.cursor = 0
        self.drive = 0.0

    def set_plan(self, local_points, origin_pose, drive):
        ox, oy, yaw = (float(value) for value in origin_pose)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        self.points = tuple((
            ox+cosine*float(x)-sine*float(y),
            oy+sine*float(x)+cosine*float(y),
        ) for x, y, _ in local_points)
        self.cursor = 0
        self.drive = float(drive)
        return bool(self.points)

    def update(self, pose):
        if not self.points:
            return TrackCommand(False, False, 0.0, 0, 0)
        x, y, yaw = (float(value) for value in pose)
        upper = min(len(self.points), self.cursor+50)
        self.cursor = min(
            range(self.cursor, upper),
            key=lambda index: math.hypot(
                self.points[index][0]-x, self.points[index][1]-y))
        nearest_distance = math.hypot(
            self.points[self.cursor][0]-x, self.points[self.cursor][1]-y)
        if nearest_distance > self.maximum_deviation:
            return TrackCommand(False, False, 0.0, 0, self.cursor)
        final_distance = math.hypot(
            self.points[-1][0]-x, self.points[-1][1]-y)
        completion = (self.reverse_completion if self.drive < 0.0 else
                      self.completion)
        if final_distance <= completion:
            return TrackCommand(True, True, 0.0, 0, len(self.points)-1)
        target = self.cursor
        while target < len(self.points)-1:
            tx, ty = self.points[target]
            if math.hypot(tx-x, ty-y) >= self.lookahead:
                break
            target += 1
        tx, ty = self.points[target]
        dx, dy = tx-x, ty-y
        lateral = -math.sin(yaw)*dx+math.cos(yaw)*dy
        distance_sq = max(dx*dx+dy*dy, 1.0e-6)
        steering = math.degrees(math.atan(
            2.0*self.wheelbase*lateral/distance_sq))
        # A generated path is never made drivable by clamping an infeasible
        # tracking command. Deviation requiring >22 deg aborts the path.
        if abs(steering) > self.limit:
            return TrackCommand(False, False, 0.0, 0, target)
        wheel = int(round(steering))
        return TrackCommand(True, False, self.drive, wheel, target)
