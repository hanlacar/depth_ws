"""Canonical LaserScan processing adapted from Lidar_ws_plus safety logic."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScanAssessment:
    fresh: bool
    valid: bool
    nearest_m: float
    hard_obstacle: bool
    avoidance_required: bool
    left_clearance_m: float
    right_clearance_m: float


class ScanSafety:
    """Cluster-count, hysteresis and clear debounce reject isolated rays."""

    def __init__(self, emergency_m=0.50, clear_m=0.60,
                 avoidance_m=1.50, half_width_m=0.60,
                 minimum_points=2, clear_scans=3):
        self.emergency_m = float(emergency_m)
        self.clear_m = float(clear_m)
        self.avoidance_m = float(avoidance_m)
        self.half_width_m = float(half_width_m)
        self.minimum_points = max(2, int(minimum_points))
        self.clear_scans = max(1, int(clear_scans))
        self.latched = False
        self.clear_count = 0

    def assess(self, points, fresh=True):
        if not fresh:
            self.latched = True
            self.clear_count = 0
            return ScanAssessment(False, False, math.inf, True, True,
                                  0.0, 0.0)
        finite = [(float(x), float(y)) for x, y in (points or ())
                  if math.isfinite(float(x)) and math.isfinite(float(y))
                  and float(x) > 0.0]
        corridor = [(x, y) for x, y in finite if abs(y) <= self.half_width_m]
        close = [(x, y) for x, y in corridor if x <= self.emergency_m]
        threat = [(x, y) for x, y in corridor if x <= self.avoidance_m]
        hard_now = len(close) >= self.minimum_points
        if hard_now:
            self.latched, self.clear_count = True, 0
        elif self.latched:
            clear_now = not corridor or min(x for x, _ in corridor) >= self.clear_m
            self.clear_count = self.clear_count+1 if clear_now else 0
            if self.clear_count >= self.clear_scans:
                self.latched, self.clear_count = False, 0
        nearest = min((x for x, _ in corridor), default=math.inf)
        left = min((x for x, y in finite if y > self.half_width_m),
                   default=math.inf)
        right = min((x for x, y in finite if y < -self.half_width_m),
                    default=math.inf)
        return ScanAssessment(True, True, nearest, self.latched,
                              len(threat) >= self.minimum_points,
                              left, right)


def optional_rear_hard_stop(active, fresh, hard_obstacle):
    """Use a rear obstacle only when an optional rear scan is available."""
    return bool(active and fresh and hard_obstacle)
