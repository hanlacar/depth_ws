"""Numerical quality checks for camera metric paths."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MetricPathQualityConfig:
    minimum_points: int = 3
    minimum_spacing_m: float = 0.05
    maximum_point_jump_m: float = 4.0
    maximum_reverse_step_m: float = 0.25
    minimum_path_length_m: float = 1.0

    def validate(self):
        values = (
            self.minimum_spacing_m, self.maximum_point_jump_m,
            self.maximum_reverse_step_m, self.minimum_path_length_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("metric path quality parameters must be finite")
        if self.minimum_points < 2:
            raise ValueError("minimum_points must be at least two")
        if self.minimum_spacing_m <= 0.0:
            raise ValueError("minimum_spacing_m must be positive")
        if self.maximum_point_jump_m <= self.minimum_spacing_m:
            raise ValueError("maximum_point_jump_m must exceed minimum spacing")
        if self.maximum_reverse_step_m < 0.0:
            raise ValueError("maximum_reverse_step_m must be nonnegative")
        if self.minimum_path_length_m <= 0.0:
            raise ValueError("minimum_path_length_m must be positive")


@dataclass(frozen=True)
class MetricPathAnalysis:
    valid: bool
    reason: str
    points: np.ndarray
    point_count: int
    minimum_x_m: float | None
    maximum_x_m: float | None
    path_length_m: float
    forward_usable_length_m: float
    maximum_curvature_per_m: float
    duplicates_removed: int
    jump_truncated: bool


def cumulative_s(points, minimum_spacing_m=1.0e-6):
    """Return strict Euclidean arc length; reject zero-length segments."""
    array = np.asarray(points, dtype=float)
    if (array.ndim != 2 or array.shape[1] != 2 or len(array) == 0 or
            not np.all(np.isfinite(array))):
        raise ValueError("path must be a nonempty finite Nx2 array")
    if len(array) == 1:
        return np.array([0.0], dtype=float)
    lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    if np.any(lengths <= minimum_spacing_m):
        raise ValueError("path contains zero-length or duplicate segments")
    return np.concatenate(([0.0], np.cumsum(lengths)))


def path_heading(points, index):
    array = np.asarray(points, dtype=float)
    if len(array) == 1:
        return 0.0
    if len(array) < 1:
        raise ValueError("at least one point is required for heading")
    index = max(0, min(int(index), len(array)-1))
    if index == 0:
        vector = array[1]-array[0]
    elif index == len(array)-1:
        vector = array[-1]-array[-2]
    else:
        vector = array[index+1]-array[index-1]
    if float(np.linalg.norm(vector)) <= 1.0e-9:
        raise ValueError("heading is undefined for duplicate points")
    return math.atan2(float(vector[1]), float(vector[0]))


def angle_difference_rad(first, second):
    return (float(first)-float(second)+math.pi) % (2.0*math.pi)-math.pi


def maximum_curvature(points):
    array = np.asarray(points, dtype=float)
    if len(array) < 3:
        return 0.0
    values = []
    for index in range(1, len(array)-1):
        before = array[index]-array[index-1]
        after = array[index+1]-array[index]
        before_length = float(np.linalg.norm(before))
        after_length = float(np.linalg.norm(after))
        if before_length <= 1.0e-9 or after_length <= 1.0e-9:
            continue
        delta = abs(angle_difference_rad(
            math.atan2(after[1], after[0]),
            math.atan2(before[1], before[0])))
        values.append(delta/max(1.0e-9, 0.5*(before_length+after_length)))
    return max(values, default=0.0)


def _orientation(a, b, c, epsilon=1.0e-9):
    value = ((b[0]-a[0])*(c[1]-a[1]) -
             (b[1]-a[1])*(c[0]-a[0]))
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0.0 else -1


def _on_segment(a, b, point, epsilon=1.0e-9):
    return bool(
        min(a[0], b[0])-epsilon <= point[0] <= max(a[0], b[0])+epsilon and
        min(a[1], b[1])-epsilon <= point[1] <= max(a[1], b[1])+epsilon)


def has_self_intersection(points):
    """Detect crossings/touches between non-adjacent polyline segments."""
    array = np.asarray(points, dtype=float)
    for first in range(len(array)-1):
        for second in range(first+2, len(array)-1):
            a, b = array[first], array[first+1]
            c, d = array[second], array[second+1]
            first_c = _orientation(a, b, c)
            first_d = _orientation(a, b, d)
            second_a = _orientation(c, d, a)
            second_b = _orientation(c, d, b)
            if (first_c*first_d < 0 and second_a*second_b < 0):
                return True
            if ((first_c == 0 and _on_segment(a, b, c)) or
                    (first_d == 0 and _on_segment(a, b, d)) or
                    (second_a == 0 and _on_segment(c, d, a)) or
                    (second_b == 0 and _on_segment(c, d, b))):
                return True
    return False


def remove_consecutive_duplicates(points, minimum_spacing_m):
    array = np.asarray(points, dtype=float)
    if len(array) == 0:
        return array.reshape((0, 2)), 0
    kept = [array[0]]
    removed = 0
    for point in array[1:]:
        if float(np.linalg.norm(point-kept[-1])) < minimum_spacing_m:
            removed += 1
        else:
            kept.append(point)
    return np.asarray(kept, dtype=float).reshape((-1, 2)), removed


def analyze_metric_path(points, config=MetricPathQualityConfig()):
    """Clean duplicates, trim a horizon jump, then validate near-to-far path."""
    config.validate()
    array = np.asarray(points, dtype=float)
    empty = np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1:] != (2,):
        return MetricPathAnalysis(False, "invalid_shape", empty, 0, None, None,
                                  0.0, 0.0, 0.0, 0, False)
    if not np.all(np.isfinite(array)):
        return MetricPathAnalysis(False, "nonfinite_point", empty, 0, None, None,
                                  0.0, 0.0, 0.0, 0, False)

    cleaned, duplicates = remove_consecutive_duplicates(
        array, config.minimum_spacing_m)
    jump_truncated = False
    if len(cleaned) >= 2:
        segment_lengths = np.linalg.norm(np.diff(cleaned, axis=0), axis=1)
        jumps = np.flatnonzero(segment_lengths > config.maximum_point_jump_m)
        if len(jumps):
            cleaned = cleaned[:int(jumps[0])+1]
            jump_truncated = True

    if len(cleaned) < config.minimum_points:
        reason = "too_few_points_after_jump" if jump_truncated else "too_few_points"
        return MetricPathAnalysis(False, reason, cleaned, len(cleaned),
                                  None, None, 0.0, 0.0, 0.0,
                                  duplicates, jump_truncated)
    if np.any(np.diff(cleaned[:, 0]) < -config.maximum_reverse_step_m):
        return MetricPathAnalysis(False, "point_order_reversal", cleaned,
                                  len(cleaned), float(np.min(cleaned[:, 0])),
                                  float(np.max(cleaned[:, 0])), 0.0, 0.0, 0.0,
                                  duplicates, jump_truncated)
    if has_self_intersection(cleaned):
        return MetricPathAnalysis(False, "self_intersection", cleaned,
                                  len(cleaned), float(np.min(cleaned[:, 0])),
                                  float(np.max(cleaned[:, 0])), 0.0, 0.0, 0.0,
                                  duplicates, jump_truncated)
    try:
        arc = cumulative_s(cleaned)
    except ValueError:
        return MetricPathAnalysis(False, "duplicate_segment", cleaned,
                                  len(cleaned), None, None, 0.0, 0.0, 0.0,
                                  duplicates, jump_truncated)
    path_length = float(arc[-1])
    minimum_x = float(np.min(cleaned[:, 0]))
    maximum_x = float(np.max(cleaned[:, 0]))
    forward_usable = max(0.0, maximum_x)
    if path_length < config.minimum_path_length_m:
        return MetricPathAnalysis(False, "path_too_short", cleaned,
                                  len(cleaned), minimum_x, maximum_x,
                                  path_length, forward_usable,
                                  maximum_curvature(cleaned), duplicates,
                                  jump_truncated)
    return MetricPathAnalysis(
        True, "ok_jump_truncated" if jump_truncated else "ok", cleaned,
        len(cleaned), minimum_x, maximum_x, path_length, forward_usable,
        maximum_curvature(cleaned), duplicates, jump_truncated)
