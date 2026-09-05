"""ROS-independent, advisory-only camera mission perception primitives."""

from dataclasses import dataclass
import math

import numpy as np


UNKNOWN = "UNKNOWN"
TRAFFIC_STATES = frozenset(("R", "G", "LEFT", UNKNOWN))


def timestamps_synchronized(stamps, slop_sec):
    values = [float(value) for value in stamps]
    return (bool(values) and all(math.isfinite(value) for value in values) and
            max(values)-min(values) <= float(slop_sec))


@dataclass(frozen=True)
class StopLineDepthConfig:
    minimum_pixels: int = 30
    depth_min_m: float = 0.20
    depth_max_m: float = 12.0
    mad_threshold: float = 3.5
    maximum_depth_mad_m: float = 0.45

    def validate(self):
        if self.minimum_pixels < 3:
            raise ValueError("minimum depth pixels must be at least three")
        values = (self.depth_min_m, self.depth_max_m, self.mad_threshold,
                  self.maximum_depth_mad_m)
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("depth parameters must be finite and positive")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("depth minimum must be below maximum")


@dataclass(frozen=True)
class StopLinePoint:
    valid: bool
    optical_xyz_m: tuple | None
    valid_pixels: int
    median_depth_m: float
    depth_mad_m: float
    sample_pixels: tuple
    reason: str


def depth_to_metres(depth, encoding):
    array = np.asarray(depth)
    normalized = str(encoding).upper()
    if normalized in ("16UC1", "MONO16"):
        return array.astype(np.float64)*0.001
    if normalized == "32FC1":
        return array.astype(np.float64)
    raise ValueError(f"unsupported depth encoding: {encoding}")


def robust_stop_line_point(mask, depth, encoding, camera_matrix,
                           config=StopLineDepthConfig()):
    """Return a robust 3-D stop-line point in REP-103 optical coordinates."""
    config.validate()
    binary = np.asarray(mask) > 0
    metric = depth_to_metres(depth, encoding)
    if binary.shape != metric.shape:
        return StopLinePoint(False, None, 0, math.nan, math.nan, (),
                             "MASK_DEPTH_SHAPE_MISMATCH")
    finite = (binary & np.isfinite(metric) & (metric >= config.depth_min_m) &
              (metric <= config.depth_max_m))
    rows, cols = np.nonzero(finite)
    values = metric[rows, cols]
    if len(values) < config.minimum_pixels:
        return StopLinePoint(False, None, len(values), math.nan, math.nan, (),
                             "INSUFFICIENT_VALID_DEPTH")
    median = float(np.median(values))
    absolute = np.abs(values-median)
    mad = float(np.median(absolute))
    robust_sigma = max(1.0e-3, 1.4826*mad)
    keep = absolute <= config.mad_threshold*robust_sigma
    rows, cols, values = rows[keep], cols[keep], values[keep]
    if len(values) < config.minimum_pixels:
        return StopLinePoint(False, None, len(values), median, mad, (),
                             "DEPTH_OUTLIER_REJECTION")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values-median)))
    if mad > config.maximum_depth_mad_m:
        return StopLinePoint(False, None, len(values), median, mad, (),
                             "DEPTH_SPREAD_EXCESS")
    matrix = np.asarray(camera_matrix, float).reshape(3, 3)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        return StopLinePoint(False, None, len(values), median, mad, (),
                             "CAMERA_INFO_INVALID")
    x = (cols-cx)*values/fx
    y = (rows-cy)*values/fy
    point = (float(np.median(x)), float(np.median(y)), median)
    order = np.linspace(0, len(rows)-1, min(80, len(rows)), dtype=int)
    samples = tuple((int(cols[i]), int(rows[i])) for i in order)
    return StopLinePoint(True, point, len(values), median, mad, samples, "OK")


def quaternion_matrix_xyzw(quaternion):
    x, y, z, w = (float(v) for v in quaternion)
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], float)


def transform_point(point, translation, quaternion):
    rotation = quaternion_matrix_xyzw(quaternion)
    return rotation@np.asarray(point, float)+np.asarray(translation, float)


def front_axle_distance(point, translation=None, quaternion=None,
                        allow_fallback=False, camera_to_front_axle_m=math.nan):
    if translation is not None and quaternion is not None:
        return float(transform_point(point, translation, quaternion)[0]), "TF"
    if (allow_fallback and math.isfinite(camera_to_front_axle_m) and
            camera_to_front_axle_m >= 0.0):
        return float(point[2]-camera_to_front_axle_m), "OFFSET_FALLBACK"
    return math.nan, "CALIBRATION_INVALID"


def front_axle_distance_from_base_point(point_base, wheelbase_m=0.73):
    """Longitudinal stop-line distance from the front axle.

    The shared T870 base_link is the four-wheel center, hence the front axle
    is always at +wheelbase/2.  This avoids introducing a second publisher for
    a front_axle TF while retaining the source-message timestamp TF lookup.
    """
    point = np.asarray(point_base, dtype=float)
    wheelbase = float(wheelbase_m)
    if (point.shape != (3,) or not np.all(np.isfinite(point)) or
            not math.isfinite(wheelbase) or wheelbase <= 0.0):
        return math.nan
    return float(point[0] - 0.5*wheelbase)


def pitch_deg_from_quaternion(quaternion):
    rotation = quaternion_matrix_xyzw(quaternion)
    return math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0))))


@dataclass(frozen=True)
class PresenceConfig:
    minimum_confidence: float = 0.50
    minimum_area_px: float = 25.0
    on_frames: int = 3
    off_frames: int = 2
    timeout_sec: float = 0.50


class DebouncedPresence:
    def __init__(self, config=PresenceConfig()):
        self.config = config
        self.detected = False
        self.on_count = self.off_count = 0
        self.last_raw_at = None
        self.last_confidence = 0.0

    def update(self, raw, confidence, area, now):
        now = float(now)
        reliable = bool(raw and math.isfinite(confidence) and
                        confidence >= self.config.minimum_confidence and
                        math.isfinite(area) and area >= self.config.minimum_area_px)
        self.last_confidence = float(confidence) if math.isfinite(confidence) else 0.0
        if reliable:
            self.last_raw_at = now
            self.on_count += 1; self.off_count = 0
            if self.on_count >= self.config.on_frames:
                self.detected = True
        else:
            self.on_count = 0; self.off_count += 1
            if self.off_count >= self.config.off_frames:
                self.detected = False
        return self.tick(now)

    def tick(self, now):
        if (self.last_raw_at is None or
                float(now)-self.last_raw_at > self.config.timeout_sec):
            self.detected = False
            self.on_count = self.off_count = 0
        return self.detected


@dataclass(frozen=True)
class MissionTrafficConfig:
    minimum_confidence: float = 0.50
    on_frames: int = 3
    switch_frames: int = 3
    timeout_sec: float = 0.50
    conflict_margin: float = 0.15


class MissionTrafficFilter:
    def __init__(self, config=MissionTrafficConfig()):
        self.config = config
        self.state = UNKNOWN
        self.pending = None
        self.pending_count = 0
        self.last_reliable_at = None
        self.last_candidate_at = None
        self.reason = "NOT_OBSERVED"
        self.conflict = False

    def update(self, red_score, green_score, left_score, other_score, now):
        red = math.isfinite(red_score) and red_score >= self.config.minimum_confidence
        green = (math.isfinite(green_score) and
                 green_score >= self.config.minimum_confidence)
        left = (math.isfinite(left_score) and
                left_score >= self.config.minimum_confidence)
        self.conflict = bool(red and green)
        if self.conflict:
            self.state = UNKNOWN
            self.pending = None; self.pending_count = 0
            self.last_reliable_at = None
            self.reason = "RED_GREEN_CONFLICT"
            return self.state
        # The course contract defines a left arrow as permissive only when it
        # is detected together with red. A lone arrow remains unsupported.
        candidate = "LEFT" if red and left else "R" if red else "G" if green else None
        if candidate is None:
            unsupported = (left or other_score >= self.config.minimum_confidence)
            self.reason = "UNSUPPORTED_LIGHT" if unsupported else "NO_LIGHT"
            return self.tick(now)
        self.last_candidate_at = float(now)
        required = (self.config.on_frames if self.state == UNKNOWN
                    else self.config.switch_frames)
        if candidate == self.state:
            self.last_reliable_at = float(now)
            self.pending = None; self.pending_count = 0
            self.reason = "CONFIRMED"
        else:
            if candidate == self.pending:
                self.pending_count += 1
            else:
                self.pending = candidate; self.pending_count = 1
            self.reason = "PENDING"
            if self.pending_count >= required:
                self.state = candidate
                self.last_reliable_at = float(now)
                self.pending = None; self.pending_count = 0
                self.reason = "STATE_CONFIRMED"
        return self.tick(now)

    def tick(self, now):
        if (self.state != UNKNOWN and self.last_reliable_at is not None and
                float(now)-self.last_reliable_at > self.config.timeout_sec):
            self.reason = "TIMEOUT"
            self.state = UNKNOWN
            self.pending = None; self.pending_count = 0
        elif (self.state == UNKNOWN and self.pending is not None and
              self.last_candidate_at is not None and
              float(now)-self.last_candidate_at > self.config.timeout_sec):
            self.reason = "TIMEOUT"
            self.pending = None; self.pending_count = 0
        assert self.state in TRAFFIC_STATES
        return self.state


@dataclass(frozen=True)
class UphillConfig:
    on_deg: float = 15.0
    off_deg: float = 12.0
    minimum_duration_sec: float = 0.25
    pitch_sign: float = 1.0


class UphillDetector:
    def __init__(self, config=UphillConfig(), reference_pitch_deg=None):
        if config.on_deg <= config.off_deg or config.minimum_duration_sec < 0:
            raise ValueError("invalid uphill hysteresis")
        if config.pitch_sign not in (-1.0, 1.0):
            raise ValueError("uphill pitch sign must be -1 or +1")
        self.config = config
        self.reference_pitch_deg = reference_pitch_deg
        self.uphill = False
        self.candidate_since = None
        self.relative_deg = math.nan

    def update(self, pitch_deg, valid, now):
        if not valid or not math.isfinite(pitch_deg):
            self.uphill = False; self.candidate_since = None
            self.relative_deg = math.nan
            return False
        if self.reference_pitch_deg is None:
            self.reference_pitch_deg = float(pitch_deg)
        self.relative_deg = self.config.pitch_sign*float(
            pitch_deg-self.reference_pitch_deg)
        if self.uphill:
            if self.relative_deg < self.config.off_deg:
                self.uphill = False; self.candidate_since = None
        elif self.relative_deg >= self.config.on_deg:
            if self.candidate_since is None:
                self.candidate_since = float(now)
            if (float(now)-self.candidate_since+1.0e-9 >=
                    self.config.minimum_duration_sec):
                self.uphill = True
        else:
            self.candidate_since = None
        return self.uphill
