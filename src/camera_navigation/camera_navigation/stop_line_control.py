"""Robust stop-line depth estimation and longitudinal state policy."""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class StopLinePhase(str, Enum):
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    STOP = "STOP"


@dataclass(frozen=True)
class StopLineConfig:
    stop_line_slowdown_distance_m: float = 2.0
    stop_line_stop_distance_m: float = 0.7
    # T870: base_link is the four-wheel centre, the vehicle nose is x=0.500 m,
    # and camera_link is x=0.015 m.  A camera-reported ground distance is
    # therefore reduced by 0.485 m before applying the bumper stop policy.
    camera_to_front_bumper_m: float = 0.485
    stop_line_release_margin_m: float = 0.2
    stop_line_confirmation_frames: int = 3
    stop_line_measurement_timeout_sec: float = 0.5
    stop_line_center_roi_width_ratio: float = 0.6
    stop_line_min_depth_m: float = 0.1
    stop_line_max_depth_m: float = 20.0
    stop_line_min_valid_depth_pixels: int = 20
    stop_line_depth_percentile: float = 50.0
    stop_line_depth_mad_scale: float = 3.5

    def validate(self):
        finite = (
            self.stop_line_slowdown_distance_m,
            self.stop_line_stop_distance_m,
            self.camera_to_front_bumper_m,
            self.stop_line_release_margin_m,
            self.stop_line_measurement_timeout_sec,
            self.stop_line_center_roi_width_ratio,
            self.stop_line_min_depth_m,
            self.stop_line_max_depth_m,
            self.stop_line_depth_percentile,
            self.stop_line_depth_mad_scale,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("stop-line parameters must be finite")
        if not (0.0 <= self.stop_line_stop_distance_m <
                self.stop_line_slowdown_distance_m):
            raise ValueError(
                "stop_line_stop_distance_m must be nonnegative and less than "
                "stop_line_slowdown_distance_m")
        if self.camera_to_front_bumper_m < 0.0:
            raise ValueError("camera_to_front_bumper_m must be nonnegative")
        if self.stop_line_release_margin_m < 0.0:
            raise ValueError("stop_line_release_margin_m must be nonnegative")
        if self.stop_line_confirmation_frames < 1:
            raise ValueError("stop_line_confirmation_frames must be at least 1")
        if self.stop_line_measurement_timeout_sec <= 0.0:
            raise ValueError("stop_line_measurement_timeout_sec must be positive")
        if not 0.0 < self.stop_line_center_roi_width_ratio <= 1.0:
            raise ValueError("stop_line_center_roi_width_ratio must be in (0, 1]")
        if not 0.0 < self.stop_line_min_depth_m < self.stop_line_max_depth_m:
            raise ValueError("stop-line depth range is invalid")
        if self.stop_line_min_valid_depth_pixels < 1:
            raise ValueError("stop_line_min_valid_depth_pixels must be positive")
        if not 0.0 <= self.stop_line_depth_percentile <= 100.0:
            raise ValueError("stop_line_depth_percentile must be in [0, 100]")
        if self.stop_line_depth_mad_scale <= 0.0:
            raise ValueError("stop_line_depth_mad_scale must be positive")


@dataclass(frozen=True)
class StopLineMeasurement:
    valid: bool
    camera_distance_m: float | None
    valid_pixel_count: int
    reason: str


@dataclass(frozen=True)
class StopLineDecision:
    phase: StopLinePhase
    front_bumper_distance_m: float | None
    stop_required: bool
    reason: str

    @property
    def maximum_drive(self):
        if self.phase == StopLinePhase.STOP:
            return 0.0
        if self.phase == StopLinePhase.SLOW:
            return 1.0
        return math.inf


def estimate_stop_line_distance(stop_line_mask, aligned_depth_m,
                                config=StopLineConfig()):
    """Return a robust camera-to-line depth from a color-aligned depth image."""
    config.validate()
    mask = np.asarray(stop_line_mask)
    depth = np.asarray(aligned_depth_m, dtype=np.float64)
    if mask.ndim != 2 or depth.ndim != 2 or mask.shape != depth.shape:
        return StopLineMeasurement(False, None, 0, "shape_mismatch")

    height, width = mask.shape
    roi_width = max(1, int(round(width*config.stop_line_center_roi_width_ratio)))
    left = max(0, (width-roi_width)//2)
    right = min(width, left+roi_width)
    selected_mask = mask[:, left:right] > 0
    if not np.any(selected_mask):
        return StopLineMeasurement(False, None, 0, "stop_line_mask_missing")

    values = depth[:, left:right][selected_mask]
    valid = np.isfinite(values)
    valid &= values >= config.stop_line_min_depth_m
    valid &= values <= config.stop_line_max_depth_m
    values = values[valid]
    minimum = config.stop_line_min_valid_depth_pixels
    if values.size < minimum:
        return StopLineMeasurement(
            False, None, int(values.size), "insufficient_valid_depth")

    median = float(np.median(values))
    absolute_deviation = np.abs(values-median)
    mad = float(np.median(absolute_deviation))
    if math.isfinite(mad) and mad > 1.0e-9:
        robust_sigma = 1.4826*mad
        values = values[
            absolute_deviation <= config.stop_line_depth_mad_scale*robust_sigma]
    if values.size < minimum:
        return StopLineMeasurement(
            False, None, int(values.size), "insufficient_depth_after_outlier_filter")

    distance = float(np.percentile(values, config.stop_line_depth_percentile))
    if not math.isfinite(distance):
        return StopLineMeasurement(False, None, int(values.size), "nonfinite_distance")
    return StopLineMeasurement(True, distance, int(values.size), "ok")


class StopLinePolicy:
    """Temporal confirmation and explicit-release stop latch."""

    def __init__(self, config=StopLineConfig()):
        config.validate()
        self.config = config
        self.stop_latched = False
        self.confirmation_count = 0
        self.last_front_bumper_distance_m = None
        self.last_measurement_at = None
        self.last_measurement_reason = "never_observed"

    def ingest_camera_distance(self, camera_distance_m, now):
        camera_distance_m = float(camera_distance_m)
        now = float(now)
        if not math.isfinite(camera_distance_m) or not math.isfinite(now):
            self.observe_unavailable("nonfinite_measurement")
            return False
        front_distance = camera_distance_m-self.config.camera_to_front_bumper_m
        self.last_front_bumper_distance_m = front_distance
        self.last_measurement_at = now
        self.last_measurement_reason = "ok"

        if not self.stop_latched:
            if front_distance <= self.config.stop_line_stop_distance_m:
                self.confirmation_count += 1
                if self.confirmation_count >= self.config.stop_line_confirmation_frames:
                    self.stop_latched = True
            elif front_distance > (self.config.stop_line_stop_distance_m +
                                   self.config.stop_line_release_margin_m):
                self.confirmation_count = 0
            # Inside the release band, preserve but do not increment the count.
        return True

    def observe_unavailable(self, reason="measurement_unavailable"):
        self.last_measurement_reason = str(reason)
        if not self.stop_latched:
            self.confirmation_count = 0

    def decision(self, now):
        now = float(now)
        if self.stop_latched:
            return StopLineDecision(
                StopLinePhase.STOP, self.last_front_bumper_distance_m,
                True, "stop_line_latched")
        if (self.last_measurement_at is None or not math.isfinite(now) or
                now-self.last_measurement_at >
                self.config.stop_line_measurement_timeout_sec):
            return StopLineDecision(
                StopLinePhase.NORMAL, None, False,
                "stop_line_measurement_unavailable")
        distance = self.last_front_bumper_distance_m
        if distance is not None and distance <= \
                self.config.stop_line_slowdown_distance_m:
            return StopLineDecision(
                StopLinePhase.SLOW, distance, False, "stop_line_slowdown")
        return StopLineDecision(
            StopLinePhase.NORMAL, distance, False, "stop_line_clear")

    def release_stop(self):
        """Explicit future behavior-arbiter hook; never called automatically."""
        self.stop_latched = False
        self.confirmation_count = 0
        self.last_front_bumper_distance_m = None
        self.last_measurement_at = None
        self.last_measurement_reason = "explicit_release"
