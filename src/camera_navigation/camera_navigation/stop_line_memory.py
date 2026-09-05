"""Stop-line tracking decoupled from the non-BEV path planner (req 2).

image_path_planner.py excludes the ego bumper/bonnet from every path-shaping
computation (road center, branch evidence, candidate path, virtual center,
jump baseline). That exclusion must NOT blind the vehicle to whether it has
already crossed a stop line: once the line's pixels vanish under the bonnet,
the planner has no way to keep tracking it. This module is that separate
memory. It is fed raw stop-line pixels/depth directly (never the planner's
excluded masks) and reasons about ego-frame longitudinal distance instead of
image geometry, so it keeps working after the line is no longer visible.

State machine
-------------
NOT_SEEN               -> no stop line ever observed (or memory expired)
DETECTED_FAR            -> visible, still far from stop_line_slowdown_distance_m
APPROACHING              -> visible (or fresh memory), inside slowdown range
NEAR_BUMPER_OCCLUDED    -> last-seen row was low enough in frame that the
                           bumper/bonnet is expected to occlude it next frame;
                           tracked purely by memory (last distance + odom)
CROSSED_FRONT_AXLE       -> memory distance, projected forward by odom, has
                           passed the front axle (distance <= 0 at the axle)
PASSED                   -> latched terminal state once crossed; stays until
                           reset() (e.g. a new mission leg / new stop line)
"""

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np


class StopLineState(str, Enum):
    NOT_SEEN = "NOT_SEEN"
    DETECTED_FAR = "DETECTED_FAR"
    APPROACHING = "APPROACHING"
    NEAR_BUMPER_OCCLUDED = "NEAR_BUMPER_OCCLUDED"
    CROSSED_FRONT_AXLE = "CROSSED_FRONT_AXLE"
    PASSED = "PASSED"


@dataclass(frozen=True)
class StopLineMemoryConfig:
    # Vehicle-fixed geometry: distance from the camera's optical center to
    # the front bumper, and from the camera to the front axle. Both must be
    # measured on the deployed vehicle; front_axle_offset_m is normally
    # smaller than camera_to_bumper_m (the axle sits behind the bumper).
    # camera x=+0.015, vehicle front x=+0.500, front axle x=+0.365.
    camera_to_bumper_m: float = 0.485
    front_axle_offset_m: float = 0.350
    # A detection whose pixel row is at/below this fraction of the frame
    # height is treated as "about to be occluded by the bumper" even while
    # still technically visible -- it primes NEAR_BUMPER_OCCLUDED instead of
    # waiting for the detection to vanish outright.
    near_bumper_row_ratio: float = 0.92
    # Slowdown/stop policy thresholds, measured from the front bumper (i.e.
    # already net of camera_to_bumper_m), matching stop_line_control.py's
    # convention so both modules can share one downstream policy consumer.
    slowdown_distance_m: float = 2.0
    stop_distance_m: float = 0.7
    # A depth/distance sample below this confidence is treated the same as
    # "not observed this frame" (falls back to memory) rather than being
    # ingested and possibly corrupting the tracked distance.
    depth_min_confidence: float = 0.35
    # If no fresh detection AND no odom update refreshes the memory for this
    # long, the memory is discarded (NOT_SEEN) rather than extrapolated
    # indefinitely on odom alone.
    memory_timeout_sec: float = 3.0
    # PASSED is latched once CROSSED_FRONT_AXLE has held for this many
    # seconds of continued forward motion (guards against a single noisy
    # odom sample flipping the terminal state early).
    passed_confirm_sec: float = 0.15

    def validate(self):
        finite = (self.camera_to_bumper_m, self.front_axle_offset_m,
                  self.near_bumper_row_ratio, self.slowdown_distance_m,
                  self.stop_distance_m, self.depth_min_confidence,
                  self.memory_timeout_sec, self.passed_confirm_sec)
        if not all(math.isfinite(v) for v in finite):
            raise ValueError("stop-line memory parameters must be finite")
        if self.camera_to_bumper_m < 0.0:
            raise ValueError("camera_to_bumper_m must be nonnegative")
        if self.front_axle_offset_m < 0.0:
            raise ValueError("front_axle_offset_m must be nonnegative")
        if not 0.0 < self.near_bumper_row_ratio <= 1.0:
            raise ValueError("near_bumper_row_ratio must be in (0, 1]")
        if not 0.0 <= self.stop_distance_m < self.slowdown_distance_m:
            raise ValueError(
                "stop_distance_m must be nonnegative and less than "
                "slowdown_distance_m")
        if not 0.0 <= self.depth_min_confidence <= 1.0:
            raise ValueError("depth_min_confidence must be in [0, 1]")
        if self.memory_timeout_sec <= 0.0:
            raise ValueError("memory_timeout_sec must be positive")
        if self.passed_confirm_sec < 0.0:
            raise ValueError("passed_confirm_sec must be nonnegative")


@dataclass(frozen=True)
class StopLineObservation:
    """One frame's raw evidence, read directly from the un-excluded masks."""
    detected: bool
    pixel_row: float | None = None
    image_height: float | None = None
    camera_distance_m: float | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class StopLineStatus:
    state: StopLineState
    camera_distance_m: float | None
    front_bumper_distance_m: float | None
    front_axle_distance_m: float | None
    crossed_front_axle: bool
    age_sec: float | None
    reason: str


def _axle_distance(camera_distance_m, cfg):
    return camera_distance_m - cfg.front_axle_offset_m


def _bumper_distance(camera_distance_m, cfg):
    return camera_distance_m - cfg.camera_to_bumper_m


class StopLineMemory:
    """ROS-independent stop-line state machine (req 2).

    Reads directly from the raw stop-line mask/depth passed by the caller --
    never from image_path_planner's ego-excluded geometry -- so bumper
    occlusion of the visual signal does not stop distance tracking. Distance
    is carried forward through occlusion using the last confident camera
    distance plus accumulated odom forward travel.
    """

    def __init__(self, config=StopLineMemoryConfig()):
        config.validate()
        self.config = config
        self.state = StopLineState.NOT_SEEN
        self.camera_distance_m = None
        self.last_update_at = None
        self.last_seen_at = None
        self.odom_reference_m = None
        self.crossed_at = None
        self.reason = "never_observed"
        self._last_seen_near_bumper = False

    def reset(self):
        self.state = StopLineState.NOT_SEEN
        self.camera_distance_m = None
        self.last_update_at = None
        self.last_seen_at = None
        self.odom_reference_m = None
        self.crossed_at = None
        self.reason = "reset"
        self._last_seen_near_bumper = False

    def update(self, observation, odom_distance_m, now_sec):
        """Advance the state machine by one frame.

        ``odom_distance_m`` is a monotonically-accumulated forward-travel
        odometer reading (meters); only its delta since the last confident
        fix is used, so wraparound/reset of the odometer itself is not a
        concern as long as it does not jump discontinuously.
        """
        now_sec = float(now_sec)
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")
        cfg = self.config

        if (self.last_update_at is not None and
                now_sec-self.last_update_at > cfg.memory_timeout_sec and
                self.state != StopLineState.PASSED):
            self.reset()
            self.reason = "memory_timeout"

        odom_valid = odom_distance_m is not None and math.isfinite(odom_distance_m)
        if odom_valid and self.odom_reference_m is not None and self.camera_distance_m is not None:
            traveled = float(odom_distance_m)-self.odom_reference_m
            if traveled > 0.0:
                self.camera_distance_m = self.camera_distance_m-traveled
        if odom_valid:
            self.odom_reference_m = float(odom_distance_m)

        near_bumper_pixel = (
            observation.detected and observation.pixel_row is not None and
            observation.image_height not in (None, 0) and
            (observation.pixel_row/observation.image_height) >=
            cfg.near_bumper_row_ratio)
        confident_fix = (
            observation.detected and observation.camera_distance_m is not None and
            math.isfinite(observation.camera_distance_m) and
            observation.confidence >= cfg.depth_min_confidence)

        if confident_fix:
            self.camera_distance_m = float(observation.camera_distance_m)
            self.last_seen_at = now_sec
            self.reason = "ok"
            self._last_seen_near_bumper = near_bumper_pixel
        elif observation.detected:
            self.reason = "low_confidence_detection"
        else:
            self.reason = "not_visible"
        self.last_update_at = now_sec

        if self.camera_distance_m is None:
            self.state = StopLineState.NOT_SEEN
            return self.status()

        front_axle_distance = _axle_distance(self.camera_distance_m, cfg)
        already_crossed = front_axle_distance <= 0.0

        if already_crossed:
            if self.crossed_at is None:
                self.crossed_at = now_sec
            elapsed = now_sec-self.crossed_at
            self.state = (StopLineState.PASSED
                          if elapsed >= cfg.passed_confirm_sec
                          else StopLineState.CROSSED_FRONT_AXLE)
        else:
            self.crossed_at = None
            if confident_fix and near_bumper_pixel:
                self.state = StopLineState.NEAR_BUMPER_OCCLUDED
            elif not confident_fix and not observation.detected:
                # Lost visually. If the last confident sighting was already
                # low enough in frame to be about to go under the
                # bonnet/bumper, that -- not merely a small residual
                # distance -- is what licenses memory-only tracking (req
                # 2.4). Otherwise the loss is unexplained (segmentation
                # dropout, e.g. exposure/shadow) and the last known state is
                # held rather than reclassified.
                if self._last_seen_near_bumper:
                    self.state = StopLineState.NEAR_BUMPER_OCCLUDED
                elif self.state in (StopLineState.NEAR_BUMPER_OCCLUDED,
                                    StopLineState.APPROACHING,
                                    StopLineState.DETECTED_FAR):
                    pass  # hold last state; distance still tracked via odom
                else:
                    self.state = StopLineState.DETECTED_FAR
            else:
                front_bumper_distance = _bumper_distance(
                    self.camera_distance_m, cfg)
                self.state = (StopLineState.APPROACHING
                              if front_bumper_distance <= cfg.slowdown_distance_m
                              else StopLineState.DETECTED_FAR)
        return self.status()

    def status(self):
        cfg = self.config
        if self.camera_distance_m is None:
            return StopLineStatus(
                self.state, None, None, None, False, None, self.reason)
        front_bumper = _bumper_distance(self.camera_distance_m, cfg)
        front_axle = _axle_distance(self.camera_distance_m, cfg)
        age = (None if self.last_update_at is None
              else max(0.0, (self.last_update_at-self.last_seen_at)
                       if self.last_seen_at is not None else 0.0))
        return StopLineStatus(
            self.state, self.camera_distance_m, front_bumper, front_axle,
            self.state in (StopLineState.CROSSED_FRONT_AXLE,
                          StopLineState.PASSED),
            age, self.reason)

    @property
    def maximum_drive(self):
        """Longitudinal ceiling matching stop_line_control's contract."""
        cfg = self.config
        status = self.status()
        if status.front_bumper_distance_m is None:
            return math.inf
        if status.crossed_front_axle:
            return math.inf
        if status.front_bumper_distance_m <= cfg.stop_distance_m:
            return 0.0
        if status.front_bumper_distance_m <= cfg.slowdown_distance_m:
            return 1.0
        return math.inf


def stop_line_pixel_row(stop_line_mask):
    """Lowest (closest-to-camera) row of a raw stop-line mask, or None."""
    mask = np.asarray(stop_line_mask) > 0
    if not np.any(mask):
        return None
    rows = np.flatnonzero(np.any(mask, axis=1))
    return float(rows.max())


def estimate_distance_from_row(pixel_row, image_height, row_to_distance_m):
    """Optional monocular fallback distance when no depth channel exists.

    ``row_to_distance_m`` is a small calibration table: an iterable of
    (row_ratio, distance_m) pairs, sorted by increasing row_ratio, built
    from a handful of measured stop-line placements for one fixed camera
    mount. This is NOT a general ground-plane/BEV projection -- it only
    ever answers "how far is the stop-line class blob", so it does not
    reintroduce BEV geometry into the non-BEV planner.
    """
    if pixel_row is None or not image_height:
        return None
    table = list(row_to_distance_m)
    if len(table) < 2:
        return None
    ratios = np.asarray([r for r, _ in table], dtype=float)
    distances = np.asarray([d for _, d in table], dtype=float)
    ratio = float(np.clip(pixel_row/float(image_height), ratios[0], ratios[-1]))
    return float(np.interp(ratio, ratios, distances))
