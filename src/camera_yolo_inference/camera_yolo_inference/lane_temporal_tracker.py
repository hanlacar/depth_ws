"""Short-lived, motion-compensated temporal support for lane masks.

Raw detector masks are never relabelled or mutated.  A tracked-only mask is
produced only after that same class has been observed, and only while the
current raw road supplies geometric support.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneTemporalConfig:
    mode: str = "none"  # none | hold | flow
    max_hold_sec: float = 0.4
    max_hold_frames: int = 12
    confidence_decay: float = 0.88
    min_flow_points: int = 18
    max_motion_px: float = 45.0
    min_road_overlap: float = 0.65
    scene_change_threshold: float = 0.30
    max_frame_gap_sec: float = 0.12
    feature_dilation_px: int = 15
    road_boundary_margin_px: int = 11
    min_scale: float = 0.82
    max_scale: float = 1.22

    def validate(self):
        if self.mode not in ("none", "hold", "flow"):
            raise ValueError("lane temporal mode must be none, hold, or flow")
        if self.max_hold_sec <= 0.0 or self.max_hold_frames < 1:
            raise ValueError("lane temporal hold limits must be positive")
        if not 0.0 < self.confidence_decay <= 1.0:
            raise ValueError("lane temporal confidence decay must be in (0, 1]")
        if self.min_flow_points < 3 or self.max_motion_px <= 0.0:
            raise ValueError("lane temporal flow limits are invalid")
        if not 0.0 <= self.min_road_overlap <= 1.0:
            raise ValueError("lane temporal road overlap must be in [0, 1]")


@dataclass
class _LaneState:
    mask: np.ndarray | None = None
    stamp: float | None = None
    age_frames: int = 0
    confidence: float = 0.0


@dataclass
class LaneTemporalResult:
    raw_white_line: np.ndarray
    tracked_white_line: np.ndarray
    raw_yellow_line: np.ndarray
    tracked_yellow_line: np.ndarray
    diagnostics: dict

    @property
    def effective_white_line(self):
        return self.raw_white_line | self.tracked_white_line

    @property
    def effective_yellow_line(self):
        return self.raw_yellow_line | self.tracked_yellow_line


def _binary(mask, shape):
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError("lane temporal mask shape mismatch")
    return ((value > 0).astype(np.uint8) * 255)


def _iou(first, second):
    union = int(np.count_nonzero((first > 0) | (second > 0)))
    return (0.0 if not union else
            float(np.count_nonzero((first > 0) & (second > 0))/union))


class LaneMaskTemporalTracker:
    """Track white and yellow masks independently across short dropouts."""

    def __init__(self, config=None):
        self.config = config or LaneTemporalConfig()
        self.config.validate()
        self._states = {"white_line": _LaneState(),
                        "yellow_line": _LaneState()}
        self._previous_gray = None
        self._previous_stamp = None

    def reset(self):
        self._states = {"white_line": _LaneState(),
                        "yellow_line": _LaneState()}
        self._previous_gray = None
        self._previous_stamp = None

    @staticmethod
    def _gray(image):
        value = np.asarray(image, np.uint8)
        return (cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
                if value.ndim == 3 else value)

    def _scene_change(self, previous, current):
        difference = float(np.mean(cv2.absdiff(previous, current))) / 255.0
        hist_previous = cv2.calcHist([previous], [0], None, [32], [0, 256])
        hist_current = cv2.calcHist([current], [0], None, [32], [0, 256])
        correlation = float(cv2.compareHist(
            hist_previous, hist_current, cv2.HISTCMP_CORREL))
        changed = (difference > self.config.scene_change_threshold or
                   correlation < 0.35)
        return changed, difference, correlation

    def _estimate_motion(self, previous_gray, current_gray, mask):
        radius = self.config.feature_dilation_px
        support = cv2.dilate(mask, np.ones((2*radius+1, 2*radius+1), np.uint8))
        points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=160, qualityLevel=0.01,
            minDistance=4, mask=support)
        if points is None or len(points) < self.config.min_flow_points:
            return None, {"flow_points": 0 if points is None else int(len(points)),
                          "flow_inliers": 0, "flow_quality": 0.0,
                          "motion_px": None, "motion_scale": None}
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      30, 0.01))
        if current is None or status is None:
            return None, {"flow_points": int(len(points)), "flow_inliers": 0,
                          "flow_quality": 0.0, "motion_px": None,
                          "motion_scale": None}
        valid = status.reshape(-1) > 0
        before = points.reshape(-1, 2)[valid]
        after = current.reshape(-1, 2)[valid]
        if len(before) < self.config.min_flow_points:
            return None, {"flow_points": int(len(before)), "flow_inliers": 0,
                          "flow_quality": 0.0, "motion_px": None,
                          "motion_scale": None}
        matrix, inliers = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2.5,
            maxIters=500, confidence=0.98)
        inlier_count = 0 if inliers is None else int(np.count_nonzero(inliers))
        quality = float(inlier_count / max(1, len(before)))
        if matrix is None or inlier_count < self.config.min_flow_points:
            return None, {"flow_points": int(len(before)),
                          "flow_inliers": inlier_count,
                          "flow_quality": quality, "motion_px": None,
                          "motion_scale": None}
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        motion = float(np.hypot(matrix[0, 2], matrix[1, 2]))
        diagnostics = {"flow_points": int(len(before)),
                       "flow_inliers": inlier_count,
                       "flow_quality": quality, "motion_px": motion,
                       "motion_scale": scale}
        if (motion > self.config.max_motion_px or
                not self.config.min_scale <= scale <= self.config.max_scale):
            return None, diagnostics
        return matrix, diagnostics

    def _track_one(self, name, raw, road, previous_gray, current_gray,
                   stamp, dt, global_reset):
        state = self._states[name]
        zero = np.zeros_like(raw)
        diagnostics = {"source": "NONE", "track_age_frames": 0,
                       "confidence": 0.0, "road_overlap": 0.0,
                       "discard_reason": ""}
        if np.any(raw):
            if state.mask is not None and self.config.mode == "flow":
                matrix, flow = self._estimate_motion(
                    previous_gray, current_gray, state.mask)
                diagnostics.update(flow)
                if matrix is not None:
                    predicted = cv2.warpAffine(
                        state.mask, matrix, (raw.shape[1], raw.shape[0]),
                        flags=cv2.INTER_NEAREST, borderValue=0)
                    union = np.count_nonzero((predicted > 0) | (raw > 0))
                    diagnostics["raw_track_iou"] = (0.0 if union == 0 else
                        float(np.count_nonzero((predicted > 0) & (raw > 0))/union))
            state.mask, state.stamp = raw.copy(), stamp
            state.age_frames, state.confidence = 0, 1.0
            diagnostics.update({"source": "RAW", "confidence": 1.0})
            return zero, diagnostics
        if self.config.mode == "none":
            state.mask = None
            diagnostics["discard_reason"] = "TRACKING_DISABLED"
            return zero, diagnostics
        if global_reset:
            state.mask = None
            diagnostics["discard_reason"] = global_reset
            return zero, diagnostics
        if state.mask is None or state.stamp is None:
            diagnostics["discard_reason"] = "NO_PRIOR_RAW_DETECTION"
            return zero, diagnostics
        age_frames = state.age_frames + 1
        age_sec = stamp - state.stamp
        if (age_frames > self.config.max_hold_frames or
                age_sec > self.config.max_hold_sec):
            state.mask = None
            diagnostics["discard_reason"] = "HOLD_LIMIT"
            return zero, diagnostics
        tracked = state.mask.copy()
        if self.config.mode == "flow":
            matrix, flow = self._estimate_motion(
                previous_gray, current_gray, state.mask)
            diagnostics.update(flow)
            if matrix is None:
                state.mask = None
                diagnostics["discard_reason"] = "OPTICAL_FLOW_INVALID"
                return zero, diagnostics
            tracked = cv2.warpAffine(
                state.mask, matrix, (raw.shape[1], raw.shape[0]),
                flags=cv2.INTER_NEAREST, borderValue=0)
        if self.config.mode == "hold":
            state.age_frames = age_frames
            state.confidence *= self.config.confidence_decay
            diagnostics.update({"source": "TRACKED",
                                "track_age_frames": age_frames,
                                "confidence": float(state.confidence)})
            return tracked, diagnostics
        support = cv2.dilate(
            road, np.ones((2*self.config.road_boundary_margin_px+1,
                           2*self.config.road_boundary_margin_px+1), np.uint8))
        area = int(np.count_nonzero(tracked))
        overlap = (0.0 if area == 0 else
                   float(np.count_nonzero((tracked > 0) & (support > 0))/area))
        diagnostics["road_overlap"] = overlap
        if area == 0 or overlap < self.config.min_road_overlap:
            state.mask = None
            diagnostics["discard_reason"] = "ROAD_SUPPORT_INVALID"
            return zero, diagnostics
        state.mask = tracked
        state.age_frames = age_frames
        state.confidence *= self.config.confidence_decay
        diagnostics.update({"source": "TRACKED",
                            "track_age_frames": age_frames,
                            "confidence": float(state.confidence)})
        return tracked, diagnostics

    def update(self, image, raw_white_line, raw_yellow_line, raw_road,
               stamp_sec):
        gray = self._gray(image)
        shape = gray.shape
        white = _binary(raw_white_line, shape)
        yellow = _binary(raw_yellow_line, shape)
        road = _binary(raw_road, shape)
        stamp = float(stamp_sec)
        reset_reason = ""
        dt = None if self._previous_stamp is None else stamp-self._previous_stamp
        scene_change = False
        scene_difference = 0.0
        histogram_correlation = 1.0
        if self._previous_gray is not None:
            scene_change, scene_difference, histogram_correlation = self._scene_change(
                self._previous_gray, gray)
        if self._previous_stamp is not None and dt <= 0.0:
            reset_reason = "TIMESTAMP_REWIND"
        elif dt is not None and dt > self.config.max_frame_gap_sec:
            reset_reason = "TIMESTAMP_GAP"
        elif scene_change:
            reset_reason = "SCENE_CHANGE"
        previous = self._previous_gray
        if previous is None:
            previous = gray
        # T1 intentionally represents a naive, time-limited same-pixel hold.
        # It resets on timestamp failures but has no motion/scene gate, making
        # it a meaningful safety baseline rather than a second copy of T2.
        class_reset = (reset_reason if self.config.mode == "flow" else
                       reset_reason if reset_reason.startswith("TIMESTAMP_") else "")
        tracked_white, white_diag = self._track_one(
            "white_line", white, road, previous, gray, stamp, dt, class_reset)
        tracked_yellow, yellow_diag = self._track_one(
            "yellow_line", yellow, road, previous, gray, stamp, dt, class_reset)
        # A newly observed opposite class always outranks a propagated mask.
        # Discard the whole propagated track instead of carving/re-labelling it;
        # that makes cross-class identity fail closed and diagnosable.
        if self.config.mode == "flow":
            white_conflict = _iou(tracked_white, yellow)
            yellow_conflict = _iou(tracked_yellow, white)
            white_diag["opposite_raw_iou"] = white_conflict
            yellow_diag["opposite_raw_iou"] = yellow_conflict
            if white_conflict > 0.15:
                tracked_white[:] = 0
                self._states["white_line"].mask = None
                white_diag.update({"source": "NONE", "track_age_frames": 0,
                                   "confidence": 0.0,
                                   "discard_reason": "OPPOSITE_CLASS_CONFLICT"})
            if yellow_conflict > 0.15:
                tracked_yellow[:] = 0
                self._states["yellow_line"].mask = None
                yellow_diag.update({"source": "NONE", "track_age_frames": 0,
                                    "confidence": 0.0,
                                    "discard_reason": "OPPOSITE_CLASS_CONFLICT"})
        self._previous_gray = gray.copy()
        self._previous_stamp = stamp
        diagnostics = {
            "mode": self.config.mode,
            "white_line_source": white_diag["source"],
            "yellow_line_source": yellow_diag["source"],
            "white_line_track_age_frames": white_diag["track_age_frames"],
            "yellow_line_track_age_frames": yellow_diag["track_age_frames"],
            "scene_change_detected": bool(scene_change),
            "scene_difference": scene_difference,
            "histogram_correlation": histogram_correlation,
            "timestamp_delta_sec": dt,
            "reset_reason": reset_reason,
            "white_line": white_diag,
            "yellow_line": yellow_diag,
        }
        return LaneTemporalResult(white, tracked_white, yellow,
                                  tracked_yellow, diagnostics)
