"""Adaptive original-image path planning extensions (no BEV geometry)."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .image_path_planner import (
    BOTH_BOUNDARIES, DEGRADED, ImagePathPlanner, INVALID, LEFT_BOUNDARY,
    PathResult, RIGHT_BOUNDARY, ROAD_CENTER, TEMPORAL_FALLBACK,
)
from .pixel_lateral_control import lookahead_offset_px, steering_from_offset_deg


@dataclass(frozen=True)
class AdaptiveNonBevConfig:
    """Pixel-domain values intentionally kept separate from metric/BEV config."""

    band_count: int = 30
    robust_fit_iterations: int = 3
    robust_fit_residual_px: float = 18.0
    robust_fit_min_inlier_ratio: float = 0.55
    near_fit_weight: float = 2.0
    road_dt_peak_ratio: float = 0.90
    road_center_gate_near_px: float = 28.0
    road_center_gate_mid_px: float = 42.0
    road_center_gate_far_px: float = 64.0
    safe_margin_near_px: float = 18.0
    safe_margin_mid_px: float = 11.0
    safe_margin_far_px: float = 5.0
    hold_time_sec: float = 0.20
    lookahead_min_ratio: float = 0.25
    lookahead_max_ratio: float = 0.78
    lookahead_recovery_step: float = 0.08

    def validate(self):
        if self.band_count < 6:
            raise ValueError("band_count must be at least 6")
        if self.robust_fit_iterations < 1:
            raise ValueError("robust_fit_iterations must be positive")
        if self.robust_fit_residual_px <= 0.0:
            raise ValueError("robust_fit_residual_px must be positive")
        if not 0.0 < self.robust_fit_min_inlier_ratio <= 1.0:
            raise ValueError("robust_fit_min_inlier_ratio must be in (0, 1]")
        if not 0.0 < self.road_dt_peak_ratio <= 1.0:
            raise ValueError("road_dt_peak_ratio must be in (0, 1]")
        if not 0.0 <= self.lookahead_min_ratio <= self.lookahead_max_ratio <= 1.0:
            raise ValueError("lookahead range must be inside [0, 1]")
        if self.hold_time_sec < 0.0:
            raise ValueError("hold_time_sec must be nonnegative")


class AdaptiveNonBevPlanner(ImagePathPlanner):
    """Adds robust bands, medial-road tracking and time-bounded fallback."""

    def __init__(self, config=None, adaptive=None):
        super().__init__(config)
        self.adaptive = adaptive or AdaptiveNonBevConfig()
        self.adaptive.validate()
        span = max(1, self.config.roi_bottom-self.config.roi_top)
        self.config.sample_interval_px = max(
            1, int(round(span/max(1, self.adaptive.band_count-1))))
        # Make the configured boot-time one-side offset available through the
        # same perspective profile used after real BOTH observations.
        self._seed_width_profile()
        self._adaptive_fit_rejected = np.empty((0, 2), dtype=float)
        self._last_fresh_timestamp = None
        self._last_fresh_result = None

    def reset(self):
        super().reset()
        self._seed_width_profile()
        self._adaptive_fit_rejected = np.empty((0, 2), dtype=float)
        self._last_fresh_timestamp = None
        self._last_fresh_result = None

    def _seed_width_profile(self):
        if self.config.lane_width_seed_px > 0.0 and not self.width_profile:
            for y in self._row_list(
                    100000, self.config.roi_top, self.config.roi_bottom,
                    self.config.sample_interval_px):
                self.width_profile[int(y)] = float(self.config.lane_width_seed_px)

    def _perspective_value(self, y, near, middle, far):
        top, bottom = self.config.roi_top, self.config.roi_bottom
        progress = np.clip((float(y)-top)/max(1.0, bottom-top), 0.0, 1.0)
        if progress >= 0.5:
            return middle+(near-middle)*(2.0*progress-1.0)
        return far+(middle-far)*(2.0*progress)

    def _road_geometry(self, road, ys):
        """Use distance-transform medial candidates inside the tracked run."""
        observations = super()._road_geometry(road, ys)
        mask = (np.asarray(road) > 0).astype(np.uint8)
        if not observations or not np.any(mask):
            return observations
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        previous = float(self.config.vehicle_center_x_px)
        previous_path = np.asarray(self.previous) if self.previous is not None else None
        for y in [int(value) for value in ys if int(value) in observations]:
            item = observations[y]
            lo, hi = int(np.ceil(item["left"])), int(np.floor(item["right"]))
            if previous_path is not None and len(previous_path):
                order = np.argsort(previous_path[:, 1])
                predicted = float(np.interp(
                    y, previous_path[order, 1], previous_path[order, 0]))
            else:
                predicted = previous
            gate = self._perspective_value(
                y, self.adaptive.road_center_gate_near_px,
                self.adaptive.road_center_gate_mid_px,
                self.adaptive.road_center_gate_far_px)
            lo = max(lo, int(np.floor(predicted-gate)))
            hi = min(hi, int(np.ceil(predicted+gate)))
            if hi < lo:
                continue
            values = distance[y, lo:hi+1]
            if not len(values) or float(np.max(values)) <= 0.0:
                continue
            peak = float(np.max(values))
            candidates = np.flatnonzero(
                values >= self.adaptive.road_dt_peak_ratio*peak)+lo
            center = float(candidates[np.argmin(np.abs(candidates-predicted))])
            item["raw_center"] = center
            item["center"] = center
            item["distance_clearance_px"] = peak
            item["center_method"] = "DISTANCE_TRANSFORM"
            previous = center
        return observations

    def _fit(self, raw, weights):
        """Reject residual outliers iteratively and weight near bands higher."""
        points = np.asarray(raw, dtype=float).reshape(-1, 2)
        base_weights = np.asarray(weights, dtype=float).reshape(-1)
        self._adaptive_fit_rejected = np.empty((0, 2), dtype=float)
        if len(points) < 3:
            return points.copy()
        y_scale = max(1.0, float(np.max(points[:, 1])))
        near = np.clip(points[:, 1]/y_scale, 0.0, 1.0)
        base_weights = base_weights*(1.0+(self.adaptive.near_fit_weight-1.0)*near)
        inliers = np.ones(len(points), dtype=bool)
        degree = min(self.config.polynomial_degree, len(points)-1)
        coefficients = None
        for _ in range(self.adaptive.robust_fit_iterations):
            if np.count_nonzero(inliers) < degree+1:
                break
            coefficients = np.polyfit(
                points[inliers, 1]/y_scale, points[inliers, 0], degree,
                w=base_weights[inliers])
            residual = np.abs(
                points[:, 0]-np.polyval(coefficients, points[:, 1]/y_scale))
            median = float(np.median(residual[inliers]))
            mad = float(np.median(np.abs(residual[inliers]-median)))
            threshold = max(
                self.adaptive.robust_fit_residual_px, median+2.0*1.4826*mad)
            candidate = residual <= threshold
            minimum = max(degree+1, int(np.ceil(
                self.adaptive.robust_fit_min_inlier_ratio*len(points))))
            if np.count_nonzero(candidate) < minimum:
                break
            if np.array_equal(candidate, inliers):
                inliers = candidate
                break
            inliers = candidate
        if coefficients is None:
            return super()._fit(points, base_weights)
        self._adaptive_fit_rejected = points[~inliers].copy()
        fitted_x = np.polyval(coefficients, points[:, 1]/y_scale)
        margin = max(0.0, self.config.fit_overshoot_margin_px)
        fitted_x = np.clip(
            fitted_x, np.min(points[inliers, 0])-margin,
            np.max(points[inliers, 0])+margin)
        return np.column_stack((fitted_x, points[:, 1]))

    def _final_safety_margin_px(self, y, roi_bottom):
        return self._perspective_value(
            y, self.adaptive.safe_margin_near_px,
            self.adaptive.safe_margin_mid_px,
            self.adaptive.safe_margin_far_px)

    def _steering_feasibility(self, points, image_width, timestamp_sec):
        """Try useful look-aheads and gate only the controller's local segment."""
        details = super()._steering_feasibility(points, image_width, timestamp_sec)
        array = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(array) < 3:
            return details
        desired = float(np.clip(
            self._lookahead_ratio(self._curvature_px(array)),
            self.adaptive.lookahead_min_ratio,
            self.adaptive.lookahead_max_ratio))
        step = max(0.01, self.adaptive.lookahead_recovery_step)
        candidates = []
        ratio = desired
        while ratio >= self.adaptive.lookahead_min_ratio-1e-9:
            candidates.append(float(max(self.adaptive.lookahead_min_ratio, ratio)))
            ratio -= step
        selected = candidates[-1]
        required = None
        offset = None
        for candidate in candidates:
            candidate_offset = lookahead_offset_px(array, image_width, candidate)
            if candidate_offset is None:
                continue
            candidate_steering = steering_from_offset_deg(
                candidate_offset, self._previous_feasibility_offset_px,
                details.get("steering_dt_sec", self.config.nominal_frame_period_sec),
                image_width, self.config.steering_proportional_gain_deg_per_norm,
                self.config.steering_derivative_gain_deg_per_norm_per_s, 1.0e6)
            selected, required, offset = candidate, candidate_steering, candidate_offset
            if abs(candidate_steering) <= self.config.maximum_steering_deg:
                break
        if required is None:
            return details
        details.update({
            "required_steering_deg": float(required),
            "steering_offset_px": float(offset),
            "steering_angle_ok": bool(
                abs(required) <= self.config.maximum_steering_deg),
            "selected_lookahead_y_ratio": float(selected),
            "lookahead_recovery_attempts": int(candidates.index(selected)),
            "lookahead_recovered": bool(selected < desired-1e-6),
        })
        return details

    @staticmethod
    def _generation_mode(sources):
        values = list(sources)
        if not values:
            return "INVALID"
        if all(value == TEMPORAL_FALLBACK for value in values):
            return "HOLD"
        counts = {value: values.count(value) for value in set(values)}
        dominant = max(counts, key=counts.get)
        return {
            BOTH_BOUNDARIES: "BOTH", LEFT_BOUNDARY: "LEFT_ONLY",
            RIGHT_BOUNDARY: "RIGHT_ONLY", ROAD_CENTER: "ROAD_ONLY",
        }.get(dominant, "INVALID")

    def plan(self, *args, timestamp_sec=None, **kwargs):
        result = super().plan(*args, timestamp_sec=timestamp_sec, **kwargs)
        now = time.monotonic() if timestamp_sec is None else float(timestamp_sec)
        mode = self._generation_mode(result.sources)
        if result.valid and mode != "HOLD":
            self._last_fresh_timestamp = now
            self._last_fresh_result = result
        elif (not result.valid and self._last_fresh_result is not None and
              self._last_fresh_timestamp is not None and
              now-self._last_fresh_timestamp <= self.adaptive.hold_time_sec):
            age = max(0.0, now-self._last_fresh_timestamp)
            cached = self._last_fresh_result
            confidence = cached.confidence*max(
                0.0, 1.0-age/max(1e-6, self.adaptive.hold_time_sec))
            diagnostics = dict(result.diagnostics or {})
            diagnostics.update({
                "generation_mode": "HOLD", "hold_age_sec": float(age),
                "hold_limit_sec": self.adaptive.hold_time_sec,
                "hold_reason": diagnostics.get(
                    "failure_reason", "CURRENT_FRAME_INVALID"),
            })
            result = PathResult(
                cached.points.copy(),
                [TEMPORAL_FALLBACK]*len(cached.points), confidence,
                DEGRADED, True, result.latency_ms, cached.road_component,
                cached.left, cached.right, cached.raw, diagnostics,
                cached.confidence_components, cached.virtual,
                cached.virtual_details)
            mode = "HOLD"
        elif mode == "HOLD":
            age = (float("inf") if self._last_fresh_timestamp is None
                   else max(0.0, now-self._last_fresh_timestamp))
            if age > self.adaptive.hold_time_sec:
                empty = np.empty((0, 2), dtype=float)
                diagnostics = dict(result.diagnostics or {})
                diagnostics.update({
                    "generation_mode": "INVALID", "failure_reason": "HOLD_EXPIRED",
                    "hold_age_sec": float(age), "hold_limit_sec": self.adaptive.hold_time_sec,
                })
                result = PathResult(
                    empty, [], 0.0, INVALID, False, result.latency_ms,
                    result.road_component, result.left, result.right, result.raw,
                    diagnostics, result.confidence_components,
                    result.virtual, result.virtual_details)
                mode = "INVALID"
            else:
                result.diagnostics["hold_age_sec"] = float(age)
        diagnostics = result.diagnostics or {}
        diagnostics.update({
            "generation_mode": mode,
            "hold_limit_sec": self.adaptive.hold_time_sec,
            "robust_fit_rejected_points": self._adaptive_fit_rejected.tolist(),
            "robust_fit_rejected_count": int(len(self._adaptive_fit_rejected)),
            "band_count_configured": self.adaptive.band_count,
            "band_interval_px": self.config.sample_interval_px,
        })
        if mode == "INVALID":
            diagnostics.setdefault("failure_reason", "NO_FEASIBLE_PATH")
        result.diagnostics = diagnostics
        return result
