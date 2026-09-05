"""Offline hybrid Direct-BEV candidates; none is wired to production ROS.

The librealsense sliding-window extractor supplies spatial lane candidates.
All metric projection, ego-road, footprint clearance, steering and state
contracts remain inherited from :class:`DirectBevPlanner`.
"""

from dataclasses import dataclass

import numpy as np

from .direct_bev_core import (DirectBevPlanner, ROAD_ONLY,
                              pure_pursuit_unclipped)
from .librealsense_bev_path import extract_sliding_window_lanes
from .metric_path_quality import maximum_curvature


@dataclass(frozen=True)
class HybridCandidateOptions:
    temporal_smoothing: bool = False
    previous_association: bool = False
    curvature_stabilization: bool = False
    mode_hysteresis_frames: int = 0
    fixed_resample_origin: bool = False
    fail_closed_hold: bool = False
    # Validation-only A6 experiment. "none" preserves commissioned S0,
    # "basic" is B1, and "gated" is the fail-closed B2 candidate.
    road_boundary_fallback: str = "none"
    # Three metric slices are the minimum already required to represent a
    # path.  Farther slices that touch the BEV ROI stay rejected, deliberately
    # shortening the path instead of inventing an image-border road edge.
    boundary_minimum_slices: int = 3
    boundary_max_width_m: float = 4.50
    boundary_width_change_m: float = 0.80
    boundary_center_jump_m: float = 0.55


class HybridDirectBevCandidate(DirectBevPlanner):
    """Configurable ablation candidate that preserves production safety gates."""

    def __init__(self, config=None, options=None):
        super().__init__(config)
        self.options = options or HybridCandidateOptions()
        self.last_raw_path = np.empty((0, 2), float)
        self.last_candidate_path = np.empty((0, 2), float)
        self.last_resampled_path = np.empty((0, 2), float)
        self.last_smoothed_path = np.empty((0, 2), float)
        self.last_previous_path = np.empty((0, 2), float)
        self.last_candidate_mode = None
        self._pending_mode = None
        self._pending_mode_frames = 0
        self._current_component = None
        self._current_distance = None
        self.road_left_boundary = np.empty((0, 2), float)
        self.road_right_boundary = np.empty((0, 2), float)
        self.boundary_center = np.empty((0, 2), float)
        self.boundary_rejected_slice_count = 0
        self.observed_road_width_m = None
        self.road_edge_confidence = 0.0
        self._candidate_path_source = "NONE"
        self._previous_path_source = "NONE"

    def reset(self):
        super().reset()
        self.last_raw_path = np.empty((0, 2), float)
        self.last_candidate_path = np.empty((0, 2), float)
        self.last_resampled_path = np.empty((0, 2), float)
        self.last_smoothed_path = np.empty((0, 2), float)
        self.last_previous_path = np.empty((0, 2), float)
        self.last_candidate_mode = None
        self._pending_mode = None
        self._pending_mode_frames = 0
        self._current_component = None
        self._current_distance = None
        self._previous_path_source = "NONE"

    def plan(self, road_mask, lane_mask, timestamp_sec):
        self.last_raw_path = np.empty((0, 2), float)
        self.last_candidate_path = np.empty((0, 2), float)
        self.last_resampled_path = np.empty((0, 2), float)
        self.last_smoothed_path = np.empty((0, 2), float)
        self.last_previous_path = (
            np.empty((0, 2), float) if self.previous is None
            else self.previous.copy())
        self.last_candidate_mode = None
        self.road_left_boundary = np.empty((0, 2), float)
        self.road_right_boundary = np.empty((0, 2), float)
        self.boundary_center = np.empty((0, 2), float)
        self.boundary_rejected_slice_count = 0
        self.observed_road_width_m = None
        self.road_edge_confidence = 0.0
        self._candidate_path_source = "NONE"
        if (self.options.previous_association and
                self.previous_timestamp is not None and
                float(timestamp_sec)-self.previous_timestamp >
                self.config.hold_time_sec):
            # A stale path cannot be a temporal association target.  The
            # production implementation otherwise compares against it forever,
            # which can deadlock reacquisition after one rejected transition.
            self.previous = None
            self.previous_timestamp = None
            self.previous_mode = None
        result = super().plan(road_mask, lane_mask, timestamp_sec)
        result.diagnostics["candidate_raw_path"] = self.last_raw_path.tolist()
        result.diagnostics["candidate_spatial_path"] = self.last_candidate_path.tolist()
        result.diagnostics["candidate_resampled_path"] = self.last_resampled_path.tolist()
        result.diagnostics["candidate_smoothed_path"] = self.last_smoothed_path.tolist()
        result.diagnostics["previous_accepted_path"] = self.last_previous_path.tolist()
        source = "NONE"
        if result.valid:
            if result.mode == "BOTH":
                source = "LANE_BOTH"
            elif result.mode == "LEFT_ONLY":
                source = "LANE_LEFT_ROAD_RIGHT"
            elif result.mode == "RIGHT_ONLY":
                source = "ROAD_LEFT_LANE_RIGHT"
            elif result.mode == ROAD_ONLY:
                source = self._candidate_path_source or "ROAD_CENTER_FALLBACK"
            elif result.mode == "HOLD":
                source = self._previous_path_source
        result.diagnostics.update({
            "path_source": source,
            "left_boundary_source": (
                "LANE" if source in ("LANE_BOTH", "LANE_LEFT_ROAD_RIGHT")
                else "ROAD" if source in ("ROAD_LEFT_LANE_RIGHT",
                                           "ROAD_BOUNDARY_BOTH") else "NONE"),
            "right_boundary_source": (
                "LANE" if source in ("LANE_BOTH", "ROAD_LEFT_LANE_RIGHT")
                else "ROAD" if source in ("LANE_LEFT_ROAD_RIGHT",
                                           "ROAD_BOUNDARY_BOTH") else "NONE"),
            "observed_road_width_m": self.observed_road_width_m,
            "minimum_safe_width_m": (self.config.vehicle_width_m +
                                      2.0*self.config.lateral_safety_margin_m),
            "boundary_valid_slice_count": int(len(self.boundary_center)),
            "boundary_rejected_slice_count": int(
                self.boundary_rejected_slice_count),
            "road_component_connected": bool(
                self._current_component is not None and
                np.any(self._current_component)),
            "road_edge_confidence": float(self.road_edge_confidence),
            "path_length_m": float(result.diagnostics.get(
                "path_length_m", np.sum(np.linalg.norm(
                    np.diff(result.points, axis=0), axis=1))
                if len(result.points) > 1 else 0.0)),
            "failure_reason": "|".join(
                result.diagnostics.get("reasons", [])),
            "road_left_boundary": self.road_left_boundary.tolist(),
            "road_right_boundary": self.road_right_boundary.tolist(),
        })
        if result.valid and result.mode != "HOLD":
            self._previous_path_source = source
        return result

    def preprocess(self, road, lane):
        values = super().preprocess(road, lane)
        self._current_component, self._current_distance = values[2], values[4]
        return values

    def _road_boundary_pairs(self):
        """Extract metric outer corridor edges from the current ego component."""
        component = self._current_component
        distance = self._current_distance
        if component is None or distance is None or not np.any(component):
            return (np.empty((0, 2), float),)*3
        cfg = self.config
        clearance = cfg.vehicle_width_m/2.0+cfg.lateral_safety_margin_m
        minimum_width = cfg.vehicle_width_m+2.0*cfg.lateral_safety_margin_m
        edge_margin = max(1, int(round(0.08/cfg.resolution_m)))
        previous_width = None
        previous_center = None
        left, right, centers, widths = [], [], [], []
        rejected = 0
        for x_m in np.linspace(cfg.x_min_m, cfg.x_max_m,
                               cfg.sliding_windows):
            row = int(round((cfg.x_max_m-x_m)/cfg.resolution_m))
            if not 0 <= row < self.rows:
                rejected += 1; continue
            cols = np.flatnonzero(component[row])
            groups = (np.split(cols, np.flatnonzero(np.diff(cols) > 1)+1)
                      if len(cols) else [])
            if not groups:
                rejected += 1; continue
            reference = (float(np.interp(x_m, self.previous[:, 0],
                                         self.previous[:, 1]))
                         if self.previous is not None else 0.0)
            reference_col = int(round((reference-cfg.y_min_m)/cfg.resolution_m))
            group = min(groups, key=lambda values: (0 if values[0] <= reference_col <= values[-1]
                                                     else min(abs(values[0]-reference_col),
                                                              abs(values[-1]-reference_col))))
            first, last = int(group[0]), int(group[-1])
            width = (last-first)*cfg.resolution_m
            safe_cols = group[distance[row, group]*cfg.resolution_m >= clearance]
            touches_roi = first <= edge_margin or last >= self.cols-1-edge_margin
            if width < minimum_width or not len(safe_cols) or touches_roi:
                rejected += 1; continue
            road_right = cfg.y_min_m+first*cfg.resolution_m
            road_left = cfg.y_min_m+last*cfg.resolution_m
            midpoint = 0.5*(road_left+road_right)
            safe_y = cfg.y_min_m+safe_cols*cfg.resolution_m
            center = float(safe_y[np.argmin(np.abs(safe_y-midpoint))])
            if self.options.road_boundary_fallback == "gated":
                if (width > self.options.boundary_max_width_m or
                        (previous_width is not None and
                         abs(width-previous_width) >
                         self.options.boundary_width_change_m) or
                        (previous_center is not None and
                         abs(center-previous_center) >
                         self.options.boundary_center_jump_m)):
                    rejected += 1; continue
                # Compare the boundary midpoint with the vehicle axis and the
                # last accepted safe path without abandoning the two edges.
                center = 0.70*center+0.30*reference
                center = float(safe_y[np.argmin(np.abs(safe_y-center))])
            left.append((x_m, road_left)); right.append((x_m, road_right))
            centers.append((x_m, center)); widths.append(width)
            previous_width, previous_center = width, center
        self.boundary_rejected_slice_count = rejected
        self.road_left_boundary = np.asarray(left, float).reshape(-1, 2)
        self.road_right_boundary = np.asarray(right, float).reshape(-1, 2)
        self.boundary_center = np.asarray(centers, float).reshape(-1, 2)
        self.observed_road_width_m = (float(np.median(widths))
                                      if widths else None)
        total = len(centers)+rejected
        confidence = len(centers)/max(1, total)
        if len(widths) > 1:
            confidence *= max(0.0, 1.0-float(np.std(widths))/max(
                minimum_width, float(np.mean(widths))))
        self.road_edge_confidence = float(np.clip(confidence, 0.0, 1.0))
        return (self.road_left_boundary, self.road_right_boundary,
                self.boundary_center)

    def _road_center(self, distance, component):
        if self.options.road_boundary_fallback == "none":
            self._candidate_path_source = "ROAD_CENTER_FALLBACK"
            return super()._road_center(distance, component)
        _, _, center = self._road_boundary_pairs()
        minimum = (self.options.boundary_minimum_slices
                   if self.options.road_boundary_fallback == "gated"
                   else self.config.minimum_path_points)
        if len(center) >= minimum:
            self._candidate_path_source = "ROAD_BOUNDARY_BOTH"
            return center
        if self.options.road_boundary_fallback == "basic":
            self._candidate_path_source = "ROAD_CENTER_FALLBACK"
            return DirectBevPlanner._road_center(self, distance, component)
        self._candidate_path_source = "NONE"
        return np.empty((0, 2), float)

    def _offset_boundary(self, points, left):
        if self.options.road_boundary_fallback == "none":
            return super()._offset_boundary(points, left)
        road_left, road_right, _ = self._road_boundary_pairs()
        opposite = road_right if left else road_left
        points = np.asarray(points, float).reshape(-1, 2)
        if len(opposite) < self.config.minimum_path_points or len(points) < 3:
            return np.empty((0, 2)), np.empty((0, 2)), float("inf")
        usable = ((points[:, 0] >= opposite[:, 0].min()) &
                  (points[:, 0] <= opposite[:, 0].max()))
        selected = points[usable]
        if len(selected) < self.config.minimum_path_points:
            return np.empty((0, 2)), np.empty((0, 2)), float("inf")
        other_y = np.interp(selected[:, 0], opposite[:, 0], opposite[:, 1])
        width = selected[:, 1]-other_y if left else other_y-selected[:, 1]
        sane = ((width >= self.config.lane_width_min_m) &
                (width <= self.config.lane_width_max_m))
        candidate = np.column_stack((selected[sane, 0],
                                     0.5*(selected[sane, 1]+other_y[sane])))
        self._candidate_path_source = ("LANE_LEFT_ROAD_RIGHT" if left else
                                       "ROAD_LEFT_LANE_RIGHT")
        return candidate, selected[~sane], 0.0

    def _sample_lane_tracks(self, lane, component):
        cfg = self.config
        result = extract_sliding_window_lanes(
            lane, component, x_max_m=cfg.x_max_m, y_min_m=cfg.y_min_m,
            resolution_m=cfg.resolution_m, windows=cfg.sliding_windows,
            margin_m=cfg.window_half_width_m,
            recenter_pixels=cfg.window_min_pixels,
            minimum_points=cfg.minimum_path_points,
            degree=cfg.fitting_degree, samples=cfg.sliding_windows)
        return result.left, result.right

    def _fallback_or_invalid(self, timestamp, reason, road, safe, component,
                             left=None, right=None, rejected=None):
        result = super()._fallback_or_invalid(
            timestamp, reason, road, safe, component, left, right, rejected)
        # The inherited production path may HOLD when ``safe`` is completely
        # empty because its containment check is conditional on np.any(safe).
        # A candidate cannot count or publish such a path as drivable.
        if (self.options.fail_closed_hold and result.valid and result.mode == "HOLD" and
                (not np.any(safe) or
                 not self._points_inside_component(result.points, safe))):
            return self._invalid("HOLD_PATH_UNSAFE", road, safe, component,
                                 left, right, rejected)
        return result

    def _crosscheck_lane_candidate(self, raw, mode, distance, component,
                                   warnings):
        candidate = np.asarray(raw, float).reshape(-1, 2)
        if mode != ROAD_ONLY:
            road = self._road_center(distance, component)
            if len(road) >= self.config.minimum_path_points:
                overlap = ((candidate[:, 0] >= road[:, 0].min()) &
                           (candidate[:, 0] <= road[:, 0].max()))
                disagreement = (float("inf") if not np.any(overlap) else
                    float(np.percentile(np.abs(
                        candidate[overlap, 1]-np.interp(
                            candidate[overlap, 0], road[:, 0], road[:, 1])), 90)))
                lane_preview, _, _ = self._resample_unstabilized(candidate)
                road_preview, _, _ = self._resample_unstabilized(road)
                curve_delta = (abs(maximum_curvature(lane_preview)-
                                   maximum_curvature(road_preview))
                               if len(lane_preview) >= 3 and len(road_preview) >= 3
                               else float("inf"))
                previous_delta = 0.0
                if self.previous is not None and len(lane_preview) >= 3:
                    previous_delta = abs(maximum_curvature(lane_preview)-
                                         maximum_curvature(self.previous))
                if (disagreement > self.config.road_center_gate_m or
                        curve_delta > self.config.temporal_curvature_gate_per_m or
                        previous_delta > self.config.temporal_curvature_gate_per_m):
                    warnings.append("LANE_ROAD_CENTER_DISAGREEMENT")
                    candidate, mode = road, ROAD_ONLY

        required = self.options.mode_hysteresis_frames
        if required > 0 and self.previous_mode is not None and mode != self.previous_mode:
            if mode == self._pending_mode:
                self._pending_mode_frames += 1
            else:
                self._pending_mode = mode
                self._pending_mode_frames = 1
            if self._pending_mode_frames < required and self.previous is not None:
                warnings.append("MODE_TRANSITION_HELD")
                return self.previous.copy(), self.previous_mode
        else:
            self._pending_mode = None
            self._pending_mode_frames = 0
        self.last_candidate_mode = mode
        self.last_candidate_path = candidate.copy()
        return candidate, mode

    def _resample_unstabilized(self, points):
        return DirectBevPlanner._resample(self, points)

    def _resample(self, points):
        fitted, rejected, residual = self._resample_unstabilized(points)
        if self.options.fixed_resample_origin and len(fitted):
            start = float(np.min(points[:, 0]))
            end = float(np.max(points[:, 0]))
            spacing = self.config.resample_spacing_m
            first = (self.config.x_min_m +
                     np.ceil((start-self.config.x_min_m)/spacing)*spacing)
            coefficients, _, _ = self._robust_fit(points)
            if coefficients is not None and first <= end+1.0e-9:
                x = np.arange(first, end+0.5*spacing, spacing)
                fitted = np.column_stack((x, np.polyval(coefficients, x)))
        self.last_resampled_path = fitted.copy()
        return fitted, rejected, residual

    def _steering(self, points):
        """Evaluate A6 steering beyond the first camera-observable path point."""
        ratios = np.linspace(self.config.lookahead_min_m,
                             self.config.lookahead_max_m, 9)
        # With no safe near field, shortening lookahead moves the target back
        # onto the centre-axis connector.  Start at the commissioned maximum
        # preview and shorten only if the steering feasibility gate requires it.
        ratios = sorted(ratios, reverse=True)
        best = None
        for lookahead in ratios:
            angle, target = pure_pursuit_unclipped(
                points, self.config.wheelbase_m, lookahead,
                lookahead_from_path_start=True)
            if angle is None:
                continue
            best = (angle, target, float(lookahead))
            if abs(angle) <= self.config.maximum_steering_deg:
                return best+(True,)
        return (best+(False,)) if best else (None, None, None, False)

    def _connect_center_axis_start(self, points, safe):
        connected, recovery, error = super()._connect_center_axis_start(points, safe)
        self.last_raw_path = connected.copy()
        self.last_smoothed_path = connected.copy()
        if (error is not None or not self.options.temporal_smoothing or
                self.previous is None or len(connected) < 3):
            return connected, recovery, error

        previous_y = np.interp(connected[:, 0], self.previous[:, 0],
                               self.previous[:, 1], left=self.previous[0, 1],
                               right=self.previous[-1, 1])
        raw_delta = abs(maximum_curvature(connected)-
                        maximum_curvature(self.previous))
        # A genuinely abrupt change remains unsmoothed and is rejected by the
        # unchanged temporal gate.  Stabilization is only for gate-scale noise.
        if raw_delta > 3.0*self.config.temporal_curvature_gate_per_m:
            return connected, recovery, error
        alphas = ([0.75, 0.60, 0.45, 0.30, 0.15]
                  if self.options.curvature_stabilization else [0.50])
        selected = connected
        for alpha in alphas:
            candidate = connected.copy()
            candidate[:, 1] = alpha*connected[:, 1]+(1.0-alpha)*previous_y
            candidate[0, 1] = self.config.path_start_lateral_m
            if not self._points_inside_component(candidate, safe):
                continue
            _, _, _, feasible = self._steering(candidate)
            if not feasible:
                continue
            if (not self.options.curvature_stabilization or
                    abs(maximum_curvature(candidate)-
                        maximum_curvature(self.previous)) <=
                    0.85*self.config.temporal_curvature_gate_per_m):
                selected = candidate
                break
        if not np.array_equal(selected, connected):
            recovery = list(recovery) + ["TEMPORAL_PATH_SMOOTHED"]
        self.last_smoothed_path = selected.copy()
        return selected, recovery, error


def ablation_planners(config):
    """Return the requested A0--A6 sequence without altering production."""
    from .direct_bev_core import DirectBevPlanner
    return {
        "A0": DirectBevPlanner(config),
        "A1": HybridDirectBevCandidate(config, HybridCandidateOptions()),
        "A2": HybridDirectBevCandidate(config, HybridCandidateOptions(
            temporal_smoothing=True, fail_closed_hold=True)),
        "A3": HybridDirectBevCandidate(config, HybridCandidateOptions(
            previous_association=True, fail_closed_hold=True)),
        "A4": HybridDirectBevCandidate(config, HybridCandidateOptions(
            temporal_smoothing=True, curvature_stabilization=True,
            fail_closed_hold=True)),
        "A5": HybridDirectBevCandidate(config, HybridCandidateOptions(
            temporal_smoothing=True, curvature_stabilization=True,
            mode_hysteresis_frames=3, fail_closed_hold=True)),
        "A6": HybridDirectBevCandidate(config, HybridCandidateOptions(
            temporal_smoothing=True, curvature_stabilization=True,
            mode_hysteresis_frames=3, fixed_resample_origin=True,
            fail_closed_hold=True)),
    }
