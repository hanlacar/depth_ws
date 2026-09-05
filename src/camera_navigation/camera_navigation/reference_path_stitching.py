"""ROS-independent odom path transformation, stitching, and pruning."""

from dataclasses import dataclass
import math

import numpy as np

from .metric_path_quality import (
    MetricPathAnalysis, MetricPathQualityConfig, angle_difference_rad,
    analyze_metric_path, cumulative_s, has_self_intersection, path_heading,
    remove_consecutive_duplicates)


@dataclass(frozen=True)
class PlanarTransform:
    x_m: float
    y_m: float
    yaw_rad: float

    def validate(self):
        if not all(math.isfinite(value) for value in
                   (self.x_m, self.y_m, self.yaw_rad)):
            raise ValueError("planar transform must be finite")


def transform_points(points, transform):
    transform.validate()
    array = np.asarray(points, dtype=float)
    if (array.ndim != 2 or array.shape[1:] != (2,) or
            not np.all(np.isfinite(array))):
        raise ValueError("points must be a finite Nx2 array")
    cosine, sine = math.cos(transform.yaw_rad), math.sin(transform.yaw_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    return array @ rotation.T + np.array([transform.x_m, transform.y_m])


@dataclass(frozen=True)
class StitchConfig:
    overlap_distance_threshold_m: float = 0.75
    stitch_max_position_error_m: float = 1.5
    stitch_max_heading_error_deg: float = 30.0
    stitch_min_overlap_points: int = 3
    reference_path_keep_behind_m: float = 5.0
    reference_path_max_total_m: float = 50.0
    reference_path_target_forward_m: float = 10.0
    minimum_spacing_m: float = 0.05

    def validate(self):
        values = (
            self.overlap_distance_threshold_m,
            self.stitch_max_position_error_m,
            self.stitch_max_heading_error_deg,
            self.reference_path_keep_behind_m,
            self.reference_path_max_total_m,
            self.reference_path_target_forward_m,
            self.minimum_spacing_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("stitching distance/angle parameters must be positive")
        if self.stitch_min_overlap_points < 2:
            raise ValueError("stitch_min_overlap_points must be at least two")
        if self.reference_path_max_total_m <= self.reference_path_keep_behind_m:
            raise ValueError("maximum total path must exceed keep-behind distance")
        if self.reference_path_target_forward_m > self.reference_path_max_total_m:
            raise ValueError("target forward distance exceeds maximum total path")


@dataclass(frozen=True)
class StitchResult:
    accepted: bool
    reason: str
    points: np.ndarray
    appended_points: int
    overlap_points: int
    maximum_position_error_m: float | None
    maximum_heading_error_deg: float | None
    stitched_length_m: float
    forward_usable_length_m: float
    headings_rad: np.ndarray


def _length(points):
    if len(points) < 2:
        return 0.0
    return float(cumulative_s(points)[-1])


class ReferencePathStitcher:
    def __init__(self, config=StitchConfig()):
        config.validate()
        self.config = config
        self.points = np.empty((0, 2), dtype=float)
        self.headings = np.empty((0,), dtype=float)

    def reset(self):
        """Forget all path state so a later mission starts from a clean segment."""
        self.points = np.empty((0, 2), dtype=float)
        self.headings = np.empty((0,), dtype=float)

    def _result(self, accepted, reason, appended=0, overlap=0,
                position_error=None, heading_error=None, vehicle_xy=None):
        length = _length(self.points)
        forward = 0.0
        if len(self.points) and vehicle_xy is not None:
            vehicle = np.asarray(vehicle_xy, dtype=float)
            nearest = int(np.argmin(np.linalg.norm(self.points-vehicle, axis=1)))
            arc = cumulative_s(self.points)
            forward = float(arc[-1]-arc[nearest])
        return StitchResult(
            accepted, reason, self.points.copy(), int(appended), int(overlap),
            position_error, heading_error, length, forward,
            self.headings.copy())

    def _prune(self, vehicle_xy):
        if len(self.points) < 2:
            return
        vehicle = np.asarray(vehicle_xy, dtype=float)
        arc = cumulative_s(self.points)
        nearest = int(np.argmin(np.linalg.norm(self.points-vehicle, axis=1)))
        vehicle_s = float(arc[nearest])
        start_s = max(0.0, vehicle_s-self.config.reference_path_keep_behind_m)
        start = int(np.searchsorted(arc, start_s, side="left"))
        end_s = start_s+self.config.reference_path_max_total_m
        end = int(np.searchsorted(arc, end_s, side="right"))
        end = max(start+2, min(len(self.points), end))
        self.points = self.points[start:end]
        self.headings = self.headings[start:end]

    @staticmethod
    def _derived_headings(points):
        return np.asarray(
            [path_heading(points, index) for index in range(len(points))],
            dtype=float)

    def update(self, new_points, vehicle_xy, new_headings=None):
        new = np.asarray(new_points, dtype=float)
        vehicle = np.asarray(vehicle_xy, dtype=float)
        if (new.ndim != 2 or new.shape[1:] != (2,) or len(new) < 2 or
                vehicle.shape != (2,) or not np.all(np.isfinite(new)) or
                not np.all(np.isfinite(vehicle))):
            return self._result(False, "invalid_stitch_input", vehicle_xy=vehicle)
        new, _removed = remove_consecutive_duplicates(
            new, self.config.minimum_spacing_m)
        if len(new) < 2 or has_self_intersection(new):
            return self._result(False, "invalid_new_path", vehicle_xy=vehicle)
        headings = (self._derived_headings(new) if new_headings is None else
                    np.asarray(new_headings, dtype=float))
        if (headings.shape != (len(new),) or
                not np.all(np.isfinite(headings))):
            return self._result(False, "invalid_new_headings", vehicle_xy=vehicle)
        headings = (headings+math.pi) % (2.0*math.pi)-math.pi

        if len(self.points) == 0:
            self.points = new.copy()
            self.headings = headings.copy()
            self._prune(vehicle)
            return self._result(True, "initialized", len(self.points),
                                vehicle_xy=vehicle)

        distances = np.linalg.norm(
            new[:, None, :]-self.points[None, :, :], axis=2)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(new)), nearest_indices]
        overlap_indices = np.flatnonzero(
            nearest_distances <= self.config.overlap_distance_threshold_m)
        unique_existing = np.unique(nearest_indices[overlap_indices])
        if (len(overlap_indices) < self.config.stitch_min_overlap_points or
                len(unique_existing) < self.config.stitch_min_overlap_points):
            return self._result(False, "insufficient_overlap", overlap=len(overlap_indices),
                                vehicle_xy=vehicle)

        matched_existing = nearest_indices[overlap_indices]
        monotonic_fraction = float(np.mean(np.diff(matched_existing) >= 0))
        if monotonic_fraction < 0.8:
            return self._result(False, "overlap_direction_reversal",
                                overlap=len(overlap_indices), vehicle_xy=vehicle)
        if int(np.max(matched_existing)) < len(self.points)-self.config.stitch_min_overlap_points:
            return self._result(False, "overlap_does_not_reach_tail",
                                overlap=len(overlap_indices), vehicle_xy=vehicle)

        position_error = float(np.max(nearest_distances[overlap_indices]))
        if position_error > self.config.stitch_max_position_error_m:
            return self._result(False, "position_error", overlap=len(overlap_indices),
                                position_error=position_error, vehicle_xy=vehicle)

        heading_errors = []
        for new_index in overlap_indices:
            old_index = int(nearest_indices[new_index])
            heading_errors.append(abs(math.degrees(angle_difference_rad(
                path_heading(new, int(new_index)),
                path_heading(self.points, old_index)))))
        heading_error = float(max(heading_errors, default=0.0))
        if heading_error > self.config.stitch_max_heading_error_deg:
            return self._result(False, "heading_error", overlap=len(overlap_indices),
                                position_error=position_error,
                                heading_error=heading_error, vehicle_xy=vehicle)

        # Continue after the new point that overlaps the furthest old point.
        tail_pair = int(np.argmax(matched_existing))
        last_overlap_new = int(overlap_indices[tail_pair])
        extension = new[last_overlap_new+1:]
        extension_headings = headings[last_overlap_new+1:]
        if len(extension):
            connection = float(np.linalg.norm(extension[0]-self.points[-1]))
            if connection > self.config.stitch_max_position_error_m:
                return self._result(False, "append_position_jump",
                                    overlap=len(overlap_indices),
                                    position_error=connection,
                                    heading_error=heading_error,
                                    vehicle_xy=vehicle)
            connection_heading = math.atan2(
                extension[0, 1]-self.points[-1, 1],
                extension[0, 0]-self.points[-1, 0])
            tail_heading_error = abs(math.degrees(angle_difference_rad(
                connection_heading, path_heading(self.points, len(self.points)-1))))
            if tail_heading_error > self.config.stitch_max_heading_error_deg:
                return self._result(False, "append_heading_jump",
                                    overlap=len(overlap_indices),
                                    position_error=connection,
                                    heading_error=tail_heading_error,
                                    vehicle_xy=vehicle)
            merged = np.vstack((self.points, extension))
            merged_headings = np.concatenate((self.headings, extension_headings))
            keep = [0]
            for index in range(1, len(merged)):
                if float(np.linalg.norm(
                        merged[index]-merged[keep[-1]])) >= self.config.minimum_spacing_m:
                    keep.append(index)
            merged = merged[keep]
            merged_headings = merged_headings[keep]
            if has_self_intersection(merged):
                return self._result(False, "append_self_intersection",
                                    overlap=len(overlap_indices),
                                    position_error=connection,
                                    heading_error=tail_heading_error,
                                    vehicle_xy=vehicle)
            self.points = merged
            self.headings = merged_headings
        appended = len(extension)
        self._prune(vehicle)
        return self._result(True, "appended" if appended else "duplicate_frame",
                            appended, len(overlap_indices), position_error,
                            heading_error, vehicle)


@dataclass(frozen=True)
class AdapterCoreResult:
    accepted: bool
    state: str
    reason: str
    metric: MetricPathAnalysis
    stitch: StitchResult | None


class ReferencePathAdapterCore:
    """Validate base_link path and consume, never create, an odom transform."""

    def __init__(self, metric_config=MetricPathQualityConfig(),
                 stitch_config=StitchConfig()):
        metric_config.validate()
        self.metric_config = metric_config
        self.stitcher = ReferencePathStitcher(stitch_config)

    def reset(self):
        self.stitcher.reset()

    def process(self, base_points, transform=None, base_headings=None):
        metric = analyze_metric_path(base_points, self.metric_config)
        if not metric.valid:
            return AdapterCoreResult(False, "metric_path_invalid", metric.reason,
                                     metric, None)
        if transform is None:
            return AdapterCoreResult(False, "waiting_for_tf", "tf_unavailable",
                                     metric, None)
        try:
            odom_points = transform_points(metric.points, transform)
        except ValueError:
            return AdapterCoreResult(False, "waiting_for_tf", "tf_invalid",
                                     metric, None)
        odom_headings = None
        if base_headings is not None:
            original = np.asarray(base_points, dtype=float)
            headings = np.asarray(base_headings, dtype=float)
            if (headings.shape != (len(original),) or
                    not np.all(np.isfinite(headings))):
                return AdapterCoreResult(False, "metric_path_invalid",
                                         "invalid_pose_orientation", metric, None)
            selected = []
            start = 0
            for point in metric.points:
                matches = np.flatnonzero(np.all(np.isclose(
                    original[start:], point, rtol=0.0, atol=1.0e-9), axis=1))
                if not len(matches):
                    return AdapterCoreResult(False, "metric_path_invalid",
                                             "orientation_alignment_failed",
                                             metric, None)
                index = start+int(matches[0])
                selected.append(headings[index]+transform.yaw_rad)
                start = index+1
            odom_headings = np.asarray(selected, dtype=float)
        stitch = self.stitcher.update(
            odom_points, np.array([transform.x_m, transform.y_m]),
            odom_headings)
        return AdapterCoreResult(stitch.accepted,
                                 "active" if stitch.accepted else "frame_rejected",
                                 stitch.reason, metric, stitch)
