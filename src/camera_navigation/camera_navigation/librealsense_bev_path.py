"""Metric-BEV adaptation of the librealsense lane-center extractor.

Reference implementation:
``~/librealsense/catkin_ws_0922/catkin_ws/src/local_planner/src/``
``lane_center_extractor.py``.  Its ``find_lane_pixels_sliding_window`` and
``fit_lane_centerline`` functions seed tracks from a near-field histogram,
follow them with bottom-to-top sliding windows, and fit a quadratic boundary.

This module preserves that algorithm, but deliberately does not copy the
reference raster constants (100x80 at 0.1 m/pixel) or its 2 m one-sided lane
offset.  Callers supply the commissioned camera_ws metric grid and vehicle
geometry.  There are no ROS imports, so identical masks can be tested offline.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlidingWindowLaneResult:
    left: np.ndarray
    right: np.ndarray
    diagnostics: dict


def _empty_points():
    return np.empty((0, 2), dtype=float)


def _histogram_base(histogram, start, stop):
    """Return a deterministic occupied peak in ``[start, stop)``."""
    start, stop = max(0, int(start)), min(len(histogram), int(stop))
    if stop <= start:
        return None
    section = np.asarray(histogram[start:stop])
    if not len(section) or float(np.max(section)) <= 0.0:
        return None
    # np.argmax's first-maximum rule matches the librealsense implementation
    # and makes equal-score forks deterministic.
    return int(start+np.argmax(section))


def _window_track(binary, base_col, windows, margin_px, recenter_pixels):
    if base_col is None:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0
    rows, cols = binary.shape
    nonzero_rows, nonzero_cols = binary.nonzero()
    height = max(1, int(np.ceil(rows/float(windows))))
    current = int(base_col)
    selected = []
    occupied_windows = 0
    # camera_ws row 0 is far and the last row is near, identical to the
    # bottom-to-top traversal used after the reference OccupancyGrid flips.
    for window in range(int(windows)):
        high = rows-window*height
        low = max(0, rows-(window+1)*height)
        if high <= 0:
            break
        indices = np.flatnonzero(
            (nonzero_rows >= low) & (nonzero_rows < high) &
            (nonzero_cols >= current-margin_px) &
            (nonzero_cols < current+margin_px))
        if len(indices):
            selected.append(indices)
            occupied_windows += 1
            if len(indices) > int(recenter_pixels):
                current = int(round(float(np.mean(nonzero_cols[indices]))))
                current = int(np.clip(current, 0, cols-1))
    if not selected:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), occupied_windows
    indices = np.concatenate(selected)
    return nonzero_rows[indices], nonzero_cols[indices], occupied_windows


def _fit_metric_track(rows, cols, *, x_max_m, y_min_m, resolution_m,
                      degree, minimum_points, samples):
    if len(rows) < int(minimum_points):
        return _empty_points(), float("inf")
    x = float(x_max_m)-np.asarray(rows, float)*float(resolution_m)
    y = float(y_min_m)+np.asarray(cols, float)*float(resolution_m)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < int(minimum_points) or float(np.ptp(x)) <= 1.0e-9:
        return _empty_points(), float("inf")
    fit_degree = min(int(degree), len(x)-1)
    coefficients = np.polyfit(x, y, fit_degree)
    fit_x = np.linspace(float(np.min(x)), float(np.max(x)), int(samples))
    fit_y = np.polyval(coefficients, fit_x)
    residual = float(np.mean(np.abs(y-np.polyval(coefficients, x))))
    return np.column_stack((fit_x, fit_y)), residual


def extract_sliding_window_lanes(lane, component, *, x_max_m, y_min_m,
                                 resolution_m, windows, margin_m,
                                 recenter_pixels, minimum_points, degree,
                                 samples):
    """Fit left/right metric lane tracks using the reference algorithm.

    Metric contract is camera_ws ``base_link``: x increases forward, y is
    positive left.  Raster rows run far-to-near and columns run y_min-to-y_max.
    """
    lane = (np.asarray(lane) > 0).astype(np.uint8)
    component = (np.asarray(component) > 0).astype(np.uint8)
    if lane.shape != component.shape or lane.ndim != 2:
        raise ValueError("lane and component must be same-shape 2-D masks")
    binary = lane & component
    rows, cols = binary.shape
    histogram = np.sum(binary[rows//2:, :], axis=0)
    center = int(round((0.0-float(y_min_m))/float(resolution_m)))
    # camera_ws y is positive-left, so left occupies columns above center;
    # the reference image used the opposite pixel sign after its flips.
    left_base = _histogram_base(histogram, center+1, cols)
    right_base = _histogram_base(histogram, 0, center)
    seed_scope = "near_half"
    # The reference depth BEV observed the vehicle-near half.  A monocular
    # ground remap can legitimately begin farther forward, leaving that half
    # empty.  Fall back to the same histogram rule over the full metric grid;
    # this changes only seed availability, not coordinates or fitted geometry.
    if left_base is None or right_base is None:
        full_histogram = np.sum(binary, axis=0)
        if left_base is None:
            left_base = _histogram_base(full_histogram, center+1, cols)
        if right_base is None:
            right_base = _histogram_base(full_histogram, 0, center)
        seed_scope = "full_grid_fallback"
    margin_px = max(1, int(round(float(margin_m)/float(resolution_m))))
    left_rows, left_cols, left_windows = _window_track(
        binary, left_base, windows, margin_px, recenter_pixels)
    right_rows, right_cols, right_windows = _window_track(
        binary, right_base, windows, margin_px, recenter_pixels)
    left, left_residual = _fit_metric_track(
        left_rows, left_cols, x_max_m=x_max_m, y_min_m=y_min_m,
        resolution_m=resolution_m, degree=degree,
        minimum_points=int(minimum_points)*max(1, int(recenter_pixels)),
        samples=samples)
    right, right_residual = _fit_metric_track(
        right_rows, right_cols, x_max_m=x_max_m, y_min_m=y_min_m,
        resolution_m=resolution_m, degree=degree,
        minimum_points=int(minimum_points)*max(1, int(recenter_pixels)),
        samples=samples)
    # A quadratic fit through a few pixels in one raster band is not a lane
    # track.  The reference enforced broad support with >100 ROI pixels and
    # recentering across windows.  Express the same structural requirement
    # using camera_ws's existing minimum-path/window parameters.
    if left_windows < int(minimum_points):
        left, left_residual = _empty_points(), float("inf")
    if right_windows < int(minimum_points):
        right, right_residual = _empty_points(), float("inf")
    return SlidingWindowLaneResult(left=left, right=right, diagnostics={
        "reference": "librealsense/local_planner/lane_center_extractor.py",
        "left_base_col": left_base,
        "right_base_col": right_base,
        "seed_scope": seed_scope,
        "left_pixels": int(len(left_rows)),
        "right_pixels": int(len(right_rows)),
        "left_occupied_windows": int(left_windows),
        "right_occupied_windows": int(right_windows),
        "left_fit_residual_m": left_residual,
        "right_fit_residual_m": right_residual,
    })
