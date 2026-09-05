"""ROS-independent tests for the librealsense Direct-BEV adaptation."""

import numpy as np

from camera_navigation.direct_bev_core import (
    BOTH, LEFT_ONLY, RIGHT_ONLY, DirectBevConfig, DirectBevPlanner,
)
from camera_navigation.librealsense_bev_path import (
    extract_sliding_window_lanes,
)


def masks(planner, center=lambda x: 0.0, left=True, right=True,
          end=8.0, half_width=0.75):
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x_m in np.arange(0.30, end, planner.config.resolution_m):
        middle = center(x_m)
        row, column = planner.metric_to_grid([[x_m, middle]])[0]
        half = int(round(half_width/planner.config.resolution_m))
        road[max(0, row-1):min(planner.rows, row+2),
             max(0, column-half):min(planner.cols, column+half+1)] = 1
        offset = int(round(0.55/planner.config.resolution_m))
        if left:
            lane[max(0, row-1):min(planner.rows, row+2),
                 column+offset-1:column+offset+2] = 1
        if right:
            lane[max(0, row-1):min(planner.rows, row+2),
                 column-offset-1:column-offset+2] = 1
    return road, lane


def extract(planner, lane, road):
    return extract_sliding_window_lanes(
        lane, road, x_max_m=planner.config.x_max_m,
        y_min_m=planner.config.y_min_m,
        resolution_m=planner.config.resolution_m,
        windows=planner.config.sliding_windows,
        margin_m=planner.config.window_half_width_m,
        recenter_pixels=planner.config.window_min_pixels,
        minimum_points=planner.config.minimum_path_points,
        degree=planner.config.fitting_degree,
        samples=planner.config.sliding_windows)


def test_reference_metric_fit_tracks_both_curved_boundaries():
    planner = DirectBevPlanner(DirectBevConfig())
    road, lane = masks(planner, center=lambda x: 0.025*x*x)
    result = extract(planner, lane, road)
    assert len(result.left) == planner.config.sliding_windows
    assert len(result.right) == planner.config.sliding_windows
    center = 0.5*(result.left[:, 1]+result.right[:, 1])
    expected = 0.025*result.left[:, 0]**2
    assert np.max(np.abs(center-expected)) < 0.08


def test_sliding_windows_reconnect_a_short_lane_gap():
    planner = DirectBevPlanner(DirectBevConfig())
    road, lane = masks(planner)
    row_a, _ = planner.metric_to_grid([[2.1, 0.0]])[0]
    row_b, _ = planner.metric_to_grid([[2.5, 0.0]])[0]
    lane[min(row_a, row_b):max(row_a, row_b)+1] = 0
    result = extract(planner, lane, road)
    assert len(result.left) == planner.config.sliding_windows
    assert len(result.right) == planner.config.sliding_windows
    assert result.diagnostics["left_occupied_windows"] >= 3
    assert result.diagnostics["right_occupied_windows"] >= 3


def test_full_grid_seed_recovers_unobservable_near_field():
    planner = DirectBevPlanner(DirectBevConfig())
    road, lane = masks(planner)
    lane[planner.rows//2:] = 0
    result = extract(planner, lane, road)
    assert len(result.left) and len(result.right)
    assert result.diagnostics["seed_scope"] == "full_grid_fallback"


def test_direct_planner_preserves_bilateral_and_one_sided_modes():
    for left, right, expected in (
            (True, True, BOTH), (True, False, LEFT_ONLY),
            (False, True, RIGHT_ONLY)):
        planner = DirectBevPlanner(DirectBevConfig())
        road, lane = masks(planner, left=left, right=right)
        result = planner.plan(road, lane, 1.0)
        assert result.valid and result.mode == expected
        grid = planner.metric_to_grid(result.points)
        assert np.all(result.component[grid[:, 0], grid[:, 1]] > 0)
        assert np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0)
        assert abs(result.diagnostics["required_steering_deg"]) <= 22.0


def test_reference_extractor_is_deterministic_at_equal_fork_scores():
    planner = DirectBevPlanner(DirectBevConfig())
    road, lane = masks(planner)
    fork = lane.copy()
    for x_m in np.arange(2.5, 7.5, planner.config.resolution_m):
        row, col = planner.metric_to_grid([[x_m, 1.10]])[0]
        fork[max(0, row-1):row+2, col-1:col+2] = 1
    outputs = [extract(planner, fork, road).left for _ in range(3)]
    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])
