"""Synthetic safety tests for the hybrid_a6-only road-edge candidate."""

from pathlib import Path

import numpy as np
import pytest

from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.direct_bev_planner_node import build_direct_bev_planner
from camera_navigation.hybrid_bev_candidate import (
    HybridCandidateOptions, HybridDirectBevCandidate,
)


def planner(mode="gated"):
    return HybridDirectBevCandidate(options=HybridCandidateOptions(
        temporal_smoothing=True, curvature_stabilization=True,
        fixed_resample_origin=True, fail_closed_hold=True,
        road_boundary_fallback=mode))


def corridor(p, center=lambda x: 0.0, half_width=lambda x: 1.0,
             end=6.0, left_lane=False, right_lane=False,
             hole=None, branch=False):
    road = np.zeros((p.rows, p.cols), np.uint8)
    lane = np.zeros_like(road)
    for x_m in np.arange(.30, end, p.config.resolution_m):
        middle = center(x_m)
        row, col = p.metric_to_grid([[x_m, middle]])[0]
        half = int(round(half_width(x_m)/p.config.resolution_m))
        road[max(0, row-1):row+2, max(0, col-half):min(p.cols, col+half+1)] = 1
        offset = int(round(.55/p.config.resolution_m))
        if left_lane:
            lane[max(0, row-1):row+2, col+offset-1:col+offset+2] = 1
        if right_lane:
            lane[max(0, row-1):row+2, col-offset-1:col-offset+2] = 1
        if branch and x_m > 3.0:
            extra = int(round((middle+1.2*(x_m-3.0)-p.config.y_min_m)/
                              p.config.resolution_m))
            road[max(0, row-1):row+2, min(col, extra):max(col, extra)+1] = 1
    if hole is not None:
        x_m, y_m, radius_m = hole
        row, col = p.metric_to_grid([[x_m, y_m]])[0]
        radius = int(round(radius_m/p.config.resolution_m))
        import cv2
        cv2.circle(road, (int(col), int(row)), radius, 0, -1)
    return road, lane


def assert_safe(p, result):
    assert result.valid
    grid = p.metric_to_grid(result.points)
    assert np.all(result.component[grid[:, 0], grid[:, 1]] > 0)
    assert np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0)
    assert np.all(np.diff(result.points[:, 0]) >= -1.0e-9)
    assert abs(result.diagnostics["required_steering_deg"]) <= 22.0


@pytest.mark.parametrize("coefficient", [0.0, 0.012, -0.012])
def test_no_lane_clear_boundaries_generate_safe_metric_center(coefficient):
    p = planner()
    road, lane = corridor(p, center=lambda x: coefficient*x*x)
    result = p.plan(road, lane, 1.0)
    assert result.diagnostics["path_source"] == "ROAD_BOUNDARY_BOTH"
    assert result.diagnostics["boundary_valid_slice_count"] >= 6
    assert result.diagnostics["left_boundary_source"] == "ROAD"
    assert result.diagnostics["right_boundary_source"] == "ROAD"
    assert_safe(p, result)


@pytest.mark.parametrize("lanes,source", [
    ((True, True), "LANE_BOTH"),
    ((True, False), "LANE_LEFT_ROAD_RIGHT"),
    ((False, True), "ROAD_LEFT_LANE_RIGHT"),
])
def test_real_lanes_have_priority_and_one_sided_uses_current_road_edge(lanes, source):
    p = planner()
    road, lane = corridor(p, left_lane=lanes[0], right_lane=lanes[1])
    result = p.plan(road, lane, 1.0)
    assert result.diagnostics["path_source"] == source
    assert_safe(p, result)


def test_wide_parking_mask_is_rejected_by_gated_but_not_basic_candidate():
    basic, gated = planner("basic"), planner("gated")
    road, lane = corridor(basic, half_width=lambda _x: 2.6)
    basic_result = basic.plan(road, lane, 1.0)
    gated_result = gated.plan(road, lane, 1.0)
    assert basic_result.valid
    assert basic_result.diagnostics["observed_road_width_m"] > 4.5
    assert not gated_result.valid
    assert gated_result.diagnostics["path_source"] == "NONE"


def test_roi_edge_contact_and_missing_current_road_fail_closed():
    p = planner()
    road, lane = corridor(p, center=lambda _x: 2.2,
                          half_width=lambda _x: 1.0)
    assert not p.plan(road, lane, 1.0).valid
    empty = np.zeros_like(road)
    missing = p.plan(empty, empty, 1.4)
    assert not missing.valid and not len(missing.points)
    assert missing.diagnostics["path_source"] == "NONE"


def test_hole_and_gradual_narrowing_do_not_leave_safe_road():
    p = planner()
    road, lane = corridor(
        p, half_width=lambda x: max(.58, 1.2-.08*x),
        hole=(3.0, .55, .20))
    result = p.plan(road, lane, 1.0)
    if result.valid:
        assert_safe(p, result)
    else:
        assert result.diagnostics["path_source"] == "NONE"


def test_branch_widening_is_not_silently_followed_as_mask_center():
    p = planner()
    road, lane = corridor(p, branch=True)
    result = p.plan(road, lane, 1.0)
    if result.valid:
        assert result.diagnostics["observed_road_width_m"] <= 4.5
        assert_safe(p, result)


def test_production_rejects_boundary_opt_in_and_defaults_remain_unchanged():
    config = DirectBevConfig()
    with pytest.raises(ValueError, match="hybrid_a6-only"):
        build_direct_bev_planner("production", config, "gated")

    launch_dir = Path(__file__).parents[1]/"launch"
    for name in ("direct_bev_video_validation.launch.py",
                 "camera_bev_standalone.launch.py"):
        text = (launch_dir/name).read_text()
        assert 'DeclareLaunchArgument("planner_variant", default_value="production")' in text
        assert 'DeclareLaunchArgument("road_boundary_fallback", default_value="none")' in text
    production = build_direct_bev_planner("production", config)
    assert production.__class__.__name__ == "DirectBevPlanner"
