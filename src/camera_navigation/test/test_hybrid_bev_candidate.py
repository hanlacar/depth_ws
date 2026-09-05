"""Targeted temporal regression tests for the offline hybrid candidate."""

import numpy as np

from camera_navigation.direct_bev_core import DirectBevConfig, HOLD
from camera_navigation.hybrid_bev_candidate import (
    HybridCandidateOptions, HybridDirectBevCandidate,
)
from camera_navigation.metric_path_quality import maximum_curvature
from camera_navigation.direct_bev_planner_node import build_direct_bev_planner


def masks(planner, center=lambda x: 0.0, left=True, right=True, end=4.0):
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x_m in np.arange(.30, end, planner.config.resolution_m):
        middle = center(x_m)
        row, column = planner.metric_to_grid([[x_m, middle]])[0]
        half = int(round(1.0/planner.config.resolution_m))
        offset = int(round(.55/planner.config.resolution_m))
        road[max(0, row-1):row+2, column-half:column+half+1] = 1
        if left:
            lane[max(0, row-1):row+2,
                 column+offset-1:column+offset+2] = 1
        if right:
            lane[max(0, row-1):row+2,
                 column-offset-1:column-offset+2] = 1
    return road, lane


def assert_safe(planner, result):
    grid = planner.metric_to_grid(result.points)
    assert np.all(result.component[grid[:, 0], grid[:, 1]] > 0)
    assert np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0)
    assert abs(result.diagnostics["required_steering_deg"]) <= 22.0


def test_temporal_curvature_jump_reproduces_and_safe_smoothing_recovers():
    plain = HybridDirectBevCandidate(options=HybridCandidateOptions())
    smooth = HybridDirectBevCandidate(options=HybridCandidateOptions(
        temporal_smoothing=True))
    for planner in (plain, smooth):
        road, lane = masks(planner)
        assert planner.plan(road, lane, 1.0).valid
    road, lane = masks(plain, center=lambda x: .02*x*x)
    rejected = plain.plan(road, lane, 1.30)
    assert not rejected.valid
    assert rejected.diagnostics["reasons"] == ["TEMPORAL_CURVATURE_JUMP"]
    road, lane = masks(smooth, center=lambda x: .02*x*x)
    recovered = smooth.plan(road, lane, 1.30)
    assert recovered.valid
    assert "TEMPORAL_PATH_SMOOTHED" in recovered.diagnostics["steering_recovery"]
    assert_safe(smooth, recovered)


def test_previous_path_association_drops_stale_prior_and_reacquires():
    planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
        previous_association=True))
    road, lane = masks(planner)
    assert planner.plan(road, lane, 1.0).valid
    road, lane = masks(planner, center=lambda x: .02*x*x)
    result = planner.plan(road, lane, 1.30)
    assert result.valid and result.mode != HOLD
    assert_safe(planner, result)


def test_resampling_is_stable_on_the_same_curve():
    planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
        temporal_smoothing=True, curvature_stabilization=True,
        fixed_resample_origin=True))
    outputs = []
    for frame in range(5):
        road, lane = masks(planner, center=lambda x: .015*x*x)
        result = planner.plan(road, lane, 1.0+frame/60.0)
        assert result.valid
        outputs.append(result.points.copy())
    assert all(np.array_equal(outputs[0][:, 0], path[:, 0])
               for path in outputs[1:])
    curvatures = [maximum_curvature(path) for path in outputs]
    assert all(abs(after-before) <= planner.config.temporal_curvature_gate_per_m
               for before, after in zip(curvatures, curvatures[1:]))


def test_mode_transition_hysteresis_holds_then_confirms():
    planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
        temporal_smoothing=True, curvature_stabilization=True,
        mode_hysteresis_frames=3))
    road, lane = masks(planner)
    first = planner.plan(road, lane, 1.0)
    assert first.valid
    modes = []
    for index in range(3):
        road, lane = masks(planner, left=False, right=False)
        result = planner.plan(road, lane, 1.02+.02*index)
        assert result.valid
        modes.append(result.mode)
    assert modes[:2] == [first.mode, first.mode]
    assert modes[2] != first.mode


def test_large_actual_curvature_change_is_not_hidden_by_smoothing():
    planner = HybridDirectBevCandidate(
        DirectBevConfig(temporal_curvature_gate_per_m=.30),
        HybridCandidateOptions(
            temporal_smoothing=True, curvature_stabilization=True))
    road, lane = masks(planner)
    assert planner.plan(road, lane, 1.0).valid
    previous = maximum_curvature(planner.previous)
    road, lane = masks(planner, center=lambda x: .02*x*x)
    rejected = planner.plan(road, lane, 1.30)
    raw_delta = abs(maximum_curvature(planner.last_raw_path)-previous)
    assert raw_delta > 3*planner.config.temporal_curvature_gate_per_m
    assert not rejected.valid
    assert rejected.diagnostics["reasons"] == ["TEMPORAL_CURVATURE_JUMP"]
    assert np.array_equal(planner.last_raw_path, planner.last_smoothed_path)


def test_timeout_mask_loss_remains_fail_safe():
    planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
        temporal_smoothing=True, previous_association=True,
        fail_closed_hold=True))
    road, lane = masks(planner)
    assert planner.plan(road, lane, 1.0).valid
    empty = np.zeros_like(road)
    held = planner.plan(empty, empty, 1.1)
    assert not held.valid
    assert held.diagnostics["reasons"] == ["HOLD_PATH_UNSAFE"]
    expired = planner.plan(empty, empty, 1.3)
    assert not expired.valid and not len(expired.points)


def test_ros_planner_selection_defaults_to_production_and_requires_opt_in():
    config = DirectBevConfig()
    production = build_direct_bev_planner("production", config)
    candidate = build_direct_bev_planner("hybrid_a6", config)
    assert production.__class__.__name__ == "DirectBevPlanner"
    assert isinstance(candidate, HybridDirectBevCandidate)


def test_a6_curved_masks_preserve_lateral_shape_and_steering_sign():
    """Positive metric y is left; MCU steering uses the opposite sign."""
    for coefficient, expected_required_sign in ((0.018, 1), (-0.018, -1)):
        planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
            fixed_resample_origin=True, fail_closed_hold=True))
        road, lane = masks(planner, center=lambda x, c=coefficient: c*x*x,
                           end=6.0)
        result = planner.plan(road, lane, 1.0)
        assert result.valid
        assert np.ptp(result.points[:, 1]) > 0.05
        assert np.sign(result.points[-1, 1]) == expected_required_sign
        assert np.sign(result.diagnostics["required_steering_deg"]) == \
            expected_required_sign
        assert maximum_curvature(result.points) > 1.0e-3


def test_a6_single_and_dual_lane_curves_follow_both_directions():
    for coefficient in (0.015, -0.015):
        for left, right in ((True, True), (True, False), (False, True),
                            (False, False)):
            planner = HybridDirectBevCandidate(options=HybridCandidateOptions(
                fixed_resample_origin=True, fail_closed_hold=True))
            road, lane = masks(
                planner, center=lambda x, c=coefficient: c*x*x,
                left=left, right=right, end=6.0)
            result = planner.plan(road, lane, 1.0)
            assert result.valid
            assert np.sign(result.points[-1, 1]) == np.sign(coefficient)
            assert np.ptp(result.points[:, 1]) > 0.04
