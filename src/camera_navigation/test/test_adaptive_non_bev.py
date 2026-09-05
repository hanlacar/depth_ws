import numpy as np

from race_interfaces.msg import ImagePath, ImagePathPoint
from camera_navigation.adaptive_non_bev_planner import (
    AdaptiveNonBevConfig, AdaptiveNonBevPlanner,
)
from camera_navigation.adaptive_pixel_controller import (
    AdaptivePixelController, DynamicLookaheadConfig,
)
from camera_navigation.camera_pixel_controller_node import PixelControllerConfig
from camera_navigation.image_path_planner import PlannerConfig


def planner(**adaptive):
    base = PlannerConfig(
        roi_top=80, roi_bottom=220, vehicle_center_x_px=160.0,
        lane_width_seed_px=100.0, ego_exclusion_enabled=False,
        minimum_component_pixels=5, valid_min_confidence=0.0,
        road_minimum_near_width_px=30.0,
        road_minimum_near_coverage_ratio=0.2,
        minimum_boundary_clearance_m=0.2,
        final_path_safety_margin_near_px=2.0,
        final_path_safety_margin_mid_px=2.0,
        final_path_safety_margin_far_px=2.0,
    )
    return AdaptiveNonBevPlanner(base, AdaptiveNonBevConfig(**adaptive))


def trapezoid_masks(left_lane=True, right_lane=True, curve=0.0):
    road = np.zeros((240, 320), np.uint8)
    lane = np.zeros_like(road)
    for y in range(60, 231):
        t = (y-60)/170.0
        bend = curve*(1.0-t)**2
        left = int(145-85*t+bend)
        right = int(175+85*t+bend)
        road[y, max(0, left):min(320, right+1)] = 1
        if left_lane:
            lane[y, max(0, left-1):min(320, left+2)] = 1
        if right_lane:
            lane[y, max(0, right-1):min(320, right+2)] = 1
    return road, lane


def test_band_count_controls_perspective_sampling():
    item = planner(band_count=15)
    assert item.config.sample_interval_px == 10


def test_both_boundaries_straight_mode():
    road, lane = trapezoid_masks()
    result = planner().plan(road, lane, np.zeros_like(lane), timestamp_sec=1.0)
    assert result.valid
    assert result.diagnostics["generation_mode"] == "BOTH"


def test_both_boundaries_curve_is_retained():
    road, lane = trapezoid_masks(curve=30.0)
    result = planner().plan(road, lane, np.zeros_like(lane), timestamp_sec=1.0)
    assert len(result.points) >= 3
    assert result.diagnostics["generation_mode"] == "BOTH"


def test_left_only_uses_normal_offset_seed_inside_road():
    road, lane = trapezoid_masks(right_lane=False)
    result = planner().plan(road, lane, np.zeros_like(lane), timestamp_sec=1.0)
    assert result.diagnostics["generation_mode"] in ("LEFT_ONLY", "ROAD_ONLY")
    assert all(road[int(round(y)), int(round(x))] for x, y in result.points)


def test_right_only_uses_normal_offset_seed_inside_road():
    road, lane = trapezoid_masks(left_lane=False)
    result = planner().plan(road, lane, np.zeros_like(lane), timestamp_sec=1.0)
    assert result.diagnostics["generation_mode"] in ("RIGHT_ONLY", "ROAD_ONLY")
    assert all(road[int(round(y)), int(round(x))] for x, y in result.points)


def test_road_only_uses_distance_transform_candidates():
    road, lane = trapezoid_masks(False, False)
    result = planner().plan(road, lane, lane, timestamp_sec=1.0)
    assert len(result.points) >= 3
    assert result.diagnostics["generation_mode"] == "ROAD_ONLY"


def test_small_road_marking_gap_is_recovered():
    road, lane = trapezoid_masks(False, False)
    road[140:148, 145:175] = 0
    result = planner().plan(road, lane, lane, timestamp_sec=1.0)
    assert len(result.points) >= 3


def test_bump_marking_refined_road_keeps_non_bev_path():
    road, lane = trapezoid_masks(False, False)
    restored = road.copy()
    broken = road.copy()
    broken[132:154] = 0
    result = planner().plan(
        restored, lane, np.zeros_like(lane), timestamp_sec=1.0)
    assert np.count_nonzero(broken) < np.count_nonzero(restored)
    assert result.valid
    assert result.diagnostics["generation_mode"] == "ROAD_ONLY"


def test_shadow_side_expansion_is_temporally_gated():
    item = planner(road_center_gate_near_px=18.0,
                   road_center_gate_mid_px=24.0,
                   road_center_gate_far_px=30.0)
    road, lane = trapezoid_masks(False, False)
    first = item.plan(road, lane, lane, timestamp_sec=1.0)
    expanded = road.copy()
    expanded[100:221, 0:100] = 1
    second = item.plan(expanded, lane, lane, timestamp_sec=1.05)
    assert abs(second.points[0, 0]-first.points[0, 0]) < 35.0


def test_robust_fit_rejects_parked_vehicle_side_jump():
    item = planner(robust_fit_residual_px=8.0)
    points = np.array([[160, 220], [160, 200], [230, 180],
                       [160, 160], [160, 140], [160, 120]], float)
    fitted = item._fit(points, np.ones(len(points)))
    assert len(item._adaptive_fit_rejected) == 1
    assert np.max(np.abs(fitted[:, 0]-160.0)) < 3.0


def test_short_path_lookahead_never_extrapolates():
    controller = AdaptivePixelController(
        PixelControllerConfig(path_timeout_sec=1.0, source_stamp_timeout_sec=1.0,
                              derivative_gain_deg_per_norm_per_s=0.0),
        DynamicLookaheadConfig())
    points = ((160.0, 220.0), (165.0, 200.0), (170.0, 180.0))
    assert controller.ingest_path(
        1_000_000_000, 320.0, points, True, 0.8, 1.0,
        [ImagePathPoint.BOTH_BOUNDARIES]*3, ImagePath.STATE_VALID)
    command = controller.step(1.01, 1_010_000_000)
    assert command.valid and -22 <= command.wheel <= 22


def test_hold_expires_by_time_not_frame_count():
    item = planner(hold_time_sec=0.10)
    road, lane = trapezoid_masks()
    assert item.plan(road, lane, np.zeros_like(lane), timestamp_sec=1.0).valid
    empty = np.zeros_like(road)
    held = item.plan(empty, empty, empty, timestamp_sec=1.05)
    expired = item.plan(empty, empty, empty, timestamp_sec=1.20)
    assert held.valid
    assert held.diagnostics["generation_mode"] == "HOLD"
    assert not expired.valid
    assert expired.diagnostics["failure_reason"] in (
        "HOLD_EXPIRED", "NO_FEASIBLE_PATH")


def test_impossible_steering_is_not_authorized_by_output_clip():
    item = planner(lookahead_min_ratio=0.25, lookahead_max_ratio=0.75)
    # With the production default P=22 a full-frame offset requests <22 deg;
    # raise the calibrated gain to exercise the mechanical infeasibility gate.
    item.config.steering_proportional_gain_deg_per_norm = 35.0
    points = np.array([[319.0, 220.0], [319.0, 180.0], [319.0, 140.0]])
    details = item._steering_feasibility(points, 320, 1.0)
    assert not details["steering_angle_ok"]
    assert abs(details["required_steering_deg"]) > 22.0
