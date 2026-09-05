import numpy as np

from camera_navigation.image_path_planner import (
    INVALID,
    ImagePathPlanner,
    PlannerConfig,
    ROAD_CENTER,
)


HEIGHT = 480
WIDTH = 640


def road_from_center(center_fn, half_width=110, spike_rows=()):
    road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        center = float(center_fn(y))
        if any(lo <= y <= hi for lo, hi in spike_rows):
            center += 55.0
        left = max(0, int(round(center-half_width)))
        right = min(WIDTH, int(round(center+half_width+1)))
        road[y, left:right] = 1
    return road


def empty_mask():
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def test_straight_road_only_curvature_noise_keeps_horizon_and_near_zero_steer():
    planner = ImagePathPlanner()
    road = road_from_center(lambda _y: 320.0, spike_rows=((292, 298),))
    result = planner.plan(road, empty_mask(), empty_mask(), timestamp_sec=0.0)
    assert result.valid
    assert result.state != INVALID
    assert set(result.sources) == {ROAD_CENTER}
    assert result.diagnostics["path_horizon_ratio"] >= 0.95
    assert result.diagnostics["rejections"]["DIRECTION_OUTLIER"] > 0
    assert abs(result.diagnostics["required_steering_deg"]) < 2.0


def test_moderate_road_only_curve_repairs_noise_below_vehicle_limit():
    planner = ImagePathPlanner()
    road = road_from_center(
        lambda y: 320.0+0.0018*(475.0-y)**2,
        spike_rows=((292, 298),))
    result = planner.plan(road, empty_mask(), empty_mask(), timestamp_sec=0.0)
    assert result.valid
    assert result.state != INVALID
    assert result.diagnostics["max_required_steering_deg"] < 22.0
    assert result.diagnostics["path_horizon_ratio"] >= 0.95


def test_curve_exceeding_configured_vehicle_limit_is_hard_invalid():
    # Higher calibrated P gain models a vehicle/camera combination for which
    # this same image-space curve physically requires more than 22 degrees.
    planner = ImagePathPlanner(PlannerConfig(
        steering_proportional_gain_deg_per_norm=40.0))
    road = road_from_center(lambda y: 320.0+0.0022*(475.0-y)**2)
    result = planner.plan(road, empty_mask(), empty_mask(), timestamp_sec=0.0)
    assert not result.valid
    assert result.state == INVALID
    assert result.diagnostics["max_required_steering_deg"] > 22.0
    assert not result.diagnostics["steering_angle_ok"]


def test_controller_equivalent_pd_detects_frame_rate_violation():
    planner = ImagePathPlanner()
    straight = np.asarray(
        [[320.0, 420.0], [320.0, 350.0], [320.0, 280.0], [320.0, 210.0]])
    first = planner._steering_feasibility(straight, WIDTH, 0.0)
    planner._commit_steering_state(first, 0.0)
    jumped = straight.copy()
    jumped[:, 0] = 600.0
    check = planner._steering_feasibility(jumped, WIDTH, 0.05)
    assert check["required_steering_deg"] > 22.0
    assert not check["steering_rate_ok"]
    assert check["steering_rate_deg_per_sec"] > 180.0


def test_persistent_spatial_continuity_warning_is_degraded_not_invalid():
    planner = ImagePathPlanner(PlannerConfig(
        max_steering_delta_deg_per_segment=0.01))
    road = road_from_center(lambda y: 320.0+0.0018*(475.0-y)**2)
    result = planner.plan(road, empty_mask(), empty_mask(), timestamp_sec=0.0)
    assert result.valid
    assert result.state != INVALID
    assert not result.diagnostics["steering_continuity_ok"]
    assert result.diagnostics["steering_repair_attempts"] == 3


def test_final_temporal_steering_rate_violation_remains_degraded_for_slew_limit():
    config = PlannerConfig(
        seed_half_width_px=300,
        steering_derivative_gain_deg_per_norm_per_s=0.0)
    planner = ImagePathPlanner(config)
    straight = road_from_center(lambda _y: 320.0, half_width=40)
    first = planner.plan(
        straight, empty_mask(), empty_mask(), timestamp_sec=0.0)
    assert first.valid
    shifted = road_from_center(lambda _y: 520.0, half_width=40)
    result = planner.plan(
        shifted, empty_mask(), empty_mask(), timestamp_sec=0.05)
    assert result.valid
    assert result.state != INVALID
    assert result.diagnostics["steering_angle_ok"]
    assert not result.diagnostics["steering_rate_ok"]


def test_final_projection_requires_vehicle_width_clearance():
    planner = ImagePathPlanner()
    narrow = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    narrow[120:476, 310:331] = 1
    points = np.asarray(
        [[320.0, 420.0], [320.0, 350.0], [320.0, 280.0]])
    projected, safe, details = planner._project_final_path_to_road(
        points, narrow, 475)
    assert not safe
    assert not len(projected)
    assert details["final_road_unrecoverable"]
