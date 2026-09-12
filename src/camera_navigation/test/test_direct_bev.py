"""Synthetic direct-BEV planner, projection and steering tests."""

import math
from pathlib import Path
import statistics
import threading

import numpy as np
import pytest
from sensor_msgs.msg import Image

from camera_navigation.direct_bev_controller import (
    BevControllerConfig, DirectBevController,
)
from camera_navigation.bev_wheel_selector_node import selected_wheel
from camera_navigation.direct_bev_drive_node import (
    DriveSafetyPolicy, drive_command_for_state,
)
from camera_navigation.direct_bev_core import (
    BOTH, DEGRADED, HOLD, INVALID, LEFT_ONLY, RIGHT_ONLY, ROAD_ONLY,
    DirectBevConfig, DirectBevPlanner, pure_pursuit_unclipped,
)
from camera_navigation.direct_bev_projection import (
    CameraModel, build_ground_remap, ground_points_to_pixels,
    project_mask_to_bev, warp_rgb_to_bev,
)
from camera_navigation.direct_bev_planner_node import (
    DirectBevPlannerNode, EventRate, LatestPlannerResultCache,
    TimestampedImageCache, evaluate_fixed_result, image_to_bgr,
    render_camera_path_overlay, safely_render_camera_overlay,
    bev_overlay_requested, wall_input_fresh,
)
from camera_navigation.ground_plane_calibration import rotation_matrix_rpy
from camera_navigation.overlay_worker import LatestOnlyWorker


def camera_model(distortion_model="none", distortion=None):
    return CameraModel(
        640, 480,
        np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]]),
        np.asarray([] if distortion is None else distortion, dtype=float),
        distortion_model)


def ros_image(stamp_ns=0, encoding="bgr8", width=2, height=2,
              step=None, pixels=None):
    message = Image()
    message.header.stamp.sec = stamp_ns//1_000_000_000
    message.header.stamp.nanosec = stamp_ns%1_000_000_000
    message.width, message.height = width, height
    message.encoding = encoding
    message.step = width*3 if step is None else step
    if pixels is None:
        pixels = np.zeros((height, message.step), np.uint8)
    message.data = np.asarray(pixels, np.uint8).tobytes()
    return message


def masks(center=lambda x: 0.0, left=True, right=True, end=8.0,
          half_width=0.75):
    planner = DirectBevPlanner(DirectBevConfig())
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x in np.arange(0.30, end, planner.config.resolution_m):
        middle = center(x)
        row, column = planner.metric_to_grid([[x, middle]])[0]
        half = int(round(half_width/planner.config.resolution_m))
        road[max(0, row-1):min(planner.rows, row+2),
             max(0, column-half):min(planner.cols, column+half+1)] = 1
        offset = int(round(0.55/planner.config.resolution_m))
        if left:
            lane[max(0, row-1):row+2, column+offset-1:column+offset+2] = 1
        if right:
            lane[max(0, row-1):row+2, column-offset-1:column-offset+2] = 1
    return planner, road, lane


def test_both_straight_is_valid():
    planner, road, lane = masks()
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.mode == BOTH and result.state == "VALID"


def test_metric_path_starts_on_vehicle_center_axis():
    planner, road, lane = masks(center=lambda x: 0.18*x)
    result = planner.plan(road, lane, 1.0)
    assert result.valid
    assert result.points[0, 0] == pytest.approx(0.32, abs=0.05)
    assert result.points[0, 1] == pytest.approx(0.0, abs=1.0e-12)
    grid = planner.metric_to_grid(result.points)
    assert np.all(result.component[grid[:, 0], grid[:, 1]] > 0)


def test_center_axis_connector_remains_within_steering_limit():
    planner, road, lane = masks(
        center=lambda x: 0.7*x, left=False, right=False, end=1.4,
        half_width=1.30)
    result = planner.plan(road, lane, 1.0)
    assert result.valid
    assert result.points[0, 1] == 0.0
    assert abs(result.diagnostics["required_steering_deg"]) <= 22.0
    assert result.diagnostics["steering_recovery"] == []


def test_no_center_axis_road_does_not_force_path():
    planner = DirectBevPlanner()
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x_m in np.arange(0.30, 2.0, planner.config.resolution_m):
        row, column = planner.metric_to_grid([[x_m, 0.80]])[0]
        half = int(round(0.55/planner.config.resolution_m))
        road[max(0, row-1):row+2, column-half:column+half+1] = 1
    result = planner.plan(road, lane, 1.0)
    assert not result.valid
    assert result.diagnostics["reasons"] == ["CENTER_AXIS_ROAD_MISSING"]


def test_both_gentle_curve_is_representable():
    planner, road, lane = masks(center=lambda x: 0.08*x*x, end=3.0)
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.mode == BOTH
    assert abs(result.diagnostics["required_steering_deg"]) > 1.0


def test_left_only_uses_metric_normal_offset():
    planner, road, lane = masks(right=False)
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.mode == LEFT_ONLY
    assert abs(np.median(result.points[:, 1])) < 0.12


def test_right_only_uses_metric_normal_offset():
    planner, road, lane = masks(left=False)
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.mode == RIGHT_ONLY
    assert abs(np.median(result.points[:, 1])) < 0.12


def test_road_only_uses_clearance_center_and_is_degraded():
    planner, road, lane = masks(left=False, right=False)
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.mode == ROAD_ONLY
    assert result.state == DEGRADED


def test_commissioned_vehicle_geometry_is_the_safe_default():
    config = DirectBevConfig()
    assert config.vehicle_width_m == pytest.approx(0.80)
    assert config.lateral_safety_margin_m == pytest.approx(0.12)
    assert config.wheelbase_m == pytest.approx(0.73)
    assert config.maximum_steering_deg == pytest.approx(22.0)
    assert config.vehicle_width_m / 2.0 + \
        config.lateral_safety_margin_m == pytest.approx(0.52)


def test_commissioned_camera_height_and_pitch_are_in_mount_config():
    mount = (Path(__file__).parents[2] / "depth_hybrid_slam" / "launch" /
             "cuvslam_only.launch.py").read_text(encoding="utf-8")
    assert '"--x", "0.015"' in mount
    assert '"--z", "0.835"' in mount
    assert '"--pitch", "0.0872665"' in mount


def test_projection_never_falls_back_from_safe_road_to_unsafe_road():
    planner = DirectBevPlanner()
    component = np.zeros((planner.rows, planner.cols), np.uint8)
    safe = np.zeros_like(component)
    row, col = planner.metric_to_grid([[1.0, 0.0]])[0]
    component[row, col-4:col+5] = 1
    projected, safe_count, road_count = planner._project_inside(
        np.asarray([[1.0, 0.0]]), component, safe)
    assert len(projected) == 0
    assert safe_count == 0
    assert road_count == 0


def test_bev_raster_near_edge_is_not_treated_as_a_physical_road_edge():
    planner = DirectBevPlanner()
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    center = planner.metric_to_grid([[planner.config.x_min_m, 0.0]])[0, 1]
    half = int(round(1.0 / planner.config.resolution_m))
    road[:, center-half:center+half+1] = 1
    _, _, _, safe, distance = planner.preprocess(road, np.zeros_like(road))
    assert safe[-1, center] == 1
    assert distance[-1, center] * planner.config.resolution_m >= 0.52


def test_hold_path_is_rejected_when_current_safe_road_conflicts():
    planner, road, lane = masks()
    first = planner.plan(road, lane, 1.0)
    assert first.valid
    component = np.ones_like(road)
    safe = np.zeros_like(road)
    safe[:, -3:] = 1
    result = planner._fallback_or_invalid(
        1.1, "TEMPORAL_LATERAL_JUMP", component, safe, component)
    assert not result.valid
    assert result.state == INVALID
    assert result.diagnostics["reasons"] == ["HOLD_PATH_UNSAFE"]


def test_direct_bev_drive_policy_slows_degraded_and_stops_invalid():
    assert drive_command_for_state("VALID", 2.0, 1.0, 0.0) == 2.0
    assert drive_command_for_state("DEGRADED", 2.0, 1.0, 0.0) == 1.0
    assert drive_command_for_state("HOLD", 2.0, 1.0, 0.0) == 0.0
    assert drive_command_for_state("INVALID", 2.0, 1.0, 0.0) == 0.0
    assert drive_command_for_state(None, 2.0, 1.0, 0.0) == 0.0


def test_drive_policy_invalid_transition_is_immediate_and_has_no_residual():
    policy = DriveSafetyPolicy(0.5, 2.0, 1.0, 0.0)
    assert policy.update({"state": "DEGRADED", "stamp_ns": 1}, 1.0)["drive"] == 1.0
    stopped = policy.update({"state": "INVALID", "stamp_ns": 2}, 1.01)
    assert stopped == {"drive": 0.0, "state": "INVALID", "stale": False}
    assert policy.evaluate(1.02)["drive"] == 0.0
    assert policy.evaluate(1.49)["drive"] == 0.0
    assert policy.evaluate(1.52) == {"drive": 0.0, "state": None, "stale": True}
    assert policy.update({"state": "DEGRADED", "stamp_ns": 3}, 2.0)["drive"] == 1.0


def test_production_direct_bev_launch_publishes_drive_and_selects_bev():
    launch = (Path(__file__).parents[1] / "launch" /
              "camera_bev_standalone.launch.py").read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("active_planner", default_value="bev")' in launch
    assert 'executable="direct_bev_drive_node"' in launch
    assert 'model = os.path.join(yolo, "models", "hanla_yolo11n_seg_best.pt")' in launch
    assert 'backend = "pytorch"' in launch


def test_diamond_hole_is_limited_and_recovered():
    planner, road, lane = masks(left=False, right=False)
    row, col = planner.metric_to_grid([[2.0, 0.0]])[0]
    road[row-3:row+4, col-3:col+4] = 0
    result = planner.plan(road, lane, 1.0)
    assert result.valid


def test_crosswalk_holes_do_not_break_ego_component():
    planner, road, lane = masks(left=False, right=False)
    for x in (1.6, 1.9, 2.2):
        row, col = planner.metric_to_grid([[x, 0.0]])[0]
        road[row-1:row+2, col-12:col+13] = 0
    assert planner.plan(road, lane, 1.0).valid


def test_bump_marking_restored_road_keeps_direct_bev_path():
    planner, road, lane = masks(left=False, right=False)
    restored = road.copy()
    row, _ = planner.metric_to_grid([[1.8, 0.0]])[0]
    broken = road.copy()
    broken[row-9:row+10] = 0
    # Common refinement returns ``restored``; the planner is deliberately
    # unaware of bump colour and consumes only that shared drivable result.
    result = planner.plan(restored, lane, 1.0)
    assert np.count_nonzero(broken) < np.count_nonzero(restored)
    assert result.valid


def test_rgb_warp_and_semantic_projection_have_pixel_alignment():
    source = np.zeros((8, 10, 3), np.uint8)
    source_mask = np.zeros((8, 10), np.uint8)
    source_mask[2:6, 4:7] = 1
    source[source_mask > 0] = (17, 91, 203)
    map_x, map_y = np.meshgrid(
        np.arange(10, dtype=np.float32), np.arange(8, dtype=np.float32))
    warped = warp_rgb_to_bev(source, map_x, map_y)
    projected = project_mask_to_bev(source_mask, map_x, map_y)
    assert np.all(warped[projected > 0] == (17, 91, 203))
    assert np.count_nonzero(warped[projected == 0]) == 0


def test_disabled_or_unsubscribed_bev_overlay_skips_render_path():
    assert not bev_overlay_requested(False, 1, True)
    assert not bev_overlay_requested(True, 0, True)
    assert not bev_overlay_requested(True, 1, False)
    assert bev_overlay_requested(True, 1, True)


def test_wide_road_does_not_pull_path_sideways():
    planner, road, lane = masks(left=False, right=False)
    first = planner.plan(road, lane, 1.0)
    road[:, :40] = 1
    second = planner.plan(road, lane, 1.05)
    assert second.valid
    assert abs(second.points[0, 1]-first.points[0, 1]) < 0.25


def test_branch_choice_is_deterministic():
    outputs = []
    for _ in range(2):
        planner, road, lane = masks(left=False, right=False)
        for x in np.arange(3.0, 7.5, 0.04):
            for center in (-0.65, 0.65):
                row, col = planner.metric_to_grid([[x, center]])[0]
                road[row-1:row+2, col-8:col+9] = 1
        result = planner.plan(road, lane, 1.0)
        outputs.append(result.points[:, 1].copy())
    assert np.allclose(outputs[0], outputs[1])


def test_short_forward_corridor_is_still_published():
    planner, road, lane = masks(left=False, right=False, end=1.35)
    result = planner.plan(road, lane, 1.0)
    assert result.valid and len(result.points) >= 3


def test_mask_loss_holds_then_expires():
    planner, road, lane = masks()
    assert planner.plan(road, lane, 1.0).valid
    empty = np.zeros_like(road)
    held = planner.plan(empty, empty, 1.10)
    expired = planner.plan(empty, empty, 1.30)
    assert held.valid and held.mode == HOLD
    assert not expired.valid and expired.mode == INVALID


def test_near_limit_and_excess_are_measured_without_clipping():
    wheelbase, x = 0.73, 0.8
    def lateral_for(angle):
        tangent = math.tan(math.radians(angle))
        return (wheelbase-math.sqrt(
            wheelbase**2-tangent**2*x**2))/tangent
    y_near = lateral_for(21.5)
    y_over = lateral_for(26.0)
    near, _ = pure_pursuit_unclipped([[x, y_near]], wheelbase, x)
    over, _ = pure_pursuit_unclipped([[x, y_over]], wheelbase, x)
    assert 21.0 < near < 22.0
    assert over > 22.0


def test_controller_rejects_actual_steering_excess():
    controller = DirectBevController()
    command = controller.command([[0.3, 0.3], [0.6, 0.6], [0.8, 0.8]],
                                 1.0, False, 1.0)
    assert not command["valid"] and command["wheel"] == 0


def test_controller_degraded_path_still_commands_wheel():
    x = np.arange(2.5, 7.6, 0.2)
    path = np.column_stack((x, 0.10*(x-2.5)**2))
    command = DirectBevController().command(path, 0.5, True, 1.0)
    assert command["valid"] and command["wheel"] != 0
    assert -22 <= command["wheel"] <= 22


@pytest.mark.parametrize("turn_sign", [1.0, -1.0])
def test_controller_bicycle_formula_and_turn_sign_are_preserved(turn_sign):
    radius = 10.0
    x = np.linspace(2.5, 7.5, 80)
    y = turn_sign*(radius-np.sqrt(radius**2-x**2))
    command = DirectBevController().command(
        np.column_stack((x, y)), 0.7, True, 1.0)
    assert command["valid"]
    assert command["raw_steering_deg"] == pytest.approx(
        command["bicycle_steering_deg"], abs=1.0e-12)
    assert np.sign(command["raw_steering_deg"]) == turn_sign
    assert np.sign(command["wheel"]) == -turn_sign


def test_controller_zero_curvature_is_exactly_neutral():
    command = DirectBevController().command(
        [[2.5, 0.0], [4.0, 0.0], [6.0, 0.0]], 0.7, True, 1.0)
    assert command["target_curvature_per_m"] == 0.0
    assert command["raw_steering_deg"] == 0.0
    assert command["required_steering_deg"] == 0.0
    assert command["wheel"] == 0


def test_controller_metric_path_sign_contract_and_limits():
    config = BevControllerConfig(lookahead_from_path_start=True)
    cases = (
        (lambda x: 0.0, 0),
        (lambda x: 0.012*x*x, -1),
        (lambda x: -0.012*x*x, 1),
        (lambda x: 0.035*x*x, -1),
        (lambda x: -0.035*x*x, 1),
    )
    x = np.arange(3.0, 8.01, 0.2)
    for curve, expected_sign in cases:
        path = np.column_stack((x, [curve(value) for value in x]))
        command = DirectBevController(config).command(path, 1.0, False, 1.0)
        assert command["valid"]
        assert np.sign(command["wheel"]) == expected_sign
        assert -22 <= command["wheel"] <= 22


@pytest.mark.parametrize("radius,turn_sign", [
    (3.0, 1), (5.0, 1), (10.0, 1), (20.0, 1),
    (3.0, -1), (5.0, -1), (10.0, -1), (20.0, -1),
])
def test_controller_matches_ackermann_on_metric_circular_paths(radius,
                                                                turn_sign):
    x = np.linspace(0.30, min(7.0, radius*0.80), 80)
    y = turn_sign*(radius-np.sqrt(radius**2-x**2))
    controller = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True))
    command = controller.command(np.column_stack((x, y)), 1.0, False, 1.0)
    expected = math.degrees(math.atan(0.73/radius))*turn_sign
    assert command["valid"]
    assert command["required_steering_deg"] == pytest.approx(expected, abs=0.08)
    assert np.sign(command["wheel"]) == -turn_sign
    assert abs(command["wheel"]) <= 22


@pytest.mark.parametrize("first_x", [2.5, 3.0, 4.0, 5.0])
def test_path_start_lookahead_preserves_curvature_when_first_point_is_far(first_x):
    radius = 10.0
    x = np.linspace(first_x, 7.5, 80)
    y = radius-np.sqrt(radius**2-x**2)
    command = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True)).command(
            np.column_stack((x, y)), 1.0, False, 1.0)
    assert command["valid"]
    assert command["required_steering_deg"] == pytest.approx(
        math.degrees(math.atan(0.73/radius)), abs=0.08)
    assert command["wheel"] < 0


def test_fractional_accumulator_average_reset_and_sign_contract():
    angle = 0.40
    x = 1.2
    tangent = math.tan(math.radians(angle))
    y = (0.73-math.sqrt(0.73**2-tangent**2*x**2))/tangent
    controller = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True, fractional_accumulator=True,
        steering_sign=1.0))
    left = [controller.command([[x, y]], 1.0, False, i/20.0)["wheel"]
            for i in range(1, 101)]
    assert statistics.fmean(left) == pytest.approx(angle, abs=0.02)
    assert set(left) <= {0, 1}
    controller.neutral()
    assert controller.fractional_residual == 0.0
    right = controller.command([[x, -y]], 1.0, False, 6.0)["wheel"]
    assert right in (0, -1)
    assert controller.fractional_sign in (0, -1)


def test_fractional_accumulator_has_no_straight_chatter():
    controller = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True, fractional_accumulator=True))
    wheels = [controller.command([[3.0, 0.0], [5.0, 0.0]],
                                  1.0, False, i/20.0)["wheel"]
              for i in range(100)]
    assert wheels == [0]*100


def test_controller_over_limit_fails_closed_and_selector_clamps():
    controller = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True))
    over = controller.command([[0.3, 0.3], [0.6, 0.6], [0.8, 0.8]],
                              1.0, False, 1.0)
    assert not over["valid"] and over["wheel"] == 0
    assert selected_wheel("bev", 19) == 19
    assert selected_wheel("bev", 31) == 22
    assert selected_wheel("bev", -31) == -22
    assert selected_wheel("none", 19) is None


def test_controller_invalid_neutral_resets_fractional_and_previous_command():
    controller = DirectBevController()
    curved = controller.command([[2.5, 0.0], [3.5, 0.4], [4.5, 0.8]],
                                0.7, True, 1.0)
    assert curved["wheel"] != 0
    stopped = controller.neutral()
    assert stopped == {"valid": False, "wheel": 0, "reason": "PATH_INVALID"}
    assert controller.previous_time is None
    assert controller.previous_steering == 0.0
    assert controller.fractional_residual == 0.0


def test_ros_path_serialization_preserves_metric_lateral_coordinates():
    from std_msgs.msg import Header
    points = np.array([[3.0, 0.0], [3.2, 0.12], [3.4, -0.08]])
    message = DirectBevPlannerNode._path_message(Header(), points)
    received = np.array([(pose.pose.position.x, pose.pose.position.y)
                         for pose in message.poses])
    assert message.header.frame_id == "base_link"
    assert np.allclose(received, points)


def test_ground_projection_remap_uses_calibrated_extrinsic():
    config = DirectBevConfig()
    camera = CameraModel(640, 480,
                         np.array([[400., 0., 320.], [0., 400., 240.],
                                   [0., 0., 1.]]), np.zeros(5))
    map_x, map_y = build_ground_remap(
        config, camera, rotation_matrix_rpy(0.0, -10.0, 0.0),
        np.array([0.015, 0.0, 0.970]))
    assert map_x.shape == (194, 151)
    assert np.count_nonzero(map_x >= 0.0) > 100
    assert np.count_nonzero(map_y >= 0.0) > 100


@pytest.mark.parametrize("model,distortion", [
    ("none", []), ("plumb_bob", [0., 0., 0., 0., 0.]),
    ("rational_polynomial", [0.]*8), ("equidistant", [0.]*4),
])
def test_ground_points_project_to_pixels_with_supported_distortion(
        model, distortion):
    pixels, indices = ground_points_to_pixels(
        [[3.0, 0.0], [4.0, 0.2]], camera_model(model, distortion),
        rotation_matrix_rpy(0.0, -10.0, 0.0), np.array([0.015, 0.0, 0.970]))
    assert indices.tolist() == [0, 1]
    assert pixels.shape == (2, 2)
    assert np.all(np.isfinite(pixels))


def test_ground_projection_rejects_nonfinite_behind_and_offscreen_points():
    pixels, indices = ground_points_to_pixels(
        [[3.0, 0.0], [3.0, np.nan], [-1.0, 0.0], [3.0, 20.0],
         [4.0, 0.0]], camera_model(),
        rotation_matrix_rpy(0.0, -10.0, 0.0), np.array([0.015, 0.0, 0.970]))
    assert len(pixels) == 2
    assert indices.tolist() == [0, 4]


def test_image_cache_prefers_exact_timestamp_and_is_bounded():
    cache = TimestampedImageCache(3)
    frames = [ros_image(stamp_ns=value) for value in (100, 200, 300, 400)]
    for frame in frames:
        cache.add(frame)
    assert len(cache) == 3
    assert cache.nearest(300, 1.0) is frames[2]


def test_image_cache_uses_nearest_frame_within_tolerance():
    cache = TimestampedImageCache(20)
    older = ros_image(stamp_ns=1_020_000_000)
    nearer = ros_image(stamp_ns=1_045_000_000)
    cache.add(older); cache.add(nearer)
    assert cache.nearest(1_050_000_000, 0.05) is nearer


def test_image_cache_skips_frame_outside_tolerance():
    cache = TimestampedImageCache(20)
    cache.add(ros_image(stamp_ns=1_000_000_000))
    assert cache.nearest(1_060_000_000, 0.05) is None


def test_image_to_bgr_handles_bgr_rgb_and_step_padding():
    padded = np.array([
        [1, 2, 3, 4, 5, 6, 99, 99],
        [7, 8, 9, 10, 11, 12, 88, 88],
    ], np.uint8)
    bgr = image_to_bgr(ros_image(step=8, pixels=padded))
    rgb = image_to_bgr(ros_image(encoding="rgb8", step=8, pixels=padded))
    assert bgr.tolist() == [[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]
    assert rgb.tolist() == [[[3, 2, 1], [6, 5, 4]],
                           [[9, 8, 7], [12, 11, 10]]]


def test_camera_overlay_does_not_join_across_offscreen_path_span():
    source = ros_image(width=640, height=480, step=640*3,
                       pixels=np.zeros((480, 640*3), np.uint8))
    points = np.array([[3.0, 0.0], [4.0, 0.0], [4.0, 20.0], [8.0, 0.0]])
    camera = camera_model()
    rotation = rotation_matrix_rpy(0.0, -10.0, 0.0)
    position = np.array([0.015, 0.0, 0.970])
    pixels, indices = ground_points_to_pixels(
        points, camera, rotation, position)
    assert indices.tolist() == [0, 1, 3]
    overlay = render_camera_path_overlay(
        source, points, camera, rotation, position, 2, 1)
    gap_midpoint = np.rint((pixels[1]+pixels[2])*0.5).astype(int)
    assert not np.any(overlay[gap_midpoint[1], gap_midpoint[0]])


def test_camera_overlay_starts_at_bottom_center_without_changing_metric_path():
    source = ros_image(width=640, height=480, step=640*3,
                       pixels=np.zeros((480, 640*3), np.uint8))
    points = np.array([[3.0, 0.0], [4.0, 0.0], [5.0, 0.1]])
    original = points.copy()
    overlay = render_camera_path_overlay(
        source, points, camera_model(), rotation_matrix_rpy(0.0, -10.0, 0.0),
        np.array([0.015, 0.0, 0.970]), 4, 3, 0.5, 1.0, True)
    assert np.any(overlay[479, 320])
    assert np.array_equal(points, original)


def test_invalid_camera_overlay_has_no_bottom_anchor():
    source = ros_image(width=640, height=480, step=640*3,
                       pixels=np.zeros((480, 640*3), np.uint8))
    overlay = render_camera_path_overlay(
        source, [[3.0, 0.0]], camera_model(),
        rotation_matrix_rpy(0.0, -10.0, 0.0),
        np.array([0.015, 0.0, 0.970]), 4, 3, 0.5, 1.0, False)
    assert not np.any(overlay)


def test_camera_overlay_error_does_not_stop_latest_only_worker():
    completed = []
    first_done = threading.Event()
    second_done = threading.Event()

    def process(job):
        safely_render_camera_overlay(
            lambda: (_ for _ in ()).throw(ValueError("render failed")),
            lambda _error: None)
        completed.append(job)
        (first_done if job == 1 else second_done).set()

    worker = LatestOnlyWorker(process, "camera-overlay-error-test")
    try:
        worker.submit(1)
        assert first_done.wait(1.0)
        worker.submit(2)
        assert second_done.wait(1.0)
    finally:
        worker.close()
    assert completed == [1, 2]


def test_fixed_result_cache_generation_and_coordinates_are_stable():
    cache = LatestPlannerResultCache()
    points = np.array([[0.3, 0.0], [0.5, 0.0], [0.7, 0.1]])
    first = cache.replace(points, "VALID", True, 0.9,
                          {"required_steering_deg": 2.0}, 123, 1.0)
    snapshot_one = cache.snapshot()
    snapshot_two = cache.snapshot()
    assert first == 1 and snapshot_one.generation == snapshot_two.generation
    assert np.array_equal(snapshot_one.points, snapshot_two.points)
    points[0, 1] = 9.0
    assert snapshot_one.points[0, 1] == 0.0
    second = cache.replace(snapshot_one.points, "DEGRADED", True, 0.7,
                           {"required_steering_deg": 3.0}, 456, 1.1)
    assert second == 2


def test_fixed_result_stale_calibration_and_steering_gates():
    cache = LatestPlannerResultCache()
    points = [[0.3, 0.0], [0.5, 0.0], [0.7, 0.1]]
    cache.replace(points, "VALID", True, 0.9,
                  {"required_steering_deg": 2.0}, 123, 1.0)
    snapshot = cache.snapshot()
    assert evaluate_fixed_result(snapshot, 1.19, True, .2, .2, 22.)[:2] == \
        (True, None)
    assert evaluate_fixed_result(snapshot, 1.21, True, .2, .2, 22.)[1] == \
        "PATH_STALE"
    assert evaluate_fixed_result(snapshot, 1.1, False, .2, .2, 22.)[1] == \
        "CALIBRATION_INVALID"
    cache.replace(points, "VALID", True, 0.9,
                  {"required_steering_deg": 22.1}, 124, 2.0)
    assert evaluate_fixed_result(cache.snapshot(), 2.0, True, .2, .2, 22.)[1] == \
        "STEERING_LIMIT_EXCEEDED"


def test_camera_info_wall_freshness_gate():
    assert wall_input_fresh(10.0, 11.9, 2.0)
    assert not wall_input_fresh(10.0, 12.01, 2.0)
    assert not wall_input_fresh(None, 10.0, 2.0)


def test_source_and_publish_timestamps_are_separate():
    cache = LatestPlannerResultCache()
    cache.replace([[0.3, 0.0], [0.5, 0.0], [0.7, 0.0]], "VALID", True,
                  1.0, {"required_steering_deg": 0.0}, 1_000, 1.0)
    header = ros_image(stamp_ns=2_000).header
    path = DirectBevPlannerNode._path_message(header, cache.snapshot().points)
    assert cache.snapshot().source_stamp_ns == 1_000
    assert path.header.stamp.nanosec == 2_000


def test_fixed_rate_meter_reports_60hz_over_ten_seconds():
    output = EventRate(700)
    semantic = EventRate(400)
    for tick in range(601):
        output.tick(tick/60.0)
    for tick in range(301):
        semantic.tick(tick/30.0)
    assert 57.0 <= output.rate() <= 63.0
    assert semantic.rate() == pytest.approx(30.0)


def test_latest_result_cache_is_atomic_under_concurrent_access():
    cache = LatestPlannerResultCache()
    failures = []

    def writer():
        for value in range(1, 201):
            cache.replace([[value, 0.0], [value+1, 0.0], [value+2, 0.0]],
                          "VALID", True, 1.0,
                          {"marker": value, "required_steering_deg": 0.0},
                          value, float(value))

    thread = threading.Thread(target=writer)
    thread.start()
    while thread.is_alive():
        snapshot = cache.snapshot()
        if snapshot is not None and \
                snapshot.points[0, 0] != snapshot.diagnostics["marker"]:
            failures.append(snapshot.generation)
    thread.join()
    assert not failures


def test_invalid_transform_never_builds_metric_remap():
    camera = CameraModel(640, 480, np.eye(3), np.zeros(5))
    with pytest.raises(ValueError):
        build_ground_remap(DirectBevConfig(), camera,
                           np.full((3, 3), np.nan), np.zeros(3))


def test_no_ego_road_is_invalid_without_previous_path():
    planner = DirectBevPlanner()
    empty = np.zeros((planner.rows, planner.cols), np.uint8)
    result = planner.plan(empty, empty, 1.0)
    assert not result.valid and "EGO_ROAD_MISSING" in result.diagnostics["reasons"]


def test_unobservable_near_field_starts_only_at_first_safe_point():
    planner, road, lane = masks(left=False, right=False)
    cutoff = planner.metric_to_grid([[2.2, 0.0]])[0, 0]
    road[cutoff:, :] = 0
    result = planner.plan(road, lane, 1.0)
    assert result.valid and result.state == DEGRADED
    grid = planner.metric_to_grid(result.points)
    assert np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0)
    assert result.points[0, 0] > 2.2


def test_far_tail_outlier_is_robustly_rejected():
    planner, road, lane = masks()
    row, col = planner.metric_to_grid([[7.5, 2.0]])[0]
    lane[row-2:row+3, col-2:col+3] = 1
    result = planner.plan(road, lane, 1.0)
    assert result.valid
    assert abs(result.diagnostics["required_steering_deg"]) <= 22.0


def test_planner_publishes_curve_below_steering_limit():
    planner, road, lane = masks(
        center=lambda x: 0.8*x, left=False, right=False, end=1.4,
        half_width=1.30)
    result = planner.plan(road, lane, 1.0)
    assert result.valid
    assert 20.0 < abs(result.diagnostics["required_steering_deg"]) <= 22.0


def test_planner_stops_only_when_recovery_cannot_meet_limit():
    planner, road, lane = masks(
        center=lambda x: 1.2*x, left=False, right=False, end=1.4,
        half_width=1.30)
    result = planner.plan(road, lane, 1.0)
    assert not result.valid
    assert result.diagnostics["reasons"] == ["PATH_START_CONNECTION_FAILED"]
