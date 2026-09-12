import math
from pathlib import Path

import numpy as np

from camera_navigation.semantic_path_contract import decode_binary_rle
from depth_hybrid_slam.csv_road_validator_core import (
    DEGRADED_LANE_UNCERTAIN, INVALID_GEOMETRY,
    INVALID_INSUFFICIENT_ROAD, INVALID_OUTSIDE_ROAD, PATH_UNAVAILABLE,
    STALE_INPUT, VALID_ROAD_AND_LANE, VALID_ROAD_ONLY, ValidatorConfig,
    classify_input_state, grid_to_metric, render_bev_overlay,
    validate_metric_bev)


CFG = ValidatorConfig()


def scene(half_road_width=1.0):
    rows, cols = np.indices((CFG.rows, CFG.cols))
    xy = grid_to_metric(rows.ravel(), cols.ravel(), CFG).reshape(
        CFG.rows, CFG.cols, 2)
    road = (np.abs(xy[:, :, 1]) <= half_road_width).astype(np.uint8)
    return xy, road, np.zeros_like(road), np.ones_like(road)


def path(offset=0.0, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    return np.asarray(((0.5, offset, yaw),
                       (4.0, offset+3.5*math.tan(yaw), yaw)))


def add_lane(lane, xy, lateral):
    lane[np.abs(xy[:, :, 1]-lateral) <= CFG.resolution_m] = 1


def test_rle_decode_uses_camera_navigation_contract():
    decoded = decode_binary_rle([1, 3, 7, 8], 2, 4)
    assert decoded.tolist() == [[0, 255, 255, 0], [0, 0, 0, 255]]


def test_a_lane_free_broad_road_is_valid_road_only():
    _, road, lane, visible = scene()
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert result.valid and result.state == VALID_ROAD_ONLY
    assert result.center_inside_ratio == 1.0
    assert result.vehicle_corridor_inside_ratio == 1.0
    assert not result.lane_visible


def test_b_vehicle_corridor_at_road_edge_is_invalid():
    _, road, lane, visible = scene()
    result = validate_metric_bev(road, lane, visible, path(0.9), CFG)
    assert not result.valid and result.state == INVALID_OUTSIDE_ROAD
    assert result.center_inside_ratio == 1.0
    assert result.vehicle_corridor_inside_ratio < 0.75


def test_c_two_lanes_centered_is_valid_road_and_lane():
    xy, road, lane, visible = scene()
    add_lane(lane, xy, -0.65)
    add_lane(lane, xy, 0.65)
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert result.valid and result.state == VALID_ROAD_AND_LANE
    assert result.lane_visible and result.lane_consistent


def test_d_one_lane_does_not_fail_valid_road():
    xy, road, lane, visible = scene()
    add_lane(lane, xy, 0.65)
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert result.valid and result.state == VALID_ROAD_AND_LANE
    assert result.lane_visible and result.lane_consistent


def test_e_lane_absent_path_outside_road_is_invalid():
    _, road, lane, visible = scene()
    result = validate_metric_bev(road, lane, visible, path(1.5), CFG)
    assert not result.valid and result.state == INVALID_OUTSIDE_ROAD


def test_f_missing_or_disconnected_road_is_insufficient():
    xy, road, lane, visible = scene()
    road[xy[:, :, 0] > 2.0] = 0
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert not result.valid and result.state == INVALID_INSUFFICIENT_ROAD
    assert result.road_confidence < CFG.minimum_road_confidence
    empty = validate_metric_bev(np.zeros_like(road), lane, visible, path(), CFG)
    assert empty.state == INVALID_INSUFFICIENT_ROAD


def test_g_lateral_offset_sweep_records_containment_drop():
    _, road, lane, visible = scene(half_road_width=0.8)
    results = [validate_metric_bev(road, lane, visible, path(value), CFG)
               for value in (0.0, 0.2, 0.4, 0.6, 1.0)]
    center = [item.center_inside_ratio for item in results]
    corridor = [item.vehicle_corridor_inside_ratio for item in results]
    assert center == sorted(center, reverse=True)
    assert corridor == sorted(corridor, reverse=True)
    assert results[0].state == VALID_ROAD_ONLY
    assert results[-1].state == INVALID_OUTSIDE_ROAD


def test_h_yaw_sweep_exits_road_at_twenty_degrees():
    _, road, lane, visible = scene()
    results = [validate_metric_bev(road, lane, visible, path(yaw_deg=value), CFG)
               for value in (0.0, 5.0, 10.0, 20.0)]
    assert all(item.valid for item in results[:3])
    assert not results[-1].valid
    assert results[-1].state == INVALID_OUTSIDE_ROAD


def test_visible_path_below_threshold_is_invalid_geometry():
    _, road, lane, visible = scene()
    visible[:CFG.rows//2, :] = 0
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert not result.valid and result.state == INVALID_GEOMETRY


def test_lane_crossing_path_is_degraded_but_road_remains_valid():
    xy, road, lane, visible = scene()
    add_lane(lane, xy, 0.0)
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    assert result.valid and result.state == DEGRADED_LANE_UNCERTAIN
    assert result.lane_visible and not result.lane_consistent


def test_invalid_calibration_and_input_timeouts_fail_closed():
    invalid = classify_input_state(False, True, True, True, 0.0, 0.0, 0.0)
    assert invalid.state == INVALID_GEOMETRY and not invalid.valid
    stale_semantic = classify_input_state(
        True, True, True, True, 0.0, 0.51, 0.0)
    assert stale_semantic.state == STALE_INPUT
    stale_path = classify_input_state(
        True, True, True, True, 0.0, 0.0, 0.51)
    assert stale_path.state == STALE_INPUT
    missing_path = classify_input_state(
        True, True, True, False, 0.0, 0.0, float("inf"))
    assert missing_path.state == PATH_UNAVAILABLE


def test_overlay_contains_debug_layers_and_text():
    _, road, lane, visible = scene()
    result = validate_metric_bev(road, lane, visible, path(), CFG)
    overlay = render_bev_overlay(road, lane, visible, path(), result, CFG)
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    assert np.count_nonzero(overlay[:, :, 1]) > 0


def test_validator_source_has_no_vehicle_command_publishers():
    source = (Path(__file__).parents[1]/"depth_hybrid_slam"/
              "csv_road_validator_node.py").read_text()
    for topic in ("/slam_drive", "/slam_wheel", "/camera_drive",
                  "/camera_wheel", "/gps_drive", "/gps_wheel",
                  "/mcu/cmd_drive", "/mcu/cmd_wheel"):
        assert topic not in source
