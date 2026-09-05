"""Synthetic acceptance tests for advisory-only mission perception."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from camera_navigation.camera_mission_perception_node import (
    CameraMissionPerceptionNode)

from camera_navigation.mission_perception_core import (
    DebouncedPresence, MissionTrafficConfig, MissionTrafficFilter,
    PresenceConfig, StopLineDepthConfig, UphillConfig, UphillDetector,
    front_axle_distance, front_axle_distance_from_base_point,
    pitch_deg_from_quaternion, robust_stop_line_point,
    timestamps_synchronized)


def camera_matrix():
    return np.array([[100.0, 0.0, 4.5], [0.0, 100.0, 3.5],
                     [0.0, 0.0, 1.0]])


def mask():
    output = np.zeros((8, 10), np.uint8)
    output[2:7, 1:9] = 255
    return output


def depth_config():
    return StopLineDepthConfig(minimum_pixels=20)


def pitch_quaternion(degrees):
    radians = math.radians(degrees)
    return (0.0, math.sin(radians/2.0), 0.0, math.cos(radians/2.0))


def test_no_stop_line_has_no_valid_depth_point():
    result = robust_stop_line_point(np.zeros((8, 10), np.uint8),
                                    np.full((8, 10), 3000, np.uint16),
                                    "16UC1", camera_matrix(), depth_config())
    assert not result.valid and result.reason == "INSUFFICIENT_VALID_DEPTH"


def test_stop_line_with_16uc1_depth_produces_robust_optical_point():
    depth = np.full((8, 10), 3000, np.uint16)
    depth[3, 3] = 9000
    result = robust_stop_line_point(mask(), depth, "16UC1", camera_matrix(),
                                    depth_config())
    assert result.valid
    assert math.isclose(result.median_depth_m, 3.0)
    assert result.valid_pixels == 39
    assert math.isclose(result.optical_xyz_m[2], 3.0)


def test_zero_and_nan_depth_never_become_valid_distance():
    zero = robust_stop_line_point(mask(), np.zeros((8, 10), np.uint16),
                                  "16UC1", camera_matrix(), depth_config())
    nan = robust_stop_line_point(mask(), np.full((8, 10), np.nan, np.float32),
                                 "32FC1", camera_matrix(), depth_config())
    assert not zero.valid and not nan.valid


def test_mask_depth_timestamp_mismatch_is_rejected():
    assert timestamps_synchronized((10.0, 10.02, 10.01), 0.05)
    assert not timestamps_synchronized((10.0, 10.08, 10.01), 0.05)


def test_tf_and_calibration_missing_return_nan():
    distance, source = front_axle_distance((0.0, 0.0, 3.0))
    assert math.isnan(distance) and source == "CALIBRATION_INVALID"


def test_explicit_offset_fallback_is_disabled_by_default():
    invalid, _ = front_axle_distance((0.0, 0.0, 3.0),
                                     camera_to_front_axle_m=0.26)
    valid, source = front_axle_distance(
        (0.0, 0.0, 3.0), allow_fallback=True,
        camera_to_front_axle_m=0.26)
    assert math.isnan(invalid)
    assert math.isclose(valid, 2.74) and source == "OFFSET_FALLBACK"


def test_front_axle_tf_distinguishes_ahead_and_behind():
    ahead, source = front_axle_distance(
        (2.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    behind, _ = front_axle_distance(
        (-0.2, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert source == "TF" and ahead > 0.0 and behind < 0.0


def test_t870_center_base_front_axle_is_half_of_073_wheelbase():
    assert math.isclose(0.73/2.0, 0.365)
    assert math.isclose(
        front_axle_distance_from_base_point((1.365, 0.0, 0.0), 0.73),
        1.0)
    assert math.isclose(
        front_axle_distance_from_base_point((0.265, 0.0, 0.0), 0.73),
        -0.1)


def test_two_physical_stop_line_components_are_sorted_and_not_duplicated():
    node = object.__new__(CameraMissionPerceptionNode)
    values = {"stop_line_component_merge_px": 2,
              "stop_line_distance_merge_m": .20}
    node.p = values.__getitem__
    node.depth_config = StopLineDepthConfig(minimum_pixels=20)
    node.camera_info = SimpleNamespace(k=[100., 0., 49.5, 0., 100., 29.5,
                                          0., 0., 1.])
    node.stop_samples = ()
    node._front_axle_distance = lambda point, frame, stamp: (
        float(point[2]), True, "TEST")
    line_mask = np.zeros((60, 100), np.uint8)
    # A small segmentation gap must merge as one physical line.
    line_mask[20:23, 10:45] = 255
    line_mask[20:23, 48:90] = 255
    line_mask[40:43, 10:90] = 255
    depth = np.zeros_like(line_mask, np.uint16)
    depth[20:23, 10:45] = 3000
    depth[20:23, 48:90] = 3000
    depth[40:43, 10:90] = 1000
    message = SimpleNamespace(encoding="16UC1",
                              header=SimpleNamespace(frame_id="optical"))
    header = SimpleNamespace(frame_id="optical", stamp=SimpleNamespace())
    result = node._component_distances(line_mask, depth, message, header, 1.0)
    assert [item["distance"] for item in result] == [1.0, 3.0]


def test_sign_one_frame_false_positive_does_not_confirm():
    state = DebouncedPresence(PresenceConfig(on_frames=3, off_frames=2))
    assert not state.update(True, .9, 100, 0.0)
    assert not state.update(False, 0.0, 0.0, 0.1)


def test_sign_consecutive_detection_and_timeout():
    state = DebouncedPresence(PresenceConfig(on_frames=3, timeout_sec=.5))
    assert not state.update(True, .9, 100, 0.0)
    assert not state.update(True, .9, 100, 0.1)
    assert state.update(True, .9, 100, 0.2)
    assert not state.tick(0.71)


def traffic():
    return MissionTrafficFilter(MissionTrafficConfig(
        minimum_confidence=.5, on_frames=3, switch_frames=3,
        timeout_sec=.5))


def test_red_and_green_require_consecutive_confirmation():
    for scores, expected in [((.8, 0, 0, 0), "R"),
                             ((0, .8, 0, 0), "G")]:
        state = traffic()
        assert state.update(*scores, 0.0) == "UNKNOWN"
        assert state.update(*scores, 0.1) == "UNKNOWN"
        assert state.update(*scores, 0.2) == expected


def test_red_to_green_switch_requires_switch_frames():
    state = traffic()
    for now in (0.0, .1, .2):
        state.update(.8, 0, 0, 0, now)
    assert state.state == "R"
    assert state.update(0, .8, 0, 0, .3) == "R"
    assert state.update(0, .8, 0, 0, .4) == "R"
    assert state.update(0, .8, 0, 0, .5) == "G"


def test_red_green_conflict_is_immediately_unknown():
    state = traffic()
    for now in (0.0, .1, .2):
        state.update(.8, 0, 0, 0, now)
    assert state.update(.8, .7, 0, 0, .3) == "UNKNOWN"
    assert state.conflict and state.reason == "RED_GREEN_CONFLICT"


def test_yellow_or_other_only_is_unknown_and_state_times_out():
    state = traffic()
    assert state.update(0, 0, 0, .9, 0.0) == "UNKNOWN"
    for now in (.1, .2, .3):
        state.update(.8, 0, 0, 0, now)
    assert state.state == "R"
    assert state.tick(.81) == "UNKNOWN"


def test_red_plus_left_confirms_left_permission_but_left_alone_does_not():
    state = traffic()
    for now in (0.0, .1, .2):
        result = state.update(.8, 0, .9, 0, now)
    assert result == "LEFT"

    state = traffic()
    for now in (0.0, .1, .2):
        result = state.update(0, 0, .9, 0, now)
    assert result == "UNKNOWN"
    assert state.reason == "UNSUPPORTED_LIGHT"


def test_pitch_quaternion_is_signed_and_invalid_quaternion_rejected():
    assert math.isclose(pitch_deg_from_quaternion(pitch_quaternion(15)), 15.0)
    try:
        pitch_deg_from_quaternion((0, 0, 0, 0))
        assert False
    except ValueError:
        pass


def test_uphill_threshold_hysteresis_downhill_and_timeout_contract():
    detector = UphillDetector(UphillConfig(
        on_deg=15.0, off_deg=12.0, minimum_duration_sec=.25), 0.0)
    assert not detector.update(0.0, True, 0.0)
    assert not detector.update(14.9, True, .1)
    assert not detector.update(15.0, True, .2)
    assert detector.update(15.1, True, .45)
    assert detector.update(12.0, True, .5)
    assert not detector.update(11.9, True, .6)
    assert not detector.update(-20.0, True, .7)
    assert not detector.update(math.nan, False, .8)


def test_calibrated_vehicle_pitch_contract_is_positive_nose_up_only():
    detector = UphillDetector(UphillConfig(
        on_deg=15.0, off_deg=12.0, minimum_duration_sec=.25), 0.0)
    assert not detector.update(14.9, True, 0.0)
    assert not detector.update(15.0, True, .1)
    assert detector.update(15.0, True, .35)
    assert detector.update(20.0, True, .4)
    assert not detector.update(11.9, True, .5)
    assert not detector.update(-15.0, True, .6)
    assert not detector.update(-25.0, True, .7)


def test_node_source_has_no_control_publishers_and_debug_defaults_off():
    root = Path(__file__).parents[1]
    source = (root/"camera_navigation"/
              "camera_mission_perception_node.py").read_text()
    config = (root/"config"/"mission_perception.yaml").read_text()
    assert 'Float32, "/camera/mission/stop_line_distance_m"' in source
    for forbidden in ('create_publisher(Float32, "/camera_drive"',
                      'create_publisher(Int32, "/camera_wheel"',
                      '"/mcu/cmd_drive"', '"/mcu/cmd_wheel"'):
        assert forbidden not in source
    assert "debug_overlay_enabled: false" in config
    assert "front_axle_x_m: 0.365" in config
    assert "wheelbase_m: 0.73" in config
    assert "allow_camera_to_front_axle_fallback" not in config
    assert "imu_reference_pitch_deg: 0.0" in config
