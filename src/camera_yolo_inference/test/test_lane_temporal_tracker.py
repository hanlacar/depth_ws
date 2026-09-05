import cv2
import numpy as np

from camera_yolo_inference.lane_temporal_tracker import (
    LaneMaskTemporalTracker, LaneTemporalConfig)


def frame_and_masks(shift=0, white=True, yellow=True):
    image = np.zeros((120, 160, 3), np.uint8)
    road = np.zeros((120, 160), np.uint8)
    cv2.rectangle(road, (30+shift, 15), (130+shift, 119), 255, -1)
    w = np.zeros_like(road)
    y = np.zeros_like(road)
    if white:
        cv2.line(w, (52+shift, 115), (68+shift, 20), 255, 4)
        cv2.line(image, (52+shift, 115), (68+shift, 20), (255, 255, 255), 4)
    if yellow:
        cv2.line(y, (108+shift, 115), (92+shift, 20), 255, 4)
        cv2.line(image, (108+shift, 115), (92+shift, 20), (0, 220, 255), 4)
    # Stable texture supplies flow points around each lane.
    for row in range(20, 115, 10):
        for col in range(38, 126, 9):
            cv2.circle(image, (col+shift, row), 1, (100, 100, 100), -1)
    return image, w, y, road


def test_classes_are_independent_and_never_cross_labelled():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
        mode="flow", min_flow_points=3, scene_change_threshold=0.8))
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 1.0)
    # The paint remains visible in RGB while only the detector mask drops.
    image2, _, yellow2, road2 = frame_and_masks(shift=2, white=True)
    result = tracker.update(image2, np.zeros_like(white), yellow2, road2, 1.033)
    assert result.diagnostics["white_line_source"] == "TRACKED"
    assert result.diagnostics["yellow_line_source"] == "RAW"
    assert np.any(result.tracked_white_line)
    assert not np.any(result.tracked_yellow_line)


def test_no_raw_history_never_synthesizes_a_line():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(mode="hold"))
    image, white, yellow, road = frame_and_masks(white=False, yellow=False)
    result = tracker.update(image, white, yellow, road, 1.0)
    assert not np.any(result.effective_white_line)
    assert not np.any(result.effective_yellow_line)
    assert result.diagnostics["white_line"]["discard_reason"] == "NO_PRIOR_RAW_DETECTION"


def test_timestamp_rewind_and_scene_change_clear_tracks():
    config = LaneTemporalConfig(mode="flow", min_flow_points=3,
                                scene_change_threshold=0.1)
    tracker = LaneMaskTemporalTracker(config)
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 1.0)
    blank = np.full_like(image, 255)
    result = tracker.update(blank, np.zeros_like(white), np.zeros_like(yellow),
                            road, 1.033)
    assert result.diagnostics["reset_reason"] == "SCENE_CHANGE"
    assert not np.any(result.effective_white_line)
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 2.0)
    result = tracker.update(image, np.zeros_like(white), np.zeros_like(yellow),
                            road, 1.9)
    assert result.diagnostics["reset_reason"] == "TIMESTAMP_REWIND"


def test_hold_expires_and_current_road_is_mandatory():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
        mode="flow", min_flow_points=3, max_hold_frames=2,
        max_hold_sec=0.5))
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 1.0)
    empty = np.zeros_like(road)
    result = tracker.update(image, np.zeros_like(white), np.zeros_like(yellow),
                            empty, 1.033)
    assert not np.any(result.effective_white_line)
    assert result.diagnostics["white_line"]["discard_reason"] == "ROAD_SUPPORT_INVALID"


def test_flow_rejects_physically_implausible_motion():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
        mode="flow", min_flow_points=3, max_motion_px=5.0,
        scene_change_threshold=0.9))
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 1.0)
    moved, _, _, moved_road = frame_and_masks(shift=25, white=False, yellow=False)
    result = tracker.update(moved, np.zeros_like(white), np.zeros_like(yellow),
                            moved_road, 1.033)
    assert not np.any(result.effective_white_line)
    assert result.diagnostics["white_line"]["discard_reason"] == "OPTICAL_FLOW_INVALID"


def test_new_opposite_class_detection_discards_propagated_track():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
        mode="flow", min_flow_points=3, scene_change_threshold=0.9))
    image, white, _, road = frame_and_masks(yellow=False)
    tracker.update(image, white, np.zeros_like(white), road, 1.0)
    # The same visible stripe is now detected as yellow.  It must not coexist
    # with a propagated white identity.
    moved, visible, _, moved_road = frame_and_masks(shift=1, yellow=False)
    result = tracker.update(moved, np.zeros_like(white), visible, moved_road, 1.033)
    assert not np.any(result.tracked_white_line)
    assert result.diagnostics["white_line"]["discard_reason"] == "OPPOSITE_CLASS_CONFLICT"
    assert result.diagnostics["yellow_line_source"] == "RAW"


def test_long_dropout_expires_but_three_frame_dropout_is_supported():
    tracker = LaneMaskTemporalTracker(LaneTemporalConfig(
        mode="hold", max_hold_frames=3, max_hold_sec=0.5,
        scene_change_threshold=0.9))
    image, white, yellow, road = frame_and_masks()
    tracker.update(image, white, yellow, road, 1.0)
    zero = np.zeros_like(white)
    for index in range(1, 4):
        result = tracker.update(image, zero, zero, road, 1.0+index/30.0)
        assert np.any(result.tracked_white_line)
    result = tracker.update(image, zero, zero, road, 1.0+4/30.0)
    assert not np.any(result.tracked_white_line)
    assert result.diagnostics["white_line"]["discard_reason"] == "HOLD_LIMIT"
