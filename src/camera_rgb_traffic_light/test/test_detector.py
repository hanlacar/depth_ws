import json
import math

import cv2
import numpy as np
import pytest

from camera_rgb_traffic_light.detector import (
    ASPECTS, STATES, Candidate, ColorTrafficLightDetector, DetectionResult, DetectorConfig,
    TemporalConfig, TemporalTrafficLightFilter, build_diagnostics,
    finite_diagnostics, normalize_to_bgr)


def canvas(width=640, height=480):
    return np.zeros((height, width, 3), np.uint8)


def circle(color, center=(320, 90), radius=14, width=640, height=480):
    image = canvas(width, height)
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    return image


def arrow(direction="left", center=(320, 90), width=640, height=480):
    image = canvas(width, height)
    cx, cy = center
    if direction == "left":
        points = [(cx-22, cy), (cx-6, cy-16), (cx-6, cy-7),
                  (cx+24, cy-7), (cx+24, cy+7), (cx-6, cy+7),
                  (cx-6, cy+16)]
    elif direction == "right":
        points = [(cx+22, cy), (cx+6, cy-16), (cx+6, cy-7),
                  (cx-24, cy-7), (cx-24, cy+7), (cx+6, cy+7),
                  (cx+6, cy+16)]
    elif direction == "down":
        points = [(cx, cy+22), (cx-16, cy+6), (cx-7, cy+6),
                  (cx-7, cy-24), (cx+7, cy-24), (cx+7, cy+6),
                  (cx+16, cy+6)]
    elif direction == "up":
        points = [(cx, cy-22), (cx-16, cy-6), (cx-7, cy-6),
                  (cx-7, cy+24), (cx+7, cy+24), (cx+7, cy-6),
                  (cx+16, cy-6)]
    else:
        raise ValueError(direction)
    cv2.fillPoly(image, [np.asarray(points, np.int32)], (0, 255, 0),
                 cv2.LINE_AA)
    return image


def red_x(center=(320, 90)):
    image = canvas(); cx, cy = center
    cv2.line(image, (cx-18, cy-18), (cx+18, cy+18), (0, 0, 255), 7,
             cv2.LINE_AA)
    cv2.line(image, (cx+18, cy-18), (cx-18, cy+18), (0, 0, 255), 7,
             cv2.LINE_AA)
    return image


@pytest.fixture
def detector():
    return ColorTrafficLightDetector()


def test_red_round_lamp_is_r(detector):
    result = detector.detect(circle((0, 0, 255)))
    assert result.raw_state == "R"
    assert result.selected.color == "red"


def test_yellow_round_lamp_is_merged_into_r(detector):
    result = detector.detect(circle((0, 255, 255)))
    assert result.raw_state == "R"
    assert result.selected.color == "yellow"


def test_red_x_is_r_with_red_x_aspect_and_rectangle_is_rejected(detector):
    result = detector.detect(red_x())
    assert result.raw_state == "R", result.rejection_reasons
    assert result.raw_aspect == "RED_X"
    rectangle = canvas()
    cv2.rectangle(rectangle, (300, 70), (340, 110), (0, 0, 255), -1)
    assert detector.detect(rectangle).raw_aspect != "RED_X"


def test_green_round_lamp_is_g(detector):
    result = detector.detect(circle((0, 255, 0)))
    assert result.raw_state == "G"


def test_clear_left_green_arrow_is_g_with_shape_diagnostic(detector):
    result = detector.detect(arrow("left"))
    assert result.raw_state == "G", result.rejection_reasons
    assert result.selected.raw_shape == "LEFT_ARROW"
    assert result.selected.left_score > result.selected.right_score


def test_down_green_arrow_is_g_and_uses_real_down_geometry(detector):
    result = detector.detect(arrow("down"))
    assert result.raw_state == "G", result.rejection_reasons
    assert result.selected.raw_shape == "DOWN_ARROW"
    assert result.selected.down_score >= detector.config.down_minimum_score


def test_right_and_unclear_green_arrows_are_g(detector):
    right = detector.detect(arrow("right"))
    unclear = detector.detect(arrow("up"))
    assert right.raw_state == "G"
    assert right.selected.raw_shape == "OTHER_GREEN_SHAPE"
    assert unclear.raw_state == "G"
    assert unclear.selected.raw_shape == "OTHER_GREEN_SHAPE"


def test_control_state_never_exposes_shape_names(detector):
    images = (circle((0, 0, 255)), circle((0, 255, 0)),
              arrow("left"), arrow("down"), arrow("right"))
    assert STATES == ("R", "G", "UNKNOWN")
    assert all(detector.detect(image).raw_state in STATES for image in images)
    assert all(detector.detect(image).raw_state not in ("LEFT", "DOWN", "CIRCLE")
               for image in images)


def test_aspect_contract_maps_all_supported_shapes(detector):
    assert ASPECTS == ("RED", "RED_X", "YELLOW", "GREEN_CIRCLE",
                       "GREEN_LEFT", "GREEN_DOWN", "GREEN_OTHER", "UNKNOWN")
    expected = ((circle((0, 0, 255)), "RED"),
                (circle((0, 255, 255)), "YELLOW"),
                (circle((0, 255, 0)), "GREEN_CIRCLE"),
                (arrow("left"), "GREEN_LEFT"),
                (arrow("down"), "GREEN_DOWN"),
                (arrow("right"), "GREEN_OTHER"))
    assert [detector.detect(image).raw_aspect for image, _ in expected] == [
        aspect for _, aspect in expected]


def test_temporal_confirmation_tracks_green_aspect_not_only_state(detector):
    tracker = TemporalTrafficLightFilter(TemporalConfig(
        confirmation_frames=2, switch_confirmation_frames=2))
    left = detector.detect(arrow("left")); down = detector.detect(arrow("down"))
    assert tracker.update(left, 0., (480, 640, 3)).aspect == "UNKNOWN"
    assert tracker.update(left, .1, (480, 640, 3)).aspect == "GREEN_LEFT"
    assert tracker.update(down, .2, (480, 640, 3)).aspect == "GREEN_LEFT"
    assert tracker.update(down, .3, (480, 640, 3)).aspect == "GREEN_DOWN"


def test_red_x_aspect_also_requires_temporal_confirmation(detector):
    tracker = TemporalTrafficLightFilter(TemporalConfig(confirmation_frames=2))
    result = detector.detect(red_x())
    assert tracker.update(result, 0., (480, 640, 3)).aspect == "UNKNOWN"
    confirmed = tracker.update(result, .1, (480, 640, 3))
    assert confirmed.state == "R" and confirmed.aspect == "RED_X"


def test_timeout_clears_detailed_aspect_with_state(detector):
    tracker = TemporalTrafficLightFilter(TemporalConfig(
        confirmation_frames=1, lost_timeout_sec=.2, input_timeout_sec=.3))
    confirmed = tracker.update(detector.detect(arrow("left")), 1.,
                               (480, 640, 3))
    assert confirmed.aspect == "GREEN_LEFT"
    timed_out = tracker.tick(1.4)
    assert (timed_out.state, timed_out.aspect) == ("UNKNOWN", "UNKNOWN")


def test_green_rectangle_is_not_confirmed(detector):
    image = canvas()
    cv2.rectangle(image, (300, 70), (340, 110), (0, 255, 0), -1)
    assert detector.detect(image).raw_state == "UNKNOWN"


def test_small_green_noise_is_unknown(detector):
    image = canvas()
    image[80:82, 300:302] = (0, 255, 0)
    assert detector.detect(image).raw_state == "UNKNOWN"


def test_large_green_sign_is_unknown(detector):
    image = canvas()
    cv2.rectangle(image, (100, 40), (540, 220), (0, 255, 0), -1)
    assert detector.detect(image).raw_state == "UNKNOWN"


def test_lower_image_green_object_is_outside_roi(detector):
    assert detector.detect(circle((0, 255, 0), center=(320, 400))).raw_state == "UNKNOWN"


def test_unlit_dark_green_object_is_unknown(detector):
    assert detector.detect(circle((0, 80, 0))).raw_state == "UNKNOWN"


def test_green_without_dark_housing_is_unknown(detector):
    image = np.full((480, 640, 3), 120, np.uint8)
    cv2.circle(image, (320, 90), 14, (0, 255, 0), -1, cv2.LINE_AA)
    result = detector.detect(image)
    assert result.raw_state == "UNKNOWN"
    assert result.rejection_reasons.get("insufficient_dark_housing", 0) >= 1


def test_low_confidence_candidate_is_unknown():
    detector = ColorTrafficLightDetector(DetectorConfig(minimum_confidence=.99))
    result = detector.detect(circle((0, 255, 0)))
    assert result.raw_state == "UNKNOWN"
    assert result.rejection_reasons.get("low_confidence", 0) >= 1


def test_red_green_similar_confidence_is_conflict_unknown(detector):
    image = circle((0, 0, 255), center=(280, 90))
    cv2.circle(image, (360, 90), 14, (0, 255, 0), -1, cv2.LINE_AA)
    result = detector.detect(image)
    assert result.raw_state == "UNKNOWN" and result.conflict


def test_same_housing_red_green_conflict_is_red_priority(detector):
    image = circle((0, 0, 255), center=(300, 90))
    cv2.circle(image, (340, 90), 14, (0, 255, 0), -1, cv2.LINE_AA)
    result = detector.detect(image)
    assert result.raw_state == "R"
    assert result.selected.color == "red"


def candidate(state, confidence, bbox=(100, 40, 20, 20), color=None):
    return Candidate(state, color or ("red" if state == "R" else "green"),
                     bbox, confidence, .9, .95, .9, .9,
                     .8 if state == "G" else .1, .1, 100., .9, .5, (),
                     "LEFT_ARROW" if state == "G" else "CIRCLE", 0.1, 0.8)


def test_high_confidence_red_has_priority_when_not_ambiguous(detector):
    red, green = candidate("R", .94), candidate("G", .62, (200, 40, 20, 20))
    result = detector._resolve((red, green), {}, (0, 0, 640, 264))
    assert result.raw_state == "R"


def test_confirmation_and_switch_require_multiple_frames():
    tracker = TemporalTrafficLightFilter(TemporalConfig(
        confirmation_frames=3, switch_confirmation_frames=2))
    red = DetectionResult("R", .9, (candidate("R", .9),),
                          candidate("R", .9), False, {}, (0, 0, 1, 1))
    assert tracker.update(red, 0.0, (480, 640, 3)).state == "UNKNOWN"
    assert tracker.update(red, .1, (480, 640, 3)).state == "UNKNOWN"
    assert tracker.update(red, .2, (480, 640, 3)).state == "R"
    green_candidate = candidate("G", .9)
    green = DetectionResult("G", .9, (green_candidate,), green_candidate,
                            False, {}, (0, 0, 1, 1))
    assert tracker.update(green, .3, (480, 640, 3)).state == "R"
    assert tracker.update(green, .4, (480, 640, 3)).state == "G"


def test_valid_green_stays_unknown_until_confirmation_count_is_met():
    tracker = TemporalTrafficLightFilter(TemporalConfig(confirmation_frames=3))
    result = ColorTrafficLightDetector().detect(arrow("down"))
    assert tracker.update(result, 0.0, (480, 640, 3)).state == "UNKNOWN"
    assert tracker.update(result, 0.1, (480, 640, 3)).state == "UNKNOWN"
    decision = tracker.update(result, 0.2, (480, 640, 3))
    assert decision.state == "G" and decision.confirmation_count >= 3


def test_non_contract_raw_state_fails_closed():
    tracker = TemporalTrafficLightFilter(TemporalConfig(confirmation_frames=1))
    item = candidate("G", .9)
    invalid = DetectionResult("LEFT", .9, (item,), item, False, {}, (0, 0, 1, 1))
    assert tracker.update(invalid, 0.0, (480, 640, 3)).state == "UNKNOWN"


def test_short_loss_holds_then_timeout_returns_unknown():
    tracker = TemporalTrafficLightFilter(TemporalConfig(
        confirmation_frames=1, switch_confirmation_frames=1,
        lost_timeout_sec=.3, input_timeout_sec=.5))
    item = candidate("G", .9)
    found = DetectionResult("G", .9, (item,), item, False, {}, (0, 0, 1, 1))
    missing = DetectionResult("UNKNOWN", 0., (), None, False, {}, (0, 0, 1, 1))
    assert tracker.update(found, 1.0, (480, 640, 3)).state == "G"
    held = tracker.update(missing, 1.2, (480, 640, 3))
    assert held.state == "G" and held.held
    assert tracker.tick(1.6).state == "UNKNOWN"


def test_conflict_drops_confirmed_state_immediately():
    tracker = TemporalTrafficLightFilter(TemporalConfig(confirmation_frames=1))
    item = candidate("R", .9)
    tracker.update(DetectionResult("R", .9, (item,), item, False, {}, (0, 0, 1, 1)),
                   1.0, (480, 640, 3))
    conflict = DetectionResult("UNKNOWN", 0., (), None, True, {}, (0, 0, 1, 1))
    assert tracker.update(conflict, 1.1, (480, 640, 3)).state == "UNKNOWN"


def test_no_input_and_input_timeout_are_unknown():
    tracker = TemporalTrafficLightFilter()
    assert tracker.tick(10.0).state == "UNKNOWN"
    tracker.last_input_time = 1.0
    assert tracker.tick(2.0).state == "UNKNOWN"


def test_tracking_position_jump_resets_confirmation():
    tracker = TemporalTrafficLightFilter(TemporalConfig(
        confirmation_frames=2, track_iou_threshold=.2,
        track_center_distance_ratio=.01))
    first = candidate("R", .9, (100, 40, 20, 20))
    second = candidate("R", .9, (500, 40, 20, 20))
    tracker.update(DetectionResult("R", .9, (first,), first, False, {}, (0, 0, 1, 1)),
                   0., (480, 640, 3))
    decision = tracker.update(
        DetectionResult("R", .9, (second,), second, False, {}, (0, 0, 1, 1)),
        .1, (480, 640, 3))
    assert decision.state == "UNKNOWN" and decision.confirmation_count == 1


def test_roi_ratios_scale_with_resolution(detector):
    small = detector.roi_bounds((480, 640, 3))
    large = detector.roi_bounds((960, 1280, 3))
    assert all(abs(actual-expected*2) <= 1
               for actual, expected in zip(large, small))


def test_rgb_and_bgr_normalization_is_exact():
    rgb = np.array([[[255, 0, 0]]], np.uint8)
    bgr = normalize_to_bgr(rgb, "rgb8")
    assert bgr.tolist() == [[[0, 0, 255]]]
    assert normalize_to_bgr(bgr, "bgr8").tolist() == bgr.tolist()


@pytest.mark.parametrize("bad", [None, np.array([], np.uint8),
                                   np.zeros((4, 4), np.float32)])
def test_invalid_images_fail_safely(detector, bad):
    result = detector.detect(bad)
    assert result.raw_state == "UNKNOWN"
    assert result.rejection_reasons == {"invalid_image": 1}


def test_overlay_preserves_geometry_and_displays_state_shape_and_rejection(
        detector, monkeypatch):
    image = circle((0, 255, 0))
    result = detector.detect(image)
    labels = []
    original = cv2.putText

    def capture(target, text, *args, **kwargs):
        labels.append(text)
        return original(target, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    overlay = detector.render_overlay(image, result, "G", 3, 30., 30., 2.)
    assert overlay.shape == image.shape and overlay.dtype == np.uint8
    rendered = " ".join(labels)
    assert "STATE=G" in rendered
    assert "SHAPE=CIRCLE" in rendered
    assert "conf=" in rendered and "reject=" in rendered


def test_diagnostics_fields_and_finiteness(detector):
    result = detector.detect(circle((0, 0, 255)))
    tracker = TemporalTrafficLightFilter(TemporalConfig(confirmation_frames=1))
    decision = tracker.update(result, 1., (480, 640, 3))
    payload = build_diagnostics(123, result, decision, 2., 3., 30., 29.)
    required = {"stamp", "state", "confidence", "input_age_ms",
                "processing_latency_ms", "candidate_count", "red_candidates",
                "yellow_candidates", "green_candidates", "left_candidates",
                "raw_state", "confirmed_state", "confirmation_count",
                "rejection_reasons", "roi", "raw_color", "raw_shape",
                "green_shape_score", "circle_score", "left_arrow_score",
                "down_arrow_score"}
    assert required <= payload.keys()
    assert payload["raw_shape"] == "CIRCLE"
    assert len(payload["selected_bbox"]) == 4
    assert finite_diagnostics(payload)
    json.dumps(payload, allow_nan=False)


def test_nonfinite_diagnostics_are_rejected():
    assert not finite_diagnostics({"bad": math.inf})
    assert not finite_diagnostics({"nested": {"bad": math.nan}})
