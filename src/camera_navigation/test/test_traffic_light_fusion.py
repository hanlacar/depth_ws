"""Traffic-light fusion synchronization, policy, and contract tests."""

import json
import math
from pathlib import Path

import pytest

from camera_navigation.traffic_light_fusion_core import (
    FusionConfig, SourceObservation, TrafficLightFusion, combine_confidence,
    normalize_rgb_diagnostics, normalize_yolo_document)


def config(**kwargs):
    values = dict(fusion_confirm_frames=2,
                  fusion_switch_confirm_frames=3,
                  fusion_single_source_confirm_frames=3,
                  fusion_max_stamp_delta_sec=.05)
    values.update(kwargs)
    return FusionConfig(**values)


def observation(source, state, aspect, confidence, stamp, received, sequence,
                bbox=(100., 50., 140., 90.), class_name=""):
    return SourceObservation(source, state, aspect, confidence, stamp,
                             received, sequence, class_name, bbox)


def paired_frame(machine, sequence, yolo_state, rgb_aspect,
                 yolo_aspect="UNKNOWN", confidence=.9, stamp=None,
                 order="yolo_first"):
    stamp = float(sequence) if stamp is None else float(stamp)
    rgb_state = "R" if rgb_aspect in ("RED", "RED_X", "YELLOW") else "G"
    yolo = observation("YOLO", yolo_state, yolo_aspect, confidence, stamp,
                       stamp, sequence, class_name="R_light" if yolo_state == "R"
                       else "G_light")
    rgb = observation("RGB", rgb_state, rgb_aspect, confidence, stamp,
                      stamp, sequence)
    items = (yolo, rgb) if order == "yolo_first" else (rgb, yolo)
    for item in items:
        assert machine.ingest(item)
    return machine.evaluate(stamp+.01)


def test_actual_yolo_classes_normalize_without_class_indices():
    document = {"timestamp": {"sec": 1, "nanosec": 0}, "detections": [
        {"class_id": 99, "class_name": "R_light", "confidence": .8,
         "xyxy": [1, 2, 10, 12]}]}
    red, reason = normalize_yolo_document(document, 1., 1)
    assert reason == "OK" and red.state == "R" and red.aspect == "UNKNOWN"
    document["detections"][0]["class_name"] = "G_light"
    green, _ = normalize_yolo_document(document, 1., 2)
    assert green.state == "G" and green.aspect == "UNKNOWN"
    document["detections"][0]["class_name"] = "Left"
    left, _ = normalize_yolo_document(document, 1., 3)
    assert left.state == "G" and left.aspect == "GREEN_LEFT"
    assert "GREEN_DOWN" not in {
        normalize_yolo_document(document, 1., 4)[0].aspect}


@pytest.mark.parametrize("yolo_state,rgb_aspect,expected_state", [
    ("R", "RED", "R"), ("R", "YELLOW", "R"),
    ("G", "GREEN_CIRCLE", "G"), ("G", "GREEN_LEFT", "G"),
    ("G", "GREEN_DOWN", "G"),
])
def test_agreement_combines_state_and_rgb_detail(
        yolo_state, rgb_aspect, expected_state):
    machine = TrafficLightFusion(config())
    assert paired_frame(machine, 1, yolo_state, rgb_aspect).state == "UNKNOWN"
    result = paired_frame(machine, 2, yolo_state, rgb_aspect)
    assert (result.state, result.aspect) == (expected_state, rgb_aspect)
    assert result.confidence == pytest.approx(combine_confidence(.9, .9))


def test_same_stamp_input_order_does_not_change_result():
    first = TrafficLightFusion(config(fusion_confirm_frames=1))
    second = TrafficLightFusion(config(fusion_confirm_frames=1))
    a = paired_frame(first, 1, "G", "GREEN_LEFT", order="yolo_first")
    b = paired_frame(second, 1, "G", "GREEN_LEFT", order="rgb_first")
    assert (a.state, a.aspect, a.confidence) == (b.state, b.aspect, b.confidence)


def test_one_source_arriving_early_does_not_double_count_pair_frames():
    machine = TrafficLightFusion(config())
    paired_frame(machine, 1, "G", "GREEN_DOWN")
    machine.ingest(observation("YOLO", "G", "UNKNOWN", .9, 1.02, 1.02, 2,
                               class_name="G_light"))
    early = machine.evaluate(1.03)
    assert early.state == "UNKNOWN"
    machine.ingest(observation("RGB", "G", "GREEN_DOWN", .9, 1.02, 1.02, 2))
    confirmed = machine.evaluate(1.04)
    assert confirmed.aspect == "GREEN_DOWN"


def test_stamp_delta_prevents_false_same_frame_conflict():
    machine = TrafficLightFusion(config(fusion_single_source_confirm_frames=1))
    machine.ingest(observation("YOLO", "R", "UNKNOWN", .9, 1., 1., 1))
    machine.ingest(observation("RGB", "G", "GREEN_CIRCLE", .9, 2., 1., 1))
    result = machine.evaluate(1.1)
    assert not result.diagnostics["sources_conflict"]
    assert (result.state, result.aspect) == ("G", "GREEN_CIRCLE")


def test_stale_missing_rewind_and_nonfinite_inputs_fail_closed():
    empty = TrafficLightFusion(config()).evaluate(1.)
    assert empty.state == "UNKNOWN" and empty.reason == "NO_INPUT"
    machine = TrafficLightFusion(config(fusion_single_source_confirm_frames=1))
    assert machine.ingest(observation("YOLO", "R", "UNKNOWN", .9, 2., 2., 1))
    assert not machine.ingest(observation("YOLO", "G", "UNKNOWN", .9, 1., 2.1, 2))
    assert machine.evaluate(3.).reason == "STALE"
    invalid, reason = normalize_yolo_document({
        "timestamp": {"sec": 1, "nanosec": 0}, "detections": [{
            "class_name": "G_light", "confidence": math.nan,
            "xyxy": [1, 2, 3, 4]}]}, 1., 1)
    assert invalid.state == "UNKNOWN" and reason == "NO_YOLO_LIGHT"


def test_single_source_needs_wait_window_and_consecutive_samples():
    machine = TrafficLightFusion(config())
    for sequence in range(1, 4):
        received = float(sequence)
        machine.ingest(observation(
            "RGB", "G", "GREEN_LEFT", .9, received, received, sequence))
        early = machine.evaluate(received+.01)
        assert early.state == "UNKNOWN"
        result = machine.evaluate(received+.06)
    assert (result.state, result.aspect) == ("G", "GREEN_LEFT")
    assert result.reason == "RGB_ONLY_CONFIRMED"
    assert result.diagnostics["single_source_used"]


@pytest.mark.parametrize("yolo_state,rgb_aspect", [
    ("R", "GREEN_CIRCLE"), ("G", "RED"),
])
def test_opposite_rg_is_unknown(yolo_state, rgb_aspect):
    machine = TrafficLightFusion(config(fusion_confirm_frames=1))
    result = paired_frame(machine, 1, yolo_state, rgb_aspect)
    assert result.state == "UNKNOWN" and result.reason == "CONFLICT"


def test_detailed_green_conflict_and_post_conflict_frame_do_not_release():
    machine = TrafficLightFusion(config(
        fusion_confirm_frames=1, fusion_single_source_confirm_frames=2))
    conflict = paired_frame(machine, 1, "G", "GREEN_DOWN",
                            yolo_aspect="GREEN_LEFT")
    assert conflict.state == "UNKNOWN" and conflict.reason == "CONFLICT"
    machine.rgb = None
    machine.ingest(observation("YOLO", "G", "UNKNOWN", .9, 2., 2., 2))
    one = machine.evaluate(2.06)
    assert one.state == "UNKNOWN" and one.reason == "WAITING_CONFIRMATION"


def test_rgb_diagnostics_position_and_json_finiteness():
    observation_value, reason = normalize_rgb_diagnostics({
        "stamp": 1_000_000_000, "state": "G", "aspect": "GREEN_DOWN",
        "confidence": .8, "selected_bbox": [10, 20, 30, 40]}, 1., 1)
    assert reason == "OK" and observation_value.bbox == (10., 20., 30., 40.)
    machine = TrafficLightFusion(config(fusion_single_source_confirm_frames=1))
    machine.ingest(observation_value)
    decision = machine.evaluate(1.06, route_mode="11")
    assert decision.diagnostics["rgb_green_down_verified"]
    json.dumps(decision.diagnostics, allow_nan=False)


def test_launch_and_node_contracts_preserve_parallel_original_topics():
    root = Path(__file__).parents[1]
    launch = (root / "launch" /
              "d456_yolo_rgb_traffic_light_fusion_validation.launch.py").read_text()
    node = (root / "camera_navigation" /
            "traffic_light_fusion_node.py").read_text()
    assert "camera_yolo_inference" in launch and "rgb_traffic_light_node" in launch
    assert launch.count("/camera/image_raw") >= 2
    for topic in ("/perception/detections_json", "/camera_traffic_light",
                  "/camera/traffic_light_rgb/diagnostics",
                  "/camera/traffic_light_fused/state",
                  "/camera/traffic_light_fused/aspect"):
        assert topic in node
    forbidden = ("/mcu_drive", "/mcu_wheel", "/lidar_drive",
                 "/lidar_wheel", "/avoidance/active")
    assert not any(topic in node for topic in forbidden)
