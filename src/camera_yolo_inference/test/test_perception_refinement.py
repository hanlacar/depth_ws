import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from camera_yolo_inference.image_contract import LatestFrameBuffer
from camera_yolo_inference.perception_refinement import (
    CommonPerceptionRefiner, RefinementConfig)


ROLES = {"white_line": {1}, "yellow_line": {2}, "road": {0},
         "words": {3}, "stop_line": {4}, "c_line": {5}}


def scene(height=120, width=160):
    image = np.full((height, width, 3), 70, np.uint8)
    masks = {name: np.zeros((height, width), np.uint8) for name in ROLES}
    masks["road"][10:height-2, 10:width-10] = 255
    return image, masks


def instance(mask, class_id, confidence=0.8):
    return {"mask": (mask > 0).astype(np.float32), "class_id": class_id,
            "confidence": confidence}


def repeat(refiner, image, instances, masks, count=3, start=1.0):
    output = None
    for index in range(count):
        output = refiner.refine(
            image, instances, masks, ROLES, start+index/60.0, 0.5)
    return output


def test_white_yolo_instance_is_reclassified_as_yellow_after_hysteresis():
    image, masks = scene()
    line = np.zeros(masks["road"].shape, np.uint8)
    line[18:112, 72:78] = 255
    image[line > 0] = (0, 190, 225)
    masks["white_line"] = line
    output = repeat(CommonPerceptionRefiner(), image, [instance(line, 1)], masks)
    assert np.count_nonzero(output.yellow_line) == np.count_nonzero(line)
    assert np.count_nonzero(output.white_line) == 0
    assert output.diagnostics["line_instances"][0]["raw_class"] == "white_line"


def test_faded_yellow_in_shadow_uses_local_lab_contrast():
    image, masks = scene()
    image[:] = (38, 42, 42)
    line = np.zeros(masks["road"].shape, np.uint8)
    line[20:110, 60:66] = 255
    image[line > 0] = (28, 80, 95)
    masks["white_line"] = line
    config = RefinementConfig(yellow_hsv_s_min=35,
                              yellow_lab_b_road_delta_min=4.0)
    output = repeat(CommonPerceptionRefiner(config), image,
                    [instance(line, 1)], masks)
    assert np.count_nonzero(output.yellow_line) > 0


def test_adjacent_white_and_yellow_instances_keep_independent_colour_statistics():
    image, masks = scene()
    white = np.zeros(masks["road"].shape, np.uint8)
    yellow = np.zeros_like(white)
    white[20:112, 68:73] = 255
    yellow[20:112, 76:81] = 255
    image[white > 0] = (230, 230, 230)
    image[yellow > 0] = (0, 190, 225)
    masks["white_line"] = white | yellow
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(white, 1), instance(yellow, 1)], masks)
    assert np.count_nonzero(output.white_line) > 0
    assert np.count_nonzero(output.yellow_line) > 0


def test_insufficient_colour_evidence_is_kept_as_unknown_line():
    image, masks = scene()
    line = np.zeros(masks["road"].shape, np.uint8)
    line[20:112, 70:76] = 255
    image[line > 0] = (75, 68, 65)
    masks["white_line"] = line
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(line, 1)], masks, count=1)
    assert np.count_nonzero(output.unknown_line) == np.count_nonzero(line)
    assert np.count_nonzero(output.white_line | output.yellow_line) == 0


@pytest.mark.parametrize("adjacent", [True, False])
def test_stop_line_detected_with_or_without_yellow_adjacency(adjacent):
    image, masks = scene()
    stop = np.zeros(masks["road"].shape, np.uint8)
    stop[72:78, 35:125] = 255
    image[stop > 0] = (235, 235, 235)
    masks["stop_line"] = stop
    instances = [instance(stop, 4)]
    if adjacent:
        yellow = np.zeros_like(stop)
        yellow[20:75, 31:35] = 255
        image[yellow > 0] = (0, 190, 225)
        masks["yellow_line"] = yellow
        instances.append(instance(yellow, 2))
    output = repeat(CommonPerceptionRefiner(), image, instances, masks, count=2)
    assert np.count_nonzero(output.stop_line) > 0
    assert output.stop_confidence >= 0.55


def test_irregular_words_inside_road_are_not_used_as_lane():
    image, masks = scene()
    words = np.zeros(masks["road"].shape, np.uint8)
    cv2.putText(words, "A", (55, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 6)
    image[words > 0] = (235, 235, 235)
    masks["words"] = words
    masks["white_line"] = words
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(words, 1), instance(words, 3)], masks)
    assert np.count_nonzero(output.words) > 0
    assert np.count_nonzero(output.white_line & words) == 0


def test_single_transverse_stripe_is_not_crosswalk():
    image, masks = scene()
    stripe = np.zeros(masks["road"].shape, np.uint8)
    stripe[70:76, 35:125] = 255
    image[stripe > 0] = 235
    masks["stop_line"] = stripe
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(stripe, 4)], masks, count=2)
    assert np.count_nonzero(output.crosswalk) == 0


def test_repeated_parallel_stripes_are_crosswalk_not_stop_line():
    image, masks = scene()
    stripes = np.zeros(masks["road"].shape, np.uint8)
    for row in (48, 61, 74, 87):
        stripes[row:row+4, 38:122] = 255
    image[stripes > 0] = 235
    masks["c_line"] = stripes
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(stripes, 5)], masks, count=2)
    assert np.count_nonzero(output.crosswalk) > 0
    assert np.count_nonzero(output.stop_line) == 0


def test_planar_marking_gap_is_restored_when_it_connects_road_components():
    image, masks = scene()
    masks["road"][54:66, 20:140] = 0
    marking = np.zeros(masks["road"].shape, np.uint8)
    marking[54:66, 20:140] = 255
    image[marking > 0] = 220
    masks["words"] = marking
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(marking, 3)], masks, count=1)
    assert np.count_nonzero(output.restored_markings) > 0
    assert np.all(output.road[56:64, 30:130] > 0)


def test_large_nonroad_region_and_obstacle_are_not_restored():
    image, masks = scene()
    masks["road"][35:95, 30:130] = 0
    marking = np.zeros(masks["road"].shape, np.uint8)
    marking[35:95, 30:130] = 255
    masks["words"] = marking
    obstacle = marking.copy()
    output = repeat(CommonPerceptionRefiner(), image,
                    [instance(marking, 3), instance(obstacle, 99)], masks,
                    count=1)
    assert np.count_nonzero(output.restored_markings) == 0
    assert np.count_nonzero(output.road[35:95, 30:130]) == 0


def test_one_frame_colour_flicker_does_not_switch_stable_track():
    image, masks = scene()
    line = np.zeros(masks["road"].shape, np.uint8)
    line[18:112, 72:78] = 255
    masks["white_line"] = line
    refiner = CommonPerceptionRefiner()
    image[line > 0] = 235
    repeat(refiner, image, [instance(line, 1)], masks, count=3)
    image[line > 0] = (0, 190, 225)
    output = repeat(refiner, image, [instance(line, 1)], masks,
                    count=1, start=1.1)
    assert np.count_nonzero(output.white_line) > 0
    assert np.count_nonzero(output.yellow_line) == 0


def test_stale_or_duplicate_timestamp_is_rejected():
    image, masks = scene()
    refiner = CommonPerceptionRefiner()
    refiner.refine(image, [], masks, ROLES, 2.0)
    with pytest.raises(ValueError, match="stale_or_duplicate"):
        refiner.refine(image, [], masks, ROLES, 2.0)


def test_latest_frame_buffer_drops_stale_and_replaces_unprocessed_frame():
    def message(sec, nanosec):
        return SimpleNamespace(header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))
    buffer = LatestFrameBuffer()
    assert buffer.push(message(1, 10)) == "accepted"
    assert buffer.push(message(1, 20)) == "replaced"
    assert buffer.push(message(1, 15)) == "stale"
    assert buffer.push(message(1, 20)) == "duplicate"
    assert buffer.take().header.stamp.nanosec == 20
    snapshot = buffer.snapshot()
    assert snapshot["replaced_before_processing"] == 1
    assert snapshot["stale"] == 1 and snapshot["duplicate"] == 1
