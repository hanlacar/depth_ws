import unittest

import numpy as np

from camera_yolo_inference.traffic_light_state import (
    PUBLISHED_STATES, UNKNOWN, TrafficLightConfig, TrafficLightFilter,
    collect_traffic_light_evidence)


ROLE_IDS = {
    "red_light": (1,),
    "green_light": (3,),
}


def detection(class_id, confidence=0.9, area=25):
    mask = np.zeros((5, 5), dtype=np.float32)
    mask.flat[:max(0, min(25, int(area)))] = 1.0
    return {
        "class_id": class_id,
        "confidence": confidence,
        "mask": mask,
        "xyxy": (0.0, 0.0, 5.0, 5.0),
    }


class TrafficLightStateTest(unittest.TestCase):

    def test_consecutive_red_and_green_confirm(self):
        for class_id, expected in ((1, "R"), (3, "G")):
            with self.subTest(expected=expected):
                state_filter = TrafficLightFilter()
                first = state_filter.update(
                    [detection(class_id)], ROLE_IDS, 1.0)
                second = state_filter.update(
                    [detection(class_id)], ROLE_IDS, 1.1)
                third = state_filter.update(
                    [detection(class_id)], ROLE_IDS, 1.2)
                self.assertEqual(first.state, UNKNOWN)
                self.assertEqual(second.state, UNKNOWN)
                self.assertEqual(third.state, expected)

    def test_one_frame_red_does_not_replace_confirmed_green(self):
        state_filter = TrafficLightFilter()
        for now in (1.0, 1.1, 1.2):
            decision = state_filter.update([detection(3)], ROLE_IDS, now)
        self.assertEqual(decision.state, "G")

        false_red = state_filter.update([detection(1)], ROLE_IDS, 1.3)
        recovered = state_filter.update([detection(3)], ROLE_IDS, 1.4)
        self.assertEqual(false_red.state, "G")
        self.assertEqual(recovered.state, "G")

    def test_no_detection_becomes_unknown_after_timeout(self):
        state_filter = TrafficLightFilter()
        for now in (1.0, 1.1, 1.2):
            state_filter.update([detection(1)], ROLE_IDS, now)
        brief_loss = state_filter.update([], ROLE_IDS, 1.3)
        timed_out = state_filter.tick(1.71)
        self.assertEqual(brief_loss.state, "R")
        self.assertEqual(timed_out.state, UNKNOWN)

    def test_equal_red_green_conflict_is_unknown(self):
        state_filter = TrafficLightFilter()
        conflict = [detection(1, 0.9), detection(3, 0.9)]
        for now in (1.0, 1.1, 1.2):
            decision = state_filter.update(conflict, ROLE_IDS, now)
        self.assertEqual(decision.state, UNKNOWN)
        self.assertEqual(decision.reason, "detection_timeout")

    def test_conflict_keeps_recent_confirmed_state_when_scores_are_close(self):
        state_filter = TrafficLightFilter()
        for now in (1.0, 1.1, 1.2):
            state_filter.update([detection(3)], ROLE_IDS, now)
        conflict = state_filter.update(
            [detection(1, 0.91), detection(3, 0.90)], ROLE_IDS, 1.3)
        self.assertEqual(conflict.state, "G")
        self.assertEqual(conflict.candidate, "G")
        self.assertEqual(conflict.reason, "conflict_keep_confirmed")

    def test_clear_confidence_winner_still_requires_confirmation(self):
        state_filter = TrafficLightFilter()
        conflict = [detection(1, 0.95), detection(3, 0.55)]
        for now in (1.0, 1.1):
            decision = state_filter.update(conflict, ROLE_IDS, now)
            self.assertEqual(decision.state, UNKNOWN)
        decision = state_filter.update(conflict, ROLE_IDS, 1.2)
        self.assertEqual(decision.state, "R")

    def test_nan_inf_and_low_confidence_are_ignored_without_crash(self):
        instances = [
            detection(1, float("nan")),
            detection(1, float("inf")),
            detection(3, 0.49),
        ]
        evidences = collect_traffic_light_evidence(instances, ROLE_IDS)
        state_filter = TrafficLightFilter()
        decision = state_filter.update(instances, ROLE_IDS, 1.0)
        self.assertEqual(evidences, ())
        self.assertEqual(decision.state, UNKNOWN)

    def test_mask_area_breaks_close_confidence_conflict(self):
        config = TrafficLightConfig(
            traffic_light_confirmation_frames=1,
            traffic_light_conflict_score_margin=0.05,
            traffic_light_mask_area_weight=0.3)
        state_filter = TrafficLightFilter(config)
        decision = state_filter.update(
            [detection(1, 0.80, area=25), detection(3, 0.82, area=2)],
            ROLE_IDS, 1.0)
        self.assertEqual(decision.state, "R")

    def test_output_contract_and_invalid_parameters(self):
        self.assertEqual(PUBLISHED_STATES, frozenset(("R", "G", UNKNOWN)))
        with self.assertRaises(ValueError):
            TrafficLightConfig(traffic_light_confirmation_frames=0).validate()
        with self.assertRaises(ValueError):
            TrafficLightConfig(traffic_light_lost_timeout_sec=0.0).validate()
        with self.assertRaises(ValueError):
            TrafficLightConfig(traffic_light_minimum_confidence=1.1).validate()


if __name__ == "__main__":
    unittest.main()
