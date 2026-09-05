import unittest

import numpy as np

from camera_navigation.camera_pixel_controller_node import (
    PixelCommand, PixelController, apply_stop_line_limit)
from camera_navigation.stop_line_control import (
    StopLineConfig, StopLinePhase, StopLinePolicy,
    estimate_stop_line_distance)


class StopLineControlTest(unittest.TestCase):

    @staticmethod
    def decision_for_camera_distance(distance, offset=0.0):
        policy = StopLinePolicy(StopLineConfig(
            camera_to_front_bumper_m=offset,
            stop_line_confirmation_frames=1))
        assert policy.ingest_camera_distance(distance, 1.0)
        return policy.decision(1.0)

    def test_required_distance_boundaries_limit_only_drive(self):
        base = PixelCommand(2.0, 4, 4.0, "ok", True)
        expected = {
            3.00: (2.0, False),
            2.01: (2.0, False),
            2.00: (1.0, False),
            1.50: (1.0, False),
            0.71: (1.0, False),
            0.70: (0.0, True),
            0.50: (0.0, True),
        }
        for distance, (drive, stop_required) in expected.items():
            with self.subTest(distance=distance):
                decision = self.decision_for_camera_distance(distance)
                output = apply_stop_line_limit(base, decision)
                self.assertEqual(output.drive, drive)
                self.assertEqual(decision.stop_required, stop_required)
                self.assertEqual(output.wheel, base.wheel)

    def test_camera_to_front_bumper_offset_is_subtracted(self):
        decision = self.decision_for_camera_distance(1.2, offset=0.5)
        output = apply_stop_line_limit(
            PixelCommand(2.0, -6, -6.0, "ok", True), decision)
        self.assertAlmostEqual(decision.front_bumper_distance_m, 0.7)
        self.assertTrue(decision.stop_required)
        self.assertEqual(output.drive, 0.0)
        self.assertEqual(output.wheel, -6)

    def test_confirmation_hysteresis_latch_and_explicit_release(self):
        policy = StopLinePolicy(StopLineConfig(
            camera_to_front_bumper_m=0.0,
            stop_line_confirmation_frames=3,
            stop_line_release_margin_m=0.2))
        policy.ingest_camera_distance(0.69, 1.0)
        self.assertEqual(policy.decision(1.0).phase, StopLinePhase.SLOW)
        policy.ingest_camera_distance(0.80, 1.1)  # inside 0.7 + 0.2 band
        self.assertEqual(policy.confirmation_count, 1)
        policy.ingest_camera_distance(0.69, 1.2)
        policy.ingest_camera_distance(0.69, 1.3)
        self.assertTrue(policy.decision(1.3).stop_required)

        policy.observe_unavailable("mask_missing")
        policy.ingest_camera_distance(3.0, 2.0)
        self.assertTrue(policy.decision(2.0).stop_required)
        policy.release_stop()
        self.assertEqual(policy.decision(2.0).phase, StopLinePhase.NORMAL)

    def test_release_band_resets_confirmation_only_above_margin(self):
        policy = StopLinePolicy(StopLineConfig(
            camera_to_front_bumper_m=0.0,
            stop_line_confirmation_frames=2))
        policy.ingest_camera_distance(0.69, 1.0)
        policy.ingest_camera_distance(0.91, 1.1)
        self.assertEqual(policy.confirmation_count, 0)
        self.assertFalse(policy.decision(1.1).stop_required)

    def test_robust_center_roi_depth_rejects_invalid_and_outliers(self):
        config = StopLineConfig(
            stop_line_center_roi_width_ratio=0.6,
            stop_line_min_valid_depth_pixels=20)
        mask = np.ones((10, 10), dtype=np.uint8)*255
        depth = np.full((10, 10), 1.5, dtype=np.float64)
        depth[0, 2] = 0.0
        depth[1, 3] = np.nan
        depth[2, 4] = np.inf
        depth[3, 5] = 10.0
        measurement = estimate_stop_line_distance(mask, depth, config)
        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.camera_distance_m, 1.5)
        self.assertGreaterEqual(measurement.valid_pixel_count, 20)

    def test_invalid_depth_and_mask_do_not_create_stop(self):
        config = StopLineConfig(stop_line_min_valid_depth_pixels=5)
        full_mask = np.ones((4, 4), dtype=np.uint8)*255
        cases = (
            (np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4)),
             "stop_line_mask_missing"),
            (full_mask, np.zeros((4, 4)), "insufficient_valid_depth"),
            (full_mask, np.full((4, 4), np.nan), "insufficient_valid_depth"),
            (full_mask, np.full((4, 4), np.inf), "insufficient_valid_depth"),
            (full_mask, np.ones((2, 2)), "shape_mismatch"),
        )
        for mask, depth, reason in cases:
            with self.subTest(reason=reason):
                measurement = estimate_stop_line_distance(mask, depth, config)
                self.assertFalse(measurement.valid)
                self.assertEqual(measurement.reason, reason)

        sparse = np.zeros((4, 4), dtype=np.uint8)
        sparse[1, 1] = 255
        measurement = estimate_stop_line_distance(sparse, np.ones((4, 4)), config)
        self.assertFalse(measurement.valid)
        self.assertEqual(measurement.reason, "insufficient_valid_depth")

    def test_invalid_path_fail_safe_has_priority_over_stop_logic(self):
        stop = self.decision_for_camera_distance(0.5)
        fail_safe = PixelController.stop("path_invalid")
        output = apply_stop_line_limit(fail_safe, stop)
        self.assertFalse(output.valid)
        self.assertEqual(output.drive, 0.0)
        self.assertEqual(output.wheel, 0)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            StopLineConfig(
                stop_line_stop_distance_m=2.0,
                stop_line_slowdown_distance_m=2.0).validate()
        with self.assertRaises(ValueError):
            StopLineConfig(camera_to_front_bumper_m=-0.1).validate()
        with self.assertRaises(ValueError):
            StopLineConfig(stop_line_confirmation_frames=0).validate()


if __name__ == "__main__":
    unittest.main()
