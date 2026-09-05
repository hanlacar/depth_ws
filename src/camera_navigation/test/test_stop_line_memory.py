"""Tests for stop_line_memory.py (req 2): pixel/depth acquisition, camera
distance, front-axle geometry, occlusion tracking via odom, and front-axle
crossing detection -- independent of image_path_planner's ego exclusion.
"""
import unittest

import numpy as np

from camera_navigation.stop_line_memory import (
    StopLineMemory,
    StopLineMemoryConfig,
    StopLineObservation,
    StopLineState,
    estimate_distance_from_row,
    stop_line_pixel_row,
)


def config(**overrides):
    values = dict(camera_to_bumper_m=0.5, front_axle_offset_m=0.8,
                  near_bumper_row_ratio=0.9, slowdown_distance_m=2.0,
                  stop_distance_m=0.7, depth_min_confidence=0.5,
                  memory_timeout_sec=1.0, passed_confirm_sec=0.0)
    values.update(overrides)
    return StopLineMemoryConfig(**values)


def seen(distance_m, row=300.0, height=480.0, confidence=1.0):
    return StopLineObservation(True, row, height, distance_m, confidence)


def not_seen():
    return StopLineObservation(False)


class StopLinePixelRowTest(unittest.TestCase):
    def test_empty_mask_returns_none(self):
        self.assertIsNone(stop_line_pixel_row(np.zeros((10, 10), np.uint8)))

    def test_returns_lowest_detected_row(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[3, 4] = 1
        mask[7, 5] = 1
        self.assertEqual(stop_line_pixel_row(mask), 7.0)


class RowToDistanceCalibrationTest(unittest.TestCase):
    def test_interpolates_between_calibration_points(self):
        table = [(0.5, 10.0), (1.0, 1.0)]
        value = estimate_distance_from_row(360.0, 480.0, table)
        # row_ratio = 0.75, halfway -> halfway between 10.0 and 1.0
        self.assertAlmostEqual(value, 5.5, places=3)

    def test_returns_none_without_enough_calibration_points(self):
        self.assertIsNone(estimate_distance_from_row(100.0, 480.0, [(0.5, 1.0)]))


class StopLineMemoryStateMachineTest(unittest.TestCase):
    def test_not_seen_before_any_detection(self):
        memory = StopLineMemory(config())
        status = memory.update(not_seen(), 0.0, 0.0)
        self.assertEqual(status.state, StopLineState.NOT_SEEN)
        self.assertFalse(status.crossed_front_axle)

    def test_far_detection_is_detected_far(self):
        memory = StopLineMemory(config())
        status = memory.update(seen(5.0), 0.0, 0.0)
        self.assertEqual(status.state, StopLineState.DETECTED_FAR)
        self.assertAlmostEqual(status.camera_distance_m, 5.0)
        self.assertAlmostEqual(status.front_bumper_distance_m, 4.5)
        self.assertAlmostEqual(status.front_axle_distance_m, 4.2)

    def test_within_slowdown_range_is_approaching(self):
        memory = StopLineMemory(config())
        # front_bumper_distance = camera_distance - 0.5 <= 2.0
        status = memory.update(seen(2.3), 0.0, 0.0)
        self.assertEqual(status.state, StopLineState.APPROACHING)

    def test_low_row_marks_near_bumper_occluded(self):
        memory = StopLineMemory(config())
        # row/height = 0.95 >= near_bumper_row_ratio(0.9), still visible but
        # close enough that occlusion is imminent.
        status = memory.update(seen(1.0, row=456.0, height=480.0), 0.0, 0.0)
        self.assertEqual(status.state, StopLineState.NEAR_BUMPER_OCCLUDED)

    def test_low_confidence_detection_is_ignored_as_a_fix(self):
        memory = StopLineMemory(config())
        memory.update(seen(5.0), 0.0, 0.0)
        status = memory.update(seen(0.1, confidence=0.1), 1.0, 1.0)
        # The low-confidence sample must not have overwritten the tracked
        # distance with 0.1 m.
        self.assertGreater(status.camera_distance_m, 1.0)

    def test_occlusion_is_tracked_via_last_distance_plus_odom(self):
        memory = StopLineMemory(config())
        # Last confident sighting is low in frame (about to go under the
        # bonnet), then the visual signal is lost.
        memory.update(seen(1.5, row=456.0, height=480.0), 0.0, 0.0)
        status = memory.update(not_seen(), 0.0, 0.1)
        self.assertEqual(status.state, StopLineState.NEAR_BUMPER_OCCLUDED)
        # Odom advances the vehicle 0.6 m forward while occluded; camera
        # distance to the (now-behind) line should shrink by the same
        # amount purely from the odom update.
        status = memory.update(not_seen(), 0.6, 0.2)
        self.assertLess(status.camera_distance_m, 1.5)

    def test_front_axle_crossing_is_detected_via_odom_after_occlusion(self):
        memory = StopLineMemory(config(front_axle_offset_m=0.8,
                                       camera_to_bumper_m=0.5))
        memory.update(seen(0.9), 0.0, 0.0)  # front_axle_distance = 0.1 m
        self.assertFalse(memory.status().crossed_front_axle)
        status = memory.update(not_seen(), 0.2, 0.1)  # +0.2 m forward travel
        self.assertTrue(status.crossed_front_axle)
        self.assertIn(status.state,
                     (StopLineState.CROSSED_FRONT_AXLE, StopLineState.PASSED))

    def test_passed_is_latched_after_confirmation_window(self):
        memory = StopLineMemory(config(passed_confirm_sec=0.2))
        memory.update(seen(0.9), 0.0, 0.0)
        status = memory.update(not_seen(), 0.2, 0.05)
        self.assertEqual(status.state, StopLineState.CROSSED_FRONT_AXLE)
        status = memory.update(not_seen(), 0.2, 0.30)
        self.assertEqual(status.state, StopLineState.PASSED)

    def test_memory_expires_after_timeout_without_updates(self):
        memory = StopLineMemory(config(memory_timeout_sec=0.5))
        memory.update(seen(5.0), 0.0, 0.0)
        status = memory.update(not_seen(), 0.0, 10.0)
        self.assertEqual(status.state, StopLineState.NOT_SEEN)

    def test_maximum_drive_reflects_stop_and_slow_bands(self):
        memory = StopLineMemory(config(stop_distance_m=0.7, slowdown_distance_m=2.0))
        memory.update(seen(0.9), 0.0, 0.0)  # bumper distance 0.4 <= stop
        self.assertEqual(memory.maximum_drive, 0.0)
        memory2 = StopLineMemory(config(stop_distance_m=0.7, slowdown_distance_m=2.0))
        memory2.update(seen(2.0), 0.0, 0.0)  # bumper distance 1.5, slow band
        self.assertEqual(memory2.maximum_drive, 1.0)

    def test_reset_clears_state(self):
        memory = StopLineMemory(config())
        memory.update(seen(0.9), 0.0, 0.0)
        memory.reset()
        self.assertEqual(memory.state, StopLineState.NOT_SEEN)
        self.assertIsNone(memory.camera_distance_m)


if __name__ == "__main__":
    unittest.main()
