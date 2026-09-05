import math
import unittest

import numpy as np

from imu_manager.imu_filter import (
    SlopeStateConfig,
    SlopeStateDetector,
    vehicle_pitch_from_accel,
)


class SlopeStateDetectorTest(unittest.TestCase):

    def setUp(self):
        self.frames = 3
        self.detector = SlopeStateDetector(SlopeStateConfig(
            uphill_enter_pitch_deg=3.0,
            uphill_exit_pitch_deg=1.5,
            uphill_confirmation_frames=self.frames,
        ))

    def feed(self, pitch, count):
        return [self.detector.update(pitch) for _ in range(count)]

    def test_level_noise_and_below_threshold_remain_false(self):
        for pitch in (0.0, 0.2, -0.3, 2.99, 0.1):
            self.assertFalse(self.detector.update(pitch))

    def test_uphill_requires_confirmation_then_stays_true(self):
        self.assertEqual(self.feed(3.1, self.frames - 1), [False, False])
        self.assertTrue(self.detector.update(3.1))
        self.assertTrue(self.detector.update(4.0))

    def test_hysteresis_prevents_chatter_and_exit_is_confirmed(self):
        self.feed(3.2, self.frames)
        for pitch in (2.9, 1.6, 2.0, 1.4, 1.6, 1.4, 1.6):
            self.assertTrue(self.detector.update(pitch))
        self.assertEqual(self.feed(1.0, self.frames - 1), [True, True])
        self.assertFalse(self.detector.update(1.0))

    def test_downhill_never_enters_uphill_state(self):
        for pitch in (-2.0, -4.0, -30.0):
            self.assertFalse(self.detector.update(pitch))
        self.feed(4.0, self.frames)
        self.assertTrue(self.detector.state)
        self.assertFalse(self.feed(-20.0, self.frames)[-1])

    def test_invalid_input_fails_closed_and_resets_confirmation(self):
        self.detector.update(4.0)
        self.assertFalse(self.detector.update(math.nan))
        self.assertFalse(self.detector.update(4.0))
        self.assertFalse(self.detector.update(math.inf))
        self.feed(4.0, self.frames)
        self.assertTrue(self.detector.state)
        self.assertFalse(self.detector.update(4.0, imu_valid=False))

    def test_pitch_sign_is_positive_for_nose_up(self):
        angle = math.radians(10.0)
        gravity_nose_up = np.array([
            -9.80665 * math.sin(angle), 0.0, 9.80665 * math.cos(angle)])
        gravity_nose_down = np.array([
            9.80665 * math.sin(angle), 0.0, 9.80665 * math.cos(angle)])
        self.assertAlmostEqual(vehicle_pitch_from_accel(gravity_nose_up), 10.0)
        self.assertAlmostEqual(vehicle_pitch_from_accel(gravity_nose_down), -10.0)

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            SlopeStateConfig(1.0, 1.0, 3),
            SlopeStateConfig(1.0, -0.1, 3),
            SlopeStateConfig(math.nan, 1.0, 3),
            SlopeStateConfig(3.0, 1.0, 0),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                SlopeStateDetector(config)


if __name__ == "__main__":
    unittest.main()
