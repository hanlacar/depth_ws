import math
import unittest

from camera_navigation.camera_pixel_controller_node import (
    PixelCommand, apply_stop_line_limit, apply_uphill_stop_limit)
from camera_navigation.stop_line_control import StopLineDecision, StopLinePhase
from camera_navigation.uphill_stop_control import (
    UphillStopConfig, UphillStopController, UphillStopState)


class UphillStopControlTest(unittest.TestCase):

    def setUp(self):
        self.policy = UphillStopController(UphillStopConfig(5.0))

    def enter_uphill(self, now=1.0):
        self.assertFalse(self.policy.update(False, True, now-0.1))
        self.assertTrue(self.policy.update(True, True, now))
        self.assertEqual(self.policy.state, UphillStopState.STOPPING)

    def test_a_flat_never_stops(self):
        for now in (0.0, 1.0, 10.0):
            self.assertFalse(self.policy.update(False, True, now))

    def test_b_new_false_to_true_edge_stops_immediately(self):
        self.enter_uphill()

    def test_c_stop_remains_active_before_five_seconds(self):
        self.enter_uphill(1.0)
        self.assertTrue(self.policy.update(True, True, 5.9))

    def test_d_five_seconds_completes_and_restores_drive(self):
        self.enter_uphill(1.0)
        self.assertFalse(self.policy.update(True, True, 6.0))
        self.assertEqual(self.policy.state, UphillStopState.PASSED)

    def test_e_same_uphill_does_not_stop_again(self):
        self.enter_uphill(1.0)
        self.policy.update(True, True, 6.0)
        for now in (6.1, 12.0, 30.0):
            self.assertFalse(self.policy.update(True, True, now))

    def test_f_flat_rearms_after_passed(self):
        self.enter_uphill(1.0)
        self.policy.update(True, True, 6.0)
        self.assertFalse(self.policy.update(False, True, 6.1))
        self.assertEqual(self.policy.state, UphillStopState.ARMED)

    def test_g_second_uphill_stops_once(self):
        self.enter_uphill(1.0)
        self.policy.update(True, True, 6.0)
        self.policy.update(False, True, 6.1)
        self.assertTrue(self.policy.update(True, True, 7.0))
        self.assertTrue(self.policy.update(True, True, 11.9))
        self.assertFalse(self.policy.update(True, True, 12.0))

    def test_h_false_during_stopping_does_not_cancel_stop(self):
        self.enter_uphill(1.0)
        self.assertTrue(self.policy.update(False, True, 3.0))
        self.assertTrue(self.policy.update(False, True, 5.9))
        self.assertFalse(self.policy.update(False, True, 6.0))
        self.assertEqual(self.policy.state, UphillStopState.ARMED)

    def test_i_invalid_imu_cannot_create_entry_edge(self):
        self.policy.update(False, True, 0.0)
        self.assertFalse(self.policy.update(True, False, 1.0))
        self.assertFalse(self.policy.update(True, True, 2.0))
        self.assertEqual(self.policy.state, UphillStopState.PASSED)

        self.policy = UphillStopController(UphillStopConfig(5.0))
        self.enter_uphill(1.0)
        self.assertTrue(self.policy.update(False, False, math.nan))
        self.assertTrue(self.policy.update(False, False, 5.9))
        self.assertFalse(self.policy.update(False, False, 6.0))
        self.assertEqual(self.policy.state, UphillStopState.PASSED)
        self.policy.update(False, True, 6.1)
        self.assertEqual(self.policy.state, UphillStopState.ARMED)

    def test_j_stop_line_and_uphill_stop_preserve_stop_contract(self):
        base = PixelCommand(2.0, -8, -8.0, "ok", True)
        stop_line = StopLineDecision(
            StopLinePhase.STOP, 0.5, True, "stop_line_latched")
        output = apply_stop_line_limit(base, stop_line)
        output = apply_uphill_stop_limit(output, True)
        self.assertEqual(output.drive, 0.0)
        self.assertEqual(output.wheel, -8)
        self.assertEqual(output.reason, "stop_line_stop")
        self.assertTrue(stop_line.stop_required)

        uphill_only = apply_uphill_stop_limit(base, True)
        self.assertEqual(uphill_only.drive, 0.0)
        self.assertEqual(uphill_only.wheel, -8)
        reverse = apply_uphill_stop_limit(
            PixelCommand(-1.0, 3, 3.0, "reverse", True), True)
        self.assertEqual(reverse.drive, 0.0)
        self.assertEqual(reverse.wheel, 3)
        self.assertFalse(StopLineDecision(
            StopLinePhase.NORMAL, None, False, "clear").stop_required)

    def test_path_fail_safe_remains_highest_priority(self):
        fail_safe = PixelCommand(0.0, 0, 0.0, "path_invalid", False)
        self.assertEqual(apply_uphill_stop_limit(fail_safe, True), fail_safe)

    def test_invalid_duration_is_rejected(self):
        for duration in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                UphillStopController(UphillStopConfig(duration))


if __name__ == "__main__":
    unittest.main()
