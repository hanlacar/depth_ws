import unittest

from race_interfaces.msg import ImagePath, ImagePathPoint

from camera_navigation.camera_pixel_controller_node import (
    DriveCommand,
    PixelCommand,
    PixelController,
    PixelControllerConfig,
    SafetyState,
)


def valid_path(controller, stamp_ns=1_000_000_000, received_at=1.0):
    assert controller.ingest_path(
        stamp_ns, 640.0,
        ((300.0, 470.0), (310.0, 350.0), (320.0, 230.0)),
        True, 0.8, received_at)


class CameraPixelControllerSafetyTest(unittest.TestCase):

    @staticmethod
    def quality_command(source, points=None, confidence=0.8):
        controller = PixelController(PixelControllerConfig(
            derivative_gain_deg_per_norm_per_s=0.0))
        points = points or (
            (320.0, 470.0), (320.0, 350.0), (320.0, 230.0))
        sources = (source,) * len(points)
        assert controller.ingest_path(
            1_000_000_000, 640.0, points, True, confidence, 1.0,
            sources)
        return controller.step(1.01, 1_010_000_000)

    def test_both_boundary_straight_path_cruises(self):
        command = self.quality_command(ImagePathPoint.BOTH_BOUNDARIES)
        self.assertEqual(command.drive, DriveCommand.CRUISE.value)
        self.assertEqual(command.wheel, 0)

    def test_road_center_path_is_slow_even_when_straight(self):
        command = self.quality_command(ImagePathPoint.ROAD_CENTER)
        self.assertEqual(command.drive, DriveCommand.SLOW.value)
        self.assertEqual(command.wheel, 0)

    def test_curved_road_center_path_is_slow_and_saturated(self):
        command = self.quality_command(
            ImagePathPoint.ROAD_CENTER,
            ((700.0, 470.0), (700.0, 350.0), (700.0, 230.0)))
        self.assertEqual(command.drive, DriveCommand.SLOW.value)
        self.assertGreater(command.wheel, 0)
        self.assertLessEqual(abs(command.wheel), 22)

    def test_single_boundary_and_low_confidence_paths_are_slow(self):
        single = self.quality_command(ImagePathPoint.LEFT_BOUNDARY)
        low_confidence = self.quality_command(
            ImagePathPoint.BOTH_BOUNDARIES, confidence=0.5)
        self.assertEqual(single.drive, DriveCommand.SLOW.value)
        self.assertEqual(low_confidence.drive, DriveCommand.SLOW.value)

    def test_degraded_path_state_is_slow(self):
        controller = PixelController(PixelControllerConfig(
            derivative_gain_deg_per_norm_per_s=0.0))
        points = ((320.0, 470.0), (320.0, 350.0), (320.0, 230.0))
        self.assertTrue(controller.ingest_path(
            1_000_000_000, 640.0, points, True, 0.9, 1.0,
            (ImagePathPoint.BOTH_BOUNDARIES,) * len(points),
            ImagePath.STATE_DEGRADED))
        command = controller.step(1.01, 1_010_000_000)
        self.assertEqual(command.drive, DriveCommand.SLOW.value)

    def test_invalid_path_state_fails_closed(self):
        controller = PixelController()
        points = ((320.0, 470.0), (320.0, 350.0), (320.0, 230.0))
        self.assertTrue(controller.ingest_path(
            1_000_000_000, 640.0, points, True, 0.9, 1.0,
            (ImagePathPoint.BOTH_BOUNDARIES,) * len(points),
            ImagePath.STATE_INVALID))
        command = controller.step(1.01, 1_010_000_000)
        self.assertEqual(command.drive, DriveCommand.STOP.value)
        self.assertEqual(command.wheel, 0)

    def test_fast_and_reverse_configuration_is_rejected(self):
        for command in (DriveCommand.FAST.value, DriveCommand.REVERSE.value):
            with self.subTest(command=command), self.assertRaises(ValueError):
                PixelControllerConfig(normal_drive_command=command).validate()

    def test_source_length_mismatch_fails_closed(self):
        controller = PixelController()
        controller.ingest_path(
            1_000_000_000, 640.0,
            ((320.0, 470.0), (320.0, 350.0), (320.0, 230.0)),
            True, 0.8, 1.0, (ImagePathPoint.ROAD_CENTER,))
        command = controller.step(1.01, 1_010_000_000)
        self.assertEqual(command.drive, DriveCommand.STOP.value)
        self.assertEqual(command.wheel, 0)

    def test_unknown_source_fails_closed(self):
        controller = PixelController()
        controller.ingest_path(
            1_000_000_000, 640.0,
            ((320.0, 470.0), (320.0, 350.0), (320.0, 230.0)),
            True, 0.8, 1.0, (99, 99, 99))
        command = controller.step(1.01, 1_010_000_000)
        self.assertEqual(command.drive, DriveCommand.STOP.value)
        self.assertEqual(command.wheel, 0)

    def test_drive_policy_uses_final_integer_wheel_with_inclusive_boundary(self):
        controller = PixelController(PixelControllerConfig(
            steering_slowdown_threshold_deg=5.0))
        expected = {
            0: DriveCommand.CRUISE.value,
            4: DriveCommand.CRUISE.value,
            -4: DriveCommand.CRUISE.value,
            5: DriveCommand.SLOW.value,
            -5: DriveCommand.SLOW.value,
            22: DriveCommand.SLOW.value,
            -22: DriveCommand.SLOW.value,
        }

        for wheel, drive in expected.items():
            with self.subTest(wheel=wheel):
                self.assertEqual(controller.drive_for_wheel(wheel), drive)

    def test_positive_path_offset_publishes_right_positive_wheel(self):
        controller = PixelController(PixelControllerConfig(
            derivative_gain_deg_per_norm_per_s=0.0))
        self.assertTrue(controller.ingest_path(
            1_000_000_000, 640.0,
            ((400.0, 470.0), (400.0, 350.0), (400.0, 230.0)),
            True, 0.8, 1.0))
        right = controller.step(1.01, 1_010_000_000)

        controller = PixelController(PixelControllerConfig(
            derivative_gain_deg_per_norm_per_s=0.0))
        self.assertTrue(controller.ingest_path(
            1_000_000_000, 640.0,
            ((240.0, 470.0), (240.0, 350.0), (240.0, 230.0)),
            True, 0.8, 1.0))
        left = controller.step(1.01, 1_010_000_000)

        self.assertGreater(right.wheel, 0)
        self.assertLess(left.wheel, 0)

    def test_waiting_invalid_and_timeout_stop_drive(self):
        controller = PixelController()
        waiting = controller.step(1.0, 1_000_000_000)
        self.assertEqual(waiting.drive, DriveCommand.STOP.value)
        self.assertEqual(
            controller.safety_state(waiting), SafetyState.WAITING_FOR_PATH)

        valid_path(controller)
        active = controller.step(1.01, 1_010_000_000)
        self.assertTrue(active.valid)
        self.assertGreater(active.drive, 0.0)
        self.assertEqual(controller.safety_state(active), SafetyState.ACTIVE)

        self.assertTrue(controller.ingest_path(
            1_020_000_000, 640.0, (), False, 0.0, 1.02))
        invalid = controller.step(1.02, 1_020_000_000)
        self.assertEqual(invalid.drive, 0.0)
        self.assertEqual(invalid.wheel, 0)
        self.assertEqual(
            controller.safety_state(invalid), SafetyState.PATH_INVALID)

        controller = PixelController()
        valid_path(controller)
        timeout = controller.step(1.16, 1_160_000_000)
        self.assertEqual(timeout.drive, 0.0)
        self.assertEqual(timeout.reason, "path_timeout")
        self.assertEqual(
            controller.safety_state(timeout), SafetyState.PATH_TIMEOUT)

    def test_final_saturation_and_time_based_slew_limit(self):
        controller = PixelController(PixelControllerConfig(
            maximum_steering_deg=22.0,
            max_steering_rate_deg_per_sec=180.0))
        controller.finalize_output(PixelController.stop("path_invalid"), 1.0)
        requested = PixelCommand(2.0, 100, 100.0, "ok", True)

        first = controller.finalize_output(requested, 1.05)
        second = controller.finalize_output(requested, 1.10)
        saturated = controller.finalize_output(requested, 2.0)

        self.assertAlmostEqual(first.steering_deg, 9.0)
        self.assertEqual(first.wheel, 9)
        self.assertAlmostEqual(second.steering_deg, 18.0)
        self.assertEqual(second.wheel, 18)
        self.assertEqual(saturated.steering_deg, 22.0)
        self.assertEqual(saturated.wheel, 22)
        self.assertLessEqual(abs(saturated.wheel), 22)

        negative = controller.finalize_output(
            PixelCommand(2.0, -100, -100.0, "ok", True), 3.0)
        self.assertEqual(negative.steering_deg, -22.0)
        self.assertEqual(negative.wheel, -22)

    def test_invalid_to_valid_recovery_is_slew_limited_from_center(self):
        controller = PixelController(PixelControllerConfig(
            max_steering_rate_deg_per_sec=180.0))
        controller.finalize_output(
            PixelCommand(2.0, 20, 20.0, "ok", True), 1.0)
        stopped = controller.finalize_output(
            PixelController.stop("path_invalid"), 1.05)
        recovered = controller.finalize_output(
            PixelCommand(2.0, -22, -22.0, "ok", True), 1.10)

        self.assertEqual(stopped.drive, 0.0)
        self.assertEqual(stopped.wheel, 0)
        self.assertEqual(recovered.drive, 1.0)
        self.assertAlmostEqual(recovered.steering_deg, -9.0)
        self.assertEqual(recovered.wheel, -9)


if __name__ == "__main__":
    unittest.main()
