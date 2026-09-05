import math
import unittest

import numpy as np

from camera_navigation.metric_path_quality import cumulative_s
from camera_navigation.reference_path_stitching import (
    PlanarTransform, ReferencePathAdapterCore, ReferencePathStitcher,
    StitchConfig, transform_points)


def straight_base_path():
    return np.column_stack((np.linspace(0.0, 10.0, 21), np.zeros(21)))


class ReferencePathStitchingTest(unittest.TestCase):

    def test_synthetic_straight_poses_extend_zero_to_thirteen(self):
        core = ReferencePathAdapterCore()
        for x_m in (0.0, 1.0, 2.0, 3.0):
            result = core.process(
                straight_base_path(), PlanarTransform(x_m, 0.0, 0.0))
            self.assertTrue(result.accepted, result.reason)
        points = core.stitcher.points
        self.assertAlmostEqual(points[0, 0], 0.0)
        self.assertAlmostEqual(points[-1, 0], 13.0)
        self.assertEqual(len(points), 27)
        self.assertTrue(np.all(np.diff(cumulative_s(points)) > 0.0))

    def test_synthetic_curve_yaws_stitch_naturally(self):
        radius = 20.0
        core = ReferencePathAdapterCore()
        for yaw_deg in (0.0, 5.0, 10.0, 15.0):
            theta = math.radians(yaw_deg)
            vehicle = np.array([
                radius*math.sin(theta), radius*(1.0-math.cos(theta))])
            forward_s = np.linspace(0.0, 10.0, 41)
            angles = theta+forward_s/radius
            global_points = np.column_stack((
                radius*np.sin(angles), radius*(1.0-np.cos(angles))))
            delta = global_points-vehicle
            cosine, sine = math.cos(theta), math.sin(theta)
            inverse = np.array([[cosine, sine], [-sine, cosine]])
            base_points = delta @ inverse.T
            result = core.process(
                base_points, PlanarTransform(vehicle[0], vehicle[1], theta))
            self.assertTrue(result.accepted, result.reason)
        stitched = core.stitcher.points
        segments = np.linalg.norm(np.diff(stitched, axis=0), axis=1)
        self.assertGreater(cumulative_s(stitched)[-1], 14.0)
        self.assertLess(float(np.max(segments)), 0.8)

    def test_same_pose_repeated_does_not_grow(self):
        core = ReferencePathAdapterCore()
        transform = PlanarTransform(0.0, 0.0, 0.0)
        first = core.process(straight_base_path(), transform)
        self.assertTrue(first.accepted)
        initial_count = len(core.stitcher.points)
        for _ in range(10):
            result = core.process(straight_base_path(), transform)
            self.assertTrue(result.accepted)
            self.assertEqual(result.stitch.appended_points, 0)
        self.assertEqual(len(core.stitcher.points), initial_count)

    def test_three_meter_lateral_jump_is_rejected_without_pollution(self):
        core = ReferencePathAdapterCore()
        self.assertTrue(core.process(
            straight_base_path(), PlanarTransform(0.0, 0.0, 0.0)).accepted)
        before = core.stitcher.points.copy()
        result = core.process(
            straight_base_path(), PlanarTransform(1.0, 3.0, 0.0))
        self.assertFalse(result.accepted)
        self.assertIn(result.reason, ("insufficient_overlap", "position_error"))
        np.testing.assert_allclose(core.stitcher.points, before)

    def test_reverse_heading_is_rejected(self):
        core = ReferencePathAdapterCore()
        core.process(straight_base_path(), PlanarTransform(0.0, 0.0, 0.0))
        before = core.stitcher.points.copy()
        result = core.process(
            straight_base_path(), PlanarTransform(10.0, 0.0, math.pi))
        self.assertFalse(result.accepted)
        np.testing.assert_allclose(core.stitcher.points, before)

    def test_missing_tf_waits_without_reference_pollution(self):
        core = ReferencePathAdapterCore()
        result = core.process(straight_base_path(), None)
        self.assertFalse(result.accepted)
        self.assertEqual(result.state, "waiting_for_tf")
        self.assertEqual(result.reason, "tf_unavailable")
        self.assertEqual(len(core.stitcher.points), 0)
        recovered = core.process(
            straight_base_path(), PlanarTransform(0.0, 0.0, 0.0))
        self.assertTrue(recovered.accepted)

    def test_reset_removes_points_and_headings_before_new_segment(self):
        core = ReferencePathAdapterCore()
        result = core.process(
            straight_base_path(), PlanarTransform(0.0, 0.0, 0.0))
        self.assertTrue(result.accepted)
        self.assertGreater(len(core.stitcher.points), 0)
        self.assertGreater(len(core.stitcher.headings), 0)
        core.reset()
        self.assertEqual(len(core.stitcher.points), 0)
        self.assertEqual(len(core.stitcher.headings), 0)

    def test_pruning_limits_total_length_and_keeps_vehicle_history(self):
        stitcher = ReferencePathStitcher(StitchConfig(
            reference_path_keep_behind_m=5.0,
            reference_path_max_total_m=20.0,
            reference_path_target_forward_m=10.0))
        for x_m in range(31):
            points = transform_points(
                straight_base_path(), PlanarTransform(float(x_m), 0.0, 0.0))
            result = stitcher.update(points, np.array([float(x_m), 0.0]))
            self.assertTrue(result.accepted, result.reason)
        self.assertLessEqual(result.stitched_length_m, 20.5)
        self.assertLessEqual(stitcher.points[0, 0], 30.0)
        self.assertGreaterEqual(stitcher.points[0, 0], 24.5)
        self.assertGreaterEqual(result.forward_usable_length_m, 9.5)

    def test_transform_is_planar_and_metric(self):
        transformed = transform_points(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            PlanarTransform(1.0, 2.0, math.pi/2.0))
        np.testing.assert_allclose(
            transformed, np.array([[1.0, 2.0], [1.0, 4.0]]), atol=1e-9)

    def test_pose_orientations_rotate_with_exact_planar_transform(self):
        points = np.column_stack((np.linspace(0.0, 2.0, 5), np.zeros(5)))
        source_headings = np.linspace(-0.2, 0.2, 5)
        result = ReferencePathAdapterCore().process(
            points, PlanarTransform(1.0, 2.0, math.pi/2.0),
            source_headings)
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose(
            result.stitch.headings_rad,
            source_headings+math.pi/2.0, atol=1e-9)

    def test_nonfinite_pose_orientation_is_rejected_without_stitching(self):
        points = straight_base_path()
        headings = np.zeros(len(points))
        headings[4] = math.nan
        core = ReferencePathAdapterCore()
        result = core.process(
            points, PlanarTransform(0.0, 0.0, 0.0), headings)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "invalid_pose_orientation")
        self.assertEqual(len(core.stitcher.points), 0)


if __name__ == "__main__":
    unittest.main()
