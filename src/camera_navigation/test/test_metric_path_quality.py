import math
import unittest

import numpy as np

from camera_navigation.ground_plane_calibration import (
    Intrinsics, project_pixel_path_to_metric, rotation_matrix_rpy)
from camera_navigation.metric_path_quality import (
    MetricPathQualityConfig, analyze_metric_path, cumulative_s,
    has_self_intersection)


class MetricPathQualityTest(unittest.TestCase):

    def test_straight_metric_path_and_strict_frenet_s(self):
        points = np.column_stack((np.linspace(0.32, 10.32, 21), np.zeros(21)))
        result = analyze_metric_path(points)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.path_length_m, 10.0)
        arc = cumulative_s(result.points)
        self.assertEqual(arc[0], 0.0)
        self.assertTrue(np.all(np.diff(arc) > 0.0))

    def test_duplicates_are_removed_without_growth(self):
        points = np.array([[0.0, 0.0], [0.01, 0.0], [0.5, 0.0],
                           [0.5, 0.0], [1.5, 0.0]])
        result = analyze_metric_path(points)
        self.assertTrue(result.valid)
        self.assertEqual(result.duplicates_removed, 2)
        self.assertEqual(result.point_count, 3)

    def test_nonfinite_reverse_and_short_paths_are_rejected(self):
        cases = (
            ([(0.0, 0.0), (math.nan, 0.0), (2.0, 0.0)], "nonfinite_point"),
            ([(0.0, 0.0), (1.0, 0.0), (0.5, 0.0)], "point_order_reversal"),
            ([(0.0, 0.0), (0.2, 0.0), (0.4, 0.0)], "path_too_short"),
        )
        for points, reason in cases:
            with self.subTest(reason=reason):
                result = analyze_metric_path(points)
                self.assertFalse(result.valid)
                self.assertEqual(result.reason, reason)

    def test_horizon_jump_is_trimmed_without_extrapolation(self):
        points = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                  (1.5, 0.0), (10.0, 0.0), (20.0, 0.0)]
        result = analyze_metric_path(points)
        self.assertTrue(result.valid)
        self.assertTrue(result.jump_truncated)
        self.assertEqual(result.reason, "ok_jump_truncated")
        self.assertAlmostEqual(result.maximum_x_m, 1.5)

    def test_self_intersection_helper(self):
        crossing = np.array([[0.0, 0.0], [2.0, 2.0],
                             [0.0, 2.0], [2.0, 0.0]])
        self.assertTrue(has_self_intersection(crossing))
        endpoint_loop = np.array([[0.0, 0.0], [1.0, 0.0],
                                  [1.0, 1.0], [0.0, 0.0]])
        self.assertTrue(has_self_intersection(endpoint_loop))

    def test_commissioned_extrinsic_produces_real_forward_points_over_10m(self):
        intrinsics = Intrinsics(fx=386.0, fy=386.0, cx=320.0, cy=240.0)
        pixels = [(320.0, float(y)) for y in range(475, 119, -10)]
        projected = project_pixel_path_to_metric(
            pixels, intrinsics, rotation_matrix_rpy(0.0, -5.0, 0.0),
            np.array([0.015, 0.0, 0.970]), max_range_m=30.0)
        result = analyze_metric_path(projected)
        self.assertTrue(result.valid)
        self.assertTrue(result.jump_truncated)
        self.assertGreaterEqual(result.maximum_x_m, 10.0)
        self.assertTrue(np.all(np.diff(result.points[:, 0]) > 0.0))
        self.assertAlmostEqual(result.minimum_x_m, 1.3338828218, places=6)
        self.assertAlmostEqual(result.maximum_x_m, 13.0437170150, places=6)


if __name__ == "__main__":
    unittest.main()
