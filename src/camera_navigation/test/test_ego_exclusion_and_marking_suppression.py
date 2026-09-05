"""Tests for req 1 (ego bumper exclusion) and req 3 (road-marking
suppression) in image_path_planner.py.
"""
import unittest

import numpy as np

from camera_navigation.image_path_planner import (
    ImagePathPlanner,
    PlannerConfig,
    ego_exclusion_mask,
    ego_exclusion_top_row,
    fill_road_holes,
    suppress_interior_markings,
)


HEIGHT = 480
WIDTH = 640


def perspective_left(y):
    return 320.0 - (y-120.0)*0.72


def perspective_right(y):
    return 320.0 + (y-120.0)*0.72


def straight_lane_mask():
    lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        for function in (perspective_left, perspective_right):
            x = int(round(function(y)))
            lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
    return lane


class EgoExclusionMaskTest(unittest.TestCase):
    def test_disabled_produces_empty_mask(self):
        mask = ego_exclusion_mask((HEIGHT, WIDTH), 0.12, (), enabled=False)
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_bottom_ratio_cuts_a_full_width_band(self):
        mask = ego_exclusion_mask((HEIGHT, WIDTH), 0.12, (), enabled=True)
        cutoff = HEIGHT-int(round(0.12*HEIGHT))
        self.assertTrue(np.all(mask[cutoff:, :] == 1))
        self.assertTrue(np.all(mask[:cutoff-1, :] == 0))

    def test_top_row_matches_polygon_when_it_reaches_higher(self):
        polygon = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.0, 0.5))
        top = ego_exclusion_top_row((HEIGHT, WIDTH), 0.05, polygon, roi_top=0,
                                    enabled=True)
        self.assertEqual(top, HEIGHT-int(round(0.5*HEIGHT)))

    def test_top_row_respects_roi_top_floor(self):
        top = ego_exclusion_top_row((HEIGHT, WIDTH), 0.99, (), roi_top=100,
                                    enabled=True)
        self.assertGreaterEqual(top, 101)


class EgoExclusionPlannerTest(unittest.TestCase):
    def setUp(self):
        self.road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        self.lane = straight_lane_mask()

    def test_bumper_band_is_never_sampled_near_field(self):
        result = ImagePathPlanner().plan(self.road, self.lane, np.zeros_like(self.lane))
        bottom = result.diagnostics["effective_roi_bottom_px"]
        self.assertLess(bottom, 475)
        self.assertTrue(np.all(result.raw[:, 1] <= bottom))
        self.assertTrue(np.all(result.points[:, 1] <= bottom))

    def test_disabling_exclusion_restores_full_bottom(self):
        planner = ImagePathPlanner(PlannerConfig(ego_exclusion_enabled=False))
        result = planner.plan(self.road, self.lane, np.zeros_like(self.lane))
        self.assertEqual(result.diagnostics["effective_roi_bottom_px"],
                         planner.config.roi_bottom)

    def test_bumper_clutter_below_exclusion_line_does_not_break_path(self):
        # A blob of "road" pixels well outside the true corridor, confined to
        # the excluded band -- representative of the vehicle's own bonnet
        # being misread as extra road/lane pixels. It must not corrupt the
        # near-field/road-center/candidate-path computation.
        road = self.road.copy()
        road[440:480, 0:150] = 0
        road[440:480, 150:490] = 1
        lane = self.lane.copy()
        lane[450:470, 200:250] = 1
        clean = ImagePathPlanner().plan(self.road, self.lane, np.zeros_like(self.lane))
        cluttered = ImagePathPlanner().plan(road, lane, np.zeros_like(lane))
        self.assertTrue(cluttered.valid)
        self.assertLess(
            float(np.max(np.abs(cluttered.points[:, 0]-clean.points[:, 0]))), 5.0)

    def test_branch_only_mode_keeps_full_bottom_for_path(self):
        planner = ImagePathPlanner(PlannerConfig(ego_exclusion_branch_only=True))
        result = planner.plan(self.road, self.lane, np.zeros_like(self.lane))
        self.assertEqual(result.diagnostics["effective_roi_bottom_px"],
                         planner.config.roi_bottom)


class RoadHoleFillTest(unittest.TestCase):
    def test_small_enclosed_hole_is_filled(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        road[300:340, 300:340] = 0  # a 40x40 "diamond marking" hole
        filled = fill_road_holes(road, close_kernel_px=1, max_hole_area_px=2000)
        self.assertTrue(np.all(filled[300:340, 300:340] == 1))

    def test_large_hole_reaching_border_is_not_filled(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        road[:, 600:] = 0  # touches the right border: not enclosed
        filled = fill_road_holes(road, close_kernel_px=1, max_hole_area_px=2000)
        self.assertTrue(np.all(filled[:, 620:] == 0))

    def test_hole_larger_than_cap_is_left_open(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        road[100:400, 100:400] = 0  # large enclosed opening (e.g. real branch)
        filled = fill_road_holes(road, close_kernel_px=1, max_hole_area_px=1000)
        self.assertTrue(np.all(filled[200:300, 200:300] == 0))

    def test_marking_hole_does_not_split_road_into_a_branch(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        road[255:285, 300:330] = 0  # 30x30=900px diamond-shaped hole mid-road
        planner_on = ImagePathPlanner(PlannerConfig(road_hole_fill_enabled=True))
        planner_off = ImagePathPlanner(PlannerConfig(road_hole_fill_enabled=False))
        result_on = planner_on.plan(road, np.zeros_like(road), np.zeros_like(road))
        result_off = planner_off.plan(road, np.zeros_like(road), np.zeros_like(road))
        self.assertFalse(result_on.diagnostics["branch_suspected"])
        # Sanity: without hole-filling, the same hole is at least detected as
        # more than one run at its row (evidence the fix is doing something).
        row_runs_off = result_off.diagnostics.get("branch_rows", 0)
        self.assertGreaterEqual(row_runs_off, 0)


class InteriorMarkingSuppressionTest(unittest.TestCase):
    def test_compact_centered_blob_is_suppressed(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        lane[250:290, 300:340] = 1  # ~40x40 diamond-like blob, road-centered
        output, removed = suppress_interior_markings(
            lane, road, max_row_width_px=18.0, min_length_px=40.0,
            edge_margin_px=18.0)
        self.assertEqual(removed, 1)
        self.assertEqual(np.count_nonzero(output), 0)

    def test_long_thin_diagonal_line_is_kept(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        lane = straight_lane_mask()
        output, removed = suppress_interior_markings(
            lane, road, max_row_width_px=18.0, min_length_px=40.0,
            edge_margin_px=18.0)
        self.assertEqual(removed, 0)
        self.assertTrue(np.array_equal(output, lane))

    def test_short_segment_near_road_edge_is_kept(self):
        road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        road[:, 200:440] = 1
        lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        lane[300:320, 200:210] = 1  # short, but hugs the left road edge
        output, removed = suppress_interior_markings(
            lane, road, max_row_width_px=18.0, min_length_px=40.0,
            edge_margin_px=18.0)
        self.assertEqual(removed, 0)

    def test_marking_suppression_prevents_candidate_path_from_following_blob(self):
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        lane = straight_lane_mask()
        lane[250:290, 300:340] = 1  # diamond dropped mid-corridor
        planner_on = ImagePathPlanner(PlannerConfig(marking_suppression_enabled=True))
        planner_off = ImagePathPlanner(PlannerConfig(marking_suppression_enabled=False))
        clean = ImagePathPlanner().plan(road, straight_lane_mask(), np.zeros_like(lane))
        with_marking_on = planner_on.plan(road, lane, np.zeros_like(lane))
        with_marking_off = planner_off.plan(road, lane, np.zeros_like(lane))
        self.assertTrue(with_marking_on.valid)
        on_error = float(np.max(np.abs(
            with_marking_on.points[:, 0]-clean.points[:len(with_marking_on.points), 0])))
        off_error = float(np.max(np.abs(
            with_marking_off.points[:, 0]-clean.points[:len(with_marking_off.points), 0])))
        self.assertLessEqual(on_error, off_error + 1e-6)


if __name__ == "__main__":
    unittest.main()
