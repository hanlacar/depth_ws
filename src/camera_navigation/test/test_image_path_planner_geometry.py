import unittest

import numpy as np

from camera_navigation.camera_image_path_node import path_computation_enabled
from camera_navigation.image_path_planner import (
    BOTH_BOUNDARIES,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY,
    ROAD_CENTER,
    ImagePathPlanner,
    PlannerConfig,
)


HEIGHT = 480
WIDTH = 640


def road_corridor(left=0, right=WIDTH):
    road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    road[:, max(0, left):min(WIDTH, right)] = 1
    return road


def lane_mask(left_fn=None, right_fn=None, spike=None):
    lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        for side, function in (("left", left_fn), ("right", right_fn)):
            if function is None:
                continue
            x = int(round(function(y)))
            if spike is not None and spike[0] == side and abs(y-spike[1]) <= 2:
                x += spike[2]
            lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
    return lane


def perspective_left(y):
    return 320.0 - (y-120.0)*0.72


def perspective_right(y):
    return 320.0 + (y-120.0)*0.72


class ImagePathPlannerGeometryTest(unittest.TestCase):
    def setUp(self):
        self.road = road_corridor()
        self.bilateral = lane_mask(perspective_left, perspective_right)

    def warm(self, planner):
        result = planner.plan(self.road, self.bilateral, np.zeros_like(self.bilateral))
        self.assertTrue(result.valid)
        return result

    def test_bilateral_straight_is_valid(self):
        result = self.warm(ImagePathPlanner())
        self.assertGreaterEqual(len(result.points), 20)
        self.assertIn(BOTH_BOUNDARIES, result.sources)
        self.assertGreater(result.sources.count(BOTH_BOUNDARIES), len(result.sources)//2)

    def test_pixel_pipeline_can_compute_without_external_control_mode(self):
        self.assertFalse(path_computation_enabled(0, False, True))
        self.assertTrue(path_computation_enabled(0, False, False))

    def test_boot_time_clipped_road_rejects_boundary_width_and_uses_road_center(self):
        planner = ImagePathPlanner()
        right_only = lane_mask(None, perspective_right)
        result = planner.plan(self.road, right_only, np.zeros_like(right_only))

        self.assertTrue(result.valid)
        self.assertEqual(set(result.sources), {ROAD_CENTER})
        self.assertEqual(len(result.virtual_details), 0)
        self.assertGreater(result.diagnostics["rejections"][
            "SINGLE_ROAD_CONFLICT"], 0)

    def test_learned_width_takes_precedence_over_road_corridor_fallback(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        result = planner.plan(
            self.road, lane_mask(None, perspective_right),
            np.zeros_like(self.bilateral))

        self.assertTrue(result.valid)
        self.assertTrue(all(item["lane_width_source"] == "width_profile"
                            for item in result.virtual_details))

    def test_sparse_pair_does_not_poison_width_while_road_center_remains_valid(self):
        planner = ImagePathPlanner()
        sparse = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for y in (400, 390, 380):
            for function in (perspective_left, perspective_right):
                x = int(round(function(y)))
                sparse[y-2:y+3, x-2:x+3] = 1

        result = planner.plan(self.road, sparse, np.zeros_like(sparse))

        self.assertTrue(result.valid)
        self.assertIn(ROAD_CENTER, result.sources)
        self.assertEqual(planner.width_profile, {})
        self.assertIsNotNone(planner.previous)

    def test_timestamp_gap_resets_temporal_path_but_keeps_width_profile(self):
        planner = ImagePathPlanner(PlannerConfig(
            temporal_state_timeout_sec=0.5))
        first = planner.plan(
            self.road, self.bilateral, np.zeros_like(self.bilateral),
            timestamp_sec=1.0)
        learned_rows = dict(planner.width_profile)
        shifted = lane_mask(
            lambda y: perspective_left(y)+80.0,
            lambda y: perspective_right(y)+80.0)

        after_gap = planner.plan(
            self.road, shifted, np.zeros_like(shifted), timestamp_sec=2.0)
        independent = ImagePathPlanner().plan(
            self.road, shifted, np.zeros_like(shifted))

        self.assertTrue(first.valid)
        self.assertTrue(after_gap.valid)
        self.assertTrue(np.allclose(after_gap.points, independent.points))
        self.assertTrue(learned_rows)
        self.assertTrue(planner.width_profile)

    def test_vehicle_body_clearance_rejects_impossible_dimensions(self):
        planner = ImagePathPlanner(PlannerConfig(
            nominal_lane_width_m=3.0,
            minimum_boundary_clearance_m=1.5,
            vehicle_width_m=3.0,
            vehicle_boundary_margin_m=0.1))
        result = planner.plan(
            self.road, self.bilateral, np.zeros_like(self.bilateral))

        self.assertFalse(result.valid)
        self.assertGreater(
            result.diagnostics["rejections"]["VEHICLE_CLEARANCE"], 0)

    def test_sparse_single_boundary_requires_road_containment(self):
        planner = ImagePathPlanner(PlannerConfig(
            valid_min_single_boundary_points=5))
        self.warm(planner)
        right_only = lane_mask(None, perspective_right)
        restricted_road = road_corridor(430, 640)

        result = planner.plan(
            restricted_road, right_only, np.zeros_like(right_only))

        self.assertFalse(result.valid)
        self.assertFalse(result.diagnostics["ego_road_component_present"])
        self.assertFalse(result.diagnostics["near_field_ok"])
        # No final path exists, so this is an unevaluated/no-output frame,
        # not an off-road path. The hard safety layer reports it separately.
        self.assertTrue(result.diagnostics["vehicle_containment_ok"])
        self.assertTrue(result.diagnostics["final_road_safety_evaluated"])
        self.assertTrue(result.diagnostics["final_road_unrecoverable"])

    def test_right_only_straight_fallback_is_valid(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        right_only = lane_mask(None, perspective_right)
        for _ in range(planner.config.source_release_frames):
            result = planner.plan(
                self.road, right_only, np.zeros_like(right_only))
        self.assertTrue(result.valid)
        self.assertIn(RIGHT_BOUNDARY, result.sources)

    def test_left_only_straight_fallback_is_valid(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        left_only = lane_mask(perspective_left, None)
        for _ in range(planner.config.source_release_frames):
            result = planner.plan(
                self.road, left_only, np.zeros_like(left_only))
        self.assertTrue(result.valid)
        self.assertIn(LEFT_BOUNDARY, result.sources)

    def test_bilateral_to_one_sided_transition_is_continuous(self):
        planner = ImagePathPlanner()
        before = self.warm(planner)
        right_only = lane_mask(None, perspective_right)
        after = planner.plan(self.road, right_only, np.zeros_like(right_only))
        self.assertTrue(after.valid)
        before_by_y = {int(y): x for x, y in before.points}
        deltas = [abs(x-before_by_y[int(y)]) for x, y in after.points
                  if int(y) in before_by_y]
        self.assertGreater(len(deltas), 10)
        self.assertLess(float(np.max(deltas)), 2.0)

    def test_small_smooth_curvature_is_valid(self):
        curve = lambda y: 0.0005*(y-300.0)**2
        lane = lane_mask(
            lambda y: perspective_left(y)+curve(y),
            lambda y: perspective_right(y)+curve(y))
        result = ImagePathPlanner().plan(self.road, lane, np.zeros_like(lane))
        self.assertTrue(result.valid)
        self.assertGreater(len(result.points), 10)

    def test_perspective_boundaries_on_same_side_of_image_center_are_tracked(self):
        shift = 60.0
        shifted = lane_mask(
            lambda y: perspective_left(y)+shift,
            lambda y: perspective_right(y)+shift)
        result = ImagePathPlanner().plan(
            self.road, shifted, np.zeros_like(shifted))
        self.assertTrue(result.valid)
        self.assertTrue(result.diagnostics["continuity_ok"])
        self.assertGreater(len(result.points), 10)

    def test_single_80_px_center_spike_is_rejected(self):
        # Isolate the raw-stage local-spike REPAIR mechanism (req 5):
        # marking_suppression_enabled and boundary_track_filter_enabled
        # target this exact "isolated short spike" shape at earlier
        # pipeline stages (req 3.C / boundary-as-track), and
        # path_corridor_enabled would also filter it via the dynamic PATH
        # ROI -- all correct in the full pipeline (see the dedicated tests
        # for each), but disabled here to test _repair_local_spikes alone.
        planner = ImagePathPlanner(PlannerConfig(
            marking_suppression_enabled=False,
            boundary_track_filter_enabled=False,
            path_corridor_enabled=False))
        lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for y in range(120, 476):
            offset = 80 if abs(y-325) <= 2 else 0
            for function in (perspective_left, perspective_right):
                x = int(round(function(y))) + offset
                lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
        result = planner.plan(self.road, lane, np.zeros_like(lane))
        # Repaired (interpolated), not deleted -- horizon length preserved.
        self.assertGreater(result.diagnostics["rejections"]["DIRECTION_OUTLIER"], 0)
        self.assertGreaterEqual(len(result.raw), 30)

    def test_alternating_boundary_spikes_repair_to_safe_road_path(self):
        # Isolate the older sample-level LATERAL_JUMP continuity gate: the
        # newer marking/track/corridor filters would all (correctly) catch
        # this exact isolated-dot-every-10-rows shape earlier.
        config = PlannerConfig(maximum_lateral_jump_px=40.0,
                               marking_suppression_enabled=False,
                               boundary_track_filter_enabled=False,
                               path_corridor_enabled=False)
        planner = ImagePathPlanner(config)
        lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for index, y in enumerate(range(475, 119, -10)):
            offset = 100 if index % 2 == 0 else -100
            lx = int(np.clip(perspective_left(y)+offset, 2, 317))
            rx = int(np.clip(perspective_right(y)+offset, 323, 637))
            lane[max(0, y-2):min(HEIGHT, y+3), lx-2:lx+3] = 1
            lane[max(0, y-2):min(HEIGHT, y+3), rx-2:rx+3] = 1
        result = planner.plan(self.road, lane, np.zeros_like(lane))
        self.assertTrue(result.valid)
        self.assertGreater(result.diagnostics["rejections"]["LATERAL_JUMP"], 0)
        self.assertGreater(result.diagnostics["rejections"]["GAP_REPAIRED"], 0)
        self.assertLess(abs(result.diagnostics["required_steering_deg"]), 2.0)

    def test_lane_width_jump_is_rejected(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        bad = lane_mask(perspective_left, lambda y: perspective_right(y)+220.0)
        result = planner.plan(self.road, bad, np.zeros_like(bad))
        # The dynamic PATH ROI (centered on the just-tracked path) now
        # screens out most of the 220px-jumped right boundary before it
        # ever reaches the pair-width sanity check, via CORRIDOR_REJECT
        # instead of INVALID_LANE_WIDTH -- either way it must not form a
        # trusted BOTH_BOUNDARIES pair.
        self.assertGreater(result.diagnostics["rejections"]["CORRIDOR_REJECT"] +
                           result.diagnostics["rejections"]["INVALID_LANE_WIDTH"], 0)
        self.assertNotIn(BOTH_BOUNDARIES, result.sources)

    def test_boundary_order_reversal_is_rejected(self):
        planner = ImagePathPlanner()
        ys = [300]
        left = np.asarray([[400.0, 300.0]])
        right = np.asarray([[200.0, 300.0]])
        points, sources, _, counters, _ = planner._raw_path(
            ys, left, right, self.road, True)
        self.assertEqual(len(points), 1)
        self.assertEqual(sources, [ROAD_CENTER])
        self.assertEqual(counters["BOUNDARY_ORDER"], 1)

    def test_gross_road_corridor_outlier_is_rejected(self):
        planner = ImagePathPlanner()
        ys = [300]
        left = np.asarray([[480.0, 300.0]])
        right = np.asarray([[580.0, 300.0]])
        points, _, _, counters, _ = planner._raw_path(
            ys, left, right, road_corridor(200, 440), True)
        # The bad lane pair is rejected, then the missing row is recovered
        # from the current ego-connected road centre as a soft fallback.
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0, 0], 319.5)
        self.assertEqual(counters["ROAD_GROSS_OUTLIER"], 1)
        self.assertEqual(counters["GAP_REPAIRED"], 1)

    def test_learned_width_keeps_sustained_single_boundary_valid(self):
        planner = ImagePathPlanner(PlannerConfig(max_single_boundary_fallback_frames=5))
        self.warm(planner)
        right_only = lane_mask(None, perspective_right)
        for _ in range(8):
            self.assertTrue(planner.plan(
                self.road, right_only, np.zeros_like(right_only)).valid)

    def test_complete_current_road_loss_stops_without_stale_path_publish(self):
        planner = ImagePathPlanner(PlannerConfig(max_temporal_fallback_frames=5))
        self.warm(planner)
        empty = np.zeros_like(self.bilateral)
        lost = planner.plan(empty, empty, empty)
        self.assertFalse(lost.valid)
        self.assertEqual(len(lost.points), 0)
        self.assertTrue(lost.diagnostics["final_road_unrecoverable"])

    def test_right_only_uses_inward_normal_offset_and_ego_anchor(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        right_only = lane_mask(None, perspective_right)
        result = planner.plan(self.road, right_only, np.zeros_like(right_only))
        self.assertTrue(result.valid)
        self.assertTrue(result.diagnostics["ego_anchor_applied"])
        self.assertGreater(len(result.virtual_details), 10)
        self.assertTrue(all(item["source"] == RIGHT_BOUNDARY
                            for item in result.virtual_details))
        for item in result.virtual_details:
            delta = item["virtual"]-item["boundary"]
            self.assertLess(item["normal"][0], 0.0)
            self.assertAlmostEqual(float(np.dot(item["tangent"], delta)),
                                   0.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(delta)),
                                   item["lane_width_px"]/2.0, places=6)
        # The anchor row is the ego-exclusion-adjusted bottom (req 1), not
        # the raw configured roi_bottom -- the bumper band is out of scope.
        self.assertTrue(np.allclose(
            result.raw[0],
            [planner.config.vehicle_center_x_px,
             result.diagnostics["effective_roi_bottom_px"]]))

    def test_left_only_uses_inward_normal_offset(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        left_only = lane_mask(perspective_left, None)
        result = planner.plan(self.road, left_only, np.zeros_like(left_only))
        self.assertTrue(result.valid)
        self.assertGreater(len(result.virtual_details), 10)
        for item in result.virtual_details:
            delta = item["virtual"]-item["boundary"]
            self.assertGreater(item["normal"][0], 0.0)
            self.assertAlmostEqual(float(np.dot(item["tangent"], delta)),
                                   0.0, places=6)

    def test_left_curve_right_only_preserves_curvature_direction(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        curve = lambda y: -0.0006*(475.0-y)**2
        right_only = lane_mask(None, lambda y: perspective_right(y)+curve(y))
        result = planner.plan(self.road, right_only, np.zeros_like(right_only))
        self.assertTrue(result.valid)
        boundary = np.asarray([item["boundary"] for item in result.virtual_details])
        virtual = np.asarray([item["virtual"] for item in result.virtual_details])
        boundary_quadratic = np.polyfit(boundary[:, 1], boundary[:, 0], 2)[0]
        virtual_quadratic = np.polyfit(virtual[:, 1], virtual[:, 0], 2)[0]
        self.assertLess(boundary_quadratic, 0.0)
        self.assertLess(virtual_quadratic, 0.0)

    def test_right_curve_left_only_preserves_curvature_direction(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        curve = lambda y: 0.0006*(475.0-y)**2
        left_only = lane_mask(lambda y: perspective_left(y)+curve(y), None)
        result = planner.plan(self.road, left_only, np.zeros_like(left_only))
        self.assertTrue(result.valid)
        boundary = np.asarray([item["boundary"] for item in result.virtual_details])
        virtual = np.asarray([item["virtual"] for item in result.virtual_details])
        boundary_quadratic = np.polyfit(boundary[:, 1], boundary[:, 0], 2)[0]
        virtual_quadratic = np.polyfit(virtual[:, 1], virtual[:, 0], 2)[0]
        self.assertGreater(boundary_quadratic, 0.0)
        self.assertGreater(virtual_quadratic, 0.0)

    def test_one_sided_to_bilateral_recovery_is_continuous(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        one_sided = planner.plan(
            self.road, lane_mask(None, perspective_right),
            np.zeros_like(self.bilateral))
        recovered = planner.plan(
            self.road, self.bilateral, np.zeros_like(self.bilateral))
        self.assertTrue(one_sided.valid)
        self.assertTrue(recovered.valid)
        one_by_y = {int(round(y)): x for x, y in one_sided.points}
        deltas = [abs(x-one_by_y[int(round(y))]) for x, y in recovered.points
                  if int(round(y)) in one_by_y]
        self.assertGreater(len(deltas), 10)
        self.assertLess(float(np.max(deltas)), 2.0)

    def test_single_boundary_spike_keeps_outlier_rejection(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        def spiked_right(y):
            return perspective_right(y)-120.0 if abs(y-325) <= 8 else perspective_right(y)
        spiked = lane_mask(None, spiked_right)
        result = planner.plan(self.road, spiked, np.zeros_like(spiked))
        self.assertTrue(result.valid)
        self.assertGreater(
            result.diagnostics["rejections"]["LATERAL_JUMP"]+
            result.diagnostics["rejections"]["DIRECTION_OUTLIER"]+
            result.diagnostics["rejections"]["CONTINUITY_FAILURE"], 0)

    def test_abnormal_width_profile_falls_back_to_safe_road_center(self):
        planner = ImagePathPlanner()
        self.warm(planner)
        planner.width_profile = {y: 1000.0 for y in range(120, 476, 10)}
        right_only = lane_mask(None, perspective_right)
        # The source state machine holds the previous safe geometry until the
        # replacement survives both confirm and release debounce.
        for _ in range(planner.config.source_release_frames):
            result = planner.plan(
                self.road, right_only, np.zeros_like(right_only))
        self.assertTrue(result.valid)
        self.assertIn(ROAD_CENTER, result.sources)
        self.assertNotIn(RIGHT_BOUNDARY, result.sources)
        self.assertEqual(result.diagnostics["virtual_center_points"], 0)


if __name__ == "__main__":
    unittest.main()
