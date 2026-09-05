import unittest

import numpy as np
from race_interfaces.msg import ImagePath, ImagePathPoint

from camera_navigation.camera_pixel_controller_node import (
    DriveCommand,
    PixelController,
    PixelControllerConfig,
)
from camera_navigation.image_path_planner import (
    BOTH_BOUNDARIES,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY,
    ROAD_CENTER,
    TEMPORAL_FALLBACK,
    ImagePathPlanner,
    PlannerConfig,
)


HEIGHT = 480
WIDTH = 640
ZERO = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def road_mask(center_fn=lambda _y: 320.0, width_fn=lambda _y: 320.0,
              top=120, bottom=479):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(max(0, top), min(HEIGHT - 1, bottom) + 1):
        center = float(center_fn(y))
        width = float(width_fn(y))
        left = max(0, int(round(center - width / 2.0)))
        right = min(WIDTH - 1, int(round(center + width / 2.0)))
        if right >= left:
            mask[y, left:right + 1] = 1
    return mask


def lane_at(left_fn=None, right_fn=None, top=120, bottom=475):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(top, bottom + 1):
        for function in (left_fn, right_fn):
            if function is None:
                continue
            x = int(round(function(y)))
            mask[y, max(0, x - 2):min(WIDTH, x + 3)] = 1
    return mask


class RoadCenterPlanningTest(unittest.TestCase):

    def test_both_boundaries_remain_primary_anchor(self):
        road = road_mask()
        lane = lane_at(lambda _y: 220.0, lambda _y: 420.0)
        result = ImagePathPlanner().plan(road, lane, ZERO)
        self.assertTrue(result.valid)
        self.assertGreater(result.sources.count(BOTH_BOUNDARIES),
                           len(result.sources) // 2)
        self.assertEqual(result.diagnostics["source_mode"], "LANE")

    def test_left_boundary_uses_observed_road_width(self):
        road = road_mask(width_fn=lambda _y: 300.0)
        lane = lane_at(lambda _y: 170.0, None)
        result = ImagePathPlanner().plan(road, lane, ZERO)
        self.assertTrue(result.valid)
        self.assertEqual(set(result.sources), {LEFT_BOUNDARY})
        self.assertTrue(np.allclose(result.raw[:, 0], 320.0, atol=1.0))
        self.assertTrue(all(item["lane_width_source"] == "road_corridor"
                            for item in result.virtual_details))

    def test_right_boundary_uses_observed_road_width(self):
        road = road_mask(width_fn=lambda _y: 300.0)
        lane = lane_at(None, lambda _y: 470.0)
        result = ImagePathPlanner().plan(road, lane, ZERO)
        self.assertTrue(result.valid)
        self.assertEqual(set(result.sources), {RIGHT_BOUNDARY})
        self.assertTrue(np.allclose(result.raw[:, 0], 320.0, atol=1.0))

    def test_lane_free_road_generates_valid_raw_and_fitted_path(self):
        result = ImagePathPlanner().plan(road_mask(), ZERO, ZERO)
        self.assertTrue(result.valid)
        self.assertGreaterEqual(len(result.raw), 6)
        self.assertEqual(set(result.sources), {ROAD_CENTER})
        self.assertEqual(result.diagnostics["source_mode"], "ROAD")

    def test_lane_free_curved_road_is_not_forced_straight(self):
        center = lambda y: 320.0 + 0.0008 * (475.0 - y) ** 2
        result = ImagePathPlanner().plan(
            road_mask(center_fn=center, width_fn=lambda _y: 220.0), ZERO, ZERO)
        self.assertTrue(result.valid)
        self.assertGreater(result.points[-1, 0] - result.points[0, 0], 40.0)

    def test_far_field_edge_noise_has_less_effect_than_near_field(self):
        clean = road_mask(width_fn=lambda _y: 260.0)
        noisy = clean.copy()
        for y in range(120, 220):
            if (y // 10) % 2:
                noisy[y, 450:500] = 1
        clean_result = ImagePathPlanner().plan(clean, ZERO, ZERO)
        noisy_result = ImagePathPlanner().plan(noisy, ZERO, ZERO)
        self.assertTrue(clean_result.valid)
        self.assertTrue(noisy_result.valid)
        self.assertLess(abs(noisy_result.points[0, 0] -
                            clean_result.points[0, 0]), 8.0)

    def test_clipped_edge_is_damped_and_downweighted(self):
        clipped = road_mask(center_fn=lambda _y: 210.0,
                            width_fn=lambda _y: 500.0)
        planner = ImagePathPlanner()
        geometry = planner._road_geometry(
            clipped, list(range(475, 119, -10)))
        first = geometry[475]
        self.assertTrue(first["left_clipped"])
        self.assertLess(abs(first["center"] - 320.0),
                        abs(first["raw_center"] - 320.0))
        result = planner.plan(clipped, ZERO, ZERO)
        self.assertTrue(result.valid)
        self.assertGreater(result.diagnostics["road_clipped_rows"], 0)

    def test_mixed_rows_keep_row_level_sources(self):
        road = road_mask(width_fn=lambda _y: 280.0)
        lane = lane_at(lambda _y: 180.0, lambda _y: 460.0,
                       top=330, bottom=475)
        result = ImagePathPlanner().plan(road, lane, ZERO)
        self.assertTrue(result.valid)
        self.assertIn(BOTH_BOUNDARIES, result.sources)
        self.assertIn(ROAD_CENTER, result.sources)
        self.assertGreater(result.diagnostics["rejections"][
            "SOURCE_TRANSITION_SMOOTHED"], 0)

    def test_lane_to_road_transition_requires_release_debounce(self):
        planner = ImagePathPlanner()
        road = road_mask(width_fn=lambda _y: 360.0)
        lane = lane_at(lambda _y: 170.0, lambda _y: 370.0)
        before = planner.plan(road, lane, ZERO)
        planner.plan(road, lane, ZERO)
        after = planner.plan(road, ZERO, ZERO)
        self.assertTrue(before.valid)
        self.assertTrue(after.valid)
        self.assertLessEqual(after.diagnostics["temporal_shift_px"], 55.0)
        self.assertFalse(after.diagnostics["source_mode_transition"])
        second = planner.plan(road, ZERO, ZERO)
        self.assertFalse(second.diagnostics["source_mode_transition"])
        released = planner.plan(road, ZERO, ZERO)
        self.assertTrue(released.diagnostics["source_mode_transition"])

    def test_road_to_lane_transition_is_smoothed(self):
        planner = ImagePathPlanner()
        road = road_mask(width_fn=lambda _y: 360.0)
        road_only = planner.plan(road, ZERO, ZERO)
        lane = lane_at(lambda _y: 270.0, lambda _y: 470.0)
        lane_result = planner.plan(road, lane, ZERO)
        self.assertTrue(road_only.valid)
        self.assertTrue(lane_result.valid)
        self.assertFalse(lane_result.diagnostics["source_mode_transition"])
        planner.plan(road, lane, ZERO)
        released = planner.plan(road, lane, ZERO)
        self.assertTrue(released.diagnostics["source_mode_transition"])
        self.assertLessEqual(lane_result.diagnostics["temporal_alpha_used"],
                             0.2)

    def test_far_road_without_near_field_is_invalid(self):
        result = ImagePathPlanner().plan(
            road_mask(top=120, bottom=340), ZERO, ZERO)
        self.assertFalse(result.valid)
        self.assertFalse(result.diagnostics["near_field_ok"])

    def test_disconnected_far_blob_is_not_used_as_ego_road(self):
        road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        road[120:280, 220:420] = 1
        result = ImagePathPlanner().plan(road, ZERO, ZERO)
        self.assertFalse(result.valid)
        self.assertFalse(result.diagnostics["ego_road_component_present"])
        self.assertEqual(len(result.raw), 0)

    def test_near_field_width_collapse_is_invalid(self):
        narrow = road_mask(width_fn=lambda _y: 30.0)
        result = ImagePathPlanner().plan(narrow, ZERO, ZERO)
        self.assertFalse(result.valid)
        self.assertFalse(result.diagnostics["near_field_ok"])

    def test_split_road_is_reported_and_not_chosen_silently(self):
        road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        road[300:480, 270:371] = 1
        road[270:301, 180:461] = 1
        road[120:271, 180:271] = 1
        road[120:271, 370:461] = 1
        result = ImagePathPlanner().plan(road, ZERO, ZERO)
        self.assertTrue(result.diagnostics["branch_suspected"])
        self.assertGreaterEqual(result.diagnostics["branch_rows"], 3)
        # req 8 (INVALID semantics): the fork here is far/mid-field only --
        # the near corridor (bottom 300:480 run) is a single, safe road --
        # so this is the same "reported, cautious, but still drivable" case
        # as test_noncritical_far_branch_is_degraded_for_controller_slowdown,
        # not a hard INVALID. It must not be silently reported as full-
        # quality VALID either.
        self.assertFalse(result.diagnostics["branch_critical"])
        self.assertEqual(result.state, "DEGRADED")

    def test_noncritical_far_branch_is_degraded_for_controller_slowdown(self):
        road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        road[300:480, 210:431] = 1
        road[120:301, 180:300] = 1
        road[120:301, 340:460] = 1
        result = ImagePathPlanner().plan(road, ZERO, ZERO)

        self.assertTrue(result.valid)
        self.assertTrue(result.diagnostics["branch_suspected"])
        self.assertFalse(result.diagnostics["branch_critical"])
        self.assertEqual(result.state, "DEGRADED")

    def test_single_row_center_spike_is_removed(self):
        road = road_mask(width_fn=lambda _y: 220.0)
        road[323:328] = 0
        road[323:328, 320:540] = 1
        result = ImagePathPlanner().plan(road, ZERO, ZERO)
        self.assertTrue(result.valid)
        self.assertGreater(result.diagnostics["road_center_spike_rows"], 0)
        self.assertLess(np.max(np.abs(np.diff(result.raw[:, 0]))), 35.0)

    def test_sustained_curve_change_is_followed(self):
        planner = ImagePathPlanner(PlannerConfig(
            temporal_hysteresis_frames=3))
        outputs = []
        for shift in (0.0, 12.0, 24.0, 36.0, 48.0):
            result = planner.plan(
                road_mask(center_fn=lambda _y, s=shift: 320.0 + s,
                          width_fn=lambda _y: 260.0), ZERO, ZERO)
            self.assertTrue(result.valid)
            outputs.append(float(np.mean(result.points[:, 0])))
        self.assertGreater(outputs[-1] - outputs[0], 15.0)

    def test_complete_current_road_loss_invalidates_temporal_immediately(self):
        planner = ImagePathPlanner(PlannerConfig(
            max_temporal_fallback_frames=2))
        self.assertTrue(planner.plan(road_mask(), ZERO, ZERO).valid)
        first = planner.plan(ZERO, ZERO, ZERO)
        self.assertFalse(first.valid)
        self.assertEqual(len(first.points), 0)
        self.assertTrue(first.diagnostics["final_road_unrecoverable"])

    def test_curved_road_only_result_reaches_pixel_drive_policy(self):
        center = lambda y: 320.0 + 0.0008 * (475.0 - y) ** 2
        result = ImagePathPlanner().plan(
            road_mask(center_fn=center, width_fn=lambda _y: 220.0), ZERO, ZERO)
        self.assertTrue(result.valid)
        self.assertEqual(set(result.sources), {ROAD_CENTER})

        source_codes = {
            BOTH_BOUNDARIES: ImagePathPoint.BOTH_BOUNDARIES,
            LEFT_BOUNDARY: ImagePathPoint.LEFT_BOUNDARY,
            RIGHT_BOUNDARY: ImagePathPoint.RIGHT_BOUNDARY,
            ROAD_CENTER: ImagePathPoint.ROAD_CENTER,
            TEMPORAL_FALLBACK: ImagePathPoint.TEMPORAL_FALLBACK,
        }
        state_codes = {
            "VALID": ImagePath.STATE_VALID,
            "DEGRADED": ImagePath.STATE_DEGRADED,
            "INVALID": ImagePath.STATE_INVALID,
        }
        controller = PixelController(PixelControllerConfig(
            derivative_gain_deg_per_norm_per_s=0.0))
        self.assertTrue(controller.ingest_path(
            1_000_000_000, WIDTH, result.points, result.valid,
            result.confidence, 1.0,
            tuple(source_codes[source] for source in result.sources),
            state_codes[result.state]))
        command = controller.step(1.01, 1_010_000_000)

        self.assertEqual(command.drive, DriveCommand.SLOW.value)
        self.assertGreater(command.wheel, 0)
        self.assertLessEqual(abs(command.wheel), 22)


if __name__ == "__main__":
    unittest.main()
