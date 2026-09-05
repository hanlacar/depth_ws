"""Tests for req 4: local-only jump rejection and the stale-previous-path
lock fix in image_path_planner.py.
"""
import unittest

import numpy as np

from camera_navigation.image_path_planner import ImagePathPlanner, PlannerConfig


HEIGHT = 480
WIDTH = 640


def perspective_left(y):
    return 320.0 - (y-120.0)*0.72


def perspective_right(y):
    return 320.0 + (y-120.0)*0.72


def lane_mask(offset_row=None, offset_px=0):
    lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        extra = offset_px if offset_row is not None and abs(y-offset_row) <= 6 else 0
        for function in (perspective_left, perspective_right):
            x = int(round(function(y)))+extra
            lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
    return lane


class LocalJumpClassificationTest(unittest.TestCase):
    def test_heading_jump_is_a_local_removal_not_a_full_invalidate(self):
        # A single-row heading kink wide/long enough (>= marking_min_length_px)
        # to survive marking suppression as a real boundary segment, but
        # sharp enough to trip the heading-angle screen.
        planner = ImagePathPlanner()
        lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for y in range(120, 476):
            kink = 60 if 260 <= y <= 300 else 0
            for function in (perspective_left, perspective_right):
                x = int(round(function(y)))+kink
                lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        result = planner.plan(road, lane, np.zeros_like(lane))
        # The horizon as a whole must still produce a usable path: local
        # outlier removal, not a wholesale discard.
        self.assertGreater(len(result.points), 10)

    def test_local_outlier_cap_prevents_over_rejection(self):
        planner = ImagePathPlanner(PlannerConfig(max_local_outlier_ratio=0.2))
        points = np.column_stack((
            320.0+50.0*np.sin(np.arange(30)),  # noisy but not degenerate
            np.linspace(475, 130, 30)))
        sources = ["ROAD"]*30
        weights = np.ones(30)
        _, _, _, rejected, _ = planner._repair_local_spikes(
            points, sources, weights)
        cap = int(np.ceil(0.2*30))
        self.assertLessEqual(rejected, cap)


class StaleTemporalLockTest(unittest.TestCase):
    def test_stale_previous_path_does_not_lock_out_a_fresh_valid_fit(self):
        config = PlannerConfig(temporal_stale_lock_max_frames=2,
                               maximum_temporal_shift_px=10.0)
        planner = ImagePathPlanner(config)
        road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        straight = lane_mask()
        # Establish a confident previous path.
        first = planner.plan(road, straight, np.zeros_like(straight), timestamp_sec=0.0)
        self.assertTrue(first.valid)

        # Feed a few frames with no road/lane evidence at all so the planner
        # is forced into complete-loss temporal fallback (fallback_age
        # grows) while `previous` stays the stale straight-line path.
        empty = np.zeros_like(straight)
        t = 0.05
        for _ in range(3):
            planner.plan(empty, empty, empty, timestamp_sec=t)
            t += 0.05

        # Now the road has genuinely curved hard to one side. Once
        # fallback_age >= temporal_stale_lock_max_frames, the new fit must be
        # trusted rather than pinned to the stale straight previous path.
        curved = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for y in range(120, 476):
            shift = (475-y)*0.9
            for function in (perspective_left, perspective_right):
                x = int(round(function(y)+shift))
                curved[y, max(0, x-2):min(WIDTH, x+3)] = 1
        result = planner.plan(road, curved, np.zeros_like(curved), timestamp_sec=t)
        # The stale-recovery path must have engaged (fallback_age was >=
        # temporal_stale_lock_max_frames) and trusted the fresh fit at a high
        # alpha, rather than pinning the blend to the stale straight path
        # (temporal_outlier_rejected / alpha=0, the pre-fix lock behavior).
        self.assertTrue(result.diagnostics["temporal_stale_recovery"])
        self.assertGreaterEqual(result.diagnostics["temporal_alpha_used"],
                                config.temporal_stale_recovery_alpha)
        self.assertFalse(result.diagnostics["temporal_outlier_rejected"])


if __name__ == "__main__":
    unittest.main()
