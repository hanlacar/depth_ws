"""Tests for the dynamic PATH ROI / corridor, boundary-as-track filtering,
ROAD_ONLY fallback, and INVALID-semantics rework (this task's requirements).
"""
import unittest

import numpy as np

from camera_navigation.image_path_planner import (
    DEGRADED,
    INVALID,
    ROAD_CENTER,
    VALID,
    ImagePathPlanner,
    PlannerConfig,
)


HEIGHT = 480
WIDTH = 640


def perspective_left(y, shift=0.0):
    return 320.0-(y-120.0)*0.72+shift


def perspective_right(y, shift=0.0):
    return 320.0+(y-120.0)*0.72+shift


def lane_mask(left_fn=None, right_fn=None, noise_rows=(), noise_px=6):
    lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        jitter = noise_px if y in noise_rows else 0
        for function in (left_fn, right_fn):
            if function is None:
                continue
            x = int(round(function(y)))+jitter
            lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
    return lane


def wide_road():
    road = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    return road


def curving_road(shift=0.0):
    """A finite-width road corridor whose center follows the same shift as
    the lane -- unlike wide_road() (full image width, always centered),
    this gives road_geometry a real curve signal too, matching a real
    camera frame where the visible road is not literally infinite."""
    road = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(120, 476):
        center = 320.0+shift
        half = 40.0+(475-y)*0.55
        left = max(0, int(center-half))
        right = min(WIDTH-1, int(center+half))
        road[y, left:right+1] = 1
    return road


class StraightRoadNoInvalidTest(unittest.TestCase):
    """정상 직선: NO_BRANCH, wide road, mild lane noise -> never INVALID."""

    def test_wide_straight_road_with_lane_noise_stays_valid_or_degraded(self):
        planner = ImagePathPlanner()
        road = wide_road()
        noise_rows = set(range(200, 260, 4))
        lane = lane_mask(perspective_left, perspective_right, noise_rows, noise_px=10)
        for _ in range(5):
            result = planner.plan(road, lane, np.zeros_like(lane))
            self.assertNotEqual(result.state, INVALID)
            self.assertIn(result.state, (VALID, DEGRADED))
        self.assertFalse(result.diagnostics["branch_critical"])

    def test_wide_straight_road_lane_free_uses_road_only(self):
        planner = ImagePathPlanner()
        road = wide_road()
        empty_lane = np.zeros_like(road)
        result = None
        for _ in range(3):
            result = planner.plan(road, empty_lane, empty_lane)
        self.assertNotEqual(result.state, INVALID)
        self.assertTrue(result.diagnostics.get("road_only_path"))
        self.assertTrue(all(source == ROAD_CENTER for source in result.sources))


class DiamondMarkingCorridorTest(unittest.TestCase):
    """마름모: diamond in the corridor center must not become a boundary or
    make the path zigzag, even though the real road boundaries are normal."""

    def test_diamond_is_not_adopted_as_boundary_and_path_stays_smooth(self):
        planner = ImagePathPlanner()
        road = wide_road()
        lane = lane_mask(perspective_left, perspective_right)
        # A diamond/letter mark: a compact, roughly centered white blob
        # spanning several rows, made of two "V" strokes so its per-row x
        # position swings sharply -- the zigzag boundary-track pattern.
        for offset, y in enumerate(range(260, 300)):
            sign = 1 if (offset//4) % 2 == 0 else -1
            x = int(320+sign*offset*3)
            lane[y, max(0, x-2):min(WIDTH, x+3)] = 1
        clean = ImagePathPlanner().plan(road, lane_mask(perspective_left, perspective_right),
                                        np.zeros_like(lane))
        result = planner.plan(road, lane, np.zeros_like(lane))
        self.assertNotEqual(result.state, INVALID)
        # The final path must stay close to the clean (no-diamond) path --
        # it must not zigzag across the frame following the mark.
        common = min(len(result.points), len(clean.points))
        if common:
            deviation = np.max(np.abs(
                result.points[:common, 0]-clean.points[:common, 0]))
            self.assertLess(deviation, 60.0)


class DynamicCorridorFollowsCurveTest(unittest.TestCase):
    """커브: the corridor must move with the previous path/road center, not
    stay pinned to a fixed screen-center trapezoid."""

    def test_corridor_center_shifts_with_previous_path_on_a_curve(self):
        planner = ImagePathPlanner()
        straight_road = curving_road(0.0)
        straight = lane_mask(perspective_left, perspective_right)
        first = planner.plan(straight_road, straight, np.zeros_like(straight))
        self.assertNotEqual(first.state, INVALID)

        curved_road = curving_road(150.0)
        curved = lane_mask(lambda y: perspective_left(y, 150.0),
                           lambda y: perspective_right(y, 150.0))
        # Feed the curve a few times so the previous path and corridor both
        # migrate toward it.
        result = None
        for _ in range(4):
            result = planner.plan(curved_road, curved, np.zeros_like(curved))
        near_bounds = None
        for item in result.diagnostics["path_corridor_bounds"]:
            if near_bounds is None or item["y"] > near_bounds["y"]:
                near_bounds = item
        self.assertIsNotNone(near_bounds)
        # The corridor's near-field center must have moved right along with
        # the curve, not stayed pinned near the original screen center.
        corridor_center = 0.5*(near_bounds["lo"]+near_bounds["hi"])
        self.assertGreater(corridor_center, 320.0+40.0)
        # And the real curved road must not have been cut off: the fitted
        # path should end up reasonably close to the true curved boundary
        # midpoint, not clipped back toward the old straight position.
        self.assertGreater(result.points[0, 0], 320.0+40.0)


class LostReacquireCorridorTest(unittest.TestCase):
    """LOST/reacquire: corridor progressively widens on loss, then narrows
    back down on reacquire -- never a single-frame snap either way."""

    def test_corridor_expands_progressively_then_shrinks_on_reacquire(self):
        config = PlannerConfig(path_corridor_expand_step=0.25,
                               path_corridor_shrink_step=0.25)
        planner = ImagePathPlanner(config)
        road = wide_road()
        lane = lane_mask(perspective_left, perspective_right)
        first = planner.plan(road, lane, np.zeros_like(lane))
        self.assertEqual(planner._corridor_state, "TRACKING")

        # A short mask dropout is tolerated as a still-"valid" (DEGRADED)
        # temporal fallback for up to max_temporal_fallback_frames -- the
        # corridor correctly does not treat that grace period as LOST.
        # Genuine LOST needs the dropout to outlast that grace period.
        empty = np.zeros_like(road)
        levels = []
        for _ in range(config.max_temporal_fallback_frames+4):
            planner.plan(empty, empty, empty)
            levels.append(planner._corridor_expand_level)
        # Progressive: strictly non-decreasing, not an instant jump to 1.0.
        self.assertTrue(all(b >= a-1e-9 for a, b in zip(levels, levels[1:])))
        self.assertGreater(levels[-1], levels[0])
        self.assertIn(planner._corridor_state, ("EXPANDING", "LOST"))

        # Reacquire: feed the same clean lane again and confirm the level
        # comes back down gradually rather than snapping to 0 in one frame.
        reacquire_levels = []
        for _ in range(config.max_temporal_fallback_frames+4):
            planner.plan(road, lane, np.zeros_like(lane))
            reacquire_levels.append(planner._corridor_expand_level)
        self.assertTrue(all(b <= a+1e-9 for a, b in zip(reacquire_levels, reacquire_levels[1:])))
        self.assertLess(reacquire_levels[-1], levels[-1])


class RoadOnlyFallbackTest(unittest.TestCase):
    """ROAD_ONLY: no lane mask at all, stable road mask -> drivable path."""

    def test_no_lane_at_all_still_produces_drivable_road_only_path(self):
        planner = ImagePathPlanner()
        road = wide_road()
        empty = np.zeros_like(road)
        result = None
        for _ in range(3):
            result = planner.plan(road, empty, empty)
        self.assertNotEqual(result.state, INVALID)
        self.assertTrue(result.valid)
        self.assertGreaterEqual(len(result.points),
                                planner.config.valid_min_road_only_points)
        self.assertTrue(all(source == ROAD_CENTER for source in result.sources))


if __name__ == "__main__":
    unittest.main()
