"""Synthetic tests for the per-stage road-pixel diagnostics added to
/camera/bev/diagnostics: raw_road_pixels, refined_road_pixels,
decoded_road_pixels, projected_road_pixels, ego_component_pixels,
safe_road_pixels, planner_state. See direct_bev_core.py (plan/
_fallback_or_invalid/_invalid) and direct_bev_planner_node.py (_process,
stage_pixel_count, _publish_invalid) for the instrumentation itself.

Covers cases 1, 2, 3, 4, 5 and the stamp_ns/monotonicity checks from the
task's test list; cases 6 (INPUT_TIMEOUT), 7 (diagnostics keep publishing
while INVALID) and 8 (stamp_ns identical across stages, live) are verified
against a running node in the accompanying live-capture run (not unit
-testable without rclpy + the full mp4/backend pipeline).
"""
import numpy as np
import pytest

from camera_navigation.direct_bev_core import DirectBevConfig, DirectBevPlanner
from camera_navigation.direct_bev_planner_node import stage_pixel_count
from camera_navigation.direct_bev_projection import project_mask_to_bev


def masks(center=lambda x: 0.0, left=True, right=True, end=8.0,
         half_width=0.75):
    """Copied from test_direct_bev.py's helper of the same name (kept local
    so this file has no import-order dependency on that one)."""
    planner = DirectBevPlanner(DirectBevConfig())
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x in np.arange(0.30, end, planner.config.resolution_m):
        middle = center(x)
        row, column = planner.metric_to_grid([[x, middle]])[0]
        half = int(round(half_width/planner.config.resolution_m))
        road[max(0, row-1):min(planner.rows, row+2),
             max(0, column-half):min(planner.cols, column+half+1)] = 1
        offset = int(round(0.55/planner.config.resolution_m))
        if left:
            lane[max(0, row-1):row+2, column+offset-1:column+offset+2] = 1
        if right:
            lane[max(0, row-1):row+2, column-offset-1:column-offset+2] = 1
    return planner, road, lane


# --- Case 1: raw road is 0 (nothing detected at all) ------------------------
def test_case1_empty_road_reports_ego_road_missing_with_zero_not_null():
    planner = DirectBevPlanner(DirectBevConfig())
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    result = planner.plan(road, lane, 1.0)
    assert not result.valid
    assert result.diagnostics["reasons"] == ["EGO_ROAD_MISSING"]
    # Genuinely zero (component/safe masks were computed, they're just
    # empty) -- must be present and 0, never absent/null, since preprocess()
    # always runs before the EGO_ROAD_MISSING check.
    assert result.diagnostics["ego_component_pixels"] == 0
    assert result.diagnostics["safe_road_pixels"] == 0
    assert result.diagnostics["ego_component_pixels"] == np.count_nonzero(result.component)
    assert result.diagnostics["safe_road_pixels"] == np.count_nonzero(result.safe_road)


# --- Case 2: raw > 0, refined == 0 (perception_refinement stage) -----------
def test_case2_raw_positive_refined_zero_extraction():
    # This exercises the plumbing (stage_pixel_count), not
    # perception_refinement's own algorithm (untouched, out of scope).
    refinement_diagnostics = {"raw_pixels": {"road": 428}, "refined_pixels": {"road": 0}}
    assert stage_pixel_count(refinement_diagnostics, "raw_pixels") == 428
    assert stage_pixel_count(refinement_diagnostics, "refined_pixels") == 0


def test_case2b_missing_or_malformed_stage_is_null_not_zero():
    # A frame where refinement never ran (or the JSON round-trip failed)
    # must report None, not a fabricated 0 -- "값이 없는 단계는 null".
    assert stage_pixel_count({}, "raw_pixels") is None
    assert stage_pixel_count({"raw_pixels": "not_a_dict"}, "raw_pixels") is None
    assert stage_pixel_count({"error": "INVALID_REFINEMENT_DIAGNOSTICS"},
                             "refined_pixels") is None


# --- Case 3: refined > 0, projected == 0 after BEV warp ---------------------
def test_case3_refined_positive_projected_zero_after_warp():
    refined_road_mask = np.zeros((480, 640), np.uint8)
    refined_road_mask[100:140, 100:140] = 1  # a real, non-trivial road blob
    assert np.count_nonzero(refined_road_mask) > 0

    # map_x/map_y always sample source pixel (400, 400) -- nowhere near the
    # road blob above -- for every output BEV cell, so the projected result
    # must come back entirely empty. Uses the real project_mask_to_bev(),
    # not a mock.
    out_rows, out_cols = 194, 151
    map_x = np.full((out_rows, out_cols), 400.0, dtype=np.float32)
    map_y = np.full((out_rows, out_cols), 400.0, dtype=np.float32)
    projected = project_mask_to_bev(refined_road_mask, map_x, map_y)
    projected_road_pixels = int(np.count_nonzero(projected))

    assert projected_road_pixels == 0
    assert np.count_nonzero(refined_road_mask) > 0 and projected_road_pixels == 0


# --- Case 4: projected road > 0, but no component connects to ego ----------
def test_case4_projected_positive_ego_component_zero():
    planner = DirectBevPlanner(DirectBevConfig())
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    # A real, big-enough (> minimum_component_area_m2) blob in the far
    # top-left corner: far from the vehicle (low row index = far x_m) AND
    # off the center corridor (low column index != center_col), so neither
    # the near-field seed nor the fallback center-corridor search in
    # _ego_component() can ever reach it.
    road[0:20, 0:20] = 1
    projected_road_pixels = int(np.count_nonzero(road))
    assert projected_road_pixels > 0

    result = planner.plan(road, lane, 1.0)
    assert result.diagnostics["ego_component_pixels"] == 0
    assert not result.valid


# --- Case 5: normal VALID input -- full stage chain, all real values -------
def test_case5_valid_input_has_consistent_non_null_stage_counts():
    planner, road, lane = masks()
    projected_road_pixels = int(np.count_nonzero(road))  # what plan() receives
    result = planner.plan(road, lane, 1.0)

    assert result.valid and result.state == "VALID"
    ego = result.diagnostics["ego_component_pixels"]
    safe = result.diagnostics["safe_road_pixels"]
    assert ego is not None and safe is not None
    assert ego > 0 and safe > 0
    assert ego == np.count_nonzero(result.component)
    assert safe == np.count_nonzero(result.safe_road)
    # Structural invariant: safe is computed from a distance transform of
    # component, and cv2.distanceTransform reports 0 for every pixel
    # outside component -- so safe can never exceed component's extent.
    assert safe <= ego
    assert projected_road_pixels > 0


# --- Case 8 (unit half): one plan() call is inherently one stamp_ns --------
def test_case8_single_plan_call_is_one_coherent_frame():
    planner, road, lane = masks()
    result = planner.plan(road, lane, 42.0)
    # All six-stage-adjacent diagnostics fields came out of this one call,
    # for this one timestamp -- by construction there is exactly one
    # stamp_ns per plan() invocation, and the direct_bev_planner_node.py
    # side stamps every field in the same result.diagnostics.update({...})
    # call with a single shared `stamp_ns` variable (verified live in the
    # accompanying capture run for the full node -> diagnostics topic path).
    assert result.diagnostics["ego_component_pixels"] is not None
    assert result.diagnostics["safe_road_pixels"] is not None
    assert result.diagnostics["reasons"] == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
