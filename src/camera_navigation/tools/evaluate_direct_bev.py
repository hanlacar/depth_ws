#!/usr/bin/env python3
"""Synthetic metric-mask evaluation for the direct BEV planner."""

from collections import Counter
import json
import time

import numpy as np

from camera_navigation.direct_bev_core import DirectBevConfig, DirectBevPlanner


def masks(planner, center, left=True, right=True, end=8.0, width=0.75):
    road = np.zeros((planner.rows, planner.cols), np.uint8)
    lane = np.zeros_like(road)
    for x in np.arange(0.3, end, planner.config.resolution_m):
        y = center(x)
        row, col = planner.metric_to_grid([[x, y]])[0]
        half = int(round(width/planner.config.resolution_m))
        road[max(0, row-1):row+2, max(0, col-half):col+half+1] = 1
        offset = int(round(0.55/planner.config.resolution_m))
        if left:
            lane[max(0, row-1):row+2, col+offset-1:col+offset+2] = 1
        if right:
            lane[max(0, row-1):row+2, col-offset-1:col-offset+2] = 1
    return road, lane


def scenarios(planner):
    straight = lambda x: 0.0
    cases = [
        ("both_straight", masks(planner, straight)),
        ("both_curve", masks(planner, lambda x: 0.08*x*x, end=3.0)),
        ("left_only", masks(planner, straight, right=False)),
        ("right_only", masks(planner, straight, left=False)),
        ("road_only", masks(planner, straight, False, False)),
        ("short_path", masks(planner, straight, False, False, end=1.35)),
    ]
    road, lane = masks(planner, straight, False, False)
    row, col = planner.metric_to_grid([[2.0, 0.0]])[0]
    road[row-3:row+4, col-3:col+4] = 0
    cases.append(("diamond_hole", (road, lane)))
    road, lane = masks(planner, straight, False, False)
    for x in (1.6, 1.9, 2.2):
        row, col = planner.metric_to_grid([[x, 0.0]])[0]
        road[row-1:row+2, col-12:col+13] = 0
    cases.append(("crosswalk_holes", (road, lane)))
    road, lane = masks(planner, straight, False, False)
    road[:, :40] = 1
    cases.append(("shadow_expansion", (road, lane)))
    cases.append(("wide_road", masks(planner, straight, False, False,
                                      width=1.35)))
    return cases


def main():
    planner = DirectBevPlanner(DirectBevConfig())
    mode_counts, state_counts, reasons = Counter(), Counter(), Counter()
    processing, steering, residuals = [], [], []
    road_violations = footprint_violations = 0
    for index, (name, (road, lane)) in enumerate(scenarios(planner)):
        planner.reset()
        started = time.perf_counter()
        result = planner.plan(road, lane, 1.0+index*0.05)
        processing.append((time.perf_counter()-started)*1000.0)
        mode_counts[result.mode] += 1
        state_counts[result.state] += 1
        reasons.update(result.diagnostics.get("reasons", []))
        steering.append(float(result.diagnostics.get(
            "required_steering_deg", 0.0) or 0.0))
        residuals.append(float(result.diagnostics.get(
            "fitting_residual_m", 0.0) or 0.0))
        for point in result.points:
            row, col = planner.metric_to_grid([point])[0]
            road_violations += int(not result.component[row, col])
            footprint_violations += int(not result.safe_road[row, col])
    deltas = np.abs(np.diff(steering))
    mean_ms = float(np.mean(processing))
    output = {
        "dataset": "synthetic_metric_masks",
        "frames": len(processing),
        "drivable_frames": int(state_counts["VALID"]+state_counts["DEGRADED"]),
        "mode_counts": dict(mode_counts), "state_counts": dict(state_counts),
        "degraded_reasons": dict(reasons),
        "road_corridor_violations": road_violations,
        "vehicle_footprint_violations": footprint_violations,
        "steering_sign_reversals": int(np.sum(
            np.sign(steering[1:])*np.sign(steering[:-1]) < 0)),
        "mean_steering_delta_deg": float(np.mean(deltas)),
        "max_steering_delta_deg": float(np.max(deltas)),
        "mean_fitting_residual_m": float(np.mean(residuals)),
        "max_fitting_residual_m": float(np.max(residuals)),
        "mean_processing_ms": mean_ms,
        "max_processing_ms": float(np.max(processing)),
        "processing_fps": 1000.0/mean_ms,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
