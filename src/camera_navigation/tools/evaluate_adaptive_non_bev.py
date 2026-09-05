#!/usr/bin/env python3
"""Deterministic synthetic before/after smoke evaluation for non-BEV paths."""

import json
import time
from collections import Counter

import numpy as np

from camera_navigation.adaptive_non_bev_planner import (
    AdaptiveNonBevConfig, AdaptiveNonBevPlanner,
)
from camera_navigation.image_path_planner import ImagePathPlanner, PlannerConfig


def frame(left=True, right=True, curve=0.0, road_only=False):
    road = np.zeros((240, 320), np.uint8)
    lane = np.zeros_like(road)
    for y in range(60, 231):
        t = (y-60)/170.0
        bend = curve*(1.0-t)**2
        lo, hi = int(145-85*t+bend), int(175+85*t+bend)
        road[y, max(0, lo):min(320, hi+1)] = 1
        if left and not road_only:
            lane[y, max(0, lo-1):lo+2] = 1
        if right and not road_only:
            lane[y, max(0, hi-1):min(320, hi+2)] = 1
    return road, lane


def scenarios():
    cases = [
        ("both_straight", frame()), ("both_curve", frame(curve=30)),
        ("left_only", frame(right=False)), ("right_only", frame(left=False)),
        ("road_only", frame(road_only=True)),
    ]
    road, lane = frame(road_only=True)
    road[135:150, 135:185] = 0
    cases.append(("marking_gap", (road, lane)))
    road, lane = frame(road_only=True)
    road[100:220, :100] = 1
    cases.append(("shadow_expansion", (road, lane)))
    road, lane = frame()
    road[120:200, 220:320] = 1
    cases.append(("parked_width_change", (road, lane)))
    road, lane = frame()
    road[:165] = 0
    lane[:165] = 0
    cases.append(("short_path", (road, lane)))
    cases.extend([("no_detection", (np.zeros_like(road), np.zeros_like(lane))),
                  ("near_limit_curve", frame(curve=55)),
                  ("impossible_curve", frame(curve=125))])
    return cases


def run(planner, cases):
    modes = Counter()
    reasons = Counter()
    steering = []
    residuals = []
    latency = []
    drivable = 0
    road_violations = 0
    previous_sign = 0
    reversals = 0
    for index, (name, (road, lane)) in enumerate(cases):
        planner.reset()
        started = time.perf_counter()
        result = planner.plan(road, lane, np.zeros_like(lane),
                              timestamp_sec=1.0+index*0.05)
        latency.append((time.perf_counter()-started)*1000.0)
        mode = (result.diagnostics or {}).get(
            "generation_mode", (result.diagnostics or {}).get("source_mode", "UNKNOWN"))
        modes[str(mode or "UNKNOWN")] += 1
        drivable += int(result.valid)
        if not result.valid:
            reasons[str((result.diagnostics or {}).get(
                "failure_reason", "UNSPECIFIED"))] += 1
        # Use the planner's ego-connected, small-hole-repaired road corridor.
        # Counting raw marking holes as off-road would mislabel the intended
        # diamond/crosswalk gap recovery scenario as a boundary violation.
        corridor = result.road_component
        for x, y in result.points:
            row = max(0, min(corridor.shape[0]-1, int(round(y))))
            col = max(0, min(corridor.shape[1]-1, int(round(x))))
            road_violations += int(not corridor[row, col])
        angle = float((result.diagnostics or {}).get("required_steering_deg", 0.0))
        steering.append(angle)
        sign = int(np.sign(angle))
        if sign and previous_sign and sign != previous_sign:
            reversals += 1
        if sign:
            previous_sign = sign
        residuals.append(float((result.diagnostics or {}).get(
            "polynomial_residual_px", 0.0)))
    deltas = np.abs(np.diff(steering))
    mean_ms = float(np.mean(latency))
    return {"frames": len(cases), "drivable_frames": drivable,
            "drivable_ratio": drivable/max(1, len(cases)),
            "mode_counts": dict(modes), "invalid_reasons": dict(reasons),
            "road_corridor_violations": road_violations,
            "steering_sign_reversals": reversals,
            "mean_steering_delta_deg": float(np.mean(deltas)) if len(deltas) else 0.0,
            "max_steering_delta_deg": float(np.max(deltas)) if len(deltas) else 0.0,
            "max_abs_steering_deg": max(map(abs, steering)),
            "mean_fit_residual_px": float(np.mean(residuals)),
            "max_fit_residual_px": max(residuals),
            "mean_processing_ms": mean_ms,
            "processing_fps": 1000.0/max(1e-6, mean_ms)}


def main():
    config = PlannerConfig(
        roi_top=80, roi_bottom=220, vehicle_center_x_px=160.0,
        lane_width_seed_px=100.0, ego_exclusion_enabled=False,
        minimum_component_pixels=5, valid_min_confidence=0.0,
        road_minimum_near_width_px=30.0, road_minimum_near_coverage_ratio=0.2,
        minimum_boundary_clearance_m=0.2)
    cases = scenarios()
    print(json.dumps({
        "dataset": "synthetic_12_scenarios",
        "before": run(ImagePathPlanner(config), cases),
        "after": run(AdaptiveNonBevPlanner(config, AdaptiveNonBevConfig()), cases),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
