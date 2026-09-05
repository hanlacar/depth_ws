#!/usr/bin/env python3
"""Compare S0 and road-edge A6 candidates on one immutable BEV cache."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import (
    HybridCandidateOptions, HybridDirectBevCandidate)


def planners(config):
    common = dict(temporal_smoothing=True, curvature_stabilization=True,
                  mode_hysteresis_frames=3, fixed_resample_origin=True,
                  fail_closed_hold=True)
    return {
        "B0": HybridDirectBevCandidate(
            config, HybridCandidateOptions(**common)),
        "B1": HybridDirectBevCandidate(
            config, HybridCandidateOptions(
                **common, road_boundary_fallback="basic")),
        "B2": HybridDirectBevCandidate(
            config, HybridCandidateOptions(
                **common, road_boundary_fallback="gated")),
    }


def length(points):
    points = np.asarray(points, float).reshape(-1, 2)
    return (float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if len(points) > 1 else 0.0)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def row_for(name, planner, result, frame, fps, road, lane, elapsed,
            previous_near):
    points = np.asarray(result.points, float).reshape(-1, 2)
    diagnostics = result.diagnostics
    grid = (planner.metric_to_grid(points) if len(points)
            else np.empty((0, 2), int))
    bounded = (not len(grid) or bool(np.all(
        (grid[:, 0] >= 0) & (grid[:, 0] < planner.rows) &
        (grid[:, 1] >= 0) & (grid[:, 1] < planner.cols))))
    road_inside = (bounded and (not len(grid) or bool(np.all(
        result.component[grid[:, 0], grid[:, 1]] > 0))))
    clearance_ok = (bounded and (not len(grid) or bool(np.all(
        result.safe_road[grid[:, 0], grid[:, 1]] > 0))))
    near = (float(np.interp(planner.config.near_required_m,
                            points[:, 0], points[:, 1]))
            if len(points) else math.nan)
    required = diagnostics.get("required_steering_deg")
    required = float(required) if required is not None else math.nan
    wheel = (int(np.clip(np.rint(-required), -22, 22))
             if result.valid and math.isfinite(required) else 0)
    path_source = diagnostics.get("path_source", "NONE")
    no_lane = not bool(np.any(lane))
    path_jump = bool(result.valid and previous_near is not None and
                     math.isfinite(near) and
                     abs(near-previous_near) >
                     planner.config.temporal_lateral_gate_m)
    intent = 0 if not len(points) else int(np.sign(points[-1, 1]))
    wheel_sign_error = bool(result.valid and intent and wheel and
                            int(np.sign(wheel)) != -intent)
    roi_misidentification = bool(
        result.valid and result.mode != "HOLD" and
        path_source == "ROAD_BOUNDARY_BOTH" and
        (diagnostics.get("observed_road_width_m") is None or
         diagnostics.get("boundary_valid_slice_count", 0) < 3))
    return {
        "frame_index": frame, "stamp_sec": frame/fps, "candidate": name,
        "valid": bool(result.valid), "state": result.state,
        "mode": result.mode, "path_source": path_source,
        "failure_reason": "|".join(diagnostics.get("reasons", [])),
        "road_pixels": int(np.count_nonzero(road)),
        "lane_pixels": int(np.count_nonzero(lane)), "no_lane": no_lane,
        "road_only": no_lane and bool(np.any(road)),
        "path_length_m": length(points), "near_path_y_m": near,
        "required_steering_deg": required, "camera_wheel": wheel,
        "road_inside": road_inside, "clearance_ok": clearance_ok,
        # Safe-road erosion already represents swept vehicle half-width plus
        # margin, so this is the explicit swept-footprint containment result.
        "swept_ok": clearance_ok,
        "steering_over_22": bool(math.isfinite(required) and
                                  abs(required) > 22.0),
        "steering_sign_error": wheel_sign_error,
        "path_jump": path_jump,
        "temporal_curvature_jump": (
            "TEMPORAL_CURVATURE_JUMP" in
            "|".join(diagnostics.get("reasons", []))),
        "invalid_nonzero": bool(not result.valid and wheel != 0),
        "boundary_valid_slices": int(
            diagnostics.get("boundary_valid_slice_count", 0)),
        "boundary_rejected_slices": int(
            diagnostics.get("boundary_rejected_slice_count", 0)),
        "observed_road_width_m": diagnostics.get("observed_road_width_m"),
        "road_edge_confidence": diagnostics.get("road_edge_confidence", 0.0),
        "boundary_misidentification": roi_misidentification,
        "processing_ms": elapsed,
    }, near if result.valid and math.isfinite(near) else None


def summarize(rows, names):
    output = {}
    by_frame = {}
    for row in rows:
        by_frame.setdefault(int(row["frame_index"]), {})[row["candidate"]] = row
    for name in names:
        selected = [r for r in rows if r["candidate"] == name]
        valid = [r for r in selected if r["valid"]]
        no_lane = [r for r in selected if r["no_lane"]]
        road_only = [r for r in selected if r["road_only"]]
        times = [float(r["processing_ms"]) for r in selected]
        recovered = [i for i, values in by_frame.items()
                     if values[name]["valid"] and not values["B0"]["valid"]]
        regressed = [i for i, values in by_frame.items()
                    if values["B0"]["valid"] and not values[name]["valid"]]
        output[name] = {
            "frames": len(selected), "drivable": len(valid),
            "drivable_ratio": len(valid)/max(1, len(selected)),
            "no_lane_frames": len(no_lane),
            "no_lane_drivable": sum(r["valid"] for r in no_lane),
            "no_lane_drivable_ratio": (sum(r["valid"] for r in no_lane) /
                                        max(1, len(no_lane))),
            "road_only_frames": len(road_only),
            "road_only_drivable": sum(r["valid"] for r in road_only),
            "road_only_drivable_ratio": (sum(r["valid"] for r in road_only) /
                                          max(1, len(road_only))),
            "recovered_vs_B0": len(recovered),
            "regressed_vs_B0": len(regressed),
            "mean_path_length_m": (statistics.fmean(
                float(r["path_length_m"]) for r in valid) if valid else 0.0),
            "paths_lt_2m": sum(0 < float(r["path_length_m"]) < 2
                                for r in valid),
            "road_outside": sum(not r["road_inside"] for r in valid),
            "clearance_violations": sum(not r["clearance_ok"] for r in valid),
            "swept_path_violations": sum(not r["swept_ok"] for r in valid),
            "steering_over_22": sum(r["steering_over_22"] for r in valid),
            "path_jumps": sum(r["path_jump"] for r in valid),
            "steering_sign_errors": sum(r["steering_sign_error"] for r in valid),
            "TEMPORAL_CURVATURE_JUMP": sum(
                r["temporal_curvature_jump"] for r in selected),
            "invalid_stale_nonzero": sum(r["invalid_nonzero"] for r in selected),
            "boundary_misidentification": sum(
                r["boundary_misidentification"] for r in valid),
            "mean_processing_ms": statistics.fmean(times),
            "processing_p95_ms": percentile(times, 95),
            "effective_fps": 1000.0/statistics.fmean(times),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = planners(DirectBevConfig())
    previous_near = {name: None for name in candidates}
    rows = []
    frame = 0
    for chunk_path in sorted(args.cache.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            for road, lane in zip(chunk["road"], chunk["lane"]):
                for name, planner in candidates.items():
                    started = time.perf_counter()
                    result = planner.plan(road, lane, frame/args.fps)
                    elapsed = 1000.0*(time.perf_counter()-started)
                    row, near = row_for(name, planner, result, frame, args.fps,
                                        road, lane, elapsed,
                                        previous_near[name])
                    rows.append(row)
                    previous_near[name] = near
                frame += 1
                if frame % 1000 == 0:
                    print(f"road-boundary candidates: {frame}", flush=True)
    csv_path = args.output/"road_boundary_lineage.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "cache": str(args.cache), "frames": frame,
        "same_semantic_frame_for_all_candidates": True,
        "candidates": summarize(rows, candidates),
    }
    (args.output/"road_boundary_summary.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["candidates"], indent=2))


if __name__ == "__main__":
    main()
