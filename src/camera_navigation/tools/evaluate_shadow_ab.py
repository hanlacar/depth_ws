#!/usr/bin/env python3
"""Deterministic, single-semantic-input A0/A6 shadow comparison."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import zlib

import numpy as np

from camera_navigation.direct_bev_controller import BevControllerConfig, DirectBevController
from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_navigation.metric_path_quality import maximum_curvature


def path_length(points):
    points = np.asarray(points, float).reshape(-1, 2)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def local_curvature(points, target):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 3 or target is None: return 0.0
    index = int(np.argmin(np.linalg.norm(points-np.asarray(target), axis=1)))
    lo, hi = max(0, index-3), min(len(points), index+4)
    sample = points[lo:hi]
    if len(sample) < 3 or np.ptp(sample[:, 0]) < 1e-6: return 0.0
    a, b, _ = np.polyfit(sample[:, 0], sample[:, 1], 2)
    x = float(points[index, 0]); slope = 2*a*x+b
    return float(2*a/(1+slope*slope)**1.5)


def constant_arc_y(x, steering_deg, wheelbase):
    curvature = math.tan(math.radians(steering_deg))/wheelbase
    if abs(curvature) < 1e-9: return 0.0
    value = 1.0-(curvature*x)**2
    if value < 0.0: return math.nan
    return float((1.0-math.sqrt(value))/curvature)


def metrics(planner, result, command, elapsed_ms):
    points = np.asarray(result.points, float).reshape(-1, 2)
    target = command.get("target_point")
    curvature = maximum_curvature(points) if len(points) >= 3 else 0.0
    local = local_curvature(points, target)
    expected = math.degrees(math.atan(planner.config.wheelbase_m*local))
    required = command.get("required_steering_deg")
    wheel = int(command.get("wheel", 0))
    target_y_error = math.nan
    if target is not None:
        predicted = constant_arc_y(float(target[0]), -float(wheel),
                                   planner.config.wheelbase_m)
        if math.isfinite(predicted): target_y_error = predicted-float(target[1])
    grid = planner.metric_to_grid(points) if len(points) else np.empty((0, 2), int)
    road_ok = bool(not len(grid) or np.all(result.component[grid[:, 0], grid[:, 1]] > 0))
    clear_ok = bool(not len(grid) or np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0))
    return {
        "state": result.state,
        "reason": "|".join(result.diagnostics.get("reasons", [])),
        "path_length_m": path_length(points),
        "lookahead_point": json.dumps(target or [], separators=(",", ":")),
        "required_steering_deg": float(required) if required is not None else math.nan,
        "wheel_command": wheel,
        "drive_command": 1.0 if result.state == "DEGRADED" else (2.0 if result.state == "VALID" else 0.0),
        "lateral_min_m": float(points[:, 1].min()) if len(points) else math.nan,
        "lateral_max_m": float(points[:, 1].max()) if len(points) else math.nan,
        "maximum_curvature_per_m": curvature,
        "local_target_curvature_per_m": local,
        "turn_radius_m": 1.0/abs(local) if abs(local) > 1e-9 else math.inf,
        "expected_steering_deg": expected,
        "steering_error_deg": (float(required)-expected if required is not None else math.nan),
        "quantized_target_lateral_error_m": target_y_error,
        "road_containment": road_ok,
        "clearance_ok": clear_ok,
        "processing_ms": elapsed_ms,
        "path_signature": np.round(points, 5).tobytes(),
        "near_y": float(np.interp(planner.config.near_required_m, points[:, 0], points[:, 1])) if len(points) else math.nan,
    }


def prefixed(name, values):
    return {f"{name}_{key}": value for key, value in values.items()
            if key not in ("path_signature", "near_y")}


def run(cache, output, fps, timestamp_rate):
    config = DirectBevConfig(); all_planners = ablation_planners(config)
    planners = {"a0": all_planners["A0"], "a6": all_planners["A6"]}
    controllers = {
        "a0": DirectBevController(BevControllerConfig()),
        "a6": DirectBevController(BevControllerConfig(lookahead_from_path_start=True)),
    }
    counts = {key: 0 for key in ("both_drivable",
                                 "a0_unsafe_hold_rejected_by_a6",
                                 "a0_only_drivable", "a6_only_drivable",
                                 "both_invalid")}
    safety = {name: {key: 0 for key in ("road_outside", "clearance_violation",
                                                "steering_over_22", "path_jump",
                                                "steering_sign_error", "invalid_nonzero",
                                                "raw_road_absent_drivable")}
              for name in planners}
    reasons = {"recovered": {}, "regressed": {}}
    special = {"recovered": [], "regressed": []}
    processing = {name: [] for name in planners}; expected_errors = []
    quantized_errors = []; digests = {name: hashlib.sha256() for name in planners}
    previous_near = {name: None for name in planners}; fields = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", newline="") as stream:
        writer = None; frame_index = 0
        for chunk_path in sorted(cache.glob("chunk_*.npz")):
            with np.load(chunk_path) as chunk:
                for road, lane in zip(chunk["road"], chunk["lane"]):
                    timestamp = frame_index/float(timestamp_rate)
                    result_by = {}; metric_by = {}
                    for name, planner in planners.items():
                        started = time.perf_counter()
                        result = planner.plan(road, lane, timestamp)
                        elapsed = (time.perf_counter()-started)*1000.0
                        command = (controllers[name].command(
                            result.points, result.confidence,
                            result.state == "DEGRADED", timestamp)
                            if result.valid else controllers[name].neutral())
                        values = metrics(planner, result, command, elapsed)
                        result_by[name] = result; metric_by[name] = values
                        processing[name].append(elapsed)
                        digests[name].update(result.state.encode())
                        digests[name].update(values["path_signature"])
                        digests[name].update(str(values["wheel_command"]).encode())
                        valid = result.valid
                        safety[name]["road_outside"] += int(valid and not values["road_containment"])
                        safety[name]["clearance_violation"] += int(valid and not values["clearance_ok"])
                        safety[name]["steering_over_22"] += int(abs(values["required_steering_deg"]) > 22 if valid else False)
                        sign_error = (valid and values["wheel_command"] != 0 and
                                      values["required_steering_deg"]*values["wheel_command"] >= 0)
                        safety[name]["steering_sign_error"] += int(sign_error)
                        safety[name]["invalid_nonzero"] += int(not valid and
                            (values["wheel_command"] != 0 or values["drive_command"] != 0))
                        safety[name]["raw_road_absent_drivable"] += int(not np.any(road) and valid)
                        near = values["near_y"] if valid else None
                        if near is not None and previous_near[name] is not None:
                            safety[name]["path_jump"] += int(abs(near-previous_near[name]) > config.temporal_lateral_gate_m)
                        previous_near[name] = near
                    a0, a6 = result_by["a0"].valid, result_by["a6"].valid
                    a6_reason = metric_by["a6"]["reason"]
                    category = ("both_drivable" if a0 and a6 else
                                "a0_unsafe_hold_rejected_by_a6"
                                if a0 and "HOLD_PATH_UNSAFE" in a6_reason else
                                "a0_only_drivable" if a0 else
                                "a6_only_drivable" if a6 else "both_invalid")
                    counts[category] += 1
                    if category == "a6_only_drivable":
                        special["recovered"].append(frame_index); key = metric_by["a6"]["reason"] or "OK"
                        reasons["recovered"][key] = reasons["recovered"].get(key, 0)+1
                    if category == "a0_only_drivable":
                        special["regressed"].append(frame_index); key = metric_by["a6"]["reason"] or "OK"
                        reasons["regressed"][key] = reasons["regressed"].get(key, 0)+1
                    if a6 and abs(metric_by["a6"]["local_target_curvature_per_m"]) > .01:
                        expected_errors.append(metric_by["a6"]["steering_error_deg"])
                        if math.isfinite(metric_by["a6"]["quantized_target_lateral_error_m"]):
                            quantized_errors.append(metric_by["a6"]["quantized_target_lateral_error_m"])
                    row = {"frame_index": frame_index,
                           "stamp_ns": int(round(frame_index/fps*1e9)),
                           "input_timestamp_sec": timestamp,
                           "same_road_mask_crc32": zlib.crc32(road.tobytes()),
                           "same_lane_mask_crc32": zlib.crc32(lane.tobytes()),
                           "category": category}
                    row.update(prefixed("a0", metric_by["a0"])); row.update(prefixed("a6", metric_by["a6"]))
                    if writer is None:
                        fields = list(row); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                    writer.writerow(row); frame_index += 1
    def distribution(values):
        return {"mean": statistics.fmean(values) if values else 0.0,
                "p50": float(np.percentile(values, 50)) if values else 0.0,
                "p95": float(np.percentile(values, 95)) if values else 0.0,
                "max_abs": max((abs(v) for v in values), default=0.0)}
    return {"frames": frame_index, "source_fps": fps,
            "timestamp_rate_fps": timestamp_rate, "sets": counts,
            "reasons": reasons, "special_frames": special,
            "safety": safety,
            "processing_ms": {name: distribution(values) for name, values in processing.items()},
            "turning_expected_minus_required_error_deg": distribution(expected_errors),
            "quantized_target_lateral_error_m": distribution(quantized_errors),
            "deterministic_digest": {name: value.hexdigest() for name, value in digests.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--timestamp-rate", type=float, default=60.0)
    args = parser.parse_args()
    summary = run(args.cache, args.output/"shadow_ab_frames.csv.gz",
                  args.fps, args.timestamp_rate)
    (args.output/"shadow_ab_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "special_frames"}, indent=2))


if __name__ == "__main__": main()
