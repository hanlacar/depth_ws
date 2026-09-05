#!/usr/bin/env python3
"""Compare opt-in hybrid_a6 steering candidates on one semantic cache."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from camera_navigation.direct_bev_controller import (
    BevControllerConfig, DirectBevController)
from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import ablation_planners


def configs():
    common = dict(lookahead_from_path_start=True, steering_lineage_enabled=True)
    adaptive = dict(common, curvature_adaptive_lookahead=True,
                    curvature_scale_per_m=0.50, lookahead_min_m=1.80)
    feedforward = dict(adaptive, feedforward_weight=0.02,
                       feedforward_max_delta_deg=4.0)
    return {
        "S0": BevControllerConfig(**common),
        "S1": BevControllerConfig(**adaptive),
        "S2": BevControllerConfig(**feedforward),
        "S3": BevControllerConfig(**feedforward,
                                  fractional_accumulator=True),
        "S4": BevControllerConfig(**feedforward, steering_gain=1.05),
    }


def arc_y(x, steering_deg, wheelbase):
    curvature = math.tan(math.radians(steering_deg))/wheelbase
    if abs(curvature) < 1.0e-12:
        return 0.0
    value = 1.0-(curvature*x)**2
    if value < 0.0:
        return math.nan
    return float((1.0-math.sqrt(value))/curvature)


def distribution(values):
    finite = np.asarray([v for v in values if math.isfinite(v)], float)
    if not len(finite):
        return {"count": 0, "mean": 0.0, "p95_abs": 0.0,
                "max_abs": 0.0}
    return {"count": len(finite), "mean": float(np.mean(finite)),
            "p95_abs": float(np.percentile(np.abs(finite), 95)),
            "max_abs": float(np.max(np.abs(finite)))}


def swept_safe(planner, result, internal_steering):
    points = np.asarray(result.points, float).reshape(-1, 2)
    if not len(points):
        return True
    x = np.linspace(max(planner.config.x_min_m, float(points[0, 0])),
                    float(points[-1, 0]), 80)
    curvature = math.tan(math.radians(internal_steering))/planner.config.wheelbase_m
    values = 1.0-(curvature*x)**2
    if np.any(values < 0.0):
        return False
    if abs(curvature) < 1.0e-12:
        y = np.zeros_like(x); heading = np.zeros_like(x)
    else:
        y = (1.0-np.sqrt(values))/curvature
        heading = np.arcsin(curvature*x)
    half = planner.config.vehicle_width_m/2.0+planner.config.lateral_safety_margin_m
    samples = np.vstack((np.column_stack((x, y)),
                         np.column_stack((x, y+half*np.cos(heading))),
                         np.column_stack((x, y-half*np.cos(heading)))))
    grid = planner.metric_to_grid(samples)
    inside = ((grid[:, 0] >= 0) & (grid[:, 0] < planner.rows) &
              (grid[:, 1] >= 0) & (grid[:, 1] < planner.cols))
    return bool(np.all(inside) and
                np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0))


def run(cache, segments_path, output, fps):
    output.mkdir(parents=True, exist_ok=True)
    turn_labels = {}
    if segments_path:
        for segment in json.loads(segments_path.read_text())["segments"]:
            for index in range(segment["start_frame"], segment["end_frame"]+1):
                turn_labels[index] = segment["class"]
    planner = ablation_planners(DirectBevConfig())["A6"]
    controllers = {name: DirectBevController(cfg)
                   for name, cfg in configs().items()}
    values = {name: {key: [] for key in (
        "expected_error", "quantization_error", "lateral_error",
        "wheel_delta", "processing_ms")} for name in controllers}
    counts = {name: {key: 0 for key in (
        "drivable", "nonzero", "sign_error", "over_22", "swept_violation",
        "straight_nonzero", "wheel_sign_flip", "invalid_nonzero")}
              for name in controllers}
    previous_wheel = {name: 0 for name in controllers}
    previous_nonzero_sign = {name: 0 for name in controllers}
    rows = []
    frame = 0
    for chunk_path in sorted(cache.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            for road, lane in zip(chunk["road"], chunk["lane"]):
                result = planner.plan(road, lane, frame/fps)
                label = turn_labels.get(frame, "other")
                base = {"frame_index": frame, "stamp_ns": int(frame/fps*1e9),
                        "turn_class": label, "state": result.state,
                        "reason": "|".join(result.diagnostics.get("reasons", []))}
                for name, controller in controllers.items():
                    started = time.perf_counter()
                    command = (controller.command(
                        result.points, result.confidence,
                        result.state == "DEGRADED", frame/fps)
                        if result.valid else controller.neutral())
                    elapsed = (time.perf_counter()-started)*1000.0
                    values[name]["processing_ms"].append(elapsed)
                    valid = bool(result.valid and command.get("valid"))
                    wheel = int(command.get("wheel", 0))
                    required = float(command.get("required_steering_deg", 0.0))
                    expected = float(command.get("feedforward_steering_deg", 0.0))
                    target = command.get("target_point")
                    lateral = math.nan
                    if target:
                        predicted = arc_y(float(target[0]), -wheel,
                                          planner.config.wheelbase_m)
                        if math.isfinite(predicted):
                            lateral = predicted-float(target[1])
                    counts[name]["drivable"] += int(valid)
                    counts[name]["nonzero"] += int(valid and wheel != 0)
                    counts[name]["sign_error"] += int(
                        valid and wheel != 0 and required*wheel >= 0.0)
                    counts[name]["over_22"] += int(abs(wheel) > 22 or
                        (valid and abs(required) > 22.0))
                    counts[name]["invalid_nonzero"] += int(not valid and wheel != 0)
                    if valid and label in ("left", "right"):
                        values[name]["expected_error"].append(required-expected)
                        values[name]["quantization_error"].append(
                            wheel-float(command.get("sign_converted_steering_deg", 0.0)))
                        values[name]["lateral_error"].append(lateral)
                        counts[name]["swept_violation"] += int(not swept_safe(
                            planner, result, -float(wheel)))
                    if valid and label == "straight":
                        counts[name]["straight_nonzero"] += int(wheel != 0)
                    delta = wheel-previous_wheel[name]
                    values[name]["wheel_delta"].append(delta)
                    sign = int(np.sign(wheel))
                    if sign and previous_nonzero_sign[name] and sign != previous_nonzero_sign[name]:
                        counts[name]["wheel_sign_flip"] += 1
                    if sign:
                        previous_nonzero_sign[name] = sign
                    previous_wheel[name] = wheel
                    for key in ("lookahead_m", "local_curvature_per_m",
                                "feedforward_steering_deg", "pure_pursuit_raw_deg",
                                "required_steering_deg", "controller_input_steering_deg",
                                "sign_converted_steering_deg",
                                "temporal_filtered_steering_deg",
                                "clamped_steering_deg", "fractional_residual_deg",
                                "rounded_int32_wheel"):
                        base[f"{name}_{key}"] = command.get(key, math.nan)
                    base[f"{name}_target_point"] = json.dumps(
                        target or [], separators=(",", ":"))
                    base[f"{name}_published_camera_wheel"] = wheel
                    base[f"{name}_target_lateral_error_m"] = lateral
                    base[f"{name}_processing_ms"] = elapsed
                rows.append(base); frame += 1
    with (output/"steering_candidate_lineage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"frames": frame, "wheel_contract": {
        "internal_required": "left_positive_right_negative",
        "camera_wheel": "right_positive_left_negative", "steering_sign": -1.0},
        "candidates": {}}
    for name in controllers:
        summary["candidates"][name] = dict(counts[name])
        summary["candidates"][name].update({
            "drivable_percent": 100.0*counts[name]["drivable"]/frame,
            "expected_minus_required_deg": distribution(values[name]["expected_error"]),
            "required_to_int32_error_deg": distribution(values[name]["quantization_error"]),
            "target_lateral_error_m": distribution(values[name]["lateral_error"]),
            "frame_wheel_delta_deg": distribution(values[name]["wheel_delta"]),
            "processing_ms": distribution(values[name]["processing_ms"]),
        })
    (output/"steering_candidate_summary.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    run(args.cache, args.segments, args.output, args.fps)


if __name__ == "__main__":
    main()
