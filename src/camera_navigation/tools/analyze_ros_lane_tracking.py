#!/usr/bin/env python3
"""Aggregate stamp-aware ROS BEV, tracking, and command validation logs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(values):
    return [float(value) for value in values
            if value is not None and np.isfinite(float(value))]


def distribution(values):
    values = finite(values)
    return ({"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            if not values else
            {"count": len(values), "mean": statistics.fmean(values),
             "p50": float(np.percentile(values, 50)),
             "p95": float(np.percentile(values, 95)), "max": max(values)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = read_jsonl(args.run/"diagnostics.jsonl")
    paths = {int(item["stamp_ns"]): item["points"]
             for item in read_jsonl(args.run/"paths.jsonl")}
    drive = read_jsonl(args.run/"drive_diagnostics.jsonl")
    controller = read_jsonl(args.run/"controller_diagnostics.jsonl")
    measurement = json.loads((args.run/"measurement_summary.json").read_text())
    states = Counter(item.get("state", "UNKNOWN") for item in diagnostics)
    reasons = Counter(reason for item in diagnostics
                      for reason in item.get("reasons", []))
    drivable = [item for item in diagnostics if item.get("path_valid")]
    tracking_sources = {lane: Counter() for lane in ("white_line", "yellow_line")}
    reset_reasons = Counter()
    scene_changes = scene_residuals = unsafe_track = conflicts = 0
    conflict_discards = 0
    timestamp_deltas = []
    for item in diagnostics:
        temporal = item.get("refinement", {}).get("lane_temporal", {})
        if not temporal:
            continue
        reset_reasons[temporal.get("reset_reason") or "NONE"] += 1
        delta = temporal.get("timestamp_delta_sec")
        if delta is not None:
            timestamp_deltas.append(delta)
        changed = bool(temporal.get("scene_change_detected"))
        scene_changes += changed
        tracked_on_change = False
        for lane in ("white_line", "yellow_line"):
            source = temporal.get(f"{lane}_source", "NONE")
            tracking_sources[lane][source] += 1
            detail = temporal.get(lane, {})
            if source == "TRACKED":
                tracked_on_change = True
                unsafe_track += float(detail.get("road_overlap", 0.0)) < .65
                conflicts += float(detail.get("opposite_raw_iou", 0.0)) > .15
            conflict_discards += detail.get("discard_reason") == "OPPOSITE_CLASS_CONFLICT"
        scene_residuals += changed and tracked_on_change

    path_jumps = 0
    previous_near = None
    signs = []
    missing_path_payload = 0
    for item in diagnostics:
        if not item.get("path_valid"):
            previous_near = None
            continue
        points = np.asarray(paths.get(int(item.get("stamp_ns", -1)), []), float).reshape(-1, 2)
        if not len(points):
            missing_path_payload += 1
            previous_near = None
        else:
            near = float(np.interp(1.5, points[:, 0], points[:, 1]))
            if previous_near is not None:
                path_jumps += abs(near-previous_near) > .65
            previous_near = near
        angle = item.get("required_steering_deg")
        if angle is not None and np.isfinite(float(angle)):
            signs.append(0 if abs(float(angle)) <= .25 else int(np.sign(float(angle))))
    reversals = sum(a and b and a != b for a, b in zip(signs, signs[1:]))
    drive_invalid_nonzero = sum(
        item.get("state") == "INVALID" and abs(float(item.get("drive", 0.0))) > 1e-9
        for item in drive)
    drive_stale_nonzero = sum(
        item.get("stale") and abs(float(item.get("drive", 0.0))) > 1e-9
        for item in drive)
    controller_wheels = [int(item.get("wheel", 0)) for item in controller]
    summary = {
        "semantic_frames": len(diagnostics),
        "path_payload_frames": len(paths),
        "missing_path_payload_for_drivable": missing_path_payload,
        "state_counts": dict(states),
        "drivable_frames": len(drivable),
        "drivable_ratio": len(drivable)/max(1, len(diagnostics)),
        "failure_reasons": dict(reasons),
        "mean_path_length_m": (statistics.fmean(
            float(item.get("path_length_m", 0.0)) for item in drivable)
            if drivable else 0.0),
        "paths_lt_2m": sum(0 < float(item.get("path_length_m", 0.0)) < 2
                            for item in drivable),
        "road_outside_paths": sum(float(item.get("safe_road_coverage", 0.0)) < .999
                                  for item in drivable),
        "clearance_violations": sum(
            item.get("minimum_clearance_m") is not None and
            float(item["minimum_clearance_m"]) < .52-1e-6
            for item in drivable),
        "steering_over_22": sum(abs(float(item.get("required_steering_deg", 0.0))) > 22
                                for item in drivable),
        "required_steering": distribution(
            item.get("required_steering_deg") for item in drivable),
        "path_jumps": int(path_jumps),
        "steering_sign_reversals": int(reversals),
        "tracking_sources": {key: dict(value) for key, value in tracking_sources.items()},
        "tracking_reset_reasons": dict(reset_reasons),
        "scene_changes": int(scene_changes),
        "scene_change_residuals": int(scene_residuals),
        "unsafe_tracked_line_frames": int(unsafe_track),
        "opposite_class_conflicts": int(conflicts),
        "opposite_class_conflict_discards": int(conflict_discards),
        "timestamp_delta_sec": distribution(timestamp_deltas),
        "timestamp_gap_over_0_12": sum(float(value) > .12 for value in timestamp_deltas),
        "semantic_fps": measurement.get("observed_fps", {}).get("semantic", 0.0),
        "planner_path_fps": measurement.get("observed_fps", {}).get("path", 0.0),
        "end_to_end_latency_ms": distribution(
            item.get("end_to_end_latency_ms") for item in diagnostics),
        "planner_processing_ms": distribution(
            item.get("processing_ms") for item in diagnostics),
        "drive_command_samples": len(drive),
        "drive_counts": dict(Counter(str(item.get("drive")) for item in drive)),
        "drive_invalid_nonzero": int(drive_invalid_nonzero),
        "drive_stale_nonzero": int(drive_stale_nonzero),
        "controller_samples": len(controller),
        "wheel_min": min(controller_wheels, default=0),
        "wheel_max": max(controller_wheels, default=0),
        "wheel_nonzero_samples": sum(value != 0 for value in controller_wheels),
        "controller_reasons": dict(Counter(
            str(item.get("reason")) for item in controller)),
        "input_timeout_frames": reasons.get("INPUT_TIMEOUT", 0),
        "calibration_invalid_frames": reasons.get("CALIBRATION_INVALID", 0),
        "measurement_summary": measurement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
