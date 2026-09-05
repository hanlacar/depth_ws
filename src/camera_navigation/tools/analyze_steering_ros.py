#!/usr/bin/env python3
"""Summarize one full ROS A6 recorder directory."""

import argparse
import json
from pathlib import Path

import numpy as np


def records(path):
    output = []
    if not path.exists(): return output
    for line in path.read_text().splitlines():
        try: output.append(json.loads(line))
        except (TypeError, ValueError): pass
    return output


def percentile(values, q):
    values = [float(value) for value in values if value is not None]
    return float(np.percentile(values, q)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("directory", type=Path)
    args = parser.parse_args(); directory = args.directory
    diagnostics = records(directory/"diagnostics.jsonl")
    controller = records(directory/"controller_diagnostics.jsonl")
    drive = records(directory/"drive_diagnostics.jsonl")
    paths = records(directory/"paths.jsonl")
    valid = [row for row in diagnostics if row.get("state") in ("VALID", "DEGRADED")]
    required = [float(row.get("required_steering_deg", 0.0) or 0.0) for row in valid]
    wheels = [int(row.get("wheel", 0)) for row in controller]
    steering = [float(row.get("steering_deg", 0.0)) for row in controller]
    nonzero_signs = np.sign([value for value in wheels if value])
    path_jumps = 0
    previous = None
    for item in paths:
        points = np.asarray(item.get("points", []), float).reshape(-1, 2)
        if len(points):
            # Match the commissioned temporal-lateral gate: compare the path
            # at configured near_required_m (1.5 m), not the naturally moving far tail.
            signature = float(np.interp(1.5, points[:, 0], points[:, 1]))
            if previous is not None and abs(signature-previous) > .35: path_jumps += 1
            previous = signature
    output = {
        "semantic_frames": len(diagnostics), "path_messages": len(paths),
        "state_counts": {state: sum(row.get("state") == state for row in diagnostics)
                         for state in ("VALID", "DEGRADED", "INVALID")},
        "drivable_ratio": len(valid)/max(1, len(diagnostics)),
        "required_steering_min_deg": min(required, default=0.0),
        "required_steering_max_deg": max(required, default=0.0),
        "camera_wheel_samples": len(wheels), "camera_wheel_min": min(wheels, default=0),
        "camera_wheel_max": max(wheels, default=0),
        "camera_wheel_nonzero": sum(value != 0 for value in wheels),
        "steering_over_22": sum(abs(value) > 22 for value in wheels),
        "maximum_frame_steering_jump_deg": (float(np.max(np.abs(np.diff(steering))))
                                             if len(steering) > 1 else 0.0),
        "steering_sign_reversals": (int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
                                     if len(nonzero_signs) > 1 else 0),
        "road_outside_paths": sum(float(row.get("safe_road_coverage", 0.0)) < .999 for row in valid),
        "clearance_violations": sum(float(row.get("minimum_clearance_m", 0.0)) < .52-1e-6 for row in valid),
        "near_path_jumps_over_0_35m": path_jumps,
        "TEMPORAL_CURVATURE_JUMP": sum("TEMPORAL_CURVATURE_JUMP" in row.get("reasons", []) for row in diagnostics),
        "invalid_or_stale_nonzero_drive": sum((row.get("stale") or row.get("state") not in ("VALID", "DEGRADED")) and abs(float(row.get("drive", 0))) > 1e-9 for row in drive),
        "invalid_or_stale_nonzero_wheel": sum(row.get("reason") in ("PATH_INVALID", "PATH_TIMEOUT", "STAMP_MISMATCH", "PATH_OR_STATUS_MISSING") and int(row.get("wheel", 0)) != 0 for row in controller),
        "planner_processing_fps_median": percentile([r.get("planner_processing_fps") for r in diagnostics], 50),
        "semantic_input_fps_median": percentile([r.get("semantic_input_fps") for r in diagnostics], 50),
        "planner_processing_ms_p50": percentile([r.get("processing_ms") for r in diagnostics], 50),
        "planner_processing_ms_p95": percentile([r.get("processing_ms") for r in diagnostics], 95),
        "end_to_end_latency_ms_p50": percentile([r.get("end_to_end_latency_ms") for r in diagnostics], 50),
        "end_to_end_latency_ms_p95": percentile([r.get("end_to_end_latency_ms") for r in diagnostics], 95),
    }
    (directory/"extended_summary.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
