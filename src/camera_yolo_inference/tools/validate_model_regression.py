#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def lines(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.validation_dir)
    diagnostics = lines(root/"diagnostics.jsonl")
    controller = lines(root/"controller_diagnostics.jsonl")
    summary = json.loads((root/"measurement_summary.json").read_text())
    frames = len(diagnostics)
    road = sum(int((row.get("raw_road_pixels") or 0) > 0) for row in diagnostics)
    paths = sum(int((row.get("path_point_count") or 0) > 0) for row in diagnostics)
    mismatch = sum(row.get("reason") == "STAMP_MISMATCH" for row in controller)
    cross_frame = sum(bool(row.get("pair_exact") is False and
                           row.get("reason") not in ("PAIR_TIMEOUT", "PATH_INVALID"))
                      for row in controller)
    states = Counter(row.get("state", "UNKNOWN") for row in diagnostics)
    wheels = summary.get("camera_wheel", {})
    drives = summary.get("camera_drive", {})
    log_text = "\n".join(path.read_text(errors="replace") for path in root.glob("*.log"))
    checks = {
        "road_detection_ratio": frames > 0 and road/frames >= 0.90,
        "path_generation_ratio": frames > 0 and paths/frames >= 0.50,
        "stamp_mismatch_ratio": paths > 0 and mismatch/paths <= 0.01,
        "different_frame_pairs": cross_frame == 0,
        "both_wheel_signs": wheels.get("min", 0) < 0 < wheels.get("max", 0),
        "wheel_limit": -27 <= wheels.get("min", -999) <= wheels.get("max", 999) <= 27,
        "invalid_drive": drives.get("invalid_nonzero_samples", -1) == 0,
        "invalid_wheel": wheels.get("invalid_nonzero_samples", -1) == 0,
        "finite": finite_tree(diagnostics) and finite_tree(controller),
        "overlay": summary.get("overlay_messages_recorded", 0) > 0,
        "node_crash": not any(token in log_text for token in (
            "Traceback (most recent call last)", "Segmentation fault", "process has died")),
    }
    report = {
        "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "frames": frames, "road_frames": road,
        "road_detection_ratio": road/max(1, frames), "path_frames": paths,
        "path_generation_ratio": paths/max(1, frames), "states": dict(states),
        "stamp_mismatch_count": mismatch,
        "stamp_mismatch_per_path": mismatch/max(1, paths),
        "different_source_frame_pairs": cross_frame,
        "camera_drive": drives, "camera_wheel": wheels,
        "overlay_messages": summary.get("overlay_messages_recorded", 0),
    }
    destination = Path(args.output or root/"regression_report.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
