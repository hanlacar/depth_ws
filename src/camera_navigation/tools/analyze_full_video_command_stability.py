#!/usr/bin/env python3
"""Quantify A6 frame-level command stability from deterministic lineage CSV."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def bounded_runs(values, predicate):
    runs = []
    start = None
    for index, value in enumerate(values + [None]):
        selected = index < len(values) and predicate(value)
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            runs.append((start, index - 1, index - start))
            start = None
    return runs


def analyze(rows, source_fps):
    wheel = np.asarray([int(row["camera_wheel_output"]) for row in rows], float)
    drive = np.asarray([finite(row["camera_drive_output"]) for row in rows], float)
    states = [row["state"] for row in rows]
    classes = [row["visual_class"] for row in rows]
    curvature = np.asarray([
        finite(row.get("path_curvature_per_m"), float("nan")) for row in rows])
    delta = np.abs(np.diff(wheel))
    nonzero_signs = np.sign(wheel[wheel != 0])
    sign_reversals = int(np.count_nonzero(np.diff(nonzero_signs) != 0))
    straight = np.asarray([name == "straight" for name in classes])
    left = np.asarray([name == "left" for name in classes])
    right = np.asarray([name == "right" for name in classes])
    expected_drive = np.asarray([
        2.0 if state == "VALID" else 1.0 if state == "DEGRADED" else 0.0
        for state in states])
    nonzero_runs = bounded_runs(drive.tolist(), lambda value: value != 0.0)
    zero_runs = bounded_runs(drive.tolist(), lambda value: value == 0.0)
    short_drive_pulses = [run for run in nonzero_runs if run[2] <= 2 and
                          run[0] > 0 and run[1] < len(drive)-1]
    short_zero_flickers = [run for run in zero_runs if run[2] <= 2 and
                           run[0] > 0 and run[1] < len(drive)-1 and
                           drive[run[0]-1] != 0 and drive[run[1]+1] != 0]
    invalid_nonzero = sum(
        states[index] == "INVALID" and (drive[index] != 0 or wheel[index] != 0)
        for index in range(len(rows)))
    stop_delays = []
    for index in range(1, len(rows)):
        if states[index] == "INVALID" and states[index-1] != "INVALID":
            cursor = index
            while cursor < len(rows) and drive[cursor] != 0:
                cursor += 1
            stop_delays.append(cursor-index)
    stable_curve_jump = int(np.count_nonzero(
        np.isfinite(curvature[1:]) & np.isfinite(curvature[:-1]) &
        (np.abs(np.diff(curvature)) <= 0.02) & (delta >= 2.0)))
    duration = len(rows)/float(source_fps)

    def sign_accuracy(mask, correct):
        values = wheel[mask & (wheel != 0)]
        return {
            "labelled_frames": int(mask.sum()),
            "nonzero_frames": int(values.size),
            "correct_nonzero": int(np.count_nonzero(correct(values))),
            "nonzero_sign_accuracy": (None if not values.size else
                                       float(np.mean(correct(values)))),
        }

    return {
        "frames": len(rows), "duration_sec": duration,
        "frame_index_min": int(rows[0]["frame_index"]),
        "frame_index_max": int(rows[-1]["frame_index"]),
        "state_counts": Counter(states),
        "failure_reason_counts": Counter(row["failure_reason"] for row in rows),
        "wheel": {
            "mean": float(wheel.mean()), "std": float(wheel.std()),
            "min": int(wheel.min()), "max": int(wheel.max()),
            "nonzero_frames": int(np.count_nonzero(wheel)),
            "delta_p95": float(np.percentile(delta, 95)),
            "delta_max": int(delta.max()),
            "sign_reversals": sign_reversals,
            "sign_reversals_per_sec": sign_reversals/duration,
            "limit_violations": int(np.count_nonzero(np.abs(wheel) > 22)),
            "straight_nonzero_ratio": (float(np.mean(wheel[straight] != 0))
                                       if straight.any() else None),
            "left": sign_accuracy(left, lambda values: values < 0),
            "right": sign_accuracy(right, lambda values: values > 0),
            "stable_curvature_jumps_ge_2deg": stable_curve_jump,
        },
        "drive": {
            "value_counts": Counter(drive.tolist()),
            "state_policy_mismatches": int(np.count_nonzero(drive != expected_drive)),
            "transitions": int(np.count_nonzero(np.diff(drive) != 0)),
            "short_nonzero_pulses_1_2_frames": len(short_drive_pulses),
            "short_zero_flickers_1_2_frames": len(short_zero_flickers),
            "invalid_entry_stop_delay_frames_mean": (
                None if not stop_delays else float(np.mean(stop_delays))),
            "invalid_entry_stop_delay_frames_max": (
                None if not stop_delays else int(max(stop_delays))),
        },
        "safety": {"invalid_nonzero_frames": int(invalid_nonzero)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--planner", default="A6")
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.lineage.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("planner") == args.planner]
    if not rows:
        raise SystemExit(f"no rows for planner={args.planner}")
    document = analyze(rows, args.source_fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
