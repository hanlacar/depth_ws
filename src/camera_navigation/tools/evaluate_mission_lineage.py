#!/usr/bin/env python3
"""Replay cached unquantized steering through the advisory mission core."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from camera_navigation.mission_decision_core import (
    MissionDecisionMachine, MissionInputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage_csv")
    parser.add_argument("--planner", default="A0")
    parser.add_argument("--sign-arm-frame", type=int, required=True)
    parser.add_argument("--output", default="mission_lineage_audit.json")
    args = parser.parse_args()
    machine = MissionDecisionMachine(); counts = Counter()
    transitions = []; previous = None; total = safety_nonzero = 0
    with Path(args.lineage_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("planner") != args.planner:
                continue
            frame = int(row["frame_index"]); now = float(row["timestamp_sec"])
            try:
                # Cached planner geometry is left-positive; mission contract
                # and physical wheel are right-positive.
                steering = -float(row["required_steering_deg"])
            except (TypeError, ValueError):
                steering = math.nan
            planner_state = row.get("state", "INVALID")
            planner_valid = planner_state in ("VALID", "DEGRADED", "HOLD")
            result = machine.update(MissionInputs(
                now=now, stamp=now, section="ACCELERATION", input_fresh=True,
                planner_valid=planner_valid, planner_state=planner_state,
                planner_drive=float(row.get("camera_drive_output") or 0.0),
                sign_detected=frame >= args.sign_arm_frame,
                required_steering_deg=steering))
            state = result.diagnostics.get("mission_state_before_safety",
                                           result.state)
            counts[state] += 1; total += 1
            if (result.safety_blocked and
                    abs(result.effective_drive) > 1.0e-6):
                safety_nonzero += 1
            if state != previous:
                transitions.append({"frame": frame, "timestamp_sec": now,
                                    "state": state,
                                    "required_steering_deg": (
                                        steering if math.isfinite(steering)
                                        else None),
                                    "drive_override": result.drive_override,
                                    "failure_reason": result.failure_reason})
                previous = state
    report = {"planner": args.planner, "frames": total,
              "sign_arm_frame": args.sign_arm_frame,
              "state_counts": dict(counts), "transitions": transitions,
              "safety_blocked_nonzero_effective": safety_nonzero,
              "note": ("Sign arm frame is from the ROS full-loop observation; "
                       "cached steering is reused without YOLO inference.")}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
