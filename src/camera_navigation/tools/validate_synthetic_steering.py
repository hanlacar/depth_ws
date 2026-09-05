#!/usr/bin/env python3
"""Metric-path steering validation without perception or vehicle actuation."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from camera_navigation.direct_bev_controller import DirectBevController
from evaluate_steering_candidates import configs


def circle(radius, sign, first=0.30):
    x = np.linspace(first, min(7.5, radius*0.80), 100)
    y = sign*(radius-np.sqrt(radius**2-x**2))
    return np.column_stack((x, y))


def scenarios():
    result = [("straight", np.column_stack((np.linspace(.3, 7.5, 80),
                                             np.zeros(80))), 0.0)]
    for sign, label in ((1, "left"), (-1, "right")):
        for radius in (3.0, 5.0, 10.0, 20.0):
            expected = sign*math.degrees(math.atan(0.73/radius))
            result.append((f"{label}_R{radius:g}", circle(radius, sign), expected))
    x = np.linspace(.3, 7.5, 100)
    result.extend([
        ("s_curve", np.column_stack((x, .20*np.sin((x-1.0)*.75))), None),
        ("gradual_curvature", np.column_stack((x, .003*x**3)), None),
        ("curvature_sign_change", np.column_stack((x, .012*(x-3.8)**3)), None),
        ("short_path", circle(10.0, 1)[:5], math.degrees(math.atan(.73/10))),
    ])
    for first in (2.5, 3.0, 4.0, 5.0):
        result.append((f"first_observed_{first:.1f}m", circle(10.0, 1, first),
                       math.degrees(math.atan(.73/10))))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate, config in configs().items():
        for index, (name, path, expected) in enumerate(scenarios()):
            command = DirectBevController(config).command(path, 1.0, False,
                                                          index+1.0)
            required = float(command.get("required_steering_deg", 0.0))
            wheel = int(command.get("wheel", 0))
            rows.append({"candidate": candidate, "scenario": name,
                         "expected_ackermann_deg": expected,
                         "required_steering_deg": required,
                         "camera_wheel": wheel,
                         "valid": command.get("valid", False),
                         "within_22": abs(wheel) <= 22,
                         "sign_ok": (wheel == 0 if expected == 0 else
                                     True if expected is None else
                                     wheel*expected <= 0),
                         "absolute_error_deg": (abs(required-expected)
                                                if expected is not None else None)})
    with (args.output/"synthetic_steering_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for candidate in configs():
        subset = [row for row in rows if row["candidate"] == candidate]
        numeric = [row["absolute_error_deg"] for row in subset
                   if row["absolute_error_deg"] is not None]
        summary[candidate] = {
            "cases": len(subset), "valid": sum(row["valid"] for row in subset),
            "sign_errors": sum(not row["sign_ok"] for row in subset),
            "over_22": sum(not row["within_22"] for row in subset),
            "ackermann_error_mean_deg": float(np.mean(numeric)),
            "ackermann_error_max_deg": float(np.max(numeric)),
        }
    (args.output/"synthetic_steering_summary.json").write_text(
        json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
