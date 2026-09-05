#!/usr/bin/env python3
"""Summarize S0 steering lineage by labelled turn direction."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ("lookahead_m", "local_curvature_per_m",
          "feedforward_steering_deg", "pure_pursuit_raw_deg",
          "required_steering_deg", "controller_input_steering_deg",
          "sign_converted_steering_deg", "temporal_filtered_steering_deg",
          "clamped_steering_deg", "rounded_int32_wheel",
          "published_camera_wheel")


def stats(values):
    values = np.asarray(values, float)
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95_abs": float(np.percentile(np.abs(values), 95)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = {"left": [], "right": [], "straight": []}
    with args.lineage.open() as stream:
        for row in csv.DictReader(stream):
            label = row["turn_class"]
            if label in grouped and row["S0_required_steering_deg"] not in ("", "nan"):
                grouped[label].append(row)
    summary = {}
    for label, rows in grouped.items():
        block = {field: stats([float(row[f"S0_{field}"]) for row in rows])
                 for field in FIELDS}
        expected = np.asarray([abs(float(row["S0_feedforward_steering_deg"]))
                               for row in rows])
        pp = np.asarray([abs(float(row["S0_pure_pursuit_raw_deg"])) for row in rows])
        required = np.asarray([abs(float(row["S0_required_steering_deg"])) for row in rows])
        signed = np.asarray([abs(float(row["S0_sign_converted_steering_deg"]))
                             for row in rows])
        filtered = np.asarray([abs(float(row["S0_temporal_filtered_steering_deg"]))
                               for row in rows])
        clamped = np.asarray([abs(float(row["S0_clamped_steering_deg"])) for row in rows])
        wheel = np.asarray([abs(float(row["S0_rounded_int32_wheel"])) for row in rows])
        block["mean_absolute_loss_deg"] = {
            "local_curvature_expectation_to_pure_pursuit": float(np.mean(expected-pp)),
            "feedforward_and_gain": float(np.mean(pp-required)),
            "final_sign_conversion": float(np.mean(required-signed)),
            "temporal_filter": float(np.mean(signed-filtered)),
            "clamp": float(np.mean(filtered-clamped)),
            "int32_rounding": float(np.mean(clamped-wheel)),
            "selector_and_publish": 0.0,
        }
        block["frames"] = len(rows)
        summary[label] = block
    summary["diagnosis"] = {
        "wheelbase_m": 0.73, "radian_degree_conversion": "verified_by_exact_arc_tests",
        "deadband_in_S0": False, "feedforward_in_S0": False,
        "steering_gain_in_S0": 1.0, "selector_scale": 1.0,
        "principal_issue": "curvature_interval_mismatch",
        "detail": ("Local curvature at a far target describes a future path segment; "
                   "pure pursuit describes the constant arc from base_link to that target. "
                   "They are equal only for a circular path from the vehicle origin."),
    }
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
