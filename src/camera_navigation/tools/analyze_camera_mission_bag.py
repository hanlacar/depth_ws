#!/usr/bin/env python3
"""Analyze an advisory mission validation bag without replaying control."""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


TOPICS = (
    "/camera/mission/section",
    "/camera/mission/stop_line_distance_m",
    "/camera/mission/stop_line_distances_m",
    "/camera/mission/sign_detected",
    "/camera/mission/traffic_light",
    "/camera/mission/uphill_detected",
    "/camera/mission/diagnostics",
    "/camera/mission/decision_state",
    "/camera/mission/drive_override_active",
    "/camera/mission/drive_override",
    "/camera/mission/decision_diagnostics",
    "/mcu/encoder",
    "/mcu/speed_mps",
    "/mcu/distance_m",
    "/mcu/speed_valid",
)


def transitions(series):
    output = []
    marker = object(); previous = marker
    for timestamp, value in series:
        key = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        if key != previous:
            output.append({"timestamp_sec": timestamp, "value": value})
            previous = key
    return output


def nearest(series, timestamp, tolerance=0.25):
    if not series:
        return None
    times = [item[0] for item in series]
    index = bisect.bisect_left(times, timestamp)
    candidates = [i for i in (index-1, index) if 0 <= i < len(series)]
    best = min(candidates, key=lambda i: abs(times[i]-timestamp))
    return series[best] if abs(times[best]-timestamp) <= tolerance else None


def percentile(values, q):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return None
    index = (len(values)-1)*q/100.0
    low = int(math.floor(index)); high = int(math.ceil(index))
    return values[low]+(values[high]-values[low])*(index-low)


def read_bag(uri):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    data = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        if topic not in data:
            continue
        message = deserialize_message(raw, get_message(types[topic]))
        value = list(message.data) if hasattr(message.data, "__iter__") and not \
            isinstance(message.data, str) else message.data
        if topic.endswith("diagnostics"):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                value = {"decode_error": True, "raw": str(value)}
        data[topic].append((timestamp_ns*1.0e-9, value))
    return data, types


def load_measurements(path):
    if path is None:
        return []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze(data, measurements=()):
    states = data["/camera/mission/decision_state"]
    overrides = data["/camera/mission/drive_override"]
    active = data["/camera/mission/drive_override_active"]
    mission_diag = data["/camera/mission/diagnostics"]
    decision_diag = data["/camera/mission/decision_diagnostics"]
    invalid_stages = [v for _, v in overrides if float(v) not in (0, 1, 2, 3)]
    unsafe = [d for _, d in decision_diag if isinstance(d, dict) and
              d.get("safety_blocked") and
              abs(float(d.get("effective_drive_if_connected", 0.0))) > 1.0e-6]
    late_actions = [str(d.get("late_red_action", "NONE")) for _, d in
                    decision_diag if isinstance(d, dict)]
    errors = []
    traffic_delays = []
    for row in measurements:
        timestamp = float(row["timestamp_sec"])
        if row.get("measured_distance_m", "").strip():
            found = nearest(data["/camera/mission/stop_line_distance_m"], timestamp)
            if found is not None and math.isfinite(float(found[1])):
                errors.append(abs(float(found[1])-float(row["measured_distance_m"])))
        expected = row.get("traffic_truth", "").strip().upper()
        if expected:
            for when, observed in data["/camera/mission/traffic_light"]:
                if when >= timestamp and observed == expected:
                    traffic_delays.append(when-timestamp); break
    sign_raw_on = [t for t, d in mission_diag if isinstance(d, dict) and
                   d.get("sign_raw_detected")]
    sign_filtered_on = [t for t, value in data["/camera/mission/sign_detected"]
                        if value]
    sign_debounce = None
    if sign_raw_on and sign_filtered_on:
        sign_debounce = max(0.0, sign_filtered_on[0]-sign_raw_on[0])
    uphill_on_angles = [d.get("imu_relative_uphill_deg") for _, d in mission_diag
                        if isinstance(d, dict) and d.get("uphill_detected") and
                        d.get("imu_relative_uphill_deg") is not None]
    return {
        "topic_counts": {topic: len(series) for topic, series in data.items()},
        "decision_state_transitions": transitions(states),
        "section_transitions": transitions(data["/camera/mission/section"]),
        "traffic_light_transitions": transitions(
            data["/camera/mission/traffic_light"]),
        "override_active_transitions": transitions(active),
        "drive_override_transitions": transitions(overrides),
        "invalid_drive_stage_count": len(invalid_stages),
        "safety_blocked_nonzero_effective_count": len(unsafe),
        "late_red_action_counts": {
            action: late_actions.count(action) for action in sorted(set(late_actions))},
        "encoder_message_count": len(data["/mcu/encoder"]),
        "speed_message_count": len(data["/mcu/speed_mps"]),
        "distance_message_count": len(data["/mcu/distance_m"]),
        "speed_valid_true_count": sum(
            1 for _, value in data["/mcu/speed_valid"] if value),
        "failure_reasons": {reason: sum(
            1 for _, d in decision_diag if isinstance(d, dict) and
            d.get("failure_reason") == reason) for reason in sorted({
                d.get("failure_reason") for _, d in decision_diag
                if isinstance(d, dict) and d.get("failure_reason")})},
        "stop_line_distance_error_m": {
            "count": len(errors),
            "mean": sum(errors)/len(errors) if errors else None,
            "p95": percentile(errors, 95), "max": max(errors) if errors else None,
        },
        "traffic_transition_delay_sec": {
            "count": len(traffic_delays),
            "mean": (sum(traffic_delays)/len(traffic_delays)
                     if traffic_delays else None),
            "p95": percentile(traffic_delays, 95),
        },
        "sign_debounce_sec": sign_debounce,
        "uphill_on_relative_angle_deg": {
            "count": len(uphill_on_angles),
            "min": min(uphill_on_angles) if uphill_on_angles else None,
            "max": max(uphill_on_angles) if uphill_on_angles else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--measurements-csv", help=(
        "optional timestamp_sec,measured_distance_m,traffic_truth CSV"))
    parser.add_argument("--output", default="mission_validation_report.json")
    args = parser.parse_args()
    data, types = read_bag(args.bag)
    report = analyze(data, load_measurements(args.measurements_csv))
    report["topic_types"] = types
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
