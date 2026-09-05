#!/usr/bin/env python3
"""Analyze a marked hybrid_a6 low-speed vehicle rosbag.

The path metric is causal: a target published at time t is transformed into
odom, then compared with the future odom pose after the vehicle has travelled
the target's arc distance.  It is not the offline quantization metric.
"""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np


TOPICS = {
    "path": "/camera/bev/path",
    "state": "/camera/bev/state",
    "bev_diag": "/camera/bev/diagnostics",
    "controller": "/camera/bev/controller_diagnostics",
    "camera_drive": "/camera_drive",
    "camera_wheel": "/camera_wheel",
    "mcu_drive": "/mcu/cmd_drive",
    "mcu_wheel": "/mcu/cmd_wheel",
    "external_mcu_drive": "/mcu_drive",
    "external_mcu_wheel": "/mcu_wheel",
    "drive_owner": "/mcu/active_drive_source",
    "wheel_owner": "/mcu/active_wheel_source",
    "safety": "/mcu/safety_state",
    "steer": "/mcu/steer_deg",
    "speed": "/mcu/speed_mps",
    "fault": "/mcu/fault",
    "fault_text": "/mcu/fault_text",
    "odom": "/odom",
    "imu_yaw": "/imu/relative_yaw_deg",
    "imu_valid": "/imu/valid",
    "manual_stop": "/manual_stop",
    "estop": "/estop_lock",
    "marker": "/a6_validation/marker",
}


def wrap_deg(value):
    return (float(value)+180.0) % 360.0-180.0


def quaternion_yaw_deg(q):
    siny = 2.0*(q.w*q.z+q.x*q.y)
    cosy = 1.0-2.0*(q.y*q.y+q.z*q.z)
    return math.degrees(math.atan2(siny, cosy))


def percentile(values):
    values = np.asarray([v for v in values if math.isfinite(v)], float)
    if not len(values):
        return {"count": 0, "mean": None, "mean_abs": None,
                "p95_abs": None, "max_abs": None}
    absolute = np.abs(values)
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "mean_abs": float(np.mean(absolute)),
            "p95_abs": float(np.percentile(absolute, 95)),
            "max_abs": float(np.max(absolute))}


def finite_or(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


_TIME_CACHE = {}


def series_times(series):
    key = id(series)
    cached = _TIME_CACHE.get(key)
    if cached is None or cached[0] != len(series):
        cached = (len(series), [item[0] for item in series])
        _TIME_CACHE[key] = cached
    return cached[1]


def nearest(series, timestamp, tolerance=0.15):
    if not series:
        return None
    times = series_times(series)
    index = bisect.bisect_left(times, timestamp)
    candidates = [i for i in (index-1, index) if 0 <= i < len(series)]
    best = min(candidates, key=lambda i: abs(times[i]-timestamp))
    return series[best][1] if abs(times[best]-timestamp) <= tolerance else None


def first_zero_after(series, timestamp, timeout=2.0):
    times = series_times(series)
    for index in range(bisect.bisect_left(times, timestamp), len(series)):
        when, value = series[index]
        if when-timestamp > timeout:
            break
        if abs(float(value)) <= 1.0e-6:
            return when-timestamp
    return None


def marker_segments(markers):
    starts = {}; segments = []
    for timestamp, text in markers:
        text = str(text).strip().upper()
        if text.endswith("_START"):
            starts[text[:-6]] = timestamp
        elif text.endswith("_END"):
            name = text[:-4]
            if name in starts and timestamp > starts[name]:
                segments.append({"name": name, "stage": name.split("_")[0],
                                 "start": starts.pop(name), "end": timestamp})
    return sorted(segments, key=lambda item: item["start"])


def transform_point(pose, point):
    x, y, yaw = pose
    angle = math.radians(yaw); c, s = math.cos(angle), math.sin(angle)
    return np.array([x+c*point[0]-s*point[1], y+s*point[0]+c*point[1]])


def read_bag(uri):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    wanted = set(TOPICS.values())
    raw = {topic: [] for topic in wanted}
    first_ns = None
    while reader.has_next():
        topic, payload, timestamp_ns = reader.read_next()
        if topic not in wanted:
            continue
        if first_ns is None:
            first_ns = timestamp_ns
        msg = deserialize_message(payload, get_message(type_map[topic]))
        timestamp = timestamp_ns/1.0e9
        if topic == TOPICS["odom"]:
            p = msg.pose.pose.position; q = msg.pose.pose.orientation
            value = {"x": float(p.x), "y": float(p.y),
                     "yaw_deg": quaternion_yaw_deg(q),
                     "speed": float(msg.twist.twist.linear.x),
                     "yaw_rate": float(msg.twist.twist.angular.z)}
        elif topic == TOPICS["path"]:
            value = np.asarray([(pose.pose.position.x, pose.pose.position.y)
                                for pose in msg.poses], float)
        elif hasattr(msg, "data"):
            value = msg.data
            if topic in (TOPICS["state"], TOPICS["controller"],
                         TOPICS["bev_diag"]):
                try:
                    value = json.loads(value)
                except (TypeError, ValueError):
                    value = {"decode_error": True, "raw": str(value)}
        else:
            continue
        raw[topic].append((timestamp, value))
    return raw, type_map, (first_ns/1.0e9 if first_ns is not None else 0.0)


def relative(raw, origin):
    return {topic: [(timestamp-origin, value) for timestamp, value in series]
            for topic, series in raw.items()}


def causal_path_errors(series):
    odom = series[TOPICS["odom"]]
    paths = series[TOPICS["path"]]
    controllers = series[TOPICS["controller"]]
    if len(odom) < 2 or not paths or not controllers:
        return []
    odom_t = [item[0] for item in odom]
    xy = np.asarray([[item[1]["x"], item[1]["y"]] for item in odom])
    cumulative = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    rows = []
    # At most 5 Hz avoids overweighting identical targets at controller rate.
    last_used = -1.0
    for timestamp, command in controllers:
        if timestamp-last_used < 0.20 or not isinstance(command, dict):
            continue
        target = command.get("target_point")
        if not target or len(target) != 2:
            continue
        index0 = bisect.bisect_left(odom_t, timestamp)
        if index0 >= len(odom):
            continue
        path = nearest(paths, timestamp, 0.20)
        if path is None or len(path) < 2:
            continue
        target = np.asarray(target, float)
        path_index = int(np.argmin(np.linalg.norm(path-target, axis=1)))
        lo, hi = max(0, path_index-1), min(len(path)-1, path_index+1)
        tangent = path[hi]-path[lo]
        if np.linalg.norm(tangent) < 1.0e-6:
            continue
        travel = float(np.linalg.norm(target))
        target_distance = cumulative[index0]+travel
        index1 = int(np.searchsorted(cumulative, target_distance))
        if index1 >= len(odom) or odom_t[index1]-timestamp > 30.0:
            continue
        start_pose = (odom[index0][1]["x"], odom[index0][1]["y"],
                      odom[index0][1]["yaw_deg"])
        target_global = transform_point(start_pose, target)
        desired_yaw = start_pose[2]+math.degrees(math.atan2(tangent[1], tangent[0]))
        actual = xy[index1]
        normal = np.array([-math.sin(math.radians(desired_yaw)),
                           math.cos(math.radians(desired_yaw))])
        rows.append({"command_time_sec": timestamp,
                     "arrival_time_sec": odom_t[index1],
                     "target_distance_m": travel,
                     "cross_track_error_m": float((actual-target_global)@normal),
                     "position_error_m": float(np.linalg.norm(actual-target_global)),
                     "heading_error_deg": wrap_deg(
                         odom[index1][1]["yaw_deg"]-desired_yaw)})
        last_used = timestamp
    return rows


def steering_response_delays(commands, feedback):
    delays = []
    previous = None
    feedback_times = [item[0] for item in feedback]
    for timestamp, value in commands:
        value = float(value)
        if previous is None or abs(value-previous) < 1.0:
            previous = value; continue
        baseline = nearest(feedback, timestamp, 0.20)
        if baseline is None:
            previous = value; continue
        threshold = float(baseline)+0.90*(value-float(baseline))
        start = bisect.bisect_left(feedback_times, timestamp)
        for when, actual in feedback[start:]:
            if when-timestamp > 2.0:
                break
            reached = actual >= threshold if value > baseline else actual <= threshold
            if reached:
                delays.append(when-timestamp); break
        previous = value
    return delays


def transition_times(series, predicate):
    output = []; previous = False
    for timestamp, value in series:
        current = bool(predicate(value))
        if current and not previous:
            output.append(timestamp)
        previous = current
    return output


def trial_metrics(segment, series, path_errors):
    start, end = segment["start"], segment["end"]
    within = lambda seq: [(t, v) for t, v in seq if start <= t <= end]
    odom = within(series[TOPICS["odom"]]); wheels = within(series[TOPICS["camera_wheel"]])
    states = within(series[TOPICS["state"]]); imu = within(series[TOPICS["imu_yaw"]])
    result = dict(segment)
    if len(odom) >= 2:
        x0, y0, yaw0 = odom[0][1]["x"], odom[0][1]["y"], odom[0][1]["yaw_deg"]
        angle = math.radians(yaw0); normal = np.array([-math.sin(angle), math.cos(angle)])
        offsets = [float((np.array([v["x"]-x0, v["y"]-y0]))@normal) for _, v in odom]
        distance = float(np.linalg.norm(np.diff(np.asarray(
            [[v["x"], v["y"]] for _, v in odom]), axis=0), axis=1).sum())
        odom_yaw = abs(wrap_deg(odom[-1][1]["yaw_deg"]-yaw0))
        imu_yaw = abs(wrap_deg(float(imu[-1][1])-float(imu[0][1]))) if len(imu) >= 2 else 0.0
        result.update({"distance_m": distance, "max_line_deviation_m": max(map(abs, offsets)),
                       "end_line_deviation_m": offsets[-1],
                       "odom_yaw_change_deg": odom_yaw,
                       "imu_yaw_change_deg": imu_yaw,
                       "odom_radius_m": (distance/math.radians(odom_yaw)
                                         if odom_yaw > 1.0 else None),
                       "imu_radius_m": (distance/math.radians(imu_yaw)
                                        if imu_yaw > 1.0 else None)})
    deltas = [(wheels[i][1]-wheels[i-1][1])/(wheels[i][0]-wheels[i-1][0])
              for i in range(1, len(wheels)) if wheels[i][0] > wheels[i-1][0]]
    result["wheel_rate_deg_s"] = percentile(deltas)
    result["invalid_count"] = sum(isinstance(v, dict) and v.get("state") == "INVALID"
                                  for _, v in states)
    trial_errors = [row for row in path_errors if start <= row["command_time_sec"] <= end]
    result["cross_track_error_m"] = percentile(
        [row["cross_track_error_m"] for row in trial_errors])
    result["heading_error_deg"] = percentile(
        [row["heading_error_deg"] for row in trial_errors])
    return result


def yaw_direction_mismatch(trial, series):
    odom_start = nearest(series[TOPICS["odom"]], trial["start"])
    odom_end = nearest(series[TOPICS["odom"]], trial["end"])
    imu_start = nearest(series[TOPICS["imu_yaw"]], trial["start"])
    imu_end = nearest(series[TOPICS["imu_yaw"]], trial["end"])
    if any(value is None for value in (odom_start, odom_end,
                                       imu_start, imu_end)):
        return False
    odom_delta = wrap_deg(odom_end["yaw_deg"]-odom_start["yaw_deg"])
    imu_delta = wrap_deg(float(imu_end)-float(imu_start))
    return abs(odom_delta) > 2.0 and abs(imu_delta) > 2.0 and \
        odom_delta*imu_delta < 0.0


def analyze(series, type_map, safe_corridor):
    camera_wheel = series[TOPICS["camera_wheel"]]
    mcu_wheel = series[TOPICS["mcu_wheel"]]
    steer = series[TOPICS["steer"]]
    aligned = []
    for timestamp, value in camera_wheel:
        row = {"time_sec": timestamp, "camera_wheel": int(value)}
        for key, topic in (("mcu_wheel", "mcu_wheel"), ("steer_deg", "steer"),
                           ("speed_mps", "speed"), ("wheel_owner", "wheel_owner"),
                           ("drive_owner", "drive_owner"), ("safety_state", "safety"),
                           ("imu_yaw_deg", "imu_yaw")):
            row[key] = nearest(series[TOPICS[topic]], timestamp)
        odom = nearest(series[TOPICS["odom"]], timestamp)
        if odom:
            row.update({f"odom_{key}": odom[key] for key in ("x", "y", "yaw_deg")})
        aligned.append(row)
    wheel_errors = [row["mcu_wheel"]-row["camera_wheel"] for row in aligned
                    if row.get("mcu_wheel") is not None]
    steer_errors = [row["steer_deg"]-row["mcu_wheel"] for row in aligned
                    if row.get("steer_deg") is not None and row.get("mcu_wheel") is not None]
    owners = [row.get("wheel_owner") for row in aligned if row.get("wheel_owner")]
    states = series[TOPICS["state"]]
    invalid_times = transition_times(states, lambda value:
        isinstance(value, dict) and value.get("state") == "INVALID")
    invalid_stop = [{"time_sec": timestamp,
                     "drive_zero_delay_sec": first_zero_after(
                         series[TOPICS["camera_drive"]], timestamp),
                     "wheel_zero_delay_sec": first_zero_after(camera_wheel, timestamp)}
                    for timestamp in invalid_times]
    invalid_samples = [(timestamp, value) for timestamp, value in states
                       if isinstance(value, dict) and value.get("state") == "INVALID"]
    invalid_nonzero_drive = sum(
        abs(float(nearest(series[TOPICS["camera_drive"]], timestamp) or 0.0)) > 1.0e-6
        for timestamp, _ in invalid_samples)
    invalid_nonzero_wheel = sum(
        abs(float(nearest(camera_wheel, timestamp) or 0.0)) > 1.0e-6
        for timestamp, _ in invalid_samples)
    manual_times = transition_times(series[TOPICS["manual_stop"]], bool)
    manual_stop = [{"time_sec": timestamp,
                    "mcu_drive_zero_delay_sec": first_zero_after(
                        series[TOPICS["mcu_drive"]], timestamp),
                    "mcu_wheel_zero_delay_sec": first_zero_after(mcu_wheel, timestamp)}
                   for timestamp in manual_times]
    path_errors = causal_path_errors(series)
    segments = marker_segments(series[TOPICS["marker"]])
    trials = [trial_metrics(segment, series, path_errors) for segment in segments]
    sign_checks = []
    for timestamp, command in series[TOPICS["controller"]]:
        if not isinstance(command, dict) or not command.get("target_point"):
            continue
        lateral = float(command["target_point"][1])
        wheel = nearest(camera_wheel, timestamp)
        if wheel is not None and abs(lateral) > 0.02 and int(wheel) != 0:
            sign_checks.append((lateral > 0.0 and wheel < 0) or
                               (lateral < 0.0 and wheel > 0))
    timeout_count = sum(
        isinstance(value, dict) and any("TIMEOUT" in str(reason)
        for reason in value.get("reasons", [])) for _, value in states)
    fault_count = sum((isinstance(value, (int, float)) and value != 0) or
                      (isinstance(value, bool) and value)
                      for _, value in series[TOPICS["fault"]])
    diagnostics = series[TOPICS["bev_diag"]]
    clearance_violations = sum(
        isinstance(value, dict) and value.get("state") in ("VALID", "DEGRADED") and
        (finite_or(value.get("safe_road_coverage"), 1.0) < 0.999 or
         finite_or(value.get("minimum_clearance_m"), safe_corridor) < safe_corridor)
        for _, value in diagnostics)
    target_laterals = [float(value["target_point"][1])
                       for _, value in series[TOPICS["controller"]]
                       if isinstance(value, dict) and value.get("target_point")]
    path_jumps = sum(abs(target_laterals[i]-target_laterals[i-1]) > 0.65
                     for i in range(1, len(target_laterals)))
    wheel_rates = [(camera_wheel[i][1]-camera_wheel[i-1][1]) /
                   (camera_wheel[i][0]-camera_wheel[i-1][0])
                   for i in range(1, len(camera_wheel))
                   if camera_wheel[i][0] > camera_wheel[i-1][0]]
    result = {
        "topic_types": type_map,
        "topic_counts": {topic: len(values) for topic, values in series.items()},
        "markers": segments,
        "wheel_contract": {"left": "negative", "right": "positive", "limit_deg": 22},
        "wheel_owner_camera_percent": (100.0*owners.count("camera")/len(owners)
                                        if owners else None),
        "camera_to_mcu_wheel_error_deg": percentile(wheel_errors),
        "mcu_command_to_steer_proxy_error_deg": percentile(steer_errors),
        "steering_response_delay_sec": percentile(
            steering_response_delays(mcu_wheel, steer)),
        "camera_wheel_over_22": sum(abs(float(v)) > 22 for _, v in camera_wheel),
        "mcu_wheel_over_22": sum(abs(float(v)) > 22 for _, v in mcu_wheel),
        "external_mcu_drive_samples": len(series[TOPICS["external_mcu_drive"]]),
        "external_mcu_wheel_samples": len(series[TOPICS["external_mcu_wheel"]]),
        "path_to_camera_wheel_sign_checks": len(sign_checks),
        "path_to_camera_wheel_sign_errors": sign_checks.count(False),
        "invalid_transitions": len(invalid_times),
        "timeout_samples": timeout_count,
        "invalid_nonzero_camera_drive_samples": invalid_nonzero_drive,
        "invalid_nonzero_camera_wheel_samples": invalid_nonzero_wheel,
        "invalid_stop_responses": invalid_stop,
        "manual_stop_responses": manual_stop,
        "mcu_fault_samples": fault_count,
        "valid_path_clearance_violation_samples": clearance_violations,
        "controller_target_jump_count": path_jumps,
        "camera_wheel_rate_deg_s": percentile(wheel_rates),
        "causal_cross_track_error_m": percentile(
            [row["cross_track_error_m"] for row in path_errors]),
        "causal_heading_error_deg": percentile(
            [row["heading_error_deg"] for row in path_errors]),
        "safe_corridor_m": safe_corridor,
        "path_departures": sum(abs(row["cross_track_error_m"]) > safe_corridor
                               for row in path_errors),
        "odom_imu_yaw_direction_mismatches": sum(
            yaw_direction_mismatch(trial, series) for trial in trials),
        "trials": trials,
        "limitations": [
            "/mcu/steer_deg is a firmware time-based estimate, not an independent steering-angle sensor.",
            "/odom is externally owned; T870_MCU publishes its steer-integrated reference on /mcu/odom.",
            "IMU yaw/radius is therefore the independent turn-response evidence.",
        ],
    }
    return result, aligned, path_errors, trials


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-corridor-m", type=float, default=0.52)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    raw, types, origin = read_bag(args.bag)
    series = relative(raw, origin)
    result, aligned, path_errors, trials = analyze(
        series, types, args.safe_corridor_m)
    (args.output/"a6_low_speed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    write_csv(args.output/"aligned_control.csv", aligned)
    write_csv(args.output/"causal_path_tracking.csv", path_errors)
    write_csv(args.output/"trial_summary.csv", trials)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
