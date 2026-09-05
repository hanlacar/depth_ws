"""Offline source-stamp, receive-stamp, synchronization and TF audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from .bag_common import load_yaml, percentile, stamp_ns


HEADER_FIRST_TYPES = {
    "sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo", "sensor_msgs/msg/Imu",
    "realsense2_camera_msgs/msg/Metadata", "nav_msgs/msg/Path",
    "nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PoseWithCovarianceStamped",
    "diagnostic_msgs/msg/DiagnosticArray",
}


def cdr_header_stamp(serialized):
    """Read a leading std_msgs/Header stamp without copying a large payload."""
    if len(serialized) < 12:
        return None
    byte_order = "<" if serialized[1] == 1 else ">"
    seconds, nanoseconds = struct.unpack_from(f"{byte_order}iI", serialized, 4)
    return seconds * 1_000_000_000 + nanoseconds


def first_header_stamp(message):
    if hasattr(message, "header") and hasattr(message.header, "stamp"):
        return stamp_ns(message.header.stamp)
    if hasattr(message, "transforms") and message.transforms:
        return stamp_ns(message.transforms[0].header.stamp)
    return None


def series_statistics(source, received):
    answer = {
        "message_count": len(received),
        "first_bag_stamp_ns": received[0] if received else None,
        "last_bag_stamp_ns": received[-1] if received else None,
        "has_source_header": bool(source),
    }
    if not source:
        return answer
    deltas = [b - a for a, b in zip(source, source[1:])]
    positive = [value / 1e9 for value in deltas if value > 0]
    rates = [1.0 / value for value in positive]
    elapsed = (source[-1] - source[0]) / 1e9
    latency_ms = [(bag - stamp) / 1e6 for stamp, bag in zip(source, received)]
    answer.update({
        "first_source_stamp_ns": source[0],
        "last_source_stamp_ns": source[-1],
        "average_hz": (len(source) - 1) / elapsed if elapsed > 0 else None,
        "minimum_instant_hz": min(rates) if rates else None,
        "p95_instant_hz": percentile(rates, 95),
        "p95_period_ms": percentile([value * 1000 for value in positive], 95),
        "duplicate_source_stamps": sum(value == 0 for value in deltas),
        "backwards_source_stamps": sum(value < 0 for value in deltas),
        "zero_source_stamps": sum(value == 0 for value in source),
        "latency_ms": {
            "mean": sum(latency_ms) / len(latency_ms),
            "p95": percentile(latency_ms, 95),
            "max": max(latency_ms),
        },
    })
    return answer


def nearest_sync(left, right, tolerance_ms):
    if not left or not right:
        return {"pair_count": 0, "mean_ms": None, "p95_ms": None,
                "max_ms": None, "over_tolerance_percent": None}
    right = sorted(right)
    index, errors = 0, []
    for stamp in sorted(left):
        while index + 1 < len(right) and abs(right[index + 1] - stamp) <= abs(right[index] - stamp):
            index += 1
        errors.append(abs(right[index] - stamp) / 1e6)
    return {
        "pair_count": len(errors),
        "mean_ms": sum(errors) / len(errors),
        "p95_ms": percentile(errors, 95),
        "max_ms": max(errors),
        "tolerance_ms": tolerance_ms,
        "over_tolerance_percent": 100.0 * sum(e > tolerance_ms for e in errors) / len(errors),
    }


def gap_statistics(stamps):
    gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:]) if b > a]
    return {
        "samples": len(stamps),
        "mean_gap_ms": sum(gaps) / len(gaps) if gaps else None,
        "p95_gap_ms": percentile(gaps, 95),
        "max_gap_ms": max(gaps) if gaps else None,
    }


def connected(edges, chain):
    graph = defaultdict(set)
    for parent, child in edges:
        graph[parent].add(child)
        graph[child].add(parent)
    for first, last in zip(chain, chain[1:]):
        seen, pending = {first}, [first]
        while pending:
            current = pending.pop()
            for neighbor in graph[current] - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        if last not in seen:
            return False
    return True


def tf_coverage(static_edges, dynamic_events, chain):
    edges = set(static_edges)
    available = 0
    for _, additions in sorted(dynamic_events):
        edges.update(additions)
        available += connected(edges, chain)
    total = len(dynamic_events)
    return {
        "chain": "->".join(chain),
        "dynamic_samples": total,
        "available_samples": available,
        "availability_percent": 100.0 * available / total if total else 0.0,
        "static_edges": sorted(f"{a}->{b}" for a, b in static_edges),
    }


def audit_bag(bag_path, recording_config):
    config = load_yaml(recording_config)
    source_bag = Path(bag_path).resolve()
    metadata = load_yaml(source_bag / "metadata.yaml")
    info = metadata.get("rosbag2_bagfile_information", {})
    temporary = None
    reader_path = source_bag
    if info.get("compression_format"):
        temporary = tempfile.TemporaryDirectory(prefix="depth_bag_audit.", dir="/tmp")
        reader_path = Path(temporary.name)
        shutil.copy2(source_bag / "metadata.yaml", reader_path / "metadata.yaml")
        for relative in info.get("relative_file_paths", []):
            shutil.copy2(source_bag / relative, reader_path / relative)
    storage = rosbag2_py.StorageOptions(uri=str(reader_path), storage_id="mcap")
    reader = rosbag2_py.SequentialCompressionReader()
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))
    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    classes = {}
    for topic, type_name in topic_types.items():
        try:
            classes[topic] = get_message(type_name)
        except (AttributeError, ModuleNotFoundError, RuntimeError, ValueError):
            classes[topic] = None

    bag_stamps, source_stamps = defaultdict(list), defaultdict(list)
    values = defaultdict(list)
    odom_positions = []
    static_edges, dynamic_events = set(), []
    decode_errors = defaultdict(int)
    while reader.has_next():
        topic, serialized, received_ns = reader.read_next()
        bag_stamps[topic].append(received_ns)
        source = None
        if topic_types[topic] in HEADER_FIRST_TYPES:
            source = cdr_header_stamp(serialized)
            if source is not None:
                source_stamps[topic].append(source)
            if topic != "/depth_slam/cuvslam/odometry":
                continue
        cls = classes.get(topic)
        if cls is None:
            continue
        try:
            message = deserialize_message(serialized, cls)
        except Exception:
            decode_errors[topic] += 1
            continue
        source = source if source is not None else first_header_stamp(message)
        if source is not None and topic_types[topic] not in HEADER_FIRST_TYPES:
            source_stamps[topic].append(source)
        if hasattr(message, "data") and topic in {
                "/depth_slam/cuvslam/tracking_valid", "/depth_slam/cuvslam/reset_count",
                "/depth_slam/cuvslam/fps"}:
            values[topic].append(message.data)
        if topic == "/depth_slam/cuvslam/odometry":
            p = message.pose.pose.position
            odom_positions.append((source or received_ns, p.x, p.y, p.z))
        if topic in ("/tf", "/tf_static"):
            additions = {(item.header.frame_id.lstrip("/"), item.child_frame_id.lstrip("/"))
                         for item in message.transforms}
            if topic == "/tf_static":
                static_edges.update(additions)
            else:
                dynamic_events.append((received_ns, additions))

    topics = {}
    for topic in sorted(topic_types):
        topics[topic] = {
            "type": topic_types[topic],
            **series_statistics(source_stamps[topic], bag_stamps[topic]),
            "decode_errors": decode_errors[topic],
        }

    tolerance = float(config["thresholds"]["sync_tolerance_ms"])
    sync = {
        "rgb_depth": nearest_sync(
            source_stamps["/camera/camera/color/metadata"],
            source_stamps["/camera/camera/depth/metadata"], tolerance),
        "infra1_infra2": nearest_sync(
            source_stamps["/camera/camera/infra1/metadata"],
            source_stamps["/camera/camera/infra2/metadata"], tolerance),
    }
    imu = {
        "accel": gap_statistics(source_stamps["/camera/camera/accel/sample"]),
        "gyro": gap_statistics(source_stamps["/camera/camera/gyro/sample"]),
        "combined": gap_statistics(source_stamps["/camera/camera/imu"]),
    }
    jumps = []
    jump_limit = float(config["thresholds"]["odometry_jump_m"])
    for before, after in zip(odom_positions, odom_positions[1:]):
        distance = math.dist(before[1:], after[1:])
        if distance > jump_limit:
            jumps.append({"stamp_ns": after[0], "distance_m": distance})
    tracking = values["/depth_slam/cuvslam/tracking_valid"]
    resets = values["/depth_slam/cuvslam/reset_count"]
    tracking_percent = 100.0 * sum(bool(v) for v in tracking) / len(tracking) if tracking else 0.0
    reset_increase = max(resets) - min(resets) if resets else None

    thresholds = config["thresholds"]
    checks = {}
    frequency_topics = {
        "color_metadata": ("/camera/camera/color/metadata", thresholds["metadata_min_hz"]),
        "depth_metadata": ("/camera/camera/depth/metadata", thresholds["metadata_min_hz"]),
        "infra1_metadata": ("/camera/camera/infra1/metadata", thresholds["metadata_min_hz"]),
        "infra2_metadata": ("/camera/camera/infra2/metadata", thresholds["metadata_min_hz"]),
        "combined_imu": ("/camera/camera/imu", thresholds["imu_min_hz"]),
        "cuvslam_odometry": ("/depth_slam/cuvslam/odometry", thresholds["cuvslam_odometry_min_hz"]),
    }
    for name, (topic, minimum) in frequency_topics.items():
        actual = topics.get(topic, {}).get("average_hz")
        checks[f"{name}_frequency"] = {
            "pass": actual is not None and actual >= minimum,
            "actual_hz": actual, "minimum_hz": minimum}
    for topic in frequency_topics.values():
        name = topic[0]
        item = topics.get(name, {})
        checks[f"{name}_stamp_order"] = {
            "pass": (
                item.get("duplicate_source_stamps", 1) == 0 and
                item.get("backwards_source_stamps", 1) == 0 and
                item.get("zero_source_stamps", 1) == 0),
        }
        p95 = (item.get("latency_ms") or {}).get("p95")
        checks[f"{name}_latency"] = {
            "pass": p95 is not None and p95 <= thresholds["latency_p95_max_ms"],
            "actual_p95_ms": p95, "maximum_ms": thresholds["latency_p95_max_ms"],
        }
    checks["tracking_valid"] = {
        "pass": tracking_percent >= thresholds["tracking_valid_min_percent"],
        "actual_percent": tracking_percent}
    checks["reset_count"] = {"pass": reset_increase is not None and
                             reset_increase <= thresholds["reset_count_max_increase"],
                             "increase": reset_increase}
    tf = tf_coverage(static_edges, dynamic_events, config["tf_chain"])
    checks["tf_chain"] = {"pass": tf["availability_percent"] > 0.0,
                          "availability_percent": tf["availability_percent"]}
    result = {
        "schema_version": 1,
        "bag_path": str(Path(bag_path).resolve()),
        "storage_id": "mcap",
        "clock_interpretation": {
            "bag_stamp": "recorder host ROS/system clock at receive time",
            "source_stamp": "message header stamp; used for intra-host ordering and sensor alignment",
            "wall_clock": "NTP/PTP state is separate evidence for multi-host alignment",
        },
        "topics": topics,
        "synchronization": sync,
        "imu_gaps": imu,
        "odometry": {"jump_threshold_m": jump_limit, "jumps": jumps,
                     "tracking_valid_percent": tracking_percent,
                     "reset_count_first": resets[0] if resets else None,
                     "reset_count_last": resets[-1] if resets else None,
                     "reset_count_increase": reset_increase},
        "tf": tf,
        "checks": checks,
        "overall_pass": all(value["pass"] for value in checks.values()),
    }
    if temporary is not None:
        temporary.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--recording-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_bag(args.bag_path, args.recording_config)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"TIME AUDIT {'PASS' if report['overall_pass'] else 'FAIL'}: {args.output}")


if __name__ == "__main__":
    main()
