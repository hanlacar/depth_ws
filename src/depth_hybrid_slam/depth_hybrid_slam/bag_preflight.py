"""Fail-closed live-graph preflight for competition recordings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

from .bag_common import load_yaml, resolve_profile, sha256, validate_id, write_yaml


SAMPLE_TOPICS = {
    "/camera/camera/color/image_raw",
    "/camera/camera/color/metadata",
    "/camera/camera/depth/image_rect_raw",
    "/camera/camera/depth/metadata",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/infra1/image_rect_raw",
    "/camera/camera/infra1/metadata",
    "/camera/camera/infra2/image_rect_raw",
    "/camera/camera/infra2/metadata",
    "/camera/camera/gyro/sample",
    "/camera/camera/accel/sample",
    "/camera/camera/imu",
    "/visual_slam/tracking/odometry",
    "/depth_slam/cuvslam/odometry",
    "/depth_slam/cuvslam/tracking_valid",
    "/depth_slam/cuvslam/reset_count",
    "/depth_slam/cuvslam/fps",
}
FORBIDDEN_CONTROL_TOPICS = ("/slam_drive", "/slam_wheel")


def run_text(command):
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_key_values(text):
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        elif "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_device():
    short = run_text(["rs-enumerate-devices", "-s"])
    full = run_text(["rs-enumerate-devices"])
    fields = parse_key_values(full)
    rows = [line.split() for line in short.splitlines() if "Intel RealSense" in line]
    row = rows[0] if rows else []
    return {
        "model": " ".join(row[:-2]) if len(row) >= 3 else fields.get("Name", "UNKNOWN"),
        "serial": row[-2] if len(row) >= 2 else fields.get("Serial Number", "UNKNOWN"),
        "firmware": row[-1] if row else fields.get("Firmware Version", "UNKNOWN"),
        "usb_mode": fields.get("Usb Type Descriptor", "UNKNOWN"),
        "physical_port": fields.get("Physical Port", "UNKNOWN"),
        "connection_type": fields.get("Connection Type", "UNKNOWN"),
    }


def read_camera_parameter(name):
    for _attempt in range(3):
        text = run_text(["ros2", "param", "get", "/camera/camera", name])
        if ":" in text:
            return text.split(":", 1)[-1].strip()
        time.sleep(0.5)
    return "UNKNOWN"


def read_clock():
    values = parse_key_values(run_text([
        "timedatectl", "show", "--property=NTPSynchronized", "--property=NTP",
        "--property=Timezone", "--property=LocalRTC", "--property=TimeUSec",
    ]))
    return {
        "source": "systemd-timesyncd/NTP" if values.get("NTP") == "yes" else "UNKNOWN",
        "ntp_enabled": values.get("NTP", "UNKNOWN"),
        "synchronized": values.get("NTPSynchronized", "UNKNOWN"),
        "timezone": values.get("Timezone", "UNKNOWN"),
        "local_rtc": values.get("LocalRTC", "UNKNOWN"),
        "time": values.get("TimeUSec", "UNKNOWN"),
        "scope_note": "NTP is wall-clock evidence; source stamps validate intra-host sensor alignment.",
    }


def file_identity(path):
    if not path:
        return None
    item = Path(path).resolve()
    if not item.is_file():
        raise FileNotFoundError(item)
    answer = {"path": str(item), "size_bytes": item.stat().st_size, "sha256": sha256(item)}
    candidates = ((item.with_suffix(".yaml"), item.parent / "metadata.yaml")
                  if item.suffix != ".db" else
                  (item.parent / "metadata.yaml", item.with_suffix(".yaml")))
    for candidate in candidates:
        if candidate.is_file():
            metadata = load_yaml(candidate)
            for key in ("case_id", "map_id", "route_id"):
                if key in metadata:
                    answer[key] = metadata[key]
            break
    checksum_file = item.parent / "checksums.sha256"
    if checksum_file.is_file():
        expected = None
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2 and fields[1].lstrip("*") == item.name:
                expected = fields[0]
                break
        answer["sealed_checksum"] = expected
        answer["checksum_valid"] = expected == answer["sha256"] if expected else False
        if not answer["checksum_valid"]:
            raise ValueError(f"checksum verification failed: {item}")
    return answer


class GraphProbe(Node):
    def __init__(self, topics):
        super().__init__("competition_bag_preflight")
        self.counts = {topic: 0 for topic in topics}
        self.values = {topic: [] for topic in topics}
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self._bag_subscriptions = []
        for topic, msg_type in topics.items():
            try:
                cls = get_message(msg_type)
            except (AttributeError, ModuleNotFoundError, RuntimeError, ValueError):
                continue

            def callback(message, name=topic):
                self.counts[name] += 1
                if hasattr(message, "data") and len(self.values[name]) < 10000:
                    self.values[name].append(message.data)

            self._bag_subscriptions.append(
                self.create_subscription(cls, topic, callback, qos))


def inspect(args):
    recording = load_yaml(args.recording_config)
    topic_config = load_yaml(args.topic_config)
    requested, expected, required = resolve_profile(topic_config, args.profile)
    validate_id(args.case_id, "case_id")
    validate_id(args.session_id, "session_id")

    expected_env = recording["expected_environment"]
    failures = []
    environment = {
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""),
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
    }
    for key, expected_value in expected_env.items():
        if environment.get(key) != str(expected_value):
            failures.append(f"ENV_MISMATCH:{key}:{environment.get(key)}!={expected_value}")

    rclpy.init(args=None)
    node = GraphProbe({t: expected[t] for t in SAMPLE_TOPICS if t in expected})
    try:
        deadline = time.monotonic() + 2.0 + float(
            recording["preflight_sample_seconds"])
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        graph = dict(node.get_topic_names_and_types())
        actual, missing_optional, qos = {}, [], {}
        recordable = set()
        for topic in requested:
            types = graph.get(topic, [])
            publishers = node.get_publishers_info_by_topic(topic) if types else []
            if not types:
                (failures if topic in required else missing_optional).append(
                    f"{'MISSING_REQUIRED' if topic in required else 'MISSING_OPTIONAL'}:{topic}")
                continue
            actual[topic] = {"types": types, "publisher_count": len(publishers)}
            qos[topic] = [{
                "node": f"/{p.node_namespace.strip('/')}/{p.node_name}".replace("//", "/"),
                "reliability": str(p.qos_profile.reliability),
                "durability": str(p.qos_profile.durability),
                "history": str(p.qos_profile.history),
                "depth": p.qos_profile.depth,
            } for p in publishers]
            if expected[topic] not in types:
                failures.append(f"TYPE_MISMATCH:{topic}:{types}!={expected[topic]}")
            if topic in required and len(publishers) != 1:
                failures.append(f"PUBLISHER_COUNT:{topic}:{len(publishers)}!=1")
            try:
                get_message(expected[topic])
                recordable.add(topic)
            except (AttributeError, ModuleNotFoundError, RuntimeError, ValueError):
                reason = f"TYPE_SUPPORT_UNAVAILABLE:{topic}:{expected[topic]}"
                (failures if topic in required else missing_optional).append(
                    reason if topic in required else f"MISSING_OPTIONAL:{reason}")

        for topic in sorted(SAMPLE_TOPICS & required):
            if topic in actual and node.counts.get(topic, 0) == 0:
                failures.append(f"NO_DATA:{topic}")
        tracking = node.values.get("/depth_slam/cuvslam/tracking_valid", [])
        resets = node.values.get("/depth_slam/cuvslam/reset_count", [])
        tracking_percent = (100.0 * sum(bool(value) for value in tracking) / len(tracking)
                            if tracking else 0.0)
        if not tracking or not bool(tracking[-1]) or tracking_percent < 99.0:
            failures.append(
                f"CUVSLAM_TRACKING_NOT_READY:latest={tracking[-1] if tracking else None}:"
                f"percent={tracking_percent:.3f}")
        if not resets or max(resets) != 0:
            failures.append(f"CUVSLAM_RESET_NOT_ZERO:{max(resets) if resets else 'NO_DATA'}")

        nodes = [f"/{ns.strip('/')}/{name}".replace("//", "/")
                 for name, ns in node.get_node_names_and_namespaces()]
        duplicates = [name for name in nodes if "competition_bag_recorder" in name]
        if duplicates:
            failures.append(f"DUPLICATE_RECORDER:{duplicates}")
        forbidden_publishers = {
            topic: len(node.get_publishers_info_by_topic(topic))
            for topic in FORBIDDEN_CONTROL_TOPICS
        }
        if any(forbidden_publishers.values()):
            failures.append(f"REAL_CONTROL_PUBLISHER_PRESENT:{forbidden_publishers}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    device = read_device()
    camera_expected = recording["expected_camera"]
    for key in ("model", "serial"):
        if str(device.get(key)) != str(camera_expected[key]):
            failures.append(f"CAMERA_{key.upper()}:{device.get(key)}!={camera_expected[key]}")
    if not str(device.get("usb_mode", "")).startswith("3"):
        failures.append(f"CAMERA_USB_MODE:{device.get('usb_mode')}")
    device["profiles"] = {
        "color": read_camera_parameter("rgb_camera.color_profile"),
        "depth": read_camera_parameter("depth_module.depth_profile"),
        "infrared": read_camera_parameter("depth_module.infra_profile"),
        "gyro_fps": read_camera_parameter("gyro_fps"),
        "accel_fps": read_camera_parameter("accel_fps"),
    }
    profile_expectations = {
        "color": str(camera_expected["color_profile"]),
        "depth": str(camera_expected["depth_profile"]),
        "infrared": str(camera_expected["infrared_profile"]),
        "gyro_fps": str(camera_expected["gyro_fps"]),
        "accel_fps": str(camera_expected["accel_fps"]),
    }
    for key, expected_value in profile_expectations.items():
        if device["profiles"][key] != expected_value:
            failures.append(
                f"CAMERA_PROFILE:{key}:{device['profiles'][key]}!={expected_value}")
    if not device.get("firmware") or device["firmware"] == "UNKNOWN":
        failures.append("CAMERA_FIRMWARE_UNKNOWN")

    bag_root = Path(args.bag_root).resolve()
    usage = shutil.disk_usage(bag_root if bag_root.exists() else bag_root.parent)
    minimum = int(recording["minimum_free_space_bytes"])
    if usage.free < minimum:
        failures.append(f"LOW_DISK:{usage.free}<{minimum}")
    session_path = (Path(args.session_path).expanduser().resolve()
                    if args.session_path else
                    bag_root / args.case_id / args.session_id)
    try:
        session_path.relative_to(bag_root)
    except ValueError:
        failures.append(f"SESSION_OUTSIDE_BAG_ROOT:{session_path}")
    if session_path.exists():
        failures.append(f"SESSION_EXISTS:{session_path}")

    map_identity = file_identity(args.map_path)
    route_identity = file_identity(args.route_path)
    if map_identity and route_identity:
        map_id = map_identity.get("map_id")
        route_map_id = route_identity.get("map_id")
        if map_id and route_map_id and map_id != route_map_id:
            failures.append(f"MAP_ROUTE_ID_MISMATCH:{map_id}!={route_map_id}")
    clock = read_clock()
    if clock["synchronized"] != "yes":
        failures.append(f"CLOCK_NOT_SYNCHRONIZED:{clock['synchronized']}")

    report = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "case_id": args.case_id,
        "session_id": args.session_id,
        "purpose": args.purpose,
        "profile": args.profile,
        "session_path": str(session_path),
        "host": {"hostname": socket.gethostname(), "kernel": platform.release()},
        "environment": environment,
        "camera": device,
        "clock": clock,
        "disk": {"free_bytes": usage.free, "minimum_bytes": minimum},
        "requested_topics": requested,
        "required_topics": sorted(required),
        "recordable_topics": sorted(recordable),
        "expected_types": expected,
        "actual_topics": actual,
        "missing_optional": missing_optional,
        "sample_counts": node.counts,
        "tracking_values": tracking,
        "tracking_valid_percent": tracking_percent,
        "reset_values": resets,
        "qos": qos,
        "forbidden_control_publishers": forbidden_publishers,
        "map": map_identity,
        "route": route_identity,
        "safety": recording["safety"],
    }
    write_yaml(args.report, report)
    Path(args.topics_output).write_text(
        "\n".join(topic for topic in requested if topic in actual and topic in recordable) + "\n",
        encoding="utf-8")
    return report


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--topic-config", required=True)
    value.add_argument("--recording-config", required=True)
    value.add_argument("--profile", default="camera_slam")
    value.add_argument("--case-id", required=True)
    value.add_argument("--session-id", required=True)
    value.add_argument("--purpose", choices=("mapping", "localization", "competition"),
                       required=True)
    value.add_argument("--bag-root", required=True)
    value.add_argument("--session-path", default="")
    value.add_argument("--map-path", default="")
    value.add_argument("--route-path", default="")
    value.add_argument("--report", required=True)
    value.add_argument("--topics-output", required=True)
    return value


def main():
    args = parser().parse_args()
    try:
        report = inspect(args)
    except Exception as error:  # preflight must fail closed
        print(f"PREFLIGHT ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    if not report["passed"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(2)
    print(f"PREFLIGHT PASS: {len(report['actual_topics'])} topics")


if __name__ == "__main__":
    main()
