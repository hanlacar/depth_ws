"""Finalize an immutable competition bag session and its provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import socket
import subprocess

from .bag_common import load_yaml, sha256, write_yaml
from .bag_time_audit import audit_bag


def run_text(command, cwd=None):
    try:
        return subprocess.run(command, cwd=cwd, check=False, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def metadata_value(metadata, key, default=None):
    return metadata.get("rosbag2_bagfile_information", {}).get(key, default)


def time_record(value):
    ns = (value or {}).get("nanoseconds_since_epoch")
    if ns is None:
        return {"utc": None, "local": None}
    instant = dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc)
    return {"utc": instant.isoformat(), "local": instant.astimezone().isoformat(), "ns": ns}


def software_versions():
    return {
        "isaac_ros": "4.6 (container image depth/isaac_ros-cuvslam:4.6-d456)",
        "cuvslam_library": "15.0.0+74f0e317-modified (observed at startup)",
        "cuvslam_container": run_text([
            "docker", "ps", "--filter", "ancestor=depth/isaac_ros-cuvslam:4.6-d456",
            "--format", "{{.Image}} {{.ID}}"]),
        "rtabmap": run_text(["dpkg-query", "-W", "-f=${Version}", "ros-jazzy-rtabmap"]),
        "rosbag2": run_text(["dpkg-query", "-W", "-f=${Version}", "ros-jazzy-rosbag2"]),
        "gpu": run_text(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    }


def finalize(args):
    bag = Path(args.bag_path).resolve()
    metadata_path = bag / "metadata.yaml"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"rosbag metadata is missing: {metadata_path}")
    for name in ("manifest.yaml", "topic_statistics.json", "checksums.sha256"):
        if (bag / name).exists():
            raise FileExistsError(f"refusing to overwrite: {bag / name}")
    metadata = load_yaml(metadata_path)
    preflight = load_yaml(args.preflight)
    audit = audit_bag(bag, args.recording_config)
    with (bag / "topic_statistics.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    info = metadata.get("rosbag2_bagfile_information", {})
    duration_ns = (info.get("duration") or {}).get("nanoseconds", 0)
    start = time_record(info.get("starting_time"))
    end_ns = start.get("ns") + duration_ns if start.get("ns") is not None else None
    end = time_record({"nanoseconds_since_epoch": end_ns}) if end_ns is not None else {
        "utc": None, "local": None}
    actual_metadata_topics = {
        entry["topic_metadata"]["name"]: {
            "type": entry["topic_metadata"]["type"], "message_count": entry["message_count"]}
        for entry in info.get("topics_with_message_count", [])
    }
    missing_required = sorted(set(preflight.get("required_topics", [])) -
                              set(actual_metadata_topics))
    git_root = Path(args.workspace).resolve()
    manifest = {
        "schema_version": 1,
        "case_id": preflight["case_id"], "session_id": preflight["session_id"],
        "purpose": preflight["purpose"], "profile": preflight["profile"],
        "start": start, "end": end,
        "host": {"hostname": socket.gethostname(), "kernel": platform.release(),
                 "os": platform.platform()},
        "ros": preflight["environment"],
        "clock": preflight["clock"],
        "camera": preflight["camera"],
        "software": software_versions(),
        "git": {
            "head": run_text(["git", "rev-parse", "HEAD"], cwd=git_root),
            "dirty": bool(run_text(["git", "status", "--porcelain"], cwd=git_root)),
        },
        "map": preflight.get("map"), "route": preflight.get("route"),
        "topics": {
            "requested": preflight["requested_topics"],
            "graph_at_preflight": preflight["actual_topics"],
            "recorded": actual_metadata_topics,
            "missing_optional": preflight["missing_optional"],
            "missing_required": missing_required,
            "policy_excluded": (["/rtabmap/mapData",
                                 "/camera/camera/aligned_depth_to_color/image_raw"]
                                if preflight["profile"] == "camera_slam" else []),
        },
        "qos_at_preflight": preflight["qos"],
        "safety": {**preflight["safety"],
                   "forbidden_control_publishers": preflight["forbidden_control_publishers"]},
        "bag": {
            "storage_id": info.get("storage_identifier"),
            "compression": "rosbag2 file Zstd over fastwrite MCAP (4 worker threads)",
            "duration_ns": duration_ns,
            "size_bytes": sum(item.stat().st_size for item in bag.rglob("*") if item.is_file()),
            "message_count": info.get("message_count"),
            "split_files": info.get("relative_file_paths", []),
            "split_count": len(info.get("relative_file_paths", [])),
            "max_split_size_bytes": 4294967296,
            "max_split_duration_seconds": 900,
            "normal_termination": args.normal_termination,
            "reindex_required": False,
        },
        "time_audit_pass": audit["overall_pass"],
        "validation_status": ("PASS" if audit["overall_pass"] and
                              not missing_required else "FAIL"),
        "resource_statistics_file": "resource_statistics.json" if
                                    (bag / "resource_statistics.json").is_file() else None,
        "control_guard_file": "control_guard.json" if
                              (bag / "control_guard.json").is_file() else None,
    }
    write_yaml(bag / "manifest.yaml", manifest)

    checksum_files = sorted(item for item in bag.rglob("*") if item.is_file() and
                            item.name != "checksums.sha256")
    with (bag / "checksums.sha256").open("x", encoding="utf-8") as stream:
        for item in checksum_files:
            stream.write(f"{sha256(item)}  {item.relative_to(bag)}\n")
    print(f"FINALIZED: {bag}")
    print(f"TIME AUDIT: {'PASS' if audit['overall_pass'] else 'FAIL'}")
    return manifest, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--recording-config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--normal-termination", action="store_true")
    args = parser.parse_args()
    finalize(args)


if __name__ == "__main__":
    main()
