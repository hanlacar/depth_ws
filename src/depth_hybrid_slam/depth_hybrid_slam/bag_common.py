"""Shared, side-effect-free helpers for competition rosbag tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import yaml


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def validate_id(value, label="identifier"):
    if not SAFE_ID.fullmatch(value or ""):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_profile(topic_config, profile):
    profiles = topic_config.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown topic profile: {profile}")
    requested, expected, required = [], {}, set()
    for group_name in profiles[profile].get("groups", []):
        group = topic_config["groups"][group_name]
        for topic, msg_type in group.get("topics", {}).items():
            if topic not in expected:
                requested.append(topic)
            expected[topic] = msg_type
            if group.get("required", False):
                required.add(topic)
    return requested, expected, required


def percentile(values, percentage):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
