"""Non-overwriting map-frame reference route recording and validation."""

import csv
import hashlib
import math
import os
from pathlib import Path

from .geometry import wrap_angle


REQUIRED_FIELDS = (
    "timestamp_sec", "timestamp_nanosec", "route_index",
    "map_x_m", "map_y_m", "map_z_m", "yaw_rad", "yaw_deg",
    "cumulative_distance_m", "direction", "localization_state",
    "localization_confidence", "tracking_valid", "reset_count",
    "nearest_rtabmap_node_id", "node_relative_x_m", "node_relative_y_m",
    "node_relative_yaw_rad", "pose_source", "valid",
)
FUTURE_FIELDS = (
    "speed_mps", "steering_deg", "encoder", "wheel_odom_x",
    "wheel_odom_y", "wheel_odom_yaw", "imu_roll", "imu_pitch", "imu_yaw",
)
# Legacy aliases contain the same map-frame values, never odom-frame values.
COMPATIBILITY_FIELDS = (
    "index", "timestamp", "x", "y", "z", "yaw", "record_valid",
    "drive_level", "mission_marker", "stop_line_id", "section_id",
)
FIELDS = REQUIRED_FIELDS + FUTURE_FIELDS + COMPATIBILITY_FIELDS


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp_ns(sample):
    if "timestamp_sec" in sample:
        return (int(sample["timestamp_sec"]) * 1_000_000_000 +
                int(sample.get("timestamp_nanosec", 0)))
    if "stamp_ns" in sample:
        return int(sample["stamp_ns"])
    return int(round(float(sample["timestamp"]) * 1.0e9))


def _coordinate(sample, new_name, old_name):
    return float(sample[new_name] if new_name in sample else sample[old_name])


class RouteRecorder:
    def __init__(self, output_path, min_interval_s=0.03,
                 min_distance_m=0.05, min_angle_rad=0.03,
                 min_yaw_deg=None, min_steering_delta_deg=1.0):
        self.path = Path(output_path).expanduser().resolve()
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(f"refusing to overwrite route: {self.path}")
        self.stream = open(self.partial, "x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self._sync()
        self.minimum_time_ns = int(float(min_interval_s) * 1.0e9)
        self.minimum_distance = float(min_distance_m)
        self.minimum_yaw = (math.radians(float(min_yaw_deg))
                            if min_yaw_deg is not None else float(min_angle_rad))
        self.minimum_steering = float(min_steering_delta_deg)
        self.last = None
        self.last_stamp_ns = None
        self.index = 0
        self.cumulative_distance = 0.0

    def _sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def _normalized(self, sample):
        stamp = _stamp_ns(sample)
        x = _coordinate(sample, "map_x_m", "x")
        y = _coordinate(sample, "map_y_m", "y")
        z = _coordinate(sample, "map_z_m", "z")
        yaw = _coordinate(sample, "yaw_rad", "yaw")
        confidence = float(sample.get("localization_confidence", 0.0))
        if not all(math.isfinite(value) for value in (x, y, z, yaw, confidence)):
            raise ValueError("route sample contains NaN or Inf")
        if self.last_stamp_ns is not None and stamp <= self.last_stamp_ns:
            raise ValueError("route timestamp must be strictly increasing")
        state = str(sample.get("localization_state",
                               sample.get("tracking_state", "INITIALIZING")))
        tracking = bool(sample.get("tracking_valid", state in ("TRACKING", "RELOCALIZED")))
        frame = str(sample.get("frame_id", "map"))
        valid = bool(sample.get("valid", tracking and frame == "map"))
        return dict(sample, stamp_ns=stamp, map_x_m=x, map_y_m=y, map_z_m=z,
                    yaw_rad=yaw, localization_state=state,
                    localization_confidence=confidence,
                    tracking_valid=tracking, valid=valid, frame_id=frame)

    def should_record(self, sample):
        current = self._normalized(sample)
        if self.last is None:
            return True
        important = (
            current["valid"] != self.last["valid"] or
            current["tracking_valid"] != self.last["tracking_valid"] or
            current["localization_state"] != self.last["localization_state"] or
            int(current.get("reset_count", 0)) != int(self.last.get("reset_count", 0)) or
            current.get("direction", "") != self.last.get("direction", "") or
            bool(current.get("mission_marker")) or
            bool(current.get("stop_line_id")) or
            current.get("section_id", "") != self.last.get("section_id", "")
        )
        moved = math.hypot(current["map_x_m"] - self.last["map_x_m"],
                           current["map_y_m"] - self.last["map_y_m"])
        turned = abs(wrap_angle(current["yaw_rad"] - self.last["yaw_rad"]))
        timed = current["stamp_ns"] - self.last["stamp_ns"]
        return important or (timed >= self.minimum_time_ns and
                             (moved >= self.minimum_distance or
                              turned >= self.minimum_yaw))

    def append(self, sample):
        current = self._normalized(sample)
        if not self.should_record(current):
            return False
        distance = 0.0
        if self.last is not None and current["valid"] and self.last["valid"]:
            distance = math.hypot(current["map_x_m"] - self.last["map_x_m"],
                                  current["map_y_m"] - self.last["map_y_m"])
        self.cumulative_distance += distance
        stamp = current["stamp_ns"]
        row = {name: current.get(name, "") for name in FIELDS}
        row.update({
            "timestamp_sec": stamp // 1_000_000_000,
            "timestamp_nanosec": stamp % 1_000_000_000,
            "route_index": self.index,
            "yaw_deg": math.degrees(current["yaw_rad"]),
            "cumulative_distance_m": self.cumulative_distance,
            "valid": current["valid"],
            "index": self.index,
            "timestamp": stamp / 1.0e9,
            "x": current["map_x_m"], "y": current["map_y_m"],
            "z": current["map_z_m"], "yaw": current["yaw_rad"],
            "record_valid": current["valid"],
        })
        self.writer.writerow(row)
        self._sync()
        self.last = current
        self.last_stamp_ns = stamp
        self.index += 1
        return True

    def finalize(self):
        self._sync()
        self.stream.close()
        os.replace(self.partial, self.path)
        directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "point_count": self.index,
            "total_distance_m": self.cumulative_distance,
            "route_csv_sha256": file_sha256(self.path),
        }

    def close(self):
        if not self.stream.closed:
            self._sync()
            self.stream.close()
