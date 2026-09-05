#!/usr/bin/env python3
"""Record planner -> camera command -> MCU-manager lineage without Arduino."""

import argparse
import csv
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String


FIELDS = (
    "wall_time_ns", "source_stamp_ns", "planner_variant", "planner_state",
    "failure_reason", "raw_road_pixels", "refined_road_pixels",
    "safe_road_pixels", "path_point_count", "path_length_m",
    "lookahead_point", "required_steering_deg", "camera_drive",
    "camera_wheel", "mcu_drive", "mcu_wheel", "active_drive_source",
    "active_wheel_source", "safety_state",
)


class FlowAuditor(Node):
    def __init__(self):
        super().__init__("camera_mcu_flow_auditor")
        self.values = {name: None for name in FIELDS}
        self.rows = []
        self.camera_drive_samples = []
        self.camera_wheel_samples = []
        self.mcu_drive_samples = []
        self.mcu_wheel_samples = []
        self.invalid_nonzero = 0
        self.create_subscription(Float32, "/camera_drive", self._camera_drive, 10)
        self.create_subscription(Int32, "/camera_wheel", self._camera_wheel, 10)
        self.create_subscription(Float32, "/mcu/cmd_drive", self._mcu_drive, 10)
        self.create_subscription(Int32, "/mcu/cmd_wheel", self._mcu_wheel, 10)
        self.create_subscription(String, "/mcu/active_drive_source",
                                 lambda m: self._set("active_drive_source", m.data), 10)
        self.create_subscription(String, "/mcu/active_wheel_source",
                                 lambda m: self._set("active_wheel_source", m.data), 10)
        self.create_subscription(String, "/mcu/safety_state",
                                 lambda m: self._set("safety_state", m.data), 10)
        self.create_subscription(String, "/camera/bev/diagnostics", self._planner, 10)
        self.create_timer(0.05, self._sample)

    def _set(self, key, value):
        self.values[key] = value

    def _camera_drive(self, message):
        value = float(message.data)
        self.values["camera_drive"] = value
        self.camera_drive_samples.append(value)

    def _camera_wheel(self, message):
        value = int(message.data)
        self.values["camera_wheel"] = value
        self.camera_wheel_samples.append(value)

    def _mcu_drive(self, message):
        value = float(message.data)
        self.values["mcu_drive"] = value
        self.mcu_drive_samples.append((value, self.values["camera_drive"]))

    def _mcu_wheel(self, message):
        value = int(message.data)
        self.values["mcu_wheel"] = value
        self.mcu_wheel_samples.append((value, self.values["camera_wheel"]))

    def _planner(self, message):
        try:
            data = json.loads(message.data)
        except (TypeError, ValueError):
            data = {"state": "INVALID", "reasons": ["JSON_DECODE_ERROR"]}
        reasons = data.get("reasons") or []
        self.values.update({
            "source_stamp_ns": data.get("source_stamp_ns", data.get("stamp_ns")),
            "planner_variant": data.get("planner_variant", data.get("variant")),
            "planner_state": data.get("state"),
            "failure_reason": reasons[0] if reasons else "",
            "raw_road_pixels": data.get("raw_road_pixels"),
            "refined_road_pixels": data.get("refined_road_pixels"),
            "safe_road_pixels": data.get("safe_road_pixels"),
            "path_point_count": data.get("path_point_count"),
            "path_length_m": data.get("path_length_m"),
            "lookahead_point": json.dumps(data.get("target_point")),
            "required_steering_deg": data.get("required_steering_deg"),
        })

    def _sample(self):
        row = dict(self.values)
        row["wall_time_ns"] = time.time_ns()
        self.rows.append(row)
        if (row.get("planner_state") == "INVALID" and
                (row.get("camera_drive") not in (None, 0, 0.0) or
                 row.get("camera_wheel") not in (None, 0))):
            self.invalid_nonzero += 1

    def summary(self):
        def matched(samples):
            comparable = [(out, source) for out, source in samples
                          if source is not None]
            return {
                "samples": len(samples), "comparable": len(comparable),
                "matches": sum(out == source for out, source in comparable),
                "mismatches": sum(out != source for out, source in comparable),
            }
        wheels = self.camera_wheel_samples
        return {
            "rows": len(self.rows),
            "camera_drive_samples": len(self.camera_drive_samples),
            "camera_wheel_samples": len(wheels),
            "camera_wheel_min": min(wheels) if wheels else None,
            "camera_wheel_max": max(wheels) if wheels else None,
            "camera_left_negative_samples": sum(value < 0 for value in wheels),
            "camera_right_positive_samples": sum(value > 0 for value in wheels),
            "camera_wheel_limit_violations": sum(abs(value) > 22 for value in wheels),
            "mcu_drive_latest_input_comparison": matched(self.mcu_drive_samples),
            "mcu_wheel_latest_input_comparison": matched(self.mcu_wheel_samples),
            "invalid_nonzero_sample_rows": self.invalid_nonzero,
            "last_active_drive_source": self.values["active_drive_source"],
            "last_active_wheel_source": self.values["active_wheel_source"],
            "last_safety_state": self.values["safety_state"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    rclpy.init()
    node = FlowAuditor()
    deadline = time.monotonic() + max(0.1, args.duration)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(node.rows)
        summary = node.summary()
        text = json.dumps(summary, indent=2, sort_keys=True)
        print(text)
        if args.summary:
            Path(args.summary).write_text(text + "\n", encoding="utf-8")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
