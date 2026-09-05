#!/usr/bin/env python3
"""Prove that a ROS Image stream changes in stamp and pixel content."""

import argparse
import hashlib
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class FrameHashDiagnostic(Node):
    def __init__(self, topic, samples):
        super().__init__("frame_hash_diagnostic")
        self.samples_required = int(samples)
        self.rows = []
        self.previous = None
        self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data)

    def _on_image(self, message):
        payload = bytes(message.data)
        array = np.frombuffer(payload, dtype=np.uint8)
        mad = None
        if self.previous is not None and self.previous.size == array.size:
            mad = float(np.mean(np.abs(
                array.astype(np.int16) - self.previous.astype(np.int16))))
        stamp_ns = (int(message.header.stamp.sec) * 1_000_000_000 +
                    int(message.header.stamp.nanosec))
        row = {
            "sample": len(self.rows), "stamp_ns": stamp_ns,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mean_abs_pixel_difference": mad,
            "width": int(message.width), "height": int(message.height),
            "encoding": message.encoding,
        }
        self.rows.append(row)
        self.previous = array.copy()
        print(json.dumps(row, separators=(",", ":")), flush=True)

    @property
    def complete(self):
        return len(self.rows) >= self.samples_required


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/image_raw")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    rclpy.init()
    node = FrameHashDiagnostic(args.topic, args.samples)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.complete:
            raise SystemExit(
                f"timeout: received {len(node.rows)}/{args.samples} frames")
        stamps = [row["stamp_ns"] for row in node.rows]
        hashes = [row["sha256"] for row in node.rows]
        summary = {
            "samples": len(node.rows),
            "strictly_increasing_stamps": all(
                right > left for left, right in zip(stamps, stamps[1:])),
            "unique_pixel_hashes": len(set(hashes)),
            "all_consecutive_pixels_changed": all(
                row["mean_abs_pixel_difference"] not in (None, 0.0)
                for row in node.rows[1:]),
        }
        print("SUMMARY " + json.dumps(summary, separators=(",", ":")))
        if (not summary["strictly_increasing_stamps"] or
                summary["unique_pixel_hashes"] != len(node.rows)):
            raise SystemExit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
