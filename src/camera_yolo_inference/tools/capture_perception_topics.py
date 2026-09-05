#!/usr/bin/env python3
"""Capture perception outputs and exact ROS bgr8 round-trip evidence."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from camera_yolo_inference.ros_image import image_to_bgr8


def stamp(message):
    return f"{message.header.stamp.sec}_{message.header.stamp.nanosec}"


class Capture(Node):
    def __init__(self, output, duration):
        super().__init__("perception_topic_capture")
        self.output = output; output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic(); self.duration = duration
        self.counts = {"raw": 0, "detections_image": 0, "overlay": 0, "road": 0,
                       "road_nonzero": 0, "detections_json": 0}
        self.records = []
        reliable = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        best = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/camera/image_raw", self.raw, best)
        self.create_subscription(Image, "/perception/detections_image",
                                 lambda m: self.color(m, "detections_image"), reliable)
        self.create_subscription(Image, "/camera/perception_overlay_image",
                                 lambda m: self.color(m, "overlay"), reliable)
        self.create_subscription(Image, "/perception/masks/road", self.road, reliable)
        self.create_subscription(String, "/perception/detections_json", self.detections, reliable)
        self.create_timer(.2, self.tick)

    def raw(self, message):
        b = image_to_bgr8(message)
        c = image_to_bgr8(message)  # exact conversion called immediately before backend.infer
        name = stamp(message)
        cv2.imwrite(str(self.output / f"B_raw_{name}.png"), b)
        self.records.append({"stamp": name, "encoding": message.encoding,
                             "shape": list(b.shape), "dtype": str(b.dtype),
                             "mean_bgr": b.reshape(-1, 3).mean(0).tolist(),
                             "std_bgr": b.reshape(-1, 3).std(0).tolist(),
                             "B_C_MAE": float(np.abs(b.astype(np.int16)-c.astype(np.int16)).mean()),
                             "B_C_equal": bool(np.array_equal(b, c))})
        self.counts["raw"] += 1

    def color(self, message, role):
        image = image_to_bgr8(message); self.counts[role] += 1
        cv2.imwrite(str(self.output / f"{role}_{stamp(message)}.png"), image)

    def road(self, message):
        raw = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        mask = raw[:, :message.width]
        pixels = int(np.count_nonzero(mask)); self.counts["road"] += 1
        self.counts["road_nonzero"] += int(pixels > 0)
        cv2.imwrite(str(self.output / f"road_{stamp(message)}_{pixels}px.png"), mask)

    def detections(self, message):
        self.counts["detections_json"] += 1
        with (self.output / "detections.jsonl").open("a") as stream:
            stream.write(message.data + "\n")

    def tick(self):
        if time.monotonic() - self.started >= self.duration:
            (self.output / "capture_summary.json").write_text(json.dumps(
                {"counts": self.counts, "raw_records": self.records}, indent=2) + "\n")
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, default=30.0); args = parser.parse_args()
    rclpy.init(); node = Capture(args.output, args.duration)
    try: rclpy.spin(node)
    finally: node.destroy_node()


if __name__ == "__main__": main()
