#!/usr/bin/env python3
"""Save a few spaced Image frames and their hashes for visual validation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self, topic, output, count, interval):
        super().__init__("image_sample_capture")
        self.topic = topic
        self.output = output
        self.target = count
        self.interval = interval
        self.bridge = CvBridge()
        self.last_saved = float("-inf")
        self.records = []
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, topic, self.receive, qos)

    def receive(self, message):
        now = time.monotonic()
        if now-self.last_saved < self.interval or len(self.records) >= self.target:
            return
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        payload = bytes(message.data)
        stamp_ns = (int(message.header.stamp.sec)*1_000_000_000 +
                    int(message.header.stamp.nanosec))
        path = self.output / f"sample_{len(self.records)+1:02d}_{stamp_ns}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")
        self.records.append({
            "path": str(path), "stamp_ns": stamp_ns,
            "encoding": message.encoding, "width": message.width,
            "height": message.height, "step": message.step,
            "data_length": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        self.last_saved = now

    def complete(self):
        return len(self.records) >= self.target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Capture(args.topic, args.output, max(1, args.count),
                   max(0.0, args.interval))
    deadline = time.monotonic()+max(0.1, args.timeout)
    try:
        while rclpy.ok() and time.monotonic() < deadline and not node.complete():
            rclpy.spin_once(node, timeout_sec=0.1)
        document = {"topic": args.topic, "samples": node.records,
                    "complete": node.complete()}
        text = json.dumps(document, indent=2, sort_keys=True)
        (args.output/"samples.json").write_text(text+"\n", encoding="utf-8")
        print(text)
        if not node.complete():
            raise SystemExit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
