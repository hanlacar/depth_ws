#!/usr/bin/env python3
"""Discover and sample every live sensor_msgs/Image topic."""

import argparse
import hashlib
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image


class TopicSample:
    def __init__(self):
        self.count = 0
        self.first_wall = None
        self.last_wall = None
        self.first_stamp = None
        self.last_stamp = None
        self.hashes = set()
        self.width = self.height = 0
        self.encoding = ""
        self.frame_id = ""
        self.step = 0
        self.data_length = 0
        self.empty_payloads = 0
        self.nonincreasing_stamps = 0
        self._previous_stamp = None
        self.first_hash = None
        self.last_hash = None

    def add(self, message):
        now = time.monotonic()
        stamp = (int(message.header.stamp.sec) * 1_000_000_000 +
                 int(message.header.stamp.nanosec))
        if self.first_wall is None:
            self.first_wall, self.first_stamp = now, stamp
        self.last_wall, self.last_stamp = now, stamp
        self.count += 1
        # Full-payload hashing distinguishes a republished frozen frame from
        # genuine pixel motion; run this bounded audit only during validation.
        payload = bytes(message.data)
        digest = hashlib.sha256(payload).hexdigest()
        self.hashes.add(digest)
        if self.first_hash is None:
            self.first_hash = digest
        self.last_hash = digest
        if not payload:
            self.empty_payloads += 1
        if self._previous_stamp is not None and stamp <= self._previous_stamp:
            self.nonincreasing_stamps += 1
        self._previous_stamp = stamp
        self.width, self.height = int(message.width), int(message.height)
        self.encoding = str(message.encoding)
        self.frame_id = str(message.header.frame_id)
        self.step = int(message.step)
        self.data_length = len(payload)

    def document(self):
        elapsed = 0.0 if self.first_wall is None else self.last_wall-self.first_wall
        return {
            "messages": self.count,
            "hz": ((self.count-1)/elapsed if self.count > 1 and elapsed > 0 else 0.0),
            "width": self.width, "height": self.height,
            "encoding": self.encoding,
            "frame_id": self.frame_id,
            "step": self.step,
            "data_length": self.data_length,
            "empty_payloads": self.empty_payloads,
            "first_stamp_ns": self.first_stamp,
            "last_stamp_ns": self.last_stamp,
            "stamp_increased": self.count > 1 and self.nonincreasing_stamps == 0,
            "nonincreasing_stamps": self.nonincreasing_stamps,
            "unique_pixel_hashes": len(self.hashes),
            "first_pixel_hash": self.first_hash,
            "last_pixel_hash": self.last_hash,
        }


class ImageTopicAuditor(Node):
    def __init__(self):
        super().__init__("image_topic_auditor")
        self.samples = {}
        self._image_subscriptions = []

    def discover(self):
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        for topic, types in sorted(self.get_topic_names_and_types()):
            if "sensor_msgs/msg/Image" not in types or topic in self.samples:
                continue
            sample = TopicSample()
            self.samples[topic] = sample
            self._image_subscriptions.append(self.create_subscription(
                Image, topic, sample.add, qos))

    def report(self):
        report = {}
        for topic, sample in sorted(self.samples.items()):
            publishers = []
            for info in self.get_publishers_info_by_topic(topic):
                profile = info.qos_profile
                publishers.append({
                    "node": f"{info.node_namespace.rstrip('/')}/{info.node_name}",
                    "reliability": getattr(profile.reliability, "name",
                                           str(profile.reliability)),
                    "durability": getattr(profile.durability, "name",
                                          str(profile.durability)),
                    "history": getattr(profile.history, "name",
                                       str(profile.history)),
                    "depth": int(profile.depth),
                })
            report[topic] = {**sample.document(),
                             "publisher_count": len(publishers),
                             "publishers": publishers,
                             "auditor_subscription_qos": {
                                 "reliability": "BEST_EFFORT",
                                 "durability": "VOLATILE",
                                 "history": "KEEP_LAST",
                                 "depth": 1,
                             }}
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rclpy.init()
    node = ImageTopicAuditor()
    deadline = time.monotonic() + max(0.1, args.duration)
    next_discovery = 0.0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if time.monotonic() >= next_discovery:
                node.discover()
                next_discovery = time.monotonic() + 0.5
            rclpy.spin_once(node, timeout_sec=0.05)
        document = node.report()
        text = json.dumps(document, indent=2, sort_keys=True)
        print(text)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(text + "\n", encoding="utf-8")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
