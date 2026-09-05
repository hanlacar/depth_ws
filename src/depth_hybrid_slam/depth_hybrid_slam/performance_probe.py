"""Measure real message arrivals, source timestamps, duplicates, and latency."""

import argparse
from collections import defaultdict
import json
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

from .ros_helpers import safe_shutdown


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, int(math.ceil(fraction*len(ordered))-1))]


class Probe(Node):
    def __init__(self, topics):
        super().__init__("depth_performance_probe")
        self.requested = list(topics)
        self.arrivals = defaultdict(list)
        self.stamps = defaultdict(list)
        self.latencies = defaultdict(list)
        self.poses = defaultdict(list)
        self.scalar_values = defaultdict(list)
        self.subscriptions_by_topic = {}

    def discover(self, timeout=5.0):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline and len(self.subscriptions_by_topic) < len(self.requested):
            rclpy.spin_once(self, timeout_sec=0.1)
            graph = dict(self.get_topic_names_and_types())
            for topic in self.requested:
                if topic in self.subscriptions_by_topic or not graph.get(topic):
                    continue
                message_type = get_message(graph[topic][0])
                self.subscriptions_by_topic[topic] = self.create_subscription(
                    message_type, topic, lambda msg, name=topic: self.receive(name, msg),
                    qos_profile_sensor_data)

    def receive(self, topic, message):
        self.arrivals[topic].append(time.monotonic())
        if hasattr(message, "header") and hasattr(message.header, "stamp"):
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1.0e-9
            self.stamps[topic].append(stamp)
            now = self.get_clock().now().nanoseconds*1.0e-9
            latency = (now-stamp)*1000.0
            if 0.0 <= latency < 60_000.0:
                self.latencies[topic].append(latency)
        if hasattr(message, "pose"):
            pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
            if hasattr(pose, "position") and hasattr(pose, "orientation"):
                p, q = pose.position, pose.orientation
                yaw = math.atan2(2.0*(q.w*q.z+q.x*q.y),
                                 1.0-2.0*(q.y*q.y+q.z*q.z))
                self.poses[topic].append((float(p.x), float(p.y), float(p.z), yaw))
        if hasattr(message, "data") and isinstance(message.data, (bool, int, float, str)):
            self.scalar_values[topic].append(message.data)

    def report(self, duration):
        result = {"duration_s": duration, "missing": sorted(set(self.requested)-set(self.subscriptions_by_topic))}
        for topic in self.requested:
            arrivals = self.arrivals[topic]
            periods = [b-a for a, b in zip(arrivals, arrivals[1:]) if b > a]
            stamps = self.stamps[topic]
            poses = self.poses[topic]
            translation_drift = None
            yaw_drift_deg = None
            path_length = None
            if len(poses) > 1:
                first, last = poses[0], poses[-1]
                translation_drift = math.sqrt(sum((last[i]-first[i])**2 for i in range(3)))
                yaw_delta = math.atan2(math.sin(last[3]-first[3]), math.cos(last[3]-first[3]))
                yaw_drift_deg = math.degrees(yaw_delta)
                path_length = sum(math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
                                  for a, b in zip(poses, poses[1:]))
            scalar = self.scalar_values[topic]
            result[topic] = {
                "count": len(arrivals),
                "average_hz": ((len(arrivals)-1)/(arrivals[-1]-arrivals[0])
                               if len(arrivals) > 1 else 0.0),
                "minimum_instant_hz": (1.0/max(periods) if periods else 0.0),
                "p95_period_ms": (1000.0*percentile(periods, .95) if periods else None),
                "average_latency_ms": (statistics.fmean(self.latencies[topic])
                                       if self.latencies[topic] else None),
                "p95_latency_ms": percentile(self.latencies[topic], .95),
                "duplicate_source_stamps": len(stamps)-len(set(stamps)),
                "translation_drift_m": translation_drift,
                "yaw_drift_deg": yaw_drift_deg,
                "path_length_m": path_length,
                "true_percent": (100.0*sum(bool(value) for value in scalar)/len(scalar)
                                 if scalar and all(isinstance(v, bool) for v in scalar) else None),
                "last_value": scalar[-1] if scalar else None,
                "maximum_value": (max(scalar) if scalar and
                                  all(isinstance(v, (bool, int, float)) for v in scalar)
                                  else None),
            }
        return result


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("topics", nargs="+")
    options, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = Probe(options.topics)
    try:
        node.discover()
        start = time.monotonic()
        try:
            while time.monotonic()-start < options.duration:
                rclpy.spin_once(node, timeout_sec=0.05)
        except KeyboardInterrupt:
            pass
        print(json.dumps(node.report(time.monotonic()-start), indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
