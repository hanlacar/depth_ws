#!/usr/bin/env python3
"""Collect low-overhead runtime rates, latency, CPU and RAM for D0/D1/D2."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String


TARGET_PROCESS_MARKERS = (
    "video_publisher.py", "fake_imu_publisher.py",
    "camera_yolo_inference_node", "direct_bev_planner_node",
    "direct_bev_controller_node", "direct_bev_drive_node",
    "bev_wheel_selector_node",
)


class PerformanceCollector(Node):
    def __init__(self):
        super().__init__("debug_performance_collector")
        self.started = time.monotonic()
        self.latest_realtime = {}
        self.latest_planner = {}
        self.pipeline_latency = []
        self.drive_times = []
        self.wheel_times = []
        self.cpu_samples = []
        self.rss_samples = []
        self.processes = {}
        self.create_subscription(String, "/camera/realtime_fps", self._realtime, 10)
        self.create_subscription(String, "/camera/bev/diagnostics", self._planner, 10)
        self.create_subscription(Float32, "/camera/pipeline_latency_ms",
                                 lambda m: self.pipeline_latency.append(float(m.data)), 10)
        self.create_subscription(Float32, "/camera_drive",
                                 lambda m: self.drive_times.append(time.monotonic()), 10)
        self.create_subscription(Int32, "/camera_wheel",
                                 lambda m: self.wheel_times.append(time.monotonic()), 10)
        self.create_timer(1.0, self._resources)

    def _realtime(self, message):
        try:
            self.latest_realtime = json.loads(message.data)
        except (TypeError, ValueError):
            pass

    def _planner(self, message):
        try:
            self.latest_planner = json.loads(message.data)
        except (TypeError, ValueError):
            pass

    def _resources(self):
        for process in psutil.process_iter(("pid", "cmdline")):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                if any(marker in command for marker in TARGET_PROCESS_MARKERS):
                    if process.pid not in self.processes:
                        self.processes[process.pid] = process
                        process.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        cpu = 0.0
        rss = 0
        for pid, process in list(self.processes.items()):
            try:
                cpu += process.cpu_percent(None)
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.processes.pop(pid, None)
        self.cpu_samples.append(cpu)
        self.rss_samples.append(rss)

    @staticmethod
    def rate(times):
        if len(times) < 2:
            return 0.0
        elapsed = times[-1]-times[0]
        return 0.0 if elapsed <= 0 else (len(times)-1)/elapsed

    def report(self, condition, configured_input_fps=None):
        latency = np.asarray(self.pipeline_latency, float)
        cpu = np.asarray(self.cpu_samples[1:], float)
        rss = np.asarray(self.rss_samples[1:], float)/(1024.0**2)
        realtime = self.latest_realtime
        inference = realtime.get("inference_unique_fps", {})
        semantic = realtime.get("semantic_unique_fps", {})
        planner = self.latest_planner
        return {
            "condition": condition,
            "duration_sec": time.monotonic()-self.started,
            "input_fps": realtime.get("input_fps", configured_input_fps),
            "semantic_fps": semantic.get("1s", {}).get(
                "header_fps", planner.get("semantic_input_fps")),
            "inference_fps": inference.get("1s", {}).get("header_fps"),
            "planner_fps": planner.get("planner_processing_fps"),
            "camera_drive_fps": self.rate(self.drive_times),
            "camera_wheel_fps": self.rate(self.wheel_times),
            "pipeline_latency_ms": {
                "count": int(latency.size),
                "mean": None if not latency.size else float(latency.mean()),
                "p95": None if not latency.size else float(np.percentile(latency, 95)),
            },
            "cpu_percent_sum": {
                "mean": None if not cpu.size else float(cpu.mean()),
                "p95": None if not cpu.size else float(np.percentile(cpu, 95)),
            },
            "rss_mib_sum": {
                "mean": None if not rss.size else float(rss.mean()),
                "max": None if not rss.size else float(rss.max()),
            },
            "gpu": "NOT RUN: NVIDIA driver unavailable",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--configured-input-fps", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = PerformanceCollector()
    deadline = time.monotonic()+args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        document = node.report(args.condition, args.configured_input_fps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(document, indent=2, sort_keys=True)
        args.output.write_text(text+"\n", encoding="utf-8")
        print(text)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
