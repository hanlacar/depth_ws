#!/usr/bin/env python3
"""Observe one circular pass of the running MP4 mission pipeline."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VideoMissionAudit(Node):
    def __init__(self, section, output, timeout):
        super().__init__("camera_mission_video_audit")
        self.section = section
        self.output = Path(output)
        self.deadline = time.monotonic()+timeout
        self.frame = None; self.initial_frame = None
        self.loop = None; self.initial_loop = None
        self.semantic_frames = set(); self.decision_frames = set()
        self.sign_raw = Counter(); self.sign_filtered = Counter()
        self.traffic_raw = Counter(); self.traffic_filtered = Counter()
        self.path_shapes = Counter(); self.turn_directions = Counter()
        self.states = Counter(); self.failure_reasons = Counter()
        self.transitions = []; self.last_state = None
        self.safety_nonzero = 0
        self.section_pub = self.create_publisher(
            String, "/camera/mission/section", 10)
        self.create_subscription(String, "/video_test/publisher_diagnostics",
                                 self.video, 10)
        self.create_subscription(String, "/camera/mission/diagnostics",
                                 self.mission, 10)
        self.create_subscription(String, "/camera/mission/decision_diagnostics",
                                 self.decision, 10)
        self.create_timer(.2, self.tick)

    @staticmethod
    def decode(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return {}

    def video(self, message):
        data = self.decode(message)
        self.frame = int(data.get("current_frame_index", -1))
        self.loop = int(data.get("loop_count", -1))
        self.source_frame_count = int(data.get("source_frame_count", 0))
        if self.initial_frame is None and self.frame >= 0 and self.loop >= 0:
            self.initial_frame, self.initial_loop = self.frame, self.loop

    def mission(self, message):
        data = self.decode(message)
        if self.frame is None:
            return
        self.semantic_frames.add((self.loop, self.frame))
        self.sign_raw[bool(data.get("sign_raw_detected"))] += 1
        self.sign_filtered[bool(data.get("sign_detected"))] += 1
        self.traffic_raw[str(data.get("traffic_light_raw_state", "UNKNOWN"))] += 1
        self.traffic_filtered[str(data.get("traffic_light_state", "UNKNOWN"))] += 1

    def decision(self, message):
        data = self.decode(message)
        if self.frame is None:
            return
        self.decision_frames.add((self.loop, self.frame))
        state = str(data.get("mission_state_before_safety",
                             data.get("decision_state", "UNKNOWN")))
        self.states[state] += 1
        self.path_shapes[str(data.get("path_shape", "UNKNOWN"))] += 1
        self.turn_directions[str(data.get("turn_direction", "NONE"))] += 1
        reason = str(data.get("failure_reason", ""))
        if reason:
            self.failure_reasons[reason] += 1
        if data.get("safety_blocked") and abs(float(
                data.get("effective_drive_if_connected", 0.0))) > 1.0e-6:
            self.safety_nonzero += 1
        if state != self.last_state:
            self.transitions.append({"loop": self.loop, "frame": self.frame,
                                     "state": state})
            self.last_state = state

    def tick(self):
        self.section_pub.publish(String(data=self.section))
        complete = (self.initial_loop is not None and self.loop is not None and
                    self.loop > self.initial_loop and
                    self.frame is not None and self.frame >= self.initial_frame)
        if complete or time.monotonic() >= self.deadline:
            report = {
                "complete_loop": complete,
                "section": self.section,
                "initial_loop": self.initial_loop,
                "initial_frame": self.initial_frame,
                "final_loop": self.loop, "final_frame": self.frame,
                "source_frame_count": getattr(self, "source_frame_count", 0),
                "observed_semantic_frame_buckets": len(self.semantic_frames),
                "observed_decision_frame_buckets": len(self.decision_frames),
                "sign_raw": dict(self.sign_raw),
                "sign_filtered": dict(self.sign_filtered),
                "traffic_raw": dict(self.traffic_raw),
                "traffic_filtered": dict(self.traffic_filtered),
                "path_shapes": dict(self.path_shapes),
                "turn_directions": dict(self.turn_directions),
                "mission_states": dict(self.states),
                "failure_reasons": dict(self.failure_reasons),
                "safety_blocked_nonzero_effective": self.safety_nonzero,
                "state_transitions": self.transitions,
            }
            self.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="ACCELERATION")
    parser.add_argument("--output", default="mission_video_audit.json")
    parser.add_argument("--timeout", type=float, default=270.0)
    args = parser.parse_args()
    rclpy.init()
    node = VideoMissionAudit(args.section.upper(), args.output, args.timeout)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
