#!/usr/bin/env python3
"""Synthetic ROS graph check for the advisory late-red decision node."""

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String


class LateRedScenario(Node):
    def __init__(self, output):
        super().__init__("late_red_synthetic_publisher")
        self.output = Path(output); self.started = time.monotonic()
        self.states = []; self.diagnostics = []; self.finished = False
        self.pubs = {
            "section": self.create_publisher(String, "/camera/mission/section", 10),
            "lines": self.create_publisher(Float32MultiArray,
                                            "/camera/mission/stop_line_distances_m", 10),
            "stop": self.create_publisher(Bool,
                                           "/camera/mission/stop_line_detected", 10),
            "traffic": self.create_publisher(String,
                                              "/camera/mission/traffic_light", 10),
            "mission": self.create_publisher(String,
                                              "/camera/mission/diagnostics", 10),
            "valid": self.create_publisher(Bool, "/camera/bev/valid", 10),
            "planner": self.create_publisher(String,
                                              "/camera/bev/diagnostics", 10),
            "drive": self.create_publisher(
                Float32, "/camera/candidate/test/drive", 10),
            "wheel": self.create_publisher(
                Int32, "/camera/candidate/test/wheel", 10),
            "encoder": self.create_publisher(Int32, "/mcu/encoder", 10),
            "speed": self.create_publisher(Float32, "/mcu/speed_mps", 10),
            "distance": self.create_publisher(Float32, "/mcu/distance_m", 10),
            "speed_valid": self.create_publisher(Bool, "/mcu/speed_valid", 10),
        }
        self.create_subscription(String, "/camera/mission/decision_state",
                                 lambda m: self.states.append(m.data), 10)
        self.create_subscription(String, "/camera/mission/decision_diagnostics",
                                 self.on_diagnostics, 10)
        self.create_timer(.05, self.tick)

    def on_diagnostics(self, message):
        try:
            self.diagnostics.append(json.loads(message.data))
        except ValueError:
            pass

    def publish(self, key, message):
        self.pubs[key].publish(message)

    def tick(self):
        elapsed = time.monotonic()-self.started
        section = "INTERSECTION"; traffic = "R"; line = 2.0
        speed = .5; distance = 10.0; tf_valid = True
        if .8 <= elapsed < 1.4:
            line, speed, distance = 1.0, .2, 10.5
        elif 1.4 <= elapsed < 1.7:
            line, speed, distance, traffic = 1.0, 0.0, 10.5, "UNKNOWN"
        elif 1.7 <= elapsed < 2.1:
            line, speed, distance, traffic = 1.0, 0.0, 10.5, "G"
        elif 2.1 <= elapsed < 2.4:
            section, line, speed, distance, traffic = \
                "NORMAL", 3.0, 1.0, 10.0, "G"
        elif 2.4 <= elapsed < 3.0:
            line, speed, distance = .3, 1.0, 20.0
        elif elapsed >= 3.0:
            line, speed, distance, traffic, tf_valid = \
                None, 1.0, 20.4, "UNKNOWN", False
        stamp = self.get_clock().now().nanoseconds
        self.publish("section", String(data=section))
        self.publish("lines", Float32MultiArray(
            data=[] if line is None else [line]))
        self.publish("stop", Bool(data=line is not None))
        self.publish("traffic", String(data=traffic))
        self.publish("mission", String(data=json.dumps({
            "input_timestamp": stamp*1.0e-9,
            "stop_line_tf_valid": tf_valid, "imu_pitch_deg": 0.0})))
        self.publish("valid", Bool(data=True))
        self.publish("planner", String(data=json.dumps({
            "state": "VALID", "required_steering_deg": -3.0,
            "source_stamp_ns": stamp})))
        self.publish("drive", Float32(data=2.0))
        self.publish("wheel", Int32(data=3))
        self.publish("encoder", Int32(data=int(1000+distance*199.8)))
        self.publish("speed", Float32(data=speed))
        self.publish("distance", Float32(data=distance))
        self.publish("speed_valid", Bool(data=True))
        if elapsed >= 3.8 and not self.finished:
            self.finished = True
            required = {"LATE_RED_CAN_STOP_AT_TARGET", "RED_DECELERATE",
                        "RED_STOPPED", "GREEN_PROCEED",
                        "LATE_RED_COMMIT_TO_CROSS", "LINE_CROSSED",
                        "INTERSECTION_COMPLETE"}
            observed = set(self.states)
            nonzero_safe = sum(1 for d in self.diagnostics if
                               d.get("safety_blocked") and abs(float(
                                   d.get("effective_drive_if_connected", 0))) > 1e-6)
            report = {"required_states": sorted(required),
                      "observed_states": sorted(observed),
                      "missing_states": sorted(required-observed),
                      "safety_blocked_nonzero_effective": nonzero_safe,
                      "diagnostic_samples": len(self.diagnostics),
                      "pass": required <= observed and nonzero_safe == 0}
            self.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="late_red_ros_report.json")
    args = parser.parse_args(); rclpy.init()
    node = LateRedScenario(args.output)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
