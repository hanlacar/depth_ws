#!/usr/bin/env python3
"""Live isolated check of the latest real manager and virtual PWM odometry."""

import json
import math
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String


class Inspector(Node):
    def __init__(self):
        super().__init__("pwm_real_manager_inspector")
        self.values = {}
        self.pub_mode = self.create_publisher(String, "/drive_mode", 10)
        self.pub_drive = self.create_publisher(Float32, "/slam_drive", 10)
        self.pub_wheel = self.create_publisher(Int32, "/slam_wheel", 10)
        self.pub_stop = self.create_publisher(Bool, "/slam_stop", 10)
        topics = (
            (Float32, "/gps_drive", "gps_drive"),
            (Int32, "/gps_wheel", "gps_wheel"),
            (Bool, "/gps_stop", "gps_stop"),
            (Float32, "/prehardware/mcu/cmd_drive", "manager_drive"),
            (Int32, "/prehardware/mcu/cmd_wheel", "manager_wheel"),
            (Bool, "/prehardware/mcu/cmd_stop", "manager_stop"),
            (Float32, "/mcu/speed_mps", "speed"),
            (Float32, "/mcu/distance_m", "distance"),
            (Int32, "/mcu/encoder", "encoder"),
            (Bool, "/mcu/connected", "connected"),
            (Float32, "/depth_slam/virtual_mcu/stop_duration_s", "stop_duration"),
            (String, "/depth_slam/virtual_mcu/state", "state"),
            (Odometry, "/odom", "odom"),
        )
        for message_type, topic, name in topics:
            self.create_subscription(
                message_type, topic,
                lambda message, key=name: self.values.__setitem__(key, message), 10)

    def command_for(self, stage, wheel, stop, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.pub_mode.publish(String(data="2"))
            self.pub_drive.publish(Float32(data=float(stage)))
            self.pub_wheel.publish(Int32(data=int(wheel)))
            self.pub_stop.publish(Bool(data=bool(stop)))
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise RuntimeError("real-manager PWM inspection timed out")


def close(value, expected, tolerance=0.002):
    return math.isclose(float(value), float(expected), abs_tol=tolerance)


def main():
    rclpy.init()
    node = Inspector()
    report = {"stages": {}}
    try:
        node.command_for(0, 0, True, 1.5)
        node.wait_for(lambda: bool(node.values.get("connected", Bool()).data))
        for stage, pwm, speed in ((1, 50, 0.527), (2, 75, 0.791),
                                  (3, 100, 1.055)):
            node.command_for(0, 0, True, 0.4)
            start_distance = float(node.values["distance"].data)
            start_encoder = int(node.values["encoder"].data)
            node.command_for(stage, 0, False, 1.0)
            actual_speed = float(node.values["speed"].data)
            actual_distance = float(node.values["distance"].data) - start_distance
            actual_encoder = int(node.values["encoder"].data) - start_encoder
            assert close(node.values["gps_drive"].data, stage)
            assert close(node.values["manager_drive"].data, stage)
            assert close(actual_speed, speed)
            report["stages"][str(stage)] = {
                "pwm": pwm, "speed_mps": actual_speed,
                "live_sample_distance_m": actual_distance,
                "encoder_delta": actual_encoder,
            }

        node.command_for(1, 12, False, 0.3)
        assert node.values["gps_wheel"].data == -12
        assert node.values["manager_wheel"].data == 12
        assert node.values["odom"].twist.twist.angular.z > 0.0
        node.command_for(1, -12, False, 0.3)
        assert node.values["gps_wheel"].data == 12
        assert node.values["manager_wheel"].data == -12
        assert node.values["odom"].twist.twist.angular.z < 0.0

        before_reverse_encoder = int(node.values["encoder"].data)
        node.command_for(-1, 0, False, 0.2)
        assert close(node.values["speed"].data, 0.0)
        node.command_for(0, 0, True, 0.4)
        held = float(node.values["stop_duration"].data)
        assert held >= 3.0
        node.command_for(-1, 0, False, 0.3)
        assert close(node.values["speed"].data, -0.527)
        assert int(node.values["encoder"].data) > before_reverse_encoder

        node.command_for(0, 0, True, 0.4)
        reverse_to_forward_hold = float(node.values["stop_duration"].data)
        assert reverse_to_forward_hold >= 3.0
        node.command_for(1, 0, False, 0.2)
        assert close(node.values["speed"].data, 0.527)

        node.command_for(1, 0, True, 0.2)
        assert node.values["gps_stop"].data
        assert node.values["manager_stop"].data
        assert close(node.values["speed"].data, 0.0)

        state = json.loads(node.values["state"].data)
        assert not state["publisher_conflicts"]
        node_names = {name for name, _namespace in node.get_node_names_and_namespaces()}
        assert "mcu_bridge" not in node_names
        assert len(node.get_publishers_info_by_topic("/odom")) == 1
        assert len(node.get_publishers_info_by_topic("/mcu/encoder")) == 1
        report.update({
            "reverse_pwm": -50,
            "reverse_speed_mps": -0.527,
            "encoder_unsigned_cumulative": True,
            "forward_to_reverse_stop_s": held,
            "reverse_to_forward_stop_s": reverse_to_forward_hold,
            "left_manager_wheel_positive": True,
            "right_manager_wheel_negative": True,
            "stop_chain": "PASS",
            "serial_bridge_running": False,
            "publisher_conflicts": [],
        })
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
