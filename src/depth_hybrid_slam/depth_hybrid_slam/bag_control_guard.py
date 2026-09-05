"""Count real vehicle command messages without publishing anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32


running = True


def stop(_signum, _frame):
    global running
    running = False


class Guard(Node):
    def __init__(self):
        super().__init__("competition_bag_control_guard")
        self.counts = {"/slam_drive": 0, "/slam_wheel": 0}
        self.create_subscription(Float32, "/slam_drive", self.drive, 10)
        self.create_subscription(Int32, "/slam_wheel", self.wheel, 10)

    def drive(self, _message):
        self.counts["/slam_drive"] += 1

    def wheel(self, _message):
        self.counts["/slam_wheel"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Guard()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    report = {"topics": node.counts, "total_real_control_commands": sum(node.counts.values())}
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
