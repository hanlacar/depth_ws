#!/usr/bin/env python3
"""Black-box camera-source forwarding/failsafe check for mcu_manager only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String


class ManagerFailsafeProbe(Node):
    def __init__(self, drive, wheel, publish_sec):
        super().__init__("mcu_manager_failsafe_probe")
        self.drive_value = float(drive)
        self.wheel_value = int(wheel)
        self.publish_sec = float(publish_sec)
        self.started = time.monotonic()
        self.stopped = None
        self.drive_zero = None
        self.wheel_zero = None
        self.forwarded_drive = False
        self.forwarded_wheel = False
        self.drive_source = None
        self.wheel_source = None
        self.drive_pub = self.create_publisher(
            Float32, "/camera/candidate/test/drive", 10)
        self.wheel_pub = self.create_publisher(
            Int32, "/camera/candidate/test/wheel", 10)
        self.create_subscription(Float32, "/mcu/cmd_drive", self._drive, 10)
        self.create_subscription(Int32, "/mcu/cmd_wheel", self._wheel, 10)
        self.create_subscription(String, "/mcu/active_drive_source",
                                 lambda m: setattr(self, "drive_source", m.data), 10)
        self.create_subscription(String, "/mcu/active_wheel_source",
                                 lambda m: setattr(self, "wheel_source", m.data), 10)
        self.timer = self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.stopped is not None:
            return
        if time.monotonic()-self.started < self.publish_sec:
            self.drive_pub.publish(Float32(data=self.drive_value))
            self.wheel_pub.publish(Int32(data=self.wheel_value))
            return
        self.stopped = time.monotonic()
        self.destroy_publisher(self.drive_pub)
        self.destroy_publisher(self.wheel_pub)

    def _drive(self, message):
        now = time.monotonic()
        self.forwarded_drive |= float(message.data) == self.drive_value
        if self.stopped is not None and float(message.data) == 0.0 and self.drive_zero is None:
            self.drive_zero = now

    def _wheel(self, message):
        now = time.monotonic()
        self.forwarded_wheel |= int(message.data) == self.wheel_value
        if self.stopped is not None and int(message.data) == 0 and self.wheel_zero is None:
            self.wheel_zero = now

    def complete(self):
        return self.drive_zero is not None and self.wheel_zero is not None

    def report(self):
        delay = lambda value: (None if value is None or self.stopped is None else
                               value-self.stopped)
        return {
            "input_drive": self.drive_value, "input_wheel": self.wheel_value,
            "forwarded_drive": self.forwarded_drive,
            "forwarded_wheel": self.forwarded_wheel,
            "active_drive_source": self.drive_source,
            "active_wheel_source": self.wheel_source,
            "drive_failsafe_sec": delay(self.drive_zero),
            "wheel_failsafe_sec": delay(self.wheel_zero),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", type=float, default=2.0)
    parser.add_argument("--wheel", type=int, default=-5)
    parser.add_argument("--publish-sec", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--maximum-failsafe-sec", type=float, default=0.6)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    rclpy.init()
    reports = []
    try:
        for trial in range(args.trials):
            node = ManagerFailsafeProbe(
                args.drive, args.wheel, args.publish_sec)
            deadline = time.monotonic()+args.timeout
            while (rclpy.ok() and time.monotonic() < deadline and
                   not node.complete()):
                rclpy.spin_once(node, timeout_sec=0.05)
            report = node.report()
            report["trial"] = trial+1
            report["passed"] = (
                report["forwarded_drive"] and report["forwarded_wheel"] and
                report["drive_failsafe_sec"] is not None and
                report["wheel_failsafe_sec"] is not None and
                report["drive_failsafe_sec"] <= args.maximum_failsafe_sec and
                report["wheel_failsafe_sec"] <= args.maximum_failsafe_sec)
            reports.append(report)
            node.destroy_node()
            time.sleep(0.1)
        delays = np.asarray([
            max(report["drive_failsafe_sec"], report["wheel_failsafe_sec"])
            for report in reports if report["drive_failsafe_sec"] is not None
            and report["wheel_failsafe_sec"] is not None], float)
        document = {
            "trials": reports,
            "summary": {
                "requested_trials": args.trials,
                "completed_trials": int(delays.size),
                "mean_sec": None if not delays.size else float(delays.mean()),
                "p95_sec": None if not delays.size else float(np.percentile(delays, 95)),
                "max_sec": None if not delays.size else float(delays.max()),
                "over_0_5_sec": int(np.count_nonzero(delays > 0.5)),
                "over_configured_maximum": int(np.count_nonzero(
                    delays > args.maximum_failsafe_sec)),
                "all_forwarded": all(report["forwarded_drive"] and
                                       report["forwarded_wheel"]
                                       for report in reports),
                "passed": len(reports) == args.trials and
                          all(report["passed"] for report in reports),
            },
        }
        text = json.dumps(document, indent=2, sort_keys=True)
        print(text)
        if args.output:
            Path(args.output).write_text(text+"\n", encoding="utf-8")
        if not document["summary"]["passed"]:
            raise SystemExit(2)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
