"""Single owner of the internal physical /slam_* command contract."""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .prehardware_core import BehaviorDecision, select_behavior


class BehaviorSelectorNode(Node):
    def __init__(self):
        super().__init__("depth_behavior_selector")
        for name, default in (
                ("armed", False), ("publish_hz", 20.0),
                ("command_timeout_s", 0.5), ("state_timeout_s", 0.5),
                ("validator_timeout_s", 0.75)):
            self.declare_parameter(name, default)
        self.armed = bool(self.get_parameter("armed").value)
        self.values = {
            "drive": 0.0, "wheel": 0, "candidate_stop": True,
            "localization": "INITIALIZING", "connector": "NOT_READY",
            "validator": "CAMERA_UNAVAILABLE", "mission_stop": False,
            "camera_stop": False, "branch_stop": False, "lidar_stop": False,
        }
        self.received = {}
        self.publisher_conflicts = []
        self._subscribe(Float32, "/depth_slam/follower/candidate_drive",
                        "drive", float)
        self._subscribe(Int32, "/depth_slam/follower/candidate_wheel",
                        "wheel", int)
        self._subscribe(Bool, "/depth_slam/follower/candidate_stop",
                        "candidate_stop", bool)
        self._subscribe(String, "/depth_slam/localization/state",
                        "localization", str)
        self._subscribe(String, "/depth_slam/csv_validation/local_path_state",
                        "connector", str)
        self._subscribe(String, "/depth_slam/csv_road_validation/state",
                        "validator", str)
        self._subscribe(Bool, "/depth_slam/mission/stop_required",
                        "mission_stop", bool)
        self._subscribe(Bool, "/camera_stop", "camera_stop", bool)
        self._subscribe(Bool, "/depth_slam/route/branch_stop",
                        "branch_stop", bool)
        self._subscribe(Bool, "/lidar_stop", "lidar_stop", bool)
        self.drive_pub = self.create_publisher(Float32, "/slam_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/slam_wheel", 10)
        self.stop_pub = self.create_publisher(Bool, "/slam_stop", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/behavior/state", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/behavior/diagnostics", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)
        self.create_timer(2.0, self._audit_publishers)

    def _audit_publishers(self):
        conflicts = []
        for topic in ("/slam_drive", "/slam_wheel", "/slam_stop"):
            for info in self.get_publishers_info_by_topic(topic):
                if info.node_name != self.get_name():
                    conflicts.append(f"{topic}:{info.node_name}")
        self.publisher_conflicts = sorted(set(conflicts))

    def _subscribe(self, message_type, topic, key, convert):
        self.create_subscription(
            message_type, topic,
            lambda message: self._set(key, convert(message.data)), 10)

    def _set(self, key, value):
        self.values[key] = value
        self.received[key] = time.monotonic()

    def _fresh(self, keys, timeout, now):
        return all(key in self.received and now-self.received[key] <= timeout
                   for key in keys)

    def _tick(self):
        now = time.monotonic()
        command_timeout = float(self.get_parameter("command_timeout_s").value)
        state_timeout = float(self.get_parameter("state_timeout_s").value)
        validator_timeout = float(
            self.get_parameter("validator_timeout_s").value)
        decision = select_behavior(
            self.values["drive"], self.values["wheel"],
            self.values["candidate_stop"],
            self._fresh(("drive", "wheel", "candidate_stop"),
                        command_timeout, now),
            self.values["localization"],
            self._fresh(("localization",), state_timeout, now),
            self.values["connector"],
            self._fresh(("connector",), state_timeout, now),
            self.values["validator"],
            self._fresh(("validator",), validator_timeout, now),
            self.values["mission_stop"], self.values["camera_stop"],
            self.values["branch_stop"], self.values["lidar_stop"])
        if self.publisher_conflicts:
            decision = BehaviorDecision(
                0.0, 0, True, "DUPLICATE_SLAM_PUBLISHER", False)
        if self.armed:
            self.drive_pub.publish(Float32(data=decision.drive))
            self.wheel_pub.publish(Int32(data=decision.wheel))
            self.stop_pub.publish(Bool(data=decision.stop))
        self.state_pub.publish(String(data=(
            decision.state if self.armed else "DISARMED")))
        self.diag_pub.publish(String(data=json.dumps({
            "armed": self.armed,
            "drive": decision.drive,
            "wheel": decision.wheel,
            "stop": decision.stop,
            "state": decision.state,
            "road_verified": decision.road_verified,
            "camera_validation_optional": True,
            "publisher_conflicts": self.publisher_conflicts,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
