#!/usr/bin/env python3
"""Single owner of the camera command contract and camera control authority."""

from dataclasses import dataclass
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String


@dataclass(frozen=True)
class SelectorResult:
    drive: float
    wheel: int
    authority: bool
    source: str
    reason: str
    stop: bool = False


class CameraCommandSelector:
    """ROS-independent mode/mission authority policy."""

    def select(self, mode, candidate_drive, candidate_wheel, candidate_fresh,
               mission_active=False, mission_drive=0.0, mission_fresh=False,
               avoidance_active=False, recovery_ready=True):
        try:
            mode = int(str(mode).strip())
        except ValueError:
            return SelectorResult(0.0, 0, False, "NONE", "MODE_INVALID")
        wheel = max(-22, min(22, int(candidate_wheel)))
        drive = float(candidate_drive)
        if mode in (7, 10):
            return SelectorResult(0.0, 0, False, "PARKING", "CAMERA_INACTIVE")
        if mode in (4, 6):
            if not mission_fresh:
                return SelectorResult(
                    0.0, 0, True, "MISSION", "MISSION_STALE_STOP", True)
            if mission_active:
                return SelectorResult(
                    0.0, 0, True, "MISSION", "INTERSECTION_STOP", True)
            return SelectorResult(0.0, 0, False, "GPS_DR", "INTERSECTION_GO")
        if not candidate_fresh or not math.isfinite(drive):
            return SelectorResult(0.0, 0, False, "NONE", "CANDIDATE_STALE")
        if drive not in (0.0, 1.0, 2.0, 3.0):
            return SelectorResult(0.0, 0, False, "NONE", "DRIVE_INVALID")
        if mode == 5 and (avoidance_active or not recovery_ready):
            return SelectorResult(0.0, 0, False, "LIDAR", "AVOIDANCE_ACTIVE")
        if mission_active:
            if not mission_fresh or not math.isfinite(float(mission_drive)):
                return SelectorResult(
                    0.0, 0, True, "MISSION", "MISSION_STALE_STOP", True)
            requested = float(mission_drive)
            if requested not in (0.0, 1.0, 2.0, 3.0):
                requested = 0.0
            return SelectorResult(
                requested, wheel, True, "MISSION", "MISSION_OVERRIDE",
                requested == 0.0)
        return SelectorResult(drive, wheel, True, "CAMERA_PATH", "OK")


class CameraCommandSelectorNode(Node):
    def __init__(self):
        super().__init__("camera_command_selector_node")
        defaults = {
            "candidate_drive_topic": "/camera/candidate/path/drive",
            "candidate_wheel_topic": "/camera/candidate/path/wheel",
            "candidate_timeout_sec": 0.20,
            "mission_timeout_sec": 0.60,
            "publish_rate_hz": 20.0,
            "avoidance_recovery_samples": 3,
            "mode_topic": "/drive_mode",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.policy = CameraCommandSelector()
        self.mode = None
        self.avoidance_active = False
        self.was_avoiding = False
        self.recovery_samples = 0
        self.values = {"drive": 0.0, "wheel": 0,
                       "mission_active": False, "mission_drive": 0.0,
                       "adapter_connected": False}
        self.received = {}
        self.drive_pub = self.create_publisher(Float32, "/camera_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/camera_wheel", 10)
        self.stop_pub = self.create_publisher(Bool, "/camera_stop", 10)
        self.authority_pub = self.create_publisher(
            Bool, "/camera/control_authority", 10)
        self.source_pub = self.create_publisher(
            String, "/camera/command_source", 10)
        self.diag_pub = self.create_publisher(
            String, "/camera/command_selector_diagnostics", 10)
        self.create_subscription(
            String, str(self.p("mode_topic")), self._mode, 10)
        self.create_subscription(Bool, "/avoidance/active", self._avoidance, 10)
        self.create_subscription(Float32, str(self.p("candidate_drive_topic")),
                                 lambda m: self._set("drive", float(m.data)), 10)
        self.create_subscription(Int32, str(self.p("candidate_wheel_topic")),
                                 lambda m: self._set("wheel", int(m.data)), 10)
        self.create_subscription(Bool, "/camera/mission/drive_override_active",
                                 lambda m: self._set("mission_active", bool(m.data)), 10)
        self.create_subscription(Float32, "/camera/mission/drive_override",
                                 lambda m: self._set("mission_drive", float(m.data)), 10)
        self.create_subscription(String, "/avoidance/route/diagnostics",
                                 self._adapter_diagnostics, 10)
        self.create_timer(1.0 / float(self.p("publish_rate_hz")), self._tick)

    def p(self, name):
        return self.get_parameter(name).value

    def _set(self, key, value):
        self.values[key] = value
        self.received[key] = time.monotonic()
        if key in ("drive", "wheel") and self.mode == 5 and not self.avoidance_active:
            if self.was_avoiding:
                self.recovery_samples += 1

    def _mode(self, message):
        try:
            new_mode = int(str(message.data).strip())
        except ValueError:
            new_mode = None
        if new_mode != self.mode:
            self.received.clear()
            self.recovery_samples = 0
            self.was_avoiding = False
        self.mode = new_mode

    def _avoidance(self, message):
        active = bool(message.data)
        if active:
            self.was_avoiding = True
            self.recovery_samples = 0
        self.avoidance_active = active

    def _adapter_diagnostics(self, message):
        try:
            data = json.loads(message.data)
            connected = (data.get("current_mode") == "5" and
                         data.get("adapter_state") == "active" and
                         bool(data.get("metric_path_valid")) and
                         bool(data.get("tf_available")) and
                         float(data.get("forward_usable_length_m", 0.0)) > 0.0)
        except (TypeError, ValueError):
            connected = False
        self._set("adapter_connected", connected)

    def _fresh(self, names, timeout, now):
        return all(name in self.received and
                   now - self.received[name] <= timeout for name in names)

    def _duplicate_publishers(self, topic):
        return [info.node_name for info in self.get_publishers_info_by_topic(topic)
                if info.node_name != self.get_name()]

    def _tick(self):
        now = time.monotonic()
        duplicate = (self._duplicate_publishers("/camera_drive") +
                     self._duplicate_publishers("/camera_wheel") +
                     self._duplicate_publishers("/camera_stop"))
        candidate_fresh = self._fresh(
            ("drive", "wheel"), float(self.p("candidate_timeout_sec")), now)
        mission_fresh = self._fresh(
            ("mission_active", "mission_drive"),
            float(self.p("mission_timeout_sec")), now)
        adapter_fresh = self._fresh(
            ("adapter_connected",), float(self.p("mission_timeout_sec")), now)
        recovery_ready = (not self.was_avoiding or (
            self.recovery_samples >= int(self.p("avoidance_recovery_samples")) and
            adapter_fresh and bool(self.values["adapter_connected"])))
        result = self.policy.select(
            self.mode, self.values["drive"], self.values["wheel"], candidate_fresh,
            self.values["mission_active"], self.values["mission_drive"],
            mission_fresh, self.avoidance_active, recovery_ready)
        if duplicate:
            result = SelectorResult(0.0, 0, False, "NONE",
                                    "FAIL_DUPLICATE_FINAL_PUBLISHER", True)
        self.drive_pub.publish(Float32(data=result.drive))
        self.wheel_pub.publish(Int32(data=result.wheel))
        self.stop_pub.publish(Bool(data=result.stop))
        self.authority_pub.publish(Bool(data=result.authority))
        self.source_pub.publish(String(data=result.source))
        self.diag_pub.publish(String(data=json.dumps({
            "mode": self.mode, "drive": result.drive, "wheel": result.wheel,
            "authority": result.authority, "source": result.source,
            "reason": result.reason, "stop": result.stop,
            "candidate_fresh": candidate_fresh,
            "mission_fresh": mission_fresh, "avoidance_active": self.avoidance_active,
            "recovery_ready": recovery_ready,
            "reference_path_connected": bool(self.values["adapter_connected"]),
            "duplicate_publishers": duplicate,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = CameraCommandSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
