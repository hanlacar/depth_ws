"""The sole publisher of /cmd_drive and /cmd_wheel."""

import json
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .command_arbiter_core import (
    arbitrate, CommandCandidate, OwnershipHandshake, safety_stop_reasons)


class CommandArbiterNode(Node):
    def __init__(self):
        super().__init__("depth_command_arbiter")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("candidate_timeout_s", 0.5)
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("owner_stop_duration_s", 1.0)
        self.declare_parameter("owner_standstill_speed_mps", 0.03)
        self.values = {
            "csv_drive": 0.0, "csv_wheel": 0,
            "lidar_drive": 0.0, "lidar_wheel": 0, "lidar_valid": False,
            "camera_drive": 0.0, "camera_wheel": 0,
            "camera_valid": False, "camera_hold": False,
            "hard": False, "mission": True, "branch": False,
            "lidar_hold": False,
            "distance_slowdown": False, "steering_slowdown": False,
            "mode": -1, "steering": 0.0,
            "speed": 0.0,
            "route_safety_state": "STOP_REQUIRED",
            "route_safety_reason": "SAFETY_NOT_READY",
        }
        self.received = {}
        for kind, topic, key, cast in (
            (Float32, "/depth_slam/follower/candidate_drive", "csv_drive", float),
            (Int32, "/depth_slam/follower/candidate_wheel", "csv_wheel", int),
            (Bool, "/depth_slam/follower/candidate_stop", "csv_stop", bool),
            (Float32, "/depth_slam/lidar/candidate_drive", "lidar_drive", float),
            (Int32, "/depth_slam/lidar/candidate_wheel", "lidar_wheel", int),
            (Bool, "/depth_slam/lidar/candidate_valid", "lidar_valid", bool),
            (Float32, "/depth_slam/camera/candidate_drive",
             "camera_drive", float),
            (Int32, "/depth_slam/camera/candidate_wheel",
             "camera_wheel", int),
            (Bool, "/depth_slam/camera/candidate_valid",
             "camera_valid", bool),
            (Bool, "/depth_slam/camera/hold", "camera_hold", bool),
            (Bool, "/depth_slam/lidar/hard_emergency", "hard", bool),
            (Bool, "/depth_slam/lidar/distance_slowdown_required",
             "distance_slowdown", bool),
            (Bool, "/depth_slam/lidar/steering_slowdown_required",
             "steering_slowdown", bool),
            (Bool, "/depth_slam/mission/stop_required", "mission", bool),
            (Bool, "/depth_slam/route/branch_stop", "branch", bool),
            (Bool, "/depth_slam/lidar/hold", "lidar_hold", bool),
            (String, "/depth_slam/safety/state", "route_safety_state", str),
            (String, "/depth_slam/safety/stop_reason",
             "route_safety_reason", str),
            (String, "/drive_mode", "mode", lambda value: int(str(value))),
            (Float32, "/mcu/steer_deg", "steering", float),
        ):
            self.create_subscription(
                kind, topic, lambda msg, k=key, c=cast: self._set(k, c(msg.data)), 10)
        self.create_subscription(Odometry, "/odom", self._odom, 20)
        self.handshake = OwnershipHandshake(
            stop_duration_s=self.get_parameter("owner_stop_duration_s").value,
            standstill_speed_mps=self.get_parameter(
                "owner_standstill_speed_mps").value)
        self.last_transition = None
        self.last_stop_reason = None
        self.drive_pub = self.create_publisher(Float32, "/cmd_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/cmd_wheel", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/command_arbiter/state", 10)
        self.owner_pub = self.create_publisher(
            String, "/depth_slam/path_owner", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/command_arbiter/diagnostics", 10)
        self.conflicts = []
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)
        self.create_timer(2.0, self._audit)

    def _set(self, key, value):
        self.values[key] = value
        self.received[key] = time.monotonic()

    def _odom(self, message):
        self.values["speed"] = float(message.twist.twist.linear.x)
        self.received["odom"] = time.monotonic()

    def _fresh(self, keys, now, timeout=None):
        timeout = (float(self.get_parameter("candidate_timeout_s").value)
                   if timeout is None else float(timeout))
        return all(key in self.received and now-self.received[key] <= timeout
                   for key in keys)

    def _audit(self):
        conflicts = []
        for topic in ("/cmd_drive", "/cmd_wheel"):
            for info in self.get_publishers_info_by_topic(topic):
                if info.node_name != self.get_name():
                    conflicts.append(topic+":"+info.node_name)
        self.conflicts = sorted(set(conflicts))

    def _tick(self):
        now = time.monotonic()
        csv = CommandCandidate(
            self.values["csv_drive"], self.values["csv_wheel"],
            not self.values.get("csv_stop", True),
            self._fresh(("csv_drive", "csv_wheel", "csv_stop"), now),
            min((self.received.get(key, 0.0) for key in
                 ("csv_drive", "csv_wheel", "csv_stop")), default=0.0))
        lidar = CommandCandidate(
            self.values["lidar_drive"], self.values["lidar_wheel"],
            self.values["lidar_valid"],
            self._fresh(("lidar_drive", "lidar_wheel", "lidar_valid"), now),
            min((self.received.get(key, 0.0) for key in
                 ("lidar_drive", "lidar_wheel", "lidar_valid")), default=0.0))
        camera = CommandCandidate(
            self.values["camera_drive"], self.values["camera_wheel"],
            self.values["camera_valid"],
            self._fresh(("camera_drive", "camera_wheel", "camera_valid"),
                        now),
            min((self.received.get(key, 0.0) for key in
                 ("camera_drive", "camera_wheel", "camera_valid")), default=0.0))
        # Loss of a safety/hold producer is not permission to move. Camera
        # road validation remains advisory and may fall back to CSV, but the
        # LiDAR, mission, and branch gates must each be fresh.
        odom_fresh = self._fresh(
            ("odom",), now, self.get_parameter("odom_timeout_s").value)
        lidar_safety_fresh = self._fresh(
            ("hard", "distance_slowdown", "steering_slowdown"), now)
        steering_fresh = self._fresh(("steering",), now)
        safety_reasons = safety_stop_reasons(
            hard_emergency=self.values["hard"], odom_fresh=odom_fresh,
            lidar_safety_fresh=lidar_safety_fresh,
            steering_fresh=steering_fresh,
            publisher_conflict=bool(self.conflicts))
        safety_stop = bool(safety_reasons)
        gated_stop = (
            self.values["mission"] or self.values["branch"] or
            self.values["lidar_hold"] or
            (self.values["camera_hold"] and
             self._fresh(("camera_hold",), now) and
             not (lidar.valid and lidar.fresh)) or
            not self._fresh(("mission",), now) or
            not self._fresh(("branch",), now) or
            not self._fresh(("lidar_hold",), now))
        decision = arbitrate(
            csv, lidar, safety_stop, gated_stop,
            self.values["distance_slowdown"], self.values["mode"],
            self.values["steering_slowdown"], camera)
        if not safety_stop and decision.owner in self.handshake.SOURCE_OWNERS:
            candidates = {"CSV": csv, "CAMERA": camera,
                          "LIDAR": lidar, "PARKING": lidar}
            previous = self.handshake.owner
            handoff = self.handshake.update(
                decision.owner, speed_mps=self.values["speed"],
                odom_fresh=odom_fresh,
                source_received_at=candidates[decision.owner].received_at,
                now=now)
            transition = (previous, self.handshake.pending_owner)
            if handoff is not None and transition != self.last_transition:
                self.get_logger().info(
                    f"[OWNER] {previous} -> {decision.owner} : STOP 1.0s")
                self.last_transition = transition
            if handoff is not None:
                decision = handoff
            elif self.handshake.owner != previous:
                self.get_logger().info(f"[OWNER] {self.handshake.owner}")
                self.last_transition = None
        self.drive_pub.publish(Float32(data=decision.drive))
        self.wheel_pub.publish(Int32(data=decision.wheel))
        self.state_pub.publish(String(data=decision.state))
        published_owner = (
            "STOP" if decision.drive == 0.0 and decision.owner not in
            ("CSV", "CAMERA", "LIDAR", "PARKING") else decision.owner)
        self.owner_pub.publish(String(data=published_owner))
        stop_reason = ""
        if published_owner == "STOP":
            stop_reason = ",".join(safety_reasons)
            if (not stop_reason and
                    self._fresh(("route_safety_state",
                                 "route_safety_reason"), now) and
                    self.values["route_safety_state"] != "READY"):
                stop_reason = self.values["route_safety_reason"]
            stop_reason = stop_reason or decision.state
        if stop_reason != self.last_stop_reason:
            if stop_reason:
                self.get_logger().warning("[STOP] "+stop_reason)
            elif self.last_stop_reason:
                self.get_logger().info("[STOP] CLEARED")
            self.last_stop_reason = stop_reason
        self.diag_pub.publish(String(data=json.dumps({
            "owner": decision.owner, "state": decision.state,
            "drive": decision.drive, "wheel": decision.wheel,
            "stop_reason": stop_reason,
            "publisher_conflicts": self.conflicts,
            "active_owner": self.handshake.owner,
            "pending_owner": self.handshake.pending_owner,
            "odom_age_s": None if "odom" not in self.received else
            max(0.0, now-self.received["odom"]),
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = CommandArbiterNode()
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
