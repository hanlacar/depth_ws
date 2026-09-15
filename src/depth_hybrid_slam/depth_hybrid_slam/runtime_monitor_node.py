"""One-Hz production health summary and unified freshness diagnostics."""

import json
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker


class RuntimeMonitorNode(Node):
    def __init__(self):
        super().__init__("depth_runtime_monitor")
        self.declare_parameter("freshness_timeout_s", 0.5)
        self.values = {"mode": -1, "owner": "STOP", "drive": 0.0,
                       "wheel": 0, "camera": False, "lidar": False,
                       "vslam": False}
        self.received = {}
        subscriptions = (
            (String, "/drive_mode", "mode", lambda value: int(str(value))),
            (String, "/depth_slam/path_owner", "owner", str),
            (Float32, "/cmd_drive", "drive", float),
            (Int32, "/cmd_wheel", "wheel", int),
            (Float32, "/mcu/steer_deg", "steering", float),
            (Bool, "/depth_slam/vslam/visual_consistent", "vslam", bool),
            (Float32, "/imu/pitch_deg", "imu", float),
            (String, "/camera/traffic_light_fused/state", "traffic", str),
            (String, "/camera/exit_branch_signal", "exit_branch", str),
        )
        for kind, topic, key, cast in subscriptions:
            self.create_subscription(
                kind, topic,
                lambda message, k=key, c=cast: self._set(k, c(message.data)),
                10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(
            String, "/depth_slam/lidar/perception", self._lidar, 10)
        self.create_subscription(
            String, "/depth_slam/camera/csv_validation", self._camera, 10)
        self.publisher = self.create_publisher(
            String, "/depth_slam/runtime/watchdog", 10)
        self.marker_publisher = self.create_publisher(
            Marker, "/depth_slam/runtime/status_marker", 10)
        self.create_timer(1.0, self._tick)

    def _set(self, key, value):
        self.values[key] = value
        self.received[key] = time.monotonic()

    def _odom(self, message):
        self._set("odom", float(message.twist.twist.linear.x))

    def _document(self, message, key):
        try:
            value = json.loads(message.data)
            self._set(key, bool(value.get("fresh", False)))
        except (TypeError, ValueError, json.JSONDecodeError):
            self.values[key] = False

    def _lidar(self, message):
        self._document(message, "lidar")

    def _camera(self, message):
        self._document(message, "camera")

    def _health(self, key, now):
        timeout = float(self.get_parameter("freshness_timeout_s").value)
        fresh = key in self.received and now-self.received[key] <= timeout
        if key in ("camera", "lidar", "vslam"):
            fresh = fresh and bool(self.values.get(key, False))
        return "OK" if fresh else "STALE"

    def _tick(self):
        now = time.monotonic()
        health = {key: self._health(key, now) for key in (
            "odom", "lidar", "steering", "camera", "traffic",
            "exit_branch", "vslam", "imu")}
        ages = {key+"_age_s": (None if key not in self.received else
                                round(max(0.0, now-self.received[key]), 3))
                for key in health}
        runtime_state = ("FAIL" if any(health[key] != "OK" for key in
                                       ("odom", "lidar", "steering")) else
                         "WARNING" if health["camera"] != "OK" or
                         health["vslam"] != "OK" else "RUNNING")
        payload = {"runtime_state": runtime_state, "segment": self.values["mode"],
                   "owner": self.values["owner"], "health": health, **ages}
        self.publisher.publish(String(data=json.dumps(
            payload, separators=(",", ":"))))
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = "runtime_status", 0
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position.z = 1.8
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.22
        marker.color.r, marker.color.g = (1.0, 0.2) if runtime_state == \
            "FAIL" else (0.2, 1.0)
        marker.color.b, marker.color.a = 0.2, 1.0
        marker.text = (f"SEG {self.values['mode']}  OWNER {self.values['owner']}\n"
                       f"D {self.values['drive']:.2f}  W {self.values['wheel']}")
        self.marker_publisher.publish(marker)
        self.get_logger().info(
            f"[RUN] SEG={self.values['mode']} OWNER={self.values['owner']} "
            f"DRIVE={self.values['drive']:.2f} WHEEL={self.values['wheel']:.1f} "
            f"LIDAR={health['lidar']} CAMERA={health['camera']} "
            f"ODOM={health['odom']} VSLAM={health['vslam']}")


def main(args=None):
    rclpy.init(args=args)
    node = RuntimeMonitorNode()
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
