"""Front/rear LaserScan policy and sole depth_ws /lidar_* source owner."""

import json
import math
import time

from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32, String

from .prehardware_core import LidarDecision, LidarPolicy


def scan_points(message):
    """Convert finite LaserScan samples to sensor-forward Cartesian points."""
    output = []
    angle = float(message.angle_min)
    for value in message.ranges:
        distance = float(value)
        if (math.isfinite(distance) and message.range_min <= distance <=
                message.range_max):
            output.append((distance*math.cos(angle), distance*math.sin(angle)))
        angle += float(message.angle_increment)
    return tuple(output)


class LidarSourceNode(Node):
    def __init__(self):
        super().__init__("depth_lidar_source")
        for name, default in (
                ("scan_timeout_s", 0.5), ("publish_hz", 20.0),
                ("half_width_m", 0.60), ("general_distance_m", 1.5),
                ("mode9_distance_m", 3.0),
                ("rear_stop_distance_m", 0.5),
                ("avoidance_wheel_deg", 18), ("clear_samples", 3)):
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value
        self.policy = LidarPolicy(
            p("half_width_m"), p("general_distance_m"),
            p("mode9_distance_m"), p("rear_stop_distance_m"),
            p("avoidance_wheel_deg"), p("clear_samples"))
        self.mode = None
        self.front = self.rear = ()
        self.front_at = self.rear_at = None
        self.local_path_at = None
        self.local_path_length = 0.0
        self.publisher_conflicts = []
        self.create_subscription(LaserScan, "/scan_front", self._front, 10)
        self.create_subscription(LaserScan, "/scan_rear", self._rear, 10)
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(
            Path, "/depth_slam/csv_validation/local_path", self._path, 10)
        self.drive_pub = self.create_publisher(Float32, "/lidar_drive", 10)
        self.wheel_pub = self.create_publisher(Int32, "/lidar_wheel", 10)
        self.stop_pub = self.create_publisher(Bool, "/lidar_stop", 10)
        self.avoid_pub = self.create_publisher(Bool, "/avoidance/active", 10)
        self.parking_pub = self.create_publisher(Bool, "/parking/active", 10)
        self.front_obstacle_pub = self.create_publisher(
            Bool, "/lidar/front_obstacle_0_5m", 10)
        self.path_safe_pub = self.create_publisher(
            Bool, "/lidar/gps_dr_path_safe", 10)
        self.path_status_pub = self.create_publisher(
            String, "/lidar/path_safety_status", 10)
        self.safety_pub = self.create_publisher(
            String, "/avoidance/safety/status", 10)
        self.route_diag_pub = self.create_publisher(
            String, "/avoidance/route/diagnostics", 10)
        hz = float(p("publish_hz"))
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)
        self.create_timer(2.0, self._audit_publishers)

    def _audit_publishers(self):
        conflicts = []
        for topic in ("/lidar_drive", "/lidar_wheel", "/lidar_stop"):
            for info in self.get_publishers_info_by_topic(topic):
                if info.node_name != self.get_name():
                    conflicts.append(f"{topic}:{info.node_name}")
        self.publisher_conflicts = sorted(set(conflicts))

    def _front(self, message):
        self.front = scan_points(message)
        self.front_at = time.monotonic()

    def _rear(self, message):
        self.rear = scan_points(message)
        self.rear_at = time.monotonic()

    def _mode(self, message):
        self.mode = str(message.data).strip()

    def _path(self, message):
        self.local_path_at = time.monotonic()
        self.local_path_length = 0.0
        for before, after in zip(message.poses, message.poses[1:]):
            self.local_path_length += math.hypot(
                after.pose.position.x-before.pose.position.x,
                after.pose.position.y-before.pose.position.y)

    def _tick(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("scan_timeout_s").value)
        front_fresh = (self.front_at is not None and
                       now-self.front_at <= timeout)
        rear_fresh = (self.rear_at is not None and
                      now-self.rear_at <= timeout)
        decision = self.policy.evaluate(
            self.mode, self.front, self.rear, front_fresh, rear_fresh)
        if self.publisher_conflicts:
            decision = LidarDecision(
                0.0, 0, True, False, False, True, False,
                "DUPLICATE_LIDAR_PUBLISHER")
        # Internal steering is physical +LEFT/-RIGHT. MCU team LiDAR input is
        # +RIGHT/-LEFT, so this boundary performs the only sign inversion.
        external_wheel = -decision.wheel_internal
        self.drive_pub.publish(Float32(data=decision.drive))
        self.wheel_pub.publish(Int32(data=external_wheel))
        self.stop_pub.publish(Bool(data=decision.stop))
        self.avoid_pub.publish(Bool(data=decision.avoidance_active))
        self.parking_pub.publish(Bool(data=decision.parking_active))
        self.front_obstacle_pub.publish(
            Bool(data=decision.front_obstacle_0_5m))
        self.path_safe_pub.publish(Bool(data=decision.gps_path_safe))
        self.path_status_pub.publish(String(data=(
            "GPS_DR_SAFE" if decision.gps_path_safe else "UNSAFE")))
        status = {
            "state": decision.state,
            "mode": self.mode,
            "front_fresh": front_fresh,
            "rear_fresh": rear_fresh,
            "drive": decision.drive,
            "wheel_internal": decision.wheel_internal,
            "wheel_external": external_wheel,
            "stop": decision.stop,
            "publisher_conflicts": self.publisher_conflicts,
        }
        self.safety_pub.publish(String(data=json.dumps(
            status, separators=(",", ":"))))
        path_fresh = (self.local_path_at is not None and
                      now-self.local_path_at <= timeout)
        self.route_diag_pub.publish(String(data=json.dumps({
            "current_mode": self.mode,
            "adapter_state": "active" if decision.avoidance_active else
            "inactive",
            "metric_path_valid": path_fresh,
            "tf_available": path_fresh,
            "forward_usable_length_m": self.local_path_length,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = LidarSourceNode()
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
