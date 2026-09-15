"""ROS adapter for conservative camera local correction over the CSV path."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .camera_correction_core import (
    CameraCorrectionMachine, camera_correction_allowed,
    plan_camera_correction)
from .lidar_path_tracker import LocalPathTracker
from .ros_helpers import quaternion_from_yaw, yaw_from_quaternion


class CameraCorrectionNode(Node):
    """Publish a camera candidate only after persistent risk and a 3 s stop."""

    def __init__(self):
        super().__init__("depth_camera_correction")
        for name, default in (
                ("publish_hz", 20.0), ("input_timeout_s", 0.5),
                ("validation_stop_s", 3.0),
                ("planner_max_steering_deg", 20.0),
                ("vehicle_width_m", 0.80)):
            self.declare_parameter(name, default)
        self.machine = CameraCorrectionMachine(
            self.get_parameter("validation_stop_s").value)
        self.tracker = LocalPathTracker()
        self.validation = "UNKNOWN"
        self.lane_confident = False
        self.left_boundary = self.right_boundary = None
        self.csv_path = ()
        self.curbs = ()
        self.lane_points = ()
        self.pose = None
        self.vehicle_stopped = False
        self.lidar_candidate_active = False
        self.lidar_hold = False
        self.mode = -1
        self.intersection_state = "INACTIVE_MODE_GATE"
        self.emergency = False
        self.received = {}
        self.plan = None
        self.rejoin_pending = False
        self.create_subscription(
            String, "/depth_slam/camera/csv_validation",
            self._validation, 10)
        self.create_subscription(
            Path, "/depth_slam/csv_validation/local_path", self._path, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self._pose, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(
            Bool, "/depth_slam/lidar/candidate_valid",
            self._lidar_candidate, 10)
        self.create_subscription(
            Bool, "/depth_slam/lidar/hold", self._lidar_hold, 10)
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(
            String, "/depth_slam/mission/traffic_gate_state",
            self._intersection, 10)
        self.create_subscription(
            Bool, "/depth_slam/lidar/hard_emergency",
            self._emergency, 10)
        self.create_subscription(
            String, "/depth_slam/lidar/perception", self._perception, 10)
        self.drive_pub = self.create_publisher(
            Float32, "/depth_slam/camera/candidate_drive", 10)
        self.wheel_pub = self.create_publisher(
            Int32, "/depth_slam/camera/candidate_wheel", 10)
        self.valid_pub = self.create_publisher(
            Bool, "/depth_slam/camera/candidate_valid", 10)
        self.hold_pub = self.create_publisher(
            Bool, "/depth_slam/camera/hold", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/camera/correction_state", 10)
        self.path_pub = self.create_publisher(
            Path, "/depth_slam/camera/correction_path", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    def _validation(self, message):
        try:
            value = json.loads(message.data)
            self.validation = str(value.get("state", "UNKNOWN")).upper()
            self.lane_confident = bool(value.get(
                "lane_geometry_confident", False))
            self.left_boundary = value.get("nearest_left_boundary_m")
            self.right_boundary = value.get("nearest_right_boundary_m")
            self.lane_points = tuple(
                (float(point[0]), float(point[1]))
                for point in value.get("lane_boundary_points", ())
                if isinstance(point, (list, tuple)) and len(point) >= 2)
            self.received["validation"] = time.monotonic()
        except (TypeError, ValueError, IndexError, json.JSONDecodeError):
            self.validation = "UNKNOWN"
            self.lane_points = ()

    def _path(self, message):
        if str(message.header.frame_id).lstrip("/") != "base_link":
            self.csv_path = ()
            return
        self.csv_path = tuple((
            float(item.pose.position.x), float(item.pose.position.y),
            yaw_from_quaternion(item.pose.orientation))
            for item in message.poses)
        self.received["path"] = time.monotonic()

    def _pose(self, message):
        if str(message.header.frame_id).lstrip("/") != "map":
            self.pose = None
            return
        value = message.pose.pose
        self.pose = (float(value.position.x), float(value.position.y),
                     yaw_from_quaternion(value.orientation))
        self.received["pose"] = time.monotonic()

    def _odom(self, message):
        twist = message.twist.twist
        self.vehicle_stopped = (
            abs(float(twist.linear.x)) <= 0.03 and
            abs(float(twist.angular.z)) <= 0.03)
        self.received["odom"] = time.monotonic()

    def _lidar_candidate(self, message):
        self.lidar_candidate_active = bool(message.data)
        self.received["lidar_candidate"] = time.monotonic()

    def _lidar_hold(self, message):
        self.lidar_hold = bool(message.data)
        self.received["lidar_hold"] = time.monotonic()

    def _mode(self, message):
        try:
            self.mode = int(str(message.data).strip())
        except ValueError:
            self.mode = -1
        self.received["mode"] = time.monotonic()

    def _emergency(self, message):
        self.emergency = bool(message.data)
        self.received["emergency"] = time.monotonic()

    def _intersection(self, message):
        self.intersection_state = str(message.data).strip().upper()
        self.received["intersection"] = time.monotonic()

    def _perception(self, message):
        try:
            value = json.loads(message.data)
            self.curbs = tuple((float(point[0]), float(point[1]))
                               for point in value.get("curbs", ()))
            self.received["perception"] = time.monotonic()
        except (TypeError, ValueError, IndexError, json.JSONDecodeError):
            self.curbs = ()

    def _fresh(self, key, now):
        timeout = float(self.get_parameter("input_timeout_s").value)
        return key in self.received and now-self.received[key] <= timeout

    def _publish_path(self):
        if self.plan is None or not self.plan.valid or self.pose is None:
            return
        ox, oy, yaw = self.pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        message = Path()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y, local_yaw in self.plan.points:
            item = PoseStamped()
            item.header = message.header
            item.pose.position.x = ox+cosine*x-sine*y
            item.pose.position.y = oy+sine*x+cosine*y
            item.pose.orientation = quaternion_from_yaw(yaw+local_yaw)
            message.poses.append(item)
        self.path_pub.publish(message)

    def _clear_plan(self):
        self.plan = None
        self.rejoin_pending = False
        self.tracker.clear()

    def _tick(self):
        now = time.monotonic()
        validation = (self.validation if self._fresh("validation", now)
                      else "UNKNOWN")
        lidar_inputs_fresh = all(self._fresh(key, now) for key in (
            "lidar_candidate", "lidar_hold", "mode"))
        lidar_active = ((self.lidar_candidate_active or self.lidar_hold or
                         self.mode in (7, 10))
                        if lidar_inputs_fresh else True)
        emergency = (self.emergency if self._fresh("emergency", now)
                     else True)
        stopped = (self.vehicle_stopped and self._fresh("odom", now))
        confident = (self.lane_confident and bool(self.lane_points) and
                     self._fresh("path", now) and
                     self._fresh("pose", now) and
                     self._fresh("perception", now))
        enabled = (self._fresh("mode", now) and
                   camera_correction_allowed(
                       self.mode, self.intersection_state if
                       self._fresh("intersection", now) else "STALE"))
        decision = self.machine.update(
            validation, confident, now=now, vehicle_stopped=stopped,
            lidar_active=lidar_active, emergency=emergency,
            rejoin_valid=self.rejoin_pending, enabled=enabled)
        if decision.discard_path:
            self._clear_plan()
        if decision.need_plan:
            self.plan = plan_camera_correction(
                self.csv_path, self.left_boundary, self.right_boundary,
                lane_points=self.lane_points, curbs=self.curbs,
                vehicle_width_m=float(self.get_parameter(
                    "vehicle_width_m").value),
                planner_limit_deg=float(self.get_parameter(
                    "planner_max_steering_deg").value))
            activated = (self.plan.valid and self.pose is not None and
                         self.tracker.set_plan(
                             self.plan.points, self.pose, 1.0))
            decision = self.machine.update(
                validation, confident, now=now,
                vehicle_stopped=stopped, plan_valid=activated,
                enabled=enabled)
            if not activated:
                self.tracker.clear()
            else:
                self._publish_path()
        drive, wheel, valid = 0.0, 0, False
        if decision.active and self.pose is not None:
            command = self.tracker.update(self.pose)
            if not command.valid:
                self.machine.state = "CAMERA_NO_VALID_PATH"
                decision = self.machine.update(
                    validation, confident, now=now,
                    vehicle_stopped=stopped, plan_valid=False,
                    enabled=enabled)
                self.tracker.clear()
            elif command.complete:
                decision = self.machine.update(
                    validation, confident, now=now,
                    vehicle_stopped=stopped, path_complete=True,
                    enabled=enabled)
                self.rejoin_pending = True
            else:
                drive, wheel, valid = command.drive, command.wheel, True
        self.drive_pub.publish(Float32(data=float(drive)))
        self.wheel_pub.publish(Int32(data=int(wheel)))
        self.valid_pub.publish(Bool(data=valid))
        self.hold_pub.publish(Bool(data=bool(decision.stop)))
        self.state_pub.publish(String(data=decision.state))


def main(args=None):
    rclpy.init(args=args)
    node = CameraCorrectionNode()
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
