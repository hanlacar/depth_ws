"""Convert active map route progress into the validator's base_link path."""

import json
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from .prehardware_core import RouteLocalConnector
from .ros_helpers import quaternion_from_yaw, yaw_from_quaternion


class RouteLocalPathNode(Node):
    def __init__(self):
        super().__init__("depth_route_local_path_connector")
        for name, default in (
                ("pose_timeout_s", 0.25), ("index_timeout_s", 0.25),
                ("publish_hz", 20.0),
                ("forward_min_m", 0.5), ("forward_max_m", 4.0)):
            self.declare_parameter(name, default)
        self.core = RouteLocalConnector(
            self.get_parameter("forward_min_m").value,
            self.get_parameter("forward_max_m").value)
        self.route = []
        self.pose = None
        self.index = None
        self.active_branch = None
        self.localization_state = "INITIALIZING"
        self.received = {}
        route_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Path, "/depth_slam/route/reference_path", self._path, route_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self._pose, 10)
        self.create_subscription(
            Int32, "/depth_slam/route/active_index", self._index, 10)
        self.create_subscription(
            String, "/depth_slam/localization/state", self._localization, 10)
        self.create_subscription(
            String, "/depth_slam/route/active_branch", self._branch, 10)
        self.path_pub = self.create_publisher(
            Path, "/depth_slam/csv_validation/local_path", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/csv_validation/local_path_state", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/csv_validation/local_path_diagnostics", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    def _path(self, message):
        if str(message.header.frame_id).lstrip("/") != "map":
            self.route = []
            return
        self.route = [(
            pose.pose.position.x, pose.pose.position.y,
            yaw_from_quaternion(pose.pose.orientation))
            for pose in message.poses]
        self.received["path"] = time.monotonic()

    def _pose(self, message):
        if str(message.header.frame_id).lstrip("/") != "map":
            self.pose = None
            return
        pose = message.pose.pose
        self.pose = (pose.position.x, pose.position.y,
                     yaw_from_quaternion(pose.orientation))
        self.received["pose"] = time.monotonic()

    def _index(self, message):
        self.index = int(message.data)
        self.received["index"] = time.monotonic()

    def _localization(self, message):
        self.localization_state = str(message.data).split(":", 1)[0]
        self.received["localization"] = time.monotonic()

    def _branch(self, message):
        branch = str(message.data).strip().upper()
        if branch != self.active_branch:
            self.active_branch = branch
            self.core.reset()
            self.index = None

    def _fresh(self, key, timeout, now):
        return key in self.received and now-self.received[key] <= timeout

    def _tick(self):
        now = time.monotonic()
        state = "READY"
        result = None
        if not self._fresh("pose", float(
                self.get_parameter("pose_timeout_s").value), now):
            state = "STALE_LOCALIZATION"
        elif not self._fresh("localization", float(
                self.get_parameter("pose_timeout_s").value), now) or \
                self.localization_state not in ("TRACKING", "RELOCALIZED"):
            state = "LOCALIZATION_LOST"
        elif not self.route or "path" not in self.received:
            state = "ACTIVE_ROUTE_UNAVAILABLE"
        elif not self._fresh("index", float(
                self.get_parameter("index_timeout_s").value), now):
            state = "STALE_ROUTE_INDEX"
        else:
            result = self.core.extract(self.route, self.pose, self.index)
            state = result.state
        if result is not None and result.valid:
            message = Path()
            message.header.frame_id = "base_link"
            message.header.stamp = self.get_clock().now().to_msg()
            for x, y, yaw in result.points:
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation = quaternion_from_yaw(yaw)
                message.poses.append(pose)
            self.path_pub.publish(message)
        self.state_pub.publish(String(data=state))
        self.diag_pub.publish(String(data=json.dumps({
            "state": state,
            "route_points": len(self.route),
            "route_index": self.index,
            "local_points": 0 if result is None else len(result.points),
            "nearest_search": False,
            "index_backtrack_allowed": False,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = RouteLocalPathNode()
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
