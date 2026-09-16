"""Mode-scoped lifecycle owner for front-LiDAR parking SLAM and Nav2."""

import json
import time

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class ParkingSlamManagerNode(Node):
    """Activate mapping/planning only in Modes 7/10 and arbitrate TF owner."""

    def __init__(self):
        super().__init__("depth_parking_slam_manager")
        self.declare_parameter("enable_parking_slam", False)
        self.declare_parameter("scan_timeout_s", 0.5)
        self.declare_parameter("map_timeout_s", 3.0)
        self.declare_parameter("maximum_unknown_ratio", 0.70)
        self.enabled = bool(self.get_parameter("enable_parking_slam").value)
        self.mode = -1
        self.entered_at = None
        self.scan_at = None
        self.map_at = None
        self.map_ready = False
        self.map_unknown_ratio = 1.0
        self.slam_state = State.PRIMARY_STATE_UNKNOWN
        self.nav2_state = State.PRIMARY_STATE_UNKNOWN
        self.transition_pending = {"slam_toolbox": False,
                                   "planner_server": False}
        self.planner_diagnostics = {}
        self.session_complete = False
        self.cleanup_requested = False
        self.slam_stop_reason = ""
        transient = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.active_pub = self.create_publisher(
            Bool, "/depth_slam/parking/slam_active", transient)
        self.map_ready_pub = self.create_publisher(
            Bool, "/depth_slam/parking/map_ready", transient)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/parking/diagnostics", 10)
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(LaserScan, "/front/scan", self._scan, 10)
        self.create_subscription(OccupancyGrid, "/map", self._map, 10)
        self.create_subscription(
            String, "/depth_slam/parking/planner_diagnostics",
            self._planner_diagnostics, 10)
        self.get_state = {}
        self.change_state = {}
        for node in ("slam_toolbox", "planner_server"):
            self.get_state[node] = self.create_client(
                GetState, f"/{node}/get_state")
            self.change_state[node] = self.create_client(
                ChangeState, f"/{node}/change_state")
        self.create_timer(0.2, self._tick)
        if not self.enabled:
            self.get_logger().info(
                "[PARKING SLAM] DISABLED - front LiDAR CSV sweep selection")

    @property
    def requested(self):
        return (self.enabled and self.mode in (7, 10) and
                not self.session_complete)

    def _mode(self, message):
        previous = self.mode
        try:
            self.mode = int(str(message.data).strip())
        except ValueError:
            self.mode = -1
        if self.mode in (7, 10) and previous != self.mode:
            self.entered_at = time.monotonic()
            self.map_at = None
            self.map_ready = False
            self.map_unknown_ratio = 1.0
            self.session_complete = False
            self.cleanup_requested = False
            self.slam_stop_reason = ""
            self.get_logger().info(
                f"[PARKING SLAM] Mode {self.mode} mapping during CSV approach")
        elif previous in (7, 10) and self.mode not in (7, 10):
            self.entered_at = None
            self.map_ready = False
            self.session_complete = False
            self.cleanup_requested = True
            self.slam_stop_reason = "MODE_EXIT"

    def _scan(self, _message):
        self.scan_at = time.monotonic()

    def _map(self, message):
        if not self.requested or not message.data:
            return
        unknown = sum(1 for value in message.data if int(value) < 0)
        self.map_unknown_ratio = unknown/max(len(message.data), 1)
        self.map_at = time.monotonic()
        self.map_ready = self.map_unknown_ratio <= float(
            self.get_parameter("maximum_unknown_ratio").value)

    def _planner_diagnostics(self, message):
        try:
            self.planner_diagnostics = json.loads(message.data)
            rejoin = str(self.planner_diagnostics.get(
                "parking_rejoin_state", ""))
            branch_locked = bool(self.planner_diagnostics.get(
                "branch_locked", False))
            if branch_locked:
                self.session_complete = True
                self.cleanup_requested = True
                self.slam_stop_reason = (
                    "NAV2_PATH_COMMITTED" if bool(
                        self.planner_diagnostics.get("nav2_path_ready", False))
                    else "CSV_FALLBACK_COMMITTED")
            elif rejoin in ("CSV_REJOIN", "COMPLETE"):
                self.session_complete = True
                self.cleanup_requested = True
                self.slam_stop_reason = "PARKING_COMPLETE"
        except (TypeError, json.JSONDecodeError):
            self.planner_diagnostics = {}

    def _query_state(self, node):
        client = self.get_state[node]
        if not client.service_is_ready():
            return
        future = client.call_async(GetState.Request())
        future.add_done_callback(
            lambda result, name=node: self._state_response(name, result))

    def _state_response(self, node, future):
        try:
            state = int(future.result().current_state.id)
        except Exception:
            return
        if node == "slam_toolbox":
            self.slam_state = state
        else:
            self.nav2_state = state
        desired = self.requested
        transition = None
        if desired and state == State.PRIMARY_STATE_UNCONFIGURED:
            transition = Transition.TRANSITION_CONFIGURE
        elif desired and state == State.PRIMARY_STATE_INACTIVE:
            transition = Transition.TRANSITION_ACTIVATE
        elif not desired and state == State.PRIMARY_STATE_ACTIVE:
            transition = Transition.TRANSITION_DEACTIVATE
        elif (not desired and self.cleanup_requested and
              state == State.PRIMARY_STATE_INACTIVE):
            # Cleanup discards the Mode-7 local map before Mode 10. Each
            # parking entry therefore starts a fresh front-LiDAR map.
            transition = Transition.TRANSITION_CLEANUP
        if transition is not None and not self.transition_pending[node]:
            self._transition(node, transition)

    def _transition(self, node, transition_id):
        client = self.change_state[node]
        if not client.service_is_ready():
            return
        request = ChangeState.Request()
        request.transition.id = int(transition_id)
        self.transition_pending[node] = True
        future = client.call_async(request)
        future.add_done_callback(
            lambda result, name=node: self._transition_response(name, result))

    def _transition_response(self, node, future):
        self.transition_pending[node] = False
        try:
            success = bool(future.result().success)
        except Exception:
            success = False
        if not success:
            self.get_logger().warning(
                f"[PARKING SLAM] lifecycle transition failed: {node}")

    def _tick(self):
        now = time.monotonic()
        if self.enabled:
            for node in ("slam_toolbox", "planner_server"):
                if not self.transition_pending[node]:
                    self._query_state(node)
        # Suspend depth ODOM map->odom before SLAM activation, and restore it
        # only after slam_toolbox has left ACTIVE.
        slam_active = self.requested or \
            self.slam_state == State.PRIMARY_STATE_ACTIVE
        self.active_pub.publish(Bool(data=slam_active))
        map_fresh = self.map_at is not None and now-self.map_at <= float(
            self.get_parameter("map_timeout_s").value)
        ready = bool(self.requested and map_fresh and self.map_ready and
                     self.slam_state == State.PRIMARY_STATE_ACTIVE)
        self.map_ready_pub.publish(Bool(data=ready))
        scan_fresh = self.scan_at is not None and now-self.scan_at <= float(
            self.get_parameter("scan_timeout_s").value)
        values = {
            "mode": self.mode,
            "parking_slam_enabled": self.enabled,
            "parking_slam_state": self.slam_state,
            "slam_active": slam_active,
            "slam_stop_reason": self.slam_stop_reason,
            "parking_map_ready": ready,
            "parking_map_unknown_ratio": self.map_unknown_ratio,
            "front_lidar_only": True,
            "rear_lidar_mapping": False,
            "front_scan_fresh": scan_fresh,
            "tf_owner": "SLAM_TOOLBOX" if slam_active else "DEPTH_ODOM",
            "duplicate_map_odom_publishers": 0,
            "nav2_lifecycle_state": self.nav2_state,
        }
        values.update(self.planner_diagnostics)
        # Lifecycle/TF ownership is authoritative here.  Planner diagnostics
        # arrive asynchronously and may still contain the previous tick's
        # slam_active value while deactivation is in progress.
        values.update({
            "mode": self.mode,
            "parking_slam_enabled": self.enabled,
            "slam_active": slam_active,
            "slam_stop_reason": self.slam_stop_reason,
            "parking_map_ready": ready,
            "tf_owner": "SLAM_TOOLBOX" if slam_active else "DEPTH_ODOM",
            "duplicate_map_odom_publishers": 0,
        })
        self.diag_pub.publish(String(data=json.dumps(
            values, separators=(",", ":"), sort_keys=True)))


def main(args=None):
    rclpy.init(args=args)
    node = ParkingSlamManagerNode()
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
