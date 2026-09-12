"""Test-only vehicle driven directly by route-follower candidate commands."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .ros_helpers import quaternion_from_yaw, safe_shutdown
from .csv_only_branching import load_csv_only_route_case
from .csv_only_recovery import lateral_offset_xy
from .route_io import load_segmented_route
from .virtual_mcu_core import VirtualAckermannVehicle


class CsvOnlyVirtualVehicleNode(Node):
    """Own test odometry/encoder without any production runtime dependency."""

    def __init__(self):
        super().__init__("depth_csv_only_virtual_vehicle")
        defaults = {
            "route_path": "", "route_metadata_path": "",
            "spawn_branch": "A",
            "prehardware_test_only": True,
            "publish_hz": 50.0, "simulation_speedup": 10.0,
            "command_timeout_s": 0.50,
            "stage_1_pwm": 0, "stage_1_speed_mps": -1.0,
            "stage_2_pwm": 0, "stage_2_speed_mps": -1.0,
            "stage_3_pwm": 0, "stage_3_speed_mps": -1.0,
            "reverse_pwm": 0, "reverse_speed_mps": -1.0,
            "counts_per_meter": 797.0, "encoder_signed": False,
            "wheelbase_m": 0.730, "max_steering_deg": 22.0,
            "direction_change_hold_s": 3.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not bool(self.p("prehardware_test_only")):
            raise ValueError("CSV-only virtual vehicle is test-only")

        spawn_branch = str(self.p("spawn_branch")).strip().upper()
        if spawn_branch not in ("A", "B"):
            raise ValueError("spawn_branch must be A or B")
        self.spawn_branch = spawn_branch
        self.route_path = str(self.p("route_path"))
        self.route_metadata_path = str(self.p("route_metadata_path"))
        route = load_segmented_route(
            self.route_path, self.route_metadata_path,
            branch=spawn_branch)
        if (not route.points or
                route.points[0].segment_id != f"START_{spawn_branch}"):
            raise ValueError("CSV-only route has the wrong START segment")
        self.route = route.points
        self.active_case = spawn_branch*4
        self.route_cache = {}
        self.active_index = None
        self.active_segment = ""
        self.cross_track_error = 0.0
        self.controller_state = ""
        self.disturbance = None

        self.pwm = {
            -1: int(self.p("reverse_pwm")), 0: 0,
            1: int(self.p("stage_1_pwm")),
            2: int(self.p("stage_2_pwm")),
            3: int(self.p("stage_3_pwm")),
        }
        if self.pwm != {-1: -50, 0: 0, 1: 50, 2: 75, 3: 100}:
            raise ValueError("CSV-only PWM calibration does not match MCU v5")
        self.vehicle = VirtualAckermannVehicle(
            {stage: float(self.p(f"stage_{stage}_speed_mps"))
             for stage in (1, 2, 3)},
            float(self.p("reverse_speed_mps")),
            wheelbase_m=float(self.p("wheelbase_m")),
            counts_per_meter=float(self.p("counts_per_meter")),
            max_steering_deg=float(self.p("max_steering_deg")),
            direction_change_hold_s=float(self.p("direction_change_hold_s")),
            encoder_signed=bool(self.p("encoder_signed")))
        first = self.route[0]
        self.vehicle.x, self.vehicle.y, self.vehicle.yaw = (
            first.x, first.y, first.yaw)

        self.values = {"drive": 0.0, "wheel": 0, "stop": True}
        self.received = {}
        self.last_trajectory = None
        self.trajectory = Path()
        self.trajectory.header.frame_id = "map"
        self.create_subscription(
            Float32, "/depth_slam/follower/candidate_drive",
            lambda message: self.command("drive", float(message.data)), 10)
        self.create_subscription(
            Int32, "/depth_slam/follower/candidate_wheel",
            lambda message: self.command("wheel", int(message.data)), 10)
        self.create_subscription(
            Bool, "/depth_slam/follower/candidate_stop", self.on_stop, 10)
        self.create_subscription(
            Float32, "/depth_slam/test/inject_lateral_offset_m",
            self.on_lateral_disturbance, 10)
        self.create_subscription(
            String, "/depth_slam/route/selected_case", self.on_case, 10)
        self.create_subscription(
            Int32, "/depth_slam/route/active_index", self.on_active_index, 10)
        self.create_subscription(
            String, "/depth_slam/route/active_segment",
            lambda message: setattr(self, "active_segment", str(message.data)),
            10)
        self.create_subscription(
            Float32, "/depth_slam/route/cross_track_error",
            self.on_cross_track, 10)
        self.create_subscription(
            String, "/depth_slam/route/controller_state",
            lambda message: setattr(
                self, "controller_state", str(message.data)), 10)

        for topic in ("/odom", "/mcu/encoder"):
            owners = self.get_publishers_info_by_topic(topic)
            if owners:
                raise RuntimeError(f"CSV-only duplicate publisher on {topic}")
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_encoder = self.create_publisher(Int32, "/mcu/encoder", 10)
        self.pub_steer = self.create_publisher(Float32, "/mcu/steer_deg", 10)
        self.pub_speed = self.create_publisher(Float32, "/mcu/speed_mps", 10)
        self.pub_distance = self.create_publisher(Float32, "/mcu/distance_m", 10)
        self.pub_connected = self.create_publisher(Bool, "/mcu/connected", 10)
        self.pub_telemetry = self.create_publisher(Bool, "/mcu/telemetry_ok", 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/csv_only/vehicle_state", 10)
        self.pub_disturbance = self.create_publisher(
            String, "/depth_slam/test/disturbance_state", 10)
        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_trajectory = self.create_publisher(
            Path, "/depth_slam/csv_only/trajectory", latched)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/depth_slam/csv_only/markers", 10)
        self.tf = TransformBroadcaster(self)

        self.publish_hz = float(self.p("publish_hz"))
        self.speedup = float(self.p("simulation_speedup"))
        if self.publish_hz <= 0.0 or not 0.0 < self.speedup <= 25.0:
            raise ValueError("invalid CSV-only publish rate or speedup")
        self.create_timer(1.0/self.publish_hz, self.tick)
        self.get_logger().warn(
            "PREHARDWARE CSV ONLY: candidate commands directly drive a virtual "
            f"vehicle; start=START_{spawn_branch}:{first.point_index} "
            f"pose=({first.x:.6f},{first.y:.6f},{first.yaw:.6f})")

    def p(self, name):
        return self.get_parameter(name).value

    def command(self, name, value):
        self.values[name] = value
        self.received[name] = time.monotonic()

    def on_stop(self, message):
        value = bool(message.data)
        self.command("stop", value)
        if self.disturbance is not None:
            self.disturbance["stop_occurred"] |= value

    def on_case(self, message):
        case = str(message.data).strip().upper()
        if len(case) == 4 and all(value in "AB" for value in case):
            self.active_case = case

    def active_route(self):
        if self.active_case not in self.route_cache:
            self.route_cache[self.active_case] = load_csv_only_route_case(
                self.route_path, self.route_metadata_path, self.active_case)
        return self.route_cache[self.active_case]

    def on_active_index(self, message):
        value = int(message.data)
        if self.disturbance is not None:
            previous = self.disturbance.get("last_active_index")
            self.disturbance["cursor_backtrack"] |= (
                previous is not None and value < previous)
            self.disturbance["last_active_index"] = value
            route = self.active_route()
            if 0 <= value < len(route):
                point = route[value]
                self.disturbance["wrong_segment_jump"] |= (
                    point.segment_id != self.disturbance["segment_id"])
        self.active_index = value

    def on_cross_track(self, message):
        self.cross_track_error = float(message.data)
        if self.disturbance is None:
            return
        now = time.monotonic()
        absolute = abs(self.cross_track_error)
        if self.disturbance["awaiting_injected_pose"]:
            threshold = max(0.05, abs(self.disturbance["offset_m"])*0.5)
            if absolute < threshold:
                return
            self.disturbance["awaiting_injected_pose"] = False
        initial = self.disturbance.get("initial_cte_m")
        if initial is None:
            initial = absolute
            self.disturbance["initial_cte_m"] = initial
            self.disturbance["maximum_cte_m"] = absolute
            self.disturbance["minimum_cte_m"] = absolute
        self.disturbance["maximum_cte_m"] = max(
            self.disturbance["maximum_cte_m"], absolute)
        self.disturbance["minimum_cte_m"] = min(
            self.disturbance["minimum_cte_m"], absolute)
        elapsed = now-self.disturbance["started_at"]
        if (self.disturbance["decrease_started_s"] is None and
                absolute <= initial-0.01):
            self.disturbance["decrease_started_s"] = elapsed
        if (self.disturbance["recovered_025_s"] is None and
                absolute <= 0.25):
            self.disturbance["recovered_025_s"] = elapsed
        self.disturbance["recovered_010"] |= absolute <= 0.10

    def on_lateral_disturbance(self, message):
        offset = float(message.data)
        if not math.isfinite(offset) or abs(offset) > 2.1+1.0e-9:
            self.get_logger().error(
                "TEST ONLY lateral disturbance must be within +/-2.1 m")
            return
        route = self.active_route()
        if self.active_index is None or not 0 <= self.active_index < len(route):
            self.get_logger().error(
                "TEST ONLY disturbance rejected: active route point unknown")
            return
        point = route[self.active_index]
        if self.active_segment and point.segment_id != self.active_segment:
            self.get_logger().error(
                "TEST ONLY disturbance rejected: route/cursor state mismatch")
            return
        self.vehicle.x, self.vehicle.y = lateral_offset_xy(
            self.vehicle.x, self.vehicle.y, point.yaw, offset)
        self.disturbance = {
            "test_only": True, "offset_m": offset,
            "side": "LEFT" if offset >= 0.0 else "RIGHT",
            "segment_id": point.segment_id,
            "point_index": point.point_index,
            "injected_x_m": self.vehicle.x, "injected_y_m": self.vehicle.y,
            "target_x_m": point.x, "target_y_m": point.y,
            "started_at": time.monotonic(), "initial_cte_m": abs(offset),
            "maximum_cte_m": abs(offset), "minimum_cte_m": abs(offset),
            "awaiting_injected_pose": True,
            "decrease_started_s": None, "recovered_025_s": None,
            "recovered_010": False, "stop_occurred": False,
            "wrong_segment_jump": False, "cursor_backtrack": False,
            "last_active_index": self.active_index,
        }
        self.get_logger().warn(
            "TEST ONLY lateral disturbance injected: "
            f"{offset:+.2f} m at {point.segment_id}:{point.point_index}")

    def command_fresh(self):
        now = time.monotonic()
        timeout = float(self.p("command_timeout_s"))
        return all(name in self.received and now-self.received[name] <= timeout
                   for name in ("drive", "wheel", "stop"))

    def step_vehicle(self, drive, wheel, stop):
        remaining = self.speedup/self.publish_hz
        state = None
        while remaining > 1.0e-12:
            step = min(0.1, remaining)
            state = self.vehicle.step(drive, wheel, stop, step)
            remaining -= step
        return state

    def tick(self):
        fresh = self.command_fresh()
        drive = self.values["drive"] if fresh else 0.0
        wheel = self.values["wheel"] if fresh else 0
        stop = self.values["stop"] if fresh else True
        state = self.step_vehicle(drive, wheel, stop)
        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = state.x, state.y
        odom.pose.pose.orientation = quaternion_from_yaw(state.yaw)
        odom.twist.twist.linear.x = state.speed_mps
        odom.twist.twist.angular.z = (
            state.speed_mps*math.tan(math.radians(state.steer_deg)) /
            self.vehicle.wheelbase)
        self.pub_odom.publish(odom)
        transform = TransformStamped()
        transform.header, transform.child_frame_id = odom.header, "base_link"
        transform.transform.translation.x = state.x
        transform.transform.translation.y = state.y
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)

        self.pub_encoder.publish(Int32(data=state.encoder))
        self.pub_steer.publish(Float32(data=state.steer_deg))
        self.pub_speed.publish(Float32(data=state.speed_mps))
        self.pub_distance.publish(Float32(data=state.distance_m))
        self.pub_connected.publish(Bool(data=True))
        self.pub_telemetry.publish(Bool(data=True))
        if (self.last_trajectory is None or math.hypot(
                state.x-self.last_trajectory[0],
                state.y-self.last_trajectory[1]) >= 0.04):
            item = PoseStamped()
            item.header = odom.header
            item.pose = odom.pose.pose
            self.trajectory.poses.append(item)
            self.last_trajectory = (state.x, state.y)
        self.trajectory.header.stamp = stamp
        self.pub_trajectory.publish(self.trajectory)
        self.pub_markers.publish(self.markers(state, stamp))
        if self.disturbance is not None:
            debug = dict(self.disturbance)
            debug.pop("started_at", None)
            debug["cte_m"] = self.cross_track_error
            debug["state"] = self.recovery_state()
            if not math.isfinite(debug["minimum_cte_m"]):
                debug["minimum_cte_m"] = None
            self.pub_disturbance.publish(String(data=json.dumps(
                debug, sort_keys=True, separators=(",", ":"))))
        self.pub_state.publish(String(data=json.dumps({
            "state": "RUNNING" if fresh else "WAITING_FOR_FOLLOWER",
            "prehardware_test_only": True, "spawn_branch": self.spawn_branch,
            "drive": drive, "wheel": wheel, "stop": bool(stop),
            "pwm": self.pwm.get(int(drive), 0),
            "speed_mps": state.speed_mps, "encoder": state.encoder,
            "distance_m": state.distance_m,
            "direction_guard_ok": state.direction_guard_ok,
        }, sort_keys=True, separators=(",", ":"))))

    def markers(self, state, stamp):
        result = MarkerArray()
        vehicle = Marker()
        vehicle.header.frame_id, vehicle.header.stamp = "map", stamp
        vehicle.ns, vehicle.id = "csv_only_vehicle", 0
        vehicle.type, vehicle.action = Marker.CUBE, Marker.ADD
        vehicle.pose.position.x, vehicle.pose.position.y = state.x, state.y
        vehicle.pose.position.z = 0.25
        vehicle.pose.orientation = quaternion_from_yaw(state.yaw)
        vehicle.scale.x, vehicle.scale.y, vehicle.scale.z = 1.40, 0.80, 0.35
        vehicle.color.r, vehicle.color.g, vehicle.color.b = 0.05, 0.45, 1.0
        vehicle.color.a = 0.90
        result.markers.append(vehicle)
        if self.disturbance is not None:
            injected = Marker()
            injected.header.frame_id, injected.header.stamp = "map", stamp
            injected.ns, injected.id = "injected_disturbance", 1
            injected.type, injected.action = Marker.SPHERE, Marker.ADD
            injected.pose.position.x = self.disturbance["injected_x_m"]
            injected.pose.position.y = self.disturbance["injected_y_m"]
            injected.pose.position.z = 0.25
            injected.pose.orientation.w = 1.0
            injected.scale.x = injected.scale.y = injected.scale.z = 0.35
            injected.color.r, injected.color.g, injected.color.b = 1.0, 0.1, 0.1
            injected.color.a = 0.95
            result.markers.append(injected)

            target = Marker()
            target.header.frame_id, target.header.stamp = "map", stamp
            target.ns, target.id = "recovery_target", 2
            target.type, target.action = Marker.SPHERE, Marker.ADD
            target.pose.position.x = self.disturbance["target_x_m"]
            target.pose.position.y = self.disturbance["target_y_m"]
            target.pose.position.z = 0.25
            target.pose.orientation.w = 1.0
            target.scale.x = target.scale.y = target.scale.z = 0.28
            target.color.r, target.color.g, target.color.b = 0.1, 1.0, 0.25
            target.color.a = 0.95
            result.markers.append(target)

            label = Marker()
            label.header.frame_id, label.header.stamp = "map", stamp
            label.ns, label.id = "recovery_status", 3
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = state.x, state.y
            label.pose.position.z = 1.4
            label.pose.orientation.w = 1.0
            label.scale.z = 0.42
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            elapsed = time.monotonic()-self.disturbance["started_at"]
            label.text = (
                f"CTE: {abs(self.cross_track_error):.2f} m\n"
                f"STATE: {self.recovery_state()}\n"
                f"MAX CTE: {self.disturbance['maximum_cte_m']:.2f} m\n"
                f"RECOVERY: {elapsed:.1f} s")
            result.markers.append(label)
        return result

    def recovery_state(self):
        absolute = abs(self.cross_track_error)
        if absolute > 1.0 or self.controller_state in (
                "ROUTE_DEVIATION_STOP", "PLAN_REJOIN"):
            return "OFF_ROUTE_STOP"
        if self.disturbance is not None and absolute > 0.25:
            return "REJOINING"
        return "TRACKING"


def main(args=None):
    rclpy.init(args=args)
    node = CsvOnlyVirtualVehicleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            safe_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
