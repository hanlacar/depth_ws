"""Test-only replacement for the serial bridge; never opens a serial port."""

import json
import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .ros_helpers import quaternion_from_yaw, safe_shutdown
from .virtual_mcu_core import VirtualAckermannVehicle


class VirtualMcuBridgeNode(Node):
    """Consumes only manager outputs and owns all virtual bridge telemetry."""

    def __init__(self):
        super().__init__("depth_virtual_mcu_bridge")
        defaults = {
            "publish_hz": 50.0, "simulation_speedup": 10.0,
            "command_timeout_s": 0.50,
            "origin_x_m": 0.0, "origin_y_m": 0.0, "origin_yaw_rad": 0.0,
            "expected_manager_node": "mcu_manager",
            "input_drive_topic": "/mcu/cmd_drive",
            "input_wheel_topic": "/mcu/cmd_wheel",
            "input_stop_topic": "/mcu/cmd_stop",
            "prehardware_test_only": True,
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
            raise ValueError("virtual MCU bridge is test-only and cannot be production-enabled")
        self.pwm_mapping = {
            -1: int(self.p("reverse_pwm")),
            0: 0,
            1: int(self.p("stage_1_pwm")),
            2: int(self.p("stage_2_pwm")),
            3: int(self.p("stage_3_pwm")),
        }
        self.vehicle = VirtualAckermannVehicle(
            {stage: float(self.p(f"stage_{stage}_speed_mps"))
             for stage in (1, 2, 3)},
            float(self.p("reverse_speed_mps")),
            wheelbase_m=float(self.p("wheelbase_m")),
            counts_per_meter=float(self.p("counts_per_meter")),
            max_steering_deg=float(self.p("max_steering_deg")),
            direction_change_hold_s=float(self.p("direction_change_hold_s")),
            encoder_signed=bool(self.p("encoder_signed")))
        if self.pwm_mapping != {-1: -50, 0: 0, 1: 50, 2: 75, 3: 100}:
            raise ValueError("virtual PWM mapping must match latest MCU contract")
        self.origin = (float(self.p("origin_x_m")), float(self.p("origin_y_m")),
                       float(self.p("origin_yaw_rad")))
        self.values = {"drive": 0.0, "wheel": 0, "stop": True}
        self.received = {}
        self.mode = "INVALID"
        self.branch = "A"
        self.route_index = -1
        self.block_reason = "WAITING_FOR_ACTUAL_MCU_MANAGER"
        self.conflicts = []
        self.last_tick = time.monotonic()
        self.last_trajectory = None
        self.trajectory = Path()
        self.trajectory.header.frame_id = "map"

        for topic in ("/odom", "/mcu/encoder"):
            existing = self.get_publishers_info_by_topic(topic)
            if existing:
                names = [f"{item.node_namespace}/{item.node_name}" for item in existing]
                raise RuntimeError(
                    f"real/duplicate bridge already owns {topic}: {names}")

        self.create_subscription(
            Float32, str(self.p("input_drive_topic")),
            lambda m: self.update("drive", float(m.data)), 10)
        self.create_subscription(
            Int32, str(self.p("input_wheel_topic")),
            lambda m: self.update("wheel", int(m.data)), 10)
        self.create_subscription(
            Bool, str(self.p("input_stop_topic")),
            lambda m: self.update("stop", bool(m.data)), 10)
        self.create_subscription(String, "/mcu/current_mode",
                                 lambda m: setattr(self, "mode", str(m.data)), 10)
        self.create_subscription(String, "/depth_slam/route/active_branch",
                                 lambda m: setattr(self, "branch", str(m.data)), 10)
        self.create_subscription(Int32, "/depth_slam/route/active_index",
                                 lambda m: setattr(self, "route_index", int(m.data)), 10)

        self.pub_encoder = self.create_publisher(Int32, "/mcu/encoder", 10)
        self.pub_steer = self.create_publisher(Float32, "/mcu/steer_deg", 10)
        self.pub_speed = self.create_publisher(Float32, "/mcu/speed_mps", 10)
        self.pub_distance = self.create_publisher(Float32, "/mcu/distance_m", 10)
        self.pub_connected = self.create_publisher(Bool, "/mcu/connected", 10)
        self.pub_telemetry = self.create_publisher(Bool, "/mcu/telemetry_ok", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_pose = self.create_publisher(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose", 10)
        self.pub_loc_state = self.create_publisher(
            String, "/depth_slam/localization/state", 10)
        self.pub_confidence = self.create_publisher(
            Float32, "/depth_slam/localization/confidence", 10)
        self.pub_pose_jump = self.create_publisher(
            Bool, "/depth_slam/localization/pose_jump", 10)
        self.pub_tracking = self.create_publisher(
            Bool, "/depth_slam/cuvslam/tracking_valid", 10)
        self.pub_stop_elapsed = self.create_publisher(
            Float32, "/depth_slam/virtual_mcu/stop_duration_s", 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/virtual_mcu/state", 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_trajectory = self.create_publisher(
            Path, "/depth_slam/virtual_mcu/trajectory", qos)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/depth_slam/virtual_mcu/markers", 10)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.publish_origin_tf()
        hz = float(self.p("publish_hz"))
        speedup = float(self.p("simulation_speedup"))
        if hz <= 0.0 or not 0.0 < speedup <= 25.0:
            raise ValueError("invalid virtual publish rate or speedup")
        self.create_timer(1.0/hz, self.tick)
        self.create_timer(1.0, self.audit_graph)
        self.get_logger().info(
            f"PWM_BASED_VIRTUAL_ODOM speeds={self.vehicle.stage_mps} "
            f"pwm={self.pwm_mapping} counts_per_meter={self.vehicle.counts_per_meter}")

    def p(self, name):
        return self.get_parameter(name).value

    def update(self, name, value):
        self.values[name] = value
        self.received[name] = time.monotonic()

    def publish_origin_tf(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.translation.x = self.origin[0]
        transform.transform.translation.y = self.origin[1]
        transform.transform.rotation = quaternion_from_yaw(self.origin[2])
        self.static_tf.sendTransform(transform)

    def _other_publishers(self, topic):
        return sorted({
            f"{item.node_namespace.rstrip('/')}/{item.node_name}"
            for item in self.get_publishers_info_by_topic(topic)
            if item.node_name != self.get_name()})

    def audit_graph(self):
        conflicts = []
        for topic in ("/odom", "/mcu/encoder"):
            conflicts.extend(f"{topic}:{name}" for name in self._other_publishers(topic))
        self.conflicts = sorted(set(conflicts))
        managers = self._other_publishers(str(self.p("input_drive_topic")))
        expected = str(self.p("expected_manager_node"))
        if self.conflicts:
            self.block_reason = "DUPLICATE_REAL_OR_VIRTUAL_BRIDGE"
        elif len(managers) != 1 or not managers[0].endswith("/"+expected):
            self.block_reason = "WAITING_FOR_ACTUAL_MCU_MANAGER"
        else:
            self.block_reason = ""

    def command_fresh(self, now):
        timeout = float(self.p("command_timeout_s"))
        return all(key in self.received and 0.0 <= now-self.received[key] <= timeout
                   for key in ("drive", "wheel", "stop"))

    def map_pose(self, state):
        cosine, sine = math.cos(self.origin[2]), math.sin(self.origin[2])
        return (self.origin[0]+cosine*state.x-sine*state.y,
                self.origin[1]+sine*state.x+cosine*state.y,
                math.atan2(math.sin(self.origin[2]+state.yaw),
                           math.cos(self.origin[2]+state.yaw)))

    def tick(self):
        now = time.monotonic()
        real_dt = max(1.0e-4, min(0.1, now-self.last_tick))
        self.last_tick = now
        safe = not self.block_reason and self.command_fresh(now)
        drive = self.values["drive"] if safe else 0.0
        wheel = self.values["wheel"] if safe else 0
        stop = self.values["stop"] if safe else True
        state = self.vehicle.step(
            drive, wheel, stop, real_dt*float(self.p("simulation_speedup")))
        if self.conflicts:
            return
        stamp = self.get_clock().now().to_msg()
        connected = not self.block_reason
        self.pub_connected.publish(Bool(data=connected))
        self.pub_telemetry.publish(Bool(data=connected))
        self.pub_encoder.publish(Int32(data=state.encoder))
        self.pub_steer.publish(Float32(data=state.steer_deg))
        self.pub_speed.publish(Float32(data=state.speed_mps))
        self.pub_distance.publish(Float32(data=state.distance_m))
        self.pub_stop_elapsed.publish(Float32(data=self.vehicle.stop_elapsed))

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = state.x, state.y
        odom.pose.pose.orientation = quaternion_from_yaw(state.yaw)
        odom.twist.twist.linear.x = state.speed_mps
        odom.twist.twist.angular.z = (state.speed_mps *
                                      math.tan(math.radians(state.steer_deg)) /
                                      self.vehicle.wheelbase)
        odom.pose.covariance[0] = odom.pose.covariance[7] = 1.0e-4
        odom.pose.covariance[35] = 1.0e-4
        self.pub_odom.publish(odom)
        transform = TransformStamped()
        transform.header, transform.child_frame_id = odom.header, "base_link"
        transform.transform.translation.x = state.x
        transform.transform.translation.y = state.y
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)

        map_x, map_y, map_yaw = self.map_pose(state)
        pose = PoseWithCovarianceStamped()
        pose.header.stamp, pose.header.frame_id = stamp, "map"
        pose.pose.pose.position.x, pose.pose.pose.position.y = map_x, map_y
        pose.pose.pose.orientation = quaternion_from_yaw(map_yaw)
        pose.pose.covariance[0] = pose.pose.covariance[7] = 1.0e-4
        pose.pose.covariance[35] = 1.0e-4
        self.pub_pose.publish(pose)
        self.pub_loc_state.publish(String(data="TRACKING"))
        self.pub_confidence.publish(Float32(data=1.0))
        self.pub_pose_jump.publish(Bool(data=False))
        self.pub_tracking.publish(Bool(data=True))
        if (self.last_trajectory is None or
                math.hypot(map_x-self.last_trajectory[0],
                           map_y-self.last_trajectory[1]) >= 0.04):
            item = PoseStamped()
            item.header = pose.header
            item.pose = pose.pose.pose
            self.trajectory.poses.append(item)
            self.last_trajectory = (map_x, map_y)
        self.trajectory.header.stamp = stamp
        self.pub_trajectory.publish(self.trajectory)
        self.pub_markers.publish(self.markers(map_x, map_y, map_yaw, state, stamp))
        self.pub_state.publish(String(data=json.dumps({
            "state": self.block_reason or "RUNNING",
            "prehardware_test_only": True,
            "mode": self.mode, "branch": self.branch,
            "route_index": self.route_index,
            "drive": drive, "wheel": wheel, "stop": bool(stop),
            "speed_mps": state.speed_mps,
            "pwm": self.pwm_mapping[int(drive)] if int(drive) in self.pwm_mapping else 0,
            "encoder": state.encoder, "distance_m": state.distance_m,
            "stop_duration_s": self.vehicle.stop_elapsed,
            "direction_guard_ok": state.direction_guard_ok,
            "publisher_conflicts": self.conflicts,
        }, sort_keys=True, separators=(",", ":"))))

    def markers(self, x, y, yaw, state, stamp):
        result = MarkerArray()
        vehicle = Marker()
        vehicle.header.frame_id, vehicle.header.stamp = "map", stamp
        vehicle.ns, vehicle.id = "virtual_vehicle", 0
        vehicle.type, vehicle.action = Marker.CUBE, Marker.ADD
        vehicle.pose.position.x, vehicle.pose.position.y = x, y
        vehicle.pose.position.z = 0.28
        vehicle.pose.orientation = quaternion_from_yaw(yaw)
        vehicle.scale.x, vehicle.scale.y, vehicle.scale.z = 1.40, 0.80, 0.35
        vehicle.color.r, vehicle.color.g = 0.05, 0.45
        vehicle.color.b, vehicle.color.a = 1.0, 0.85
        result.markers.append(vehicle)
        text = Marker()
        text.header, text.ns, text.id = vehicle.header, "virtual_status", 1
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y = x, y
        text.pose.position.z, text.pose.orientation.w = 1.15, 1.0
        text.scale.z = 0.32
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = (f"PREHARDWARE TEST ONLY\nMODE: {self.mode}  BRANCH: {self.branch} "
                     f"INDEX: {self.route_index}\nDRIVE: {self.values['drive']} "
                     f"WHEEL: {self.values['wheel']} STOP: {self.values['stop']}\n"
                     f"ENCODER: {state.encoder} DISTANCE: {state.distance_m:.2f} m")
        result.markers.append(text)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = VirtualMcuBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
