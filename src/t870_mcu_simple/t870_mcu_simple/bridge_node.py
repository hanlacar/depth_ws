#!/usr/bin/env python3
"""Final T870 serial bridge and sole measured production ODOM owner."""

import json
import math
import os
from pathlib import Path
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import TransformBroadcaster

from .odom_core import MeasuredEncoderOdom


def find_serial_port(preferred="/dev/t870_mcu"):
    if preferred and os.path.exists(preferred):
        return os.path.realpath(preferred)
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        accepted = ("arduino", "mega", "ch340", "ch341", "ch910", "wch")
        rejected = ("gps", "gnss", "u-blox", "ublox")
        matches = []
        for path in by_id.iterdir():
            name = path.name.lower()
            if any(word in name for word in rejected):
                continue
            score = (100 if "arduino" in name or "mega" in name else
                     50 if any(word in name for word in accepted) else 0)
            if score:
                matches.append((score, str(path.resolve())))
        if matches:
            return sorted(matches, reverse=True)[0][1]
    acm = sorted(Path("/dev").glob("ttyACM*"))
    return str(acm[0]) if len(acm) == 1 else None


class SimpleMcuBridge(Node):
    def __init__(self):
        super().__init__("t870_mcu_simple_bridge")
        defaults = (
            ("port", "auto"), ("preferred_symlink", "/dev/t870_mcu"),
            ("baud", 115200), ("reset_wait_s", 2.0),
            ("reconnect_s", 1.0), ("tx_hz", 10.0),
            ("drive_timeout_s", 0.5), ("steer_center_adc", 484),
            ("steer_counts_per_deg", 18.0), ("max_steer_deg", 22),
            ("steer_pwm", 130), ("steer_tolerance_adc", 4),
            ("steer_settle_ms", 250), ("steer_fine_band_adc", 50),
            ("steer_fine_on_ms", 30), ("steer_fine_cycle_ms", 90),
            ("steer_timeout_ms", 5000),
            ("firmware_drive_timeout_ms", 700),
            ("firmware_status_ms", 200), ("front_forward_level", 0),
            ("rear_forward_level", 0), ("steer_left_level", 1),
            ("reverse_pwm", 50), ("stage1_pwm", 50),
            ("stage2_pwm", 75), ("stage3_pwm", 100),
            ("encoder_cpr", 163.0), ("counts_per_meter", 797.0),
            ("max_encoder_delta_counts", 5000),
            ("wheelbase_m", 0.73), ("base_from_rear_m", 0.365),
            ("odom_publish_hz", 30.0), ("odom_input_timeout_s", 0.5),
            ("odom_frame", "odom"), ("base_frame", "base_link"),
        )
        for name, value in defaults:
            self.declare_parameter(name, value)

        def p(name):
            return self.get_parameter(name).value

        self.port_parameter = str(p("port"))
        self.preferred_symlink = str(p("preferred_symlink"))
        self.baud = int(p("baud"))
        self.reset_wait_s = float(p("reset_wait_s"))
        self.reconnect_s = float(p("reconnect_s"))
        self.drive_timeout_s = float(p("drive_timeout_s"))
        self.center_adc = int(p("steer_center_adc"))
        self.counts_per_degree = float(p("steer_counts_per_deg"))
        self.max_degree = int(p("max_steer_deg"))
        if self.max_degree != 22:
            raise ValueError("T870 production steering limit must be 22 degrees")
        self.steer_pwm = int(p("steer_pwm"))
        self.steer_tolerance = int(p("steer_tolerance_adc"))
        self.steer_settle_ms = int(p("steer_settle_ms"))
        self.steer_fine_band = int(p("steer_fine_band_adc"))
        self.steer_fine_on_ms = int(p("steer_fine_on_ms"))
        self.steer_fine_cycle_ms = int(p("steer_fine_cycle_ms"))
        self.steer_timeout_ms = int(p("steer_timeout_ms"))
        self.firmware_drive_timeout_ms = int(p("firmware_drive_timeout_ms"))
        self.firmware_status_ms = int(p("firmware_status_ms"))
        self.front_forward = 1 if int(p("front_forward_level")) else 0
        self.rear_forward = 1 if int(p("rear_forward_level")) else 0
        self.steer_left = 1 if int(p("steer_left_level")) else 0
        self.reverse_pwm = int(p("reverse_pwm"))
        self.stages = {
            0: 0, 1: int(p("stage1_pwm")), 2: int(p("stage2_pwm")),
            3: int(p("stage3_pwm")),
        }
        self.encoder_cpr = float(p("encoder_cpr"))
        self.odom_frame = str(p("odom_frame")).lstrip("/")
        self.base_frame = str(p("base_frame")).lstrip("/")
        self.odom_timeout = float(p("odom_input_timeout_s"))
        self.odom_model = MeasuredEncoderOdom(
            counts_per_meter=float(p("counts_per_meter")),
            wheelbase_m=float(p("wheelbase_m")),
            base_from_rear_m=float(p("base_from_rear_m")),
            max_steer_deg=self.max_degree,
            max_encoder_delta_counts=int(p("max_encoder_delta_counts")))

        self.serial = None
        self.port = None
        self.opened_at = 0.0
        self.ready = False
        self.last_connect_attempt = 0.0
        self.rx_buffer = bytearray()
        self.commanded_drive = 0
        self.last_drive_received = None
        self.stop_active = False
        self.last_wheel_command = 0
        self.last_encoder = None
        self.last_encoder_at = None
        self.last_status_at = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.applied_stage = 0
        self.applied_pwm = 0
        self.firmware_drive_pwm = None
        self.firmware_armed = None
        self.firmware_steer_active = None
        self.last_command_log = None

        self.create_subscription(Float32, "/cmd_drive", self.on_drive, 10)
        self.create_subscription(Int32, "/cmd_wheel", self.on_wheel, 10)
        self.create_subscription(Bool, "/cmd_stop", self.on_stop, 10)
        self.connected_pub = self.create_publisher(Bool, "/mcu/connected", 10)
        self.ready_pub = self.create_publisher(Bool, "/mcu/ready", 10)
        self.applied_drive_pub = self.create_publisher(
            Float32, "/mcu/applied_drive", 10)
        self.applied_wheel_pub = self.create_publisher(
            Int32, "/mcu/applied_wheel", 10)
        self.stop_pub = self.create_publisher(Bool, "/mcu/stop_active", 10)
        self.steer_adc_pub = self.create_publisher(Int32, "/mcu/steer_a0", 10)
        self.steer_pub = self.create_publisher(Float32, "/mcu/steer_deg", 10)
        self.encoder_pub = self.create_publisher(Int32, "/mcu/encoder", 10)
        self.encoder_a_pub = self.create_publisher(Int32, "/mcu/encA", 10)
        self.encoder_rate_pub = self.create_publisher(
            Float32, "/mcu/encoder_rate", 10)
        self.rpm_pub = self.create_publisher(Float32, "/mcu/rpm", 10)
        self.raw_pub = self.create_publisher(String, "/mcu/raw_status", 10)
        self.firmware_pub = self.create_publisher(String, "/mcu/fw_message", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.odom_state_pub = self.create_publisher(
            String, "/mcu/odom_state", 10)
        self.odom_distance_pub = self.create_publisher(
            Float32, "/mcu/odom_distance_m", 10)
        self.firmware_drive_pub = self.create_publisher(
            Int32, "/mcu/firmware_drive_pwm", 10)
        self.firmware_armed_pub = self.create_publisher(
            Bool, "/mcu/firmware_armed", 10)
        self.command_diag_pub = self.create_publisher(
            String, "/mcu/command_diagnostics", 10)
        self.transform = TransformBroadcaster(self)

        tx_hz = float(p("tx_hz"))
        odom_hz = float(p("odom_publish_hz"))
        if min(tx_hz, odom_hz) <= 0.0:
            raise ValueError("publish frequencies must be positive")
        self.create_timer(0.02, self.io_tick)
        self.create_timer(1.0/tx_hz, self.tx_tick)
        self.create_timer(1.0/odom_hz, self.publish_odom)
        self.get_logger().info(
            "MCU contract: sole /cmd_drive + /cmd_wheel consumer; "
            "measured encoder/steering /odom owner")

    def resolve_port(self):
        if self.port_parameter.lower() not in ("auto", "", "none"):
            return self.port_parameter
        return find_serial_port(self.preferred_symlink)

    def open_serial(self):
        now = time.monotonic()
        if now-self.last_connect_attempt < self.reconnect_s:
            return
        self.last_connect_attempt = now
        port = self.resolve_port()
        if not port:
            return
        try:
            self.serial = serial.Serial(
                port, self.baud, timeout=0, write_timeout=0.2)
            self.port = port
            self.opened_at = now
            self.ready = False
            self.rx_buffer.clear()
            self.odom_model.reset_encoder_baseline()
            self.publish_bool(self.connected_pub, True)
            self.publish_bool(self.ready_pub, False)
            self.get_logger().info(
                f"serial opened: {port} @ {self.baud}")
        except Exception as error:
            self.serial = None
            self.publish_bool(self.connected_pub, False)
            self.get_logger().warning(
                f"serial open failed: {type(error).__name__}: {error}")

    def close_serial(self, reason=""):
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.ready = False
        self.port = None
        self.last_status_at = None
        self.publish_bool(self.connected_pub, False)
        self.publish_bool(self.ready_pub, False)
        if reason:
            self.get_logger().warning(reason)

    def send(self, value):
        if self.serial is None:
            return False
        try:
            self.serial.write((str(value)+"\n").encode("ascii"))
            return True
        except Exception as error:
            self.close_serial(
                f"serial write failed: {type(error).__name__}: {error}")
            return False

    def configure_and_arm(self):
        commands = (
            f"CFG,CENTER,{self.center_adc}",
            f"CFG,COUNTS_PER_DEG,{self.counts_per_degree:.6f}",
            f"CFG,MAX_DEG,{self.max_degree}",
            f"CFG,STEER_PWM,{self.steer_pwm}",
            f"CFG,STEER_TOL,{self.steer_tolerance}",
            f"CFG,STEER_SETTLE_MS,{self.steer_settle_ms}",
            f"CFG,STEER_FINE_BAND,{self.steer_fine_band}",
            f"CFG,STEER_FINE_ON_MS,{self.steer_fine_on_ms}",
            f"CFG,STEER_FINE_CYCLE_MS,{self.steer_fine_cycle_ms}",
            f"CFG,STEER_TIMEOUT_MS,{self.steer_timeout_ms}",
            f"CFG,DRIVE_TIMEOUT_MS,{self.firmware_drive_timeout_ms}",
            f"CFG,STATUS_MS,{self.firmware_status_ms}",
            f"CFG,FRONT_FWD,{self.front_forward}",
            f"CFG,REAR_FWD,{self.rear_forward}",
            f"CFG,STEER_LEFT,{self.steer_left}",
            "X", "ARM", "D,0",
        )
        if not all(self.send(command) for command in commands):
            return
        # The commissioned firmware zeros its encoder on ARM. Discard the
        # pre-ARM counter (for example 153048 -> 0) before publishing ODOM.
        self.odom_model.reset_encoder_baseline()
        self.last_encoder = None
        self.last_encoder_at = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.ready = True
        self.publish_bool(self.ready_pub, True)
        self.get_logger().info("Arduino READY and armed at drive=0")

    def io_tick(self):
        if self.serial is None:
            self.open_serial()
            return
        if not self.ready and time.monotonic()-self.opened_at >= self.reset_wait_s:
            self.configure_and_arm()
        try:
            waiting = self.serial.in_waiting
            if waiting:
                self.rx_buffer.extend(self.serial.read(waiting))
                if len(self.rx_buffer) > 8192:
                    self.rx_buffer.clear()
                while b"\n" in self.rx_buffer:
                    line, _, remaining = self.rx_buffer.partition(b"\n")
                    self.rx_buffer = bytearray(remaining)
                    value = line.decode("utf-8", errors="replace").strip()
                    if value:
                        self.handle_line(value)
        except Exception as error:
            self.close_serial(
                f"serial read failed: {type(error).__name__}: {error}")

    def tx_tick(self):
        self.publish_bool(self.connected_pub, self.serial is not None)
        self.publish_bool(self.ready_pub, self.ready)
        if not self.ready:
            self.publish_applied(0, 0, False)
            return
        now = time.monotonic()
        fresh = (self.last_drive_received is not None and
                 now-self.last_drive_received <=
                 self.drive_timeout_s)
        stage = self.commanded_drive if fresh and not self.stop_active else 0
        pwm = (-abs(self.reverse_pwm) if stage == -1 else
               abs(self.stages.get(stage, 0)))
        self.send(f"D,{pwm}")
        self.publish_applied(stage, pwm, fresh)

    def on_drive(self, message):
        raw = float(message.data)
        value = int(round(raw))
        if abs(raw-value) > 1.0e-6 or value not in (-1, 0, 1, 2, 3):
            self.get_logger().warning(
                f"cmd_drive ignored: {raw}; allowed -1,0,1,2,3")
            return
        self.commanded_drive = value
        self.odom_model.set_direction_from_stage(value)
        self.last_drive_received = time.monotonic()

    def on_wheel(self, message):
        degree = max(-22, min(22, int(message.data)))
        self.last_wheel_command = degree
        if self.ready and not self.stop_active:
            self.send(f"W,{degree}")
            self.applied_wheel_pub.publish(Int32(data=degree))

    def on_stop(self, message):
        requested = bool(message.data)
        if requested and not self.stop_active:
            self.stop_active = True
            self.last_drive_received = None
            if self.ready:
                self.send("X")
                self.send("D,0")
        elif not requested and self.stop_active:
            self.stop_active = False
        self.publish_bool(self.stop_pub, self.stop_active)

    def handle_line(self, value):
        if value.startswith("STAT,"):
            self.handle_status(value)
            return
        self.firmware_pub.publish(String(data=value))
        if value.startswith(("ERR,", "EVT,")):
            self.get_logger().warning("MCU: "+value)

    def handle_status(self, value):
        # The uploaded 0914 firmware emits one duplicated drive field. Accept
        # that exact 9-field line and the corrected documented 8-field line.
        fields = value.split(",")
        if len(fields) not in (8, 9):
            return
        try:
            adc = int(fields[1])
            firmware_pwm = int(fields[4])
            steer_active = bool(int(fields[6] if len(fields) == 9 else
                                    fields[5]))
            firmware_armed = bool(int(fields[7] if len(fields) == 9 else
                                      fields[6]))
            encoder = int(fields[8] if len(fields) == 9 else fields[7])
        except ValueError:
            return
        now = time.monotonic()
        steering = max(-22.0, min(
            22.0, (adc-self.center_adc)/self.counts_per_degree))
        self.odom_model.set_steering_deg(steering)
        distance, delta_yaw = self.odom_model.update_encoder(encoder)
        elapsed = (None if self.last_encoder_at is None else
                   now-self.last_encoder_at)
        if self.odom_model.last_discontinuity:
            self.get_logger().warning(
                "encoder counter discontinuity ignored; ODOM re-baselined")
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            encoder_rate = 0.0
        elif elapsed is not None and elapsed > 1.0e-3:
            self.linear_velocity = distance/elapsed
            self.angular_velocity = delta_yaw/elapsed
            encoder_rate = (0.0 if self.last_encoder is None else
                            (encoder-self.last_encoder)/elapsed)
        else:
            encoder_rate = 0.0
        self.last_encoder = encoder
        self.last_encoder_at = now
        self.last_status_at = now
        self.firmware_drive_pwm = firmware_pwm
        self.firmware_armed = firmware_armed
        self.firmware_steer_active = steer_active
        self.raw_pub.publish(String(data=value))
        self.firmware_drive_pub.publish(Int32(data=firmware_pwm))
        self.firmware_armed_pub.publish(Bool(data=firmware_armed))
        self.steer_adc_pub.publish(Int32(data=adc))
        self.steer_pub.publish(Float32(data=steering))
        self.encoder_pub.publish(Int32(data=encoder))
        self.encoder_a_pub.publish(Int32(data=encoder))
        self.encoder_rate_pub.publish(Float32(data=float(encoder_rate)))
        rpm = (encoder_rate*60.0/self.encoder_cpr
               if self.encoder_cpr > 0.0 else 0.0)
        self.rpm_pub.publish(Float32(data=float(rpm)))

    def publish_odom(self):
        now = time.monotonic()
        fresh = (self.ready and self.last_status_at is not None and
                 now-self.last_status_at <= self.odom_timeout)
        self.odom_state_pub.publish(String(data=(
            "TRACKING_REAL_ENCODER_STEERING" if fresh else
            "WAITING_FOR_REAL_ENCODER_STEERING")))
        if not fresh:
            return
        x, y, yaw = self.odom_model.base_pose()
        stamp = self.get_clock().now().to_msg()
        orientation_z = math.sin(yaw/2.0)
        orientation_w = math.cos(yaw/2.0)
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = orientation_z
        message.pose.pose.orientation.w = orientation_w
        message.twist.twist.linear.x = self.linear_velocity
        message.twist.twist.angular.z = self.angular_velocity
        message.pose.covariance[0] = 0.01
        message.pose.covariance[7] = 0.01
        message.pose.covariance[35] = 0.02
        self.odom_pub.publish(message)
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation.z = orientation_z
        transform.transform.rotation.w = orientation_w
        self.transform.sendTransform(transform)
        self.odom_distance_pub.publish(Float32(
            data=float(self.odom_model.state.distance_m)))

    def publish_applied(self, stage, pwm=0, command_fresh=False):
        self.applied_stage = int(stage)
        self.applied_pwm = int(pwm)
        self.applied_drive_pub.publish(Float32(data=float(stage)))
        self.publish_bool(self.stop_pub, self.stop_active)
        now = time.monotonic()
        command_age = (None if self.last_drive_received is None else
                       max(0.0, now-self.last_drive_received))
        status_age = (None if self.last_status_at is None else
                      max(0.0, now-self.last_status_at))
        values = {
            "serial_connected": self.serial is not None,
            "bridge_ready": self.ready,
            "serial_port": self.port,
            "command_fresh": bool(command_fresh),
            "command_age_s": command_age,
            "commanded_stage": self.commanded_drive,
            "applied_stage": self.applied_stage,
            "applied_pwm": self.applied_pwm,
            "stop_active": self.stop_active,
            "firmware_armed": self.firmware_armed,
            "firmware_drive_pwm": self.firmware_drive_pwm,
            "firmware_steer_active": self.firmware_steer_active,
            "status_age_s": status_age,
            "encoder": self.last_encoder,
        }
        self.command_diag_pub.publish(String(data=json.dumps(
            values, separators=(",", ":"), allow_nan=False)))
        signature = (
            self.applied_stage, self.applied_pwm, bool(command_fresh),
            self.stop_active, self.firmware_armed,
            self.firmware_drive_pwm)
        if signature != self.last_command_log:
            self.get_logger().info(
                "[MCU COMMAND] "+json.dumps(
                    values, separators=(",", ":"), allow_nan=False))
            self.last_command_log = signature

    @staticmethod
    def publish_bool(publisher, value):
        publisher.publish(Bool(data=bool(value)))

    def shutdown(self):
        if self.serial is not None:
            try:
                self.send("X")
                self.send("DISARM")
                self.serial.close()
            except Exception:
                pass
            self.serial = None


def main(args=None):
    rclpy.init(args=args)
    node = SimpleMcuBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
