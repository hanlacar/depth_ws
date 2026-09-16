#!/usr/bin/env python3
"""T870 FIXED_0914 serial bridge with measured production odometry."""

import json
import math
import os
import time
from pathlib import Path

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import TransformBroadcaster

import serial

from .odom_core import MeasuredEncoderOdom, steering_from_adc


def find_serial_port(preferred: str = "/dev/t870_mcu") -> str | None:
    if preferred and os.path.exists(preferred):
        return os.path.realpath(preferred)

    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        preferred_words = ("arduino", "mega", "ch340", "ch341", "ch910", "wch")
        reject_words = ("gps", "gnss", "u-blox", "ublox")
        scored = []
        for p in by_id.iterdir():
            name = p.name.lower()
            if any(w in name for w in reject_words):
                continue
            score = 0
            if "arduino" in name or "mega" in name:
                score = 100
            elif any(w in name for w in preferred_words):
                score = 50
            if score:
                scored.append((score, str(p.resolve())))
        if scored:
            scored.sort(reverse=True)
            return scored[0][1]

    # Mega 계열은 일반적으로 ttyACM. ttyUSB는 GPS/CP210x 오탐 위험 때문에 자동 사용 안 함.
    acms = sorted(Path("/dev").glob("ttyACM*"))
    if len(acms) == 1:
        return str(acms[0])
    return None


class SimpleMcuBridge(Node):
    def __init__(self):
        super().__init__("t870_mcu_simple_bridge")

        # Serial / safety
        self.declare_parameter("port", "auto")
        self.declare_parameter("preferred_symlink", "/dev/t870_mcu")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("reset_wait_s", 2.0)
        self.declare_parameter("reconnect_s", 1.0)
        self.declare_parameter("tx_hz", 10.0)
        self.declare_parameter("drive_timeout_s", 0.5)

        # Vehicle calibration
        self.declare_parameter("steer_center_adc", 484)
        self.declare_parameter("steer_counts_per_deg", 18.0)
        self.declare_parameter("max_steer_deg", 22)
        self.declare_parameter("steer_pwm", 130)
        self.declare_parameter("steer_tolerance_adc", 4)
        self.declare_parameter("steer_settle_ms", 250)
        self.declare_parameter("steer_fine_band_adc", 50)
        self.declare_parameter("steer_fine_on_ms", 30)
        self.declare_parameter("steer_fine_cycle_ms", 90)
        self.declare_parameter("steer_timeout_ms", 5000)
        self.declare_parameter("firmware_drive_timeout_ms", 700)
        self.declare_parameter("firmware_status_ms", 200)

        # Direction polarity. 1=HIGH when forward/left, 0=LOW.
        self.declare_parameter("front_forward_level", 1)
        self.declare_parameter("rear_forward_level", 1)
        self.declare_parameter("steer_left_level", 1)

        # Drive stage mapping
        self.declare_parameter("reverse_pwm", 50)
        self.declare_parameter("stage1_pwm", 50)
        self.declare_parameter("stage2_pwm", 75)
        self.declare_parameter("stage3_pwm", 100)
        self.declare_parameter("encoder_cpr", 163.0)

        # Measured encoder + steering Ackermann odometry.  These parameters do
        # not synthesize motion; pose advances only on a real ENC_A count.
        self.declare_parameter("counts_per_meter", 797.0)
        self.declare_parameter("max_encoder_delta_counts", 5000)
        self.declare_parameter("wheelbase_m", 0.73)
        self.declare_parameter("base_from_rear_m", 0.365)
        self.declare_parameter("odom_publish_hz", 30.0)
        self.declare_parameter("odom_input_timeout_s", 0.5)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        gp = lambda k: self.get_parameter(k).value
        self.port_param = str(gp("port"))
        self.preferred_symlink = str(gp("preferred_symlink"))
        self.baud = int(gp("baud"))
        self.reset_wait_s = float(gp("reset_wait_s"))
        self.reconnect_s = float(gp("reconnect_s"))
        self.tx_period = 1.0 / max(1.0, float(gp("tx_hz")))
        self.drive_timeout_s = float(gp("drive_timeout_s"))

        self.center_adc = int(gp("steer_center_adc"))
        self.counts_per_deg = float(gp("steer_counts_per_deg"))
        self.max_deg = int(gp("max_steer_deg"))
        if self.max_deg != 22:
            raise ValueError("T870 production steering limit must be 22 degrees")
        self.steer_pwm = int(gp("steer_pwm"))
        self.steer_tol = int(gp("steer_tolerance_adc"))
        self.steer_settle_ms = int(gp("steer_settle_ms"))
        self.steer_fine_band = int(gp("steer_fine_band_adc"))
        self.steer_fine_on_ms = int(gp("steer_fine_on_ms"))
        self.steer_fine_cycle_ms = int(gp("steer_fine_cycle_ms"))
        self.steer_timeout_ms = int(gp("steer_timeout_ms"))
        self.fw_drive_timeout_ms = int(gp("firmware_drive_timeout_ms"))
        self.fw_status_ms = int(gp("firmware_status_ms"))
        self.front_fwd = 1 if int(gp("front_forward_level")) else 0
        self.rear_fwd = 1 if int(gp("rear_forward_level")) else 0
        self.steer_left = 1 if int(gp("steer_left_level")) else 0

        self.reverse_pwm = int(gp("reverse_pwm"))
        self.stage_pwm = {
            0: 0,
            1: int(gp("stage1_pwm")),
            2: int(gp("stage2_pwm")),
            3: int(gp("stage3_pwm")),
        }
        self.encoder_cpr = float(gp("encoder_cpr"))
        self.odom_timeout = float(gp("odom_input_timeout_s"))
        self.odom_frame = str(gp("odom_frame"))
        self.base_frame = str(gp("base_frame"))
        odom_hz = float(gp("odom_publish_hz"))
        if odom_hz <= 0.0 or self.odom_timeout <= 0.0:
            raise ValueError("odometry rate and timeout must be positive")
        self.odom_model = MeasuredEncoderOdom(
            counts_per_meter=float(gp("counts_per_meter")),
            wheelbase_m=float(gp("wheelbase_m")),
            base_from_rear_m=float(gp("base_from_rear_m")),
            max_steer_deg=float(self.max_deg),
            max_encoder_delta_counts=int(gp("max_encoder_delta_counts")),
        )

        self.ser = None
        self.port = None
        self.opened_at = 0.0
        self.ready = False
        self.last_connect_attempt = 0.0
        self.rx_buf = bytearray()

        self.cmd_drive = 0
        self.last_drive_rx = None
        self.stop_active = False
        self.last_wheel_cmd = 0
        self.last_sent_drive_pwm = None

        self.last_enc = None
        self.last_enc_t = None
        self.last_status_at = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.firmware_drive_pwm = None
        self.firmware_armed = None
        self.firmware_steer_active = None
        self.applied_stage = 0
        self.applied_pwm = 0
        self.last_command_log = None

        # Inputs: upper layer already decided everything.
        self.create_subscription(Float32, "/cmd_drive", self.cb_drive, 10)
        self.create_subscription(Int32, "/cmd_wheel", self.cb_wheel, 10)
        self.create_subscription(Bool, "/cmd_stop", self.cb_stop, 10)

        # Minimal feedback.
        self.pub_connected = self.create_publisher(Bool, "/mcu/connected", 10)
        self.pub_ready = self.create_publisher(Bool, "/mcu/ready", 10)
        self.pub_applied_drive = self.create_publisher(Float32, "/mcu/applied_drive", 10)
        self.pub_applied_wheel = self.create_publisher(Int32, "/mcu/applied_wheel", 10)
        self.pub_stop = self.create_publisher(Bool, "/mcu/stop_active", 10)
        self.pub_a0 = self.create_publisher(Int32, "/mcu/steer_a0", 10)
        self.pub_steer_deg = self.create_publisher(Float32, "/mcu/steer_deg", 10)
        self.pub_encoder = self.create_publisher(Int32, "/mcu/encoder", 10)
        self.pub_encA = self.create_publisher(Int32, "/mcu/encA", 10)
        self.pub_encoder_rate = self.create_publisher(Float32, "/mcu/encoder_rate", 10)
        self.pub_rpm = self.create_publisher(Float32, "/mcu/rpm", 10)
        self.pub_speed_mps = self.create_publisher(Float32, "/mcu/speed_mps", 10)
        self.pub_distance_m = self.create_publisher(Float32, "/mcu/distance_m", 10)
        self.pub_odom_distance_m = self.create_publisher(
            Float32, "/mcu/odom_distance_m", 10)
        self.pub_raw = self.create_publisher(String, "/mcu/raw_status", 10)
        self.pub_fw = self.create_publisher(String, "/mcu/fw_message", 10)
        self.pub_firmware_drive = self.create_publisher(
            Int32, "/mcu/firmware_drive_pwm", 10)
        self.pub_firmware_armed = self.create_publisher(
            Bool, "/mcu/firmware_armed", 10)
        self.pub_command_diagnostics = self.create_publisher(
            String, "/mcu/command_diagnostics", 10)
        self.pub_odom_state = self.create_publisher(
            String, "/mcu/odom_state", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.transform = TransformBroadcaster(self)

        self.io_timer = self.create_timer(0.02, self.io_tick)
        self.tx_timer = self.create_timer(self.tx_period, self.tx_tick)
        self.odom_timer = self.create_timer(1.0 / odom_hz, self.publish_odom)

        self.get_logger().info(
            "simple bridge: /cmd_drive + /cmd_wheel + /cmd_stop only; "
            "no manager, no mode/arbitration, no automatic drive")
        self.get_logger().info(
            f"steering center={self.center_adc}, +LEFT/-RIGHT, clamp ±{self.max_deg}deg; "
            f"settle={self.steer_settle_ms}ms tol=±{self.steer_tol} ADC")
        self.get_logger().info("drive encoder: ENC_A only, Arduino Mega D2, RISING; /mcu/encA = /mcu/encoder")
        self.get_logger().info(
            "MCU contract: sole /cmd_drive + /cmd_wheel consumer; "
            "measured encoder/steering /odom owner")

    def resolve_port(self):
        if self.port_param.lower() not in ("auto", "", "none"):
            return self.port_param
        return find_serial_port(self.preferred_symlink)

    def open_serial(self):
        now = time.monotonic()
        if now - self.last_connect_attempt < self.reconnect_s:
            return
        self.last_connect_attempt = now
        port = self.resolve_port()
        if not port:
            return
        try:
            self.ser = serial.Serial(port, self.baud, timeout=0, write_timeout=0.2)
            self.port = port
            self.opened_at = now
            self.ready = False
            self.rx_buf.clear()
            self.last_sent_drive_pwm = None
            self.last_status_at = None
            self.odom_model.reset_encoder_baseline()
            self.publish_bool(self.pub_connected, True)
            self.publish_bool(self.pub_ready, False)
            self.get_logger().info(f"serial opened: {port} @ {self.baud}; reset wait {self.reset_wait_s:.1f}s")
        except Exception as e:
            self.ser = None
            self.publish_bool(self.pub_connected, False)
            self.get_logger().warn(f"serial open failed: {type(e).__name__}: {e}")

    def close_serial(self, reason=""):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.ready = False
        self.port = None
        self.last_sent_drive_pwm = None
        self.last_status_at = None
        self.publish_bool(self.pub_connected, False)
        self.publish_bool(self.pub_ready, False)
        if reason:
            self.get_logger().warn(reason)

    def send(self, text: str):
        if self.ser is None:
            return False
        try:
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as e:
            self.close_serial(f"serial write failed: {type(e).__name__}: {e}")
            return False

    def configure_and_arm(self):
        # No steering movement here. Only configuration + drive 0 + ARM.
        cmds = [
            f"CFG,CENTER,{self.center_adc}",
            f"CFG,COUNTS_PER_DEG,{self.counts_per_deg:.6f}",
            f"CFG,MAX_DEG,{self.max_deg}",
            f"CFG,STEER_PWM,{self.steer_pwm}",
            f"CFG,STEER_TOL,{self.steer_tol}",
            f"CFG,STEER_SETTLE_MS,{self.steer_settle_ms}",
            f"CFG,STEER_FINE_BAND,{self.steer_fine_band}",
            f"CFG,STEER_FINE_ON_MS,{self.steer_fine_on_ms}",
            f"CFG,STEER_FINE_CYCLE_MS,{self.steer_fine_cycle_ms}",
            f"CFG,STEER_TIMEOUT_MS,{self.steer_timeout_ms}",
            f"CFG,DRIVE_TIMEOUT_MS,{self.fw_drive_timeout_ms}",
            f"CFG,STATUS_MS,{self.fw_status_ms}",
            f"CFG,FRONT_FWD,{self.front_fwd}",
            f"CFG,REAR_FWD,{self.rear_fwd}",
            f"CFG,STEER_LEFT,{self.steer_left}",
            "X",
            "ARM",
            "D,0",
        ]
        for c in cmds:
            if not self.send(c):
                return False
        # ARM resets the commissioned firmware encoder.  Never interpret the
        # old counter -> zero transition as physical travel.
        self.odom_model.reset_encoder_baseline()
        self.last_enc = None
        self.last_enc_t = None
        self.last_status_at = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.ready = True
        self.publish_bool(self.pub_ready, True)
        self.get_logger().info("Arduino READY and armed at drive=0; waiting ROS commands")
        return True

    def io_tick(self):
        if self.ser is None:
            self.open_serial()
            return

        if not self.ready and (time.monotonic() - self.opened_at >= self.reset_wait_s):
            self.configure_and_arm()

        try:
            waiting = self.ser.in_waiting
            if waiting:
                chunk = self.ser.read(waiting)
                self.rx_buf.extend(chunk)
                if len(self.rx_buf) > 8192:
                    self.rx_buf.clear()
                while b"\n" in self.rx_buf:
                    line, _, rest = self.rx_buf.partition(b"\n")
                    self.rx_buf = bytearray(rest)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self.handle_line(text)
        except Exception as e:
            self.close_serial(f"serial read failed: {type(e).__name__}: {e}")

    def tx_tick(self):
        self.publish_bool(self.pub_connected, self.ser is not None)
        self.publish_bool(self.pub_ready, self.ready)
        if not self.ready:
            self.publish_applied(0)
            return

        now = time.monotonic()
        fresh = self.last_drive_rx is not None and (now - self.last_drive_rx <= self.drive_timeout_s)
        stage = self.cmd_drive if (fresh and not self.stop_active) else 0
        signed_pwm = self.stage_to_pwm(stage)

        # Repeat at tx_hz so the Arduino watchdog is fed only while ROS command is fresh.
        self.send(f"D,{signed_pwm}")
        self.last_sent_drive_pwm = signed_pwm
        self.publish_applied(stage)

    def stage_to_pwm(self, stage: int) -> int:
        if stage == -1:
            return -abs(self.reverse_pwm)
        return abs(self.stage_pwm.get(stage, 0))

    def cb_drive(self, msg: Float32):
        raw = float(msg.data)
        v = int(round(raw))
        if abs(raw - v) > 1e-6 or v not in (-1, 0, 1, 2, 3):
            self.get_logger().warn(f"cmd_drive ignored: {msg.data} (allowed exactly -1,0,1,2,3)")
            return
        self.cmd_drive = v
        self.odom_model.set_direction_from_stage(v)
        self.last_drive_rx = time.monotonic()

    def cb_wheel(self, msg: Int32):
        deg = max(-self.max_deg, min(self.max_deg, int(msg.data)))
        self.last_wheel_cmd = deg
        if self.ready and not self.stop_active:
            self.send(f"W,{deg}")
            m = Int32(); m.data = deg; self.pub_applied_wheel.publish(m)

    def cb_stop(self, msg: Bool):
        requested = bool(msg.data)
        if requested and not self.stop_active:
            self.stop_active = True
            # STOP 해제 후 과거 drive 명령이 자동 재적용되지 않게 폐기.
            # 상위가 계속 drive를 발행 중이면 다음 새 메시지에서 즉시 재개된다.
            self.last_drive_rx = None
            if self.ready:
                self.send("X")
                self.send("D,0")
        elif not requested and self.stop_active:
            self.stop_active = False
        self.publish_bool(self.pub_stop, self.stop_active)

    def handle_line(self, text: str):
        if text.startswith("STAT,"):
            self.handle_status(text)
            return

        # READY/OK/EVT/ERR are low-rate only. No 20-lines-per-second logger.
        m = String(); m.data = text; self.pub_fw.publish(m)
        if text.startswith("ERR,") or text.startswith("EVT,"):
            self.get_logger().warn(f"MCU: {text}")

    def handle_status(self, text: str):
        # FIXED_0914 firmware contract:
        # STAT,adc,target_adc,target_deg,drive_signed_pwm,steer_active,armed,encA
        parts = text.split(",")
        if len(parts) != 8:
            return
        try:
            adc = int(parts[1])
            int(parts[2])  # target_adc: syntax validation
            int(parts[3])  # target_deg: syntax validation
            signed_pwm = int(parts[4])
            steer_active = bool(int(parts[5]))
            firmware_armed = bool(int(parts[6]))
            encA = int(parts[7])
        except ValueError:
            return

        now = time.monotonic()
        steering = steering_from_adc(
            adc, self.center_adc, self.counts_per_deg, self.max_deg)
        self.odom_model.set_steering_deg(steering)
        distance, delta_yaw = self.odom_model.update_encoder(encA)
        elapsed = None if self.last_enc_t is None else now - self.last_enc_t
        if self.odom_model.last_discontinuity:
            self.get_logger().warning(
                "encoder counter discontinuity ignored; ODOM re-baselined")
            rate = 0.0
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
        elif elapsed is not None and elapsed > 1.0e-3:
            rate = 0.0 if self.last_enc is None else (encA - self.last_enc) / elapsed
            self.linear_velocity = distance / elapsed
            self.angular_velocity = delta_yaw / elapsed
        else:
            rate = 0.0

        self.last_enc = encA
        self.last_enc_t = now
        self.last_status_at = now
        self.firmware_drive_pwm = signed_pwm
        self.firmware_steer_active = steer_active
        self.firmware_armed = firmware_armed

        m = String(); m.data = text; self.pub_raw.publish(m)
        m = Int32(); m.data = adc; self.pub_a0.publish(m)
        m = Float32(); m.data = float(steering); self.pub_steer_deg.publish(m)
        m = Int32(); m.data = encA; self.pub_encoder.publish(m)
        m = Int32(); m.data = encA; self.pub_encA.publish(m)
        rpm = rate * 60.0 / self.encoder_cpr if self.encoder_cpr > 0.0 else 0.0
        m = Float32(); m.data = float(rate); self.pub_encoder_rate.publish(m)
        m = Float32(); m.data = float(rpm); self.pub_rpm.publish(m)
        m = Float32(); m.data = float(self.linear_velocity); self.pub_speed_mps.publish(m)
        m = Float32(); m.data = float(self.odom_model.state.distance_m); self.pub_distance_m.publish(m)
        self.pub_odom_distance_m.publish(m)
        m = Int32(); m.data = signed_pwm; self.pub_firmware_drive.publish(m)
        m = Bool(); m.data = firmware_armed; self.pub_firmware_armed.publish(m)

    def publish_odom(self):
        now = time.monotonic()
        fresh = (self.ready and self.last_status_at is not None and
                 now - self.last_status_at <= self.odom_timeout)
        self.pub_odom_state.publish(String(data=(
            "TRACKING_REAL_ENCODER_STEERING" if fresh else
            "WAITING_FOR_REAL_ENCODER_STEERING")))
        if not fresh:
            return

        x, y, yaw = self.odom_model.base_pose()
        stamp = self.get_clock().now().to_msg()
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(yaw / 2.0)
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
        transform.transform.rotation.z = message.pose.pose.orientation.z
        transform.transform.rotation.w = message.pose.pose.orientation.w
        self.transform.sendTransform(transform)

    def publish_applied(self, stage: int):
        self.applied_stage = int(stage)
        self.applied_pwm = self.stage_to_pwm(stage)
        m = Float32(); m.data = float(stage); self.pub_applied_drive.publish(m)
        self.publish_bool(self.pub_stop, self.stop_active)
        now = time.monotonic()
        command_fresh = (self.last_drive_rx is not None and
                         now - self.last_drive_rx <= self.drive_timeout_s)
        values = {
            "serial_connected": self.ser is not None,
            "bridge_ready": self.ready,
            "serial_port": self.port,
            "command_fresh": command_fresh,
            "command_age_s": (None if self.last_drive_rx is None else
                              max(0.0, now - self.last_drive_rx)),
            "commanded_stage": self.cmd_drive,
            "applied_stage": self.applied_stage,
            "applied_pwm": self.applied_pwm,
            "stop_active": self.stop_active,
            "firmware_armed": self.firmware_armed,
            "firmware_drive_pwm": self.firmware_drive_pwm,
            "firmware_steer_active": self.firmware_steer_active,
            "status_age_s": (None if self.last_status_at is None else
                             max(0.0, now - self.last_status_at)),
            "encoder": self.last_enc,
        }
        self.pub_command_diagnostics.publish(String(data=json.dumps(
            values, separators=(",", ":"), allow_nan=False)))
        signature = (
            self.applied_stage, self.applied_pwm, command_fresh,
            self.stop_active, self.firmware_armed,
            self.firmware_drive_pwm)
        if signature != self.last_command_log:
            self.get_logger().info(
                "[MCU COMMAND] " + json.dumps(
                    values, separators=(",", ":"), allow_nan=False))
            self.last_command_log = signature

    @staticmethod
    def publish_bool(pub, value: bool):
        m = Bool(); m.data = bool(value); pub.publish(m)

    def shutdown(self):
        if self.ser is not None:
            try:
                self.send("X")
                self.send("DISARM")
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


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
