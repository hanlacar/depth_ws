#!/usr/bin/env python3
"""Subscribe to D456 raw IMU topics and publish vehicle-frame estimates."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

from .imu_filter import (
    FilterConfig,
    ImuFilter,
    CALIBRATING,
    CALIBRATION_FAILED,
    READY,
    SlopeStateConfig,
    SlopeStateDetector,
    StartupCalibrationConfig,
    StartupCalibrator,
    calibration_gate_ready,
    angle_message_fields,
    apply_sensor_axis_transform,
    axis_matrix_from_row_major,
    quaternion_from_rpy,
    transform_sensor_to_base,
)


KNOWN_MODES = {"IDLE", "NORMAL", "PARALLEL_PARK", "T_PARK", "SLOPE"}


def manager_transform_vector(raw_vector, axis_matrix, mounting_rpy_deg):
    """Manager adapter to the single shared column-vector transform."""
    return transform_sensor_to_base(raw_vector, axis_matrix, mounting_rpy_deg)


def make_angle_message(stamp_sec, frame_id, roll_deg, pitch_deg, yaw_deg):
    """Build unrounded base_link RPY stamped with the fusion sensor time."""
    msg = Vector3Stamped()
    seconds, nanoseconds, frame, vector = angle_message_fields(
        stamp_sec, frame_id, roll_deg, pitch_deg, yaw_deg)
    msg.header.stamp.sec = seconds
    msg.header.stamp.nanosec = nanoseconds
    msg.header.frame_id = frame
    msg.vector.x, msg.vector.y, msg.vector.z = vector
    return msg


class ImuManagerNode(Node):
    def __init__(self):
        super().__init__("imu_manager_node")
        defaults = {
            "gyro_topic": "/camera/camera/gyro/sample",
            "accel_topic": "/camera/camera/accel/sample",
            "mode_topic": "/vehicle_mode",
            "output_frame": "base_link",
            "publish_rate_hz": 50.0,
            "stale_timeout_sec": 0.2,
            "complementary_alpha": 0.98,
            "max_dt_sec": 0.1,
            "stationary_gyro_threshold": 0.03,
            "stationary_accel_tolerance": 0.35,
            "bias_sample_count": 200,
            "mounting_roll_deg": 0.0,
            "mounting_pitch_deg": 0.0,
            "mounting_yaw_deg": 0.0,
            "sensor_axis_matrix": [],
            "orientation_roll_pitch_variance": 0.02,
            "orientation_yaw_variance": 1.0,
            "calibration_validated": False,
            "startup_calibration_enabled": True,
            "startup_calibration_duration_sec": 3.0,
            "startup_calibration_timeout_sec": 15.0,
            "startup_gyro_still_threshold_rad_s": 0.05,
            "startup_accel_stddev_limit_mps2": 0.10,
            "startup_accel_norm_min_mps2": 8.5,
            "startup_accel_norm_max_mps2": 11.0,
            "assume_vehicle_level_at_start": True,
            "uphill_enter_pitch_deg": 3.0,
            "uphill_exit_pitch_deg": 1.5,
            "uphill_confirmation_frames": 10,
        }
        for name, value in defaults.items():
            if name == "sensor_axis_matrix":
                self.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY)
            else:
                self.declare_parameter(name, value)
        axis_values = list(self.get_parameter("sensor_axis_matrix").value)
        axis = axis_matrix_from_row_major(axis_values) if len(axis_values) == 9 else None
        cfg = FilterConfig(**{
            key: self.get_parameter(key).value for key in (
                "complementary_alpha", "max_dt_sec", "stationary_gyro_threshold",
                "stationary_accel_tolerance", "bias_sample_count",
            )
        })
        self.filter = ImuFilter(cfg, axis, (
            self.get_parameter("mounting_roll_deg").value,
            self.get_parameter("mounting_pitch_deg").value,
            self.get_parameter("mounting_yaw_deg").value,
        ))
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.stale_timeout = float(self.get_parameter("stale_timeout_sec").value)
        self.orientation_rp_variance = float(self.get_parameter("orientation_roll_pitch_variance").value)
        self.orientation_yaw_variance = float(self.get_parameter("orientation_yaw_variance").value)
        self.calibration_validated = bool(self.get_parameter("calibration_validated").value)
        self.startup_calibration_enabled = bool(self.get_parameter("startup_calibration_enabled").value)
        self.mounting_yaw_deg = float(self.get_parameter("mounting_yaw_deg").value)
        self.slope_detector = SlopeStateDetector(SlopeStateConfig(
            uphill_enter_pitch_deg=float(
                self.get_parameter("uphill_enter_pitch_deg").value),
            uphill_exit_pitch_deg=float(
                self.get_parameter("uphill_exit_pitch_deg").value),
            uphill_confirmation_frames=int(
                self.get_parameter("uphill_confirmation_frames").value),
        ))
        self.calibrator = None
        self.startup_state = READY
        if self.startup_calibration_enabled:
            self.startup_state = CALIBRATING
            self.calibrator = StartupCalibrator(StartupCalibrationConfig(
                duration_sec=float(self.get_parameter("startup_calibration_duration_sec").value),
                timeout_sec=float(self.get_parameter("startup_calibration_timeout_sec").value),
                gyro_still_threshold_rad_s=float(self.get_parameter("startup_gyro_still_threshold_rad_s").value),
                accel_stddev_limit_mps2=float(self.get_parameter("startup_accel_stddev_limit_mps2").value),
                accel_norm_min_mps2=float(self.get_parameter("startup_accel_norm_min_mps2").value),
                accel_norm_max_mps2=float(self.get_parameter("startup_accel_norm_max_mps2").value),
            ), started_monotonic=time.monotonic())
        self.mode = "NORMAL"
        self.receipt_gyro = None
        self.receipt_accel = None
        self.safe_invalid = not self.filter.transform_valid
        self._calibration_failure_logged = False
        self.create_subscription(Imu, str(self.get_parameter("gyro_topic").value), self.gyro_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, str(self.get_parameter("accel_topic").value), self.accel_callback, qos_profile_sensor_data)
        self.create_subscription(String, str(self.get_parameter("mode_topic").value), self.mode_callback, 10)
        self.data_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.rpy_pub = self.create_publisher(Vector3Stamped, "/imu/rpy_deg", 10)
        self.angle_pub = self.create_publisher(Vector3Stamped, "/imu/angle", 10)
        self.yaw_pub = self.create_publisher(Float32, "/imu/relative_yaw_deg", 10)
        self.pitch_pub = self.create_publisher(Float32, "/imu/pitch_deg", 10)
        self.pitch_compat_pub = self.create_publisher(Float32, "/imu/pitch", 10)
        self.roll_pub = self.create_publisher(Float32, "/imu/roll_deg", 10)
        self.roll_compat_pub = self.create_publisher(Float32, "/imu/roll", 10)
        self.yaw_compat_pub = self.create_publisher(Float32, "/imu/yaw", 10)
        self.yaw_rate_compat_pub = self.create_publisher(Float32, "/imu/yaw_rate", 10)
        self.yaw_rate_pub = self.create_publisher(Float32, "/imu/angular_velocity_z", 10)
        self.valid_pub = self.create_publisher(Bool, "/imu/valid", 10)
        self.valid_compat_pub = self.create_publisher(Bool, "/imu_valid", 10)
        self.stationary_pub = self.create_publisher(Bool, "/imu/stationary", 10)
        self.slope_pub = self.create_publisher(Bool, "/imu/slope", 10)
        self.slope_compat_pub = self.create_publisher(Bool, "/slope_state", 10)
        self.pitch_vehicle_pub = self.create_publisher(Float32, "/imu_pitch", 10)
        self.roll_vehicle_pub = self.create_publisher(Float32, "/imu_roll", 10)
        self.yaw_vehicle_pub = self.create_publisher(Float32, "/imu_yaw", 10)
        self.yaw_rate_vehicle_pub = self.create_publisher(Float32, "/imu_yaw_rate", 10)
        self.create_service(Trigger, "/imu/reset_reference", self.reset_reference_callback)
        self.create_service(Trigger, "/imu/reset", self.reset_reference_callback)
        self.create_service(Trigger, "/imu_reset", self.reset_reference_callback)
        self.create_timer(1.0 / float(self.get_parameter("publish_rate_hz").value), self.publish_outputs)
        if self.safe_invalid:
            self.get_logger().error("sensor_axis_matrix must contain nine finite values; outputs remain invalid")
        if self.startup_state == CALIBRATING:
            self.get_logger().warning(
                "Keep vehicle stationary on level ground during startup calibration.")

    @staticmethod
    def _stamp(msg):
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _vector(field):
        return np.array([field.x, field.y, field.z], dtype=float)

    def gyro_callback(self, msg):
        if self.startup_state == CALIBRATING:
            self._calibration_sample(msg, "gyro")
            return
        if self.startup_state != READY or not self.validate_timestamp(msg, "gyro"):
            return
        if self.filter.update_gyro(self._vector(msg.angular_velocity), self._stamp(msg)):
            self.receipt_gyro = time.monotonic()
            if self.filter.last_attitude_stamp is not None:
                self.slope_detector.update(
                    self.filter.pitch_deg, self._valid())

    def accel_callback(self, msg):
        if self.startup_state == CALIBRATING:
            self._calibration_sample(msg, "accel")
            return
        if self.startup_state != READY or not self.validate_timestamp(msg, "accel"):
            return
        if self.filter.update_accel(self._vector(msg.linear_acceleration), self._stamp(msg)):
            self.receipt_accel = time.monotonic()

    def _calibration_sample(self, msg, stream):
        raw = self._vector(msg.angular_velocity if stream == "gyro" else msg.linear_acceleration)
        stamp = self._stamp(msg)
        try:
            axis_vector = apply_sensor_axis_transform(self.filter.axis_matrix, raw)
        except ValueError:
            return
        now = time.monotonic()
        accepted = (self.calibrator.add_gyro(axis_vector, stamp, now) if stream == "gyro"
                    else self.calibrator.add_accel(axis_vector, stamp, now))
        if accepted:
            if stream == "gyro":
                self.receipt_gyro = now
            else:
                self.receipt_accel = now
        result = self.calibrator.try_complete(now, self.mounting_yaw_deg)
        if result is not None:
            mounting = (result.mounting_roll_deg, result.mounting_pitch_deg,
                        result.mounting_yaw_deg)
            if not self.filter.apply_startup_calibration(mounting, result.gyro_bias_axis):
                self.startup_state = CALIBRATION_FAILED
                return
            self.startup_state = READY
            self.filter.reset_reference()
            self.slope_detector.invalidate()
            self.get_logger().info(
                "Startup calibration READY: gravity_axis_mean=%s norm=%.6f "
                "roll_correction=%.6f pitch_correction=%.6f gyro_bias_axis=%s "
                "duration=%.3f restarts=%d" % (
                    result.gravity_axis_mean.tolist(), result.gravity_norm,
                    result.mounting_roll_deg, result.mounting_pitch_deg,
                    result.gyro_bias_axis.tolist(), result.duration_sec,
                    result.restart_count))

    def mode_callback(self, msg):
        requested = str(msg.data).strip().upper()
        new_mode = requested if requested in KNOWN_MODES else "NORMAL"
        if requested not in KNOWN_MODES:
            self.get_logger().warning(f"Undefined vehicle mode {requested!r}; using NORMAL")
        self.handle_mode_transition(self.mode, new_mode)
        self.mode = new_mode

    def validate_timestamp(self, msg, stream):
        vector = msg.angular_velocity if stream == "gyro" else msg.linear_acceleration
        if not self.filter.finite_vector(self._vector(vector)):
            self.enter_safe_invalid_state(f"non-finite {stream} sample")
            return False
        previous = self.filter.last_gyro_stamp if stream == "gyro" else self.filter.last_accel_stamp
        stamp = self._stamp(msg)
        return math.isfinite(stamp) and stamp > 0.0 and (previous is None or stamp > previous)

    def estimate_stationary_bias(self):
        return self.filter.estimate_stationary_bias()

    def transform_sensor_to_vehicle_frame(self, vector):
        return self.filter.transform_sensor_to_vehicle_frame(vector)

    def update_complementary_filter(self, gyro, dt):
        self.filter.update_complementary_filter(gyro, dt)

    def integrate_relative_yaw(self, yaw_rate, dt):
        self.filter.integrate_relative_yaw(yaw_rate, dt)

    def detect_stationary_state(self):
        return self.filter.detect_stationary_state()

    def handle_mode_transition(self, old_mode, new_mode):
        self.filter.handle_mode_transition(old_mode, new_mode)

    def reset_reference_callback(self, request, response):
        del request
        self.filter.reset_reference()
        response.success = True
        response.message = "Relative yaw reference reset"
        return response

    def enter_safe_invalid_state(self, reason):
        self.safe_invalid = True
        self.get_logger().error(reason)

    def _valid(self):
        now = time.monotonic()
        fresh = (
            self.receipt_gyro is not None and self.receipt_accel is not None
            and now - self.receipt_gyro <= self.stale_timeout
            and now - self.receipt_accel <= self.stale_timeout
        )
        calibration_ready = calibration_gate_ready(
            self.startup_calibration_enabled, self.startup_state,
            self.calibration_validated)
        return (calibration_ready and fresh and self.filter.bias_ready
                and self.filter.transform_valid and not self.safe_invalid)

    def publish_outputs(self):
        if self.startup_state == CALIBRATING:
            self.startup_state = self.calibrator.check_timeout(time.monotonic())
        if self.startup_state == CALIBRATION_FAILED and not self._calibration_failure_logged:
            self._calibration_failure_logged = True
            self.get_logger().error(
                "Startup calibration failed; IMU remains invalid: "
                f"{self.calibrator.failure_reason or 'unspecified calibration failure'}. "
                "Keep vehicle stationary on level ground during startup calibration.")
        now = self.get_clock().now().to_msg()
        valid = self._valid()
        if not valid:
            self.slope_detector.invalidate()
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = now, self.output_frame
        q = quaternion_from_rpy(self.filter.roll_deg, self.filter.pitch_deg, self.filter.relative_yaw_deg)
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = q
        imu.orientation_covariance[0] = self.orientation_rp_variance
        imu.orientation_covariance[4] = self.orientation_rp_variance
        # Relative gyro yaw has no absolute-heading reference.
        imu.orientation_covariance[8] = self.orientation_yaw_variance
        if self.filter.gyro is not None:
            corrected = self.filter.gyro - self.filter.gyro_bias
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = corrected.tolist()
        if self.filter.accel is not None:
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = self.filter.accel.tolist()
        self.data_pub.publish(imu)
        rpy = Vector3Stamped()
        rpy.header.stamp, rpy.header.frame_id = now, self.output_frame
        rpy.vector.x, rpy.vector.y, rpy.vector.z = self.filter.roll_deg, self.filter.pitch_deg, self.filter.relative_yaw_deg
        self.rpy_pub.publish(rpy)
        if self.filter.last_attitude_stamp is not None:
            self.angle_pub.publish(make_angle_message(
                self.filter.last_attitude_stamp, self.output_frame,
                self.filter.roll_deg, self.filter.pitch_deg,
                self.filter.relative_yaw_deg,
            ))
        for publisher, value in ((self.yaw_pub, self.filter.relative_yaw_deg), (self.pitch_pub, self.filter.pitch_deg),
                                 (self.roll_pub, self.filter.roll_deg),
                                 (self.yaw_rate_pub, 0.0 if self.filter.gyro is None else self.filter.gyro[2] - self.filter.gyro_bias[2])):
            msg = Float32(); msg.data = float(value); publisher.publish(msg)
        yaw_rate_deg_s = 0.0 if self.filter.gyro is None else math.degrees(self.filter.gyro[2] - self.filter.gyro_bias[2])
        for publisher, value in ((self.pitch_compat_pub, self.filter.pitch_deg),
                                 (self.roll_compat_pub, self.filter.roll_deg),
                                 (self.yaw_compat_pub, self.filter.relative_yaw_deg),
                                 (self.yaw_rate_compat_pub, yaw_rate_deg_s)):
            msg = Float32(); msg.data = float(round(value, 2)); publisher.publish(msg)
        for publisher, value in ((self.pitch_vehicle_pub, self.filter.pitch_deg),
                                 (self.roll_vehicle_pub, self.filter.roll_deg),
                                 (self.yaw_vehicle_pub, self.filter.relative_yaw_deg),
                                 (self.yaw_rate_vehicle_pub, yaw_rate_deg_s)):
            msg = Float32(); msg.data = float(round(value, 2)); publisher.publish(msg)
        msg = Bool(); msg.data = bool(valid); self.valid_pub.publish(msg); self.valid_compat_pub.publish(msg)
        msg = Bool(); msg.data = bool(self.filter.stationary); self.stationary_pub.publish(msg)
        slope = Bool(); slope.data = bool(valid and self.slope_detector.state)
        self.slope_pub.publish(slope); self.slope_compat_pub.publish(slope)


def main(args=None):
    rclpy.init(args=args)
    node = ImuManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
