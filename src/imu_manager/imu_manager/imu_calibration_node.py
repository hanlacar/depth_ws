#!/usr/bin/env python3
"""Observe existing D456 ROS IMU topics; never opens the camera device."""

from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
import yaml

from .imu_filter import (
    axis_matrix_from_row_major,
    compute_level_alignment,
    transform_sensor_to_base,
    validate_axis_matrix,
    validate_level_samples,
    validate_rotation_matrix,
)


def calibration_transform_vector(raw_vector, axis_matrix, mounting_rpy_deg):
    """Calibration adapter to the single shared column-vector transform."""
    return transform_sensor_to_base(raw_vector, axis_matrix, mounting_rpy_deg)


class StreamStats:
    def __init__(self, expected_hz):
        self.expected_hz = expected_hz
        self.stamps = []
        self.duplicate_or_reverse = 0
        self.gaps_100ms = 0

    def add(self, stamp):
        if self.stamps:
            dt = stamp - self.stamps[-1]
            if dt <= 0.0:
                self.duplicate_or_reverse += 1
                return False
            if dt >= 0.1:
                self.gaps_100ms += 1
        self.stamps.append(stamp)
        return True

    def result(self):
        intervals = np.diff(self.stamps)
        elapsed = self.stamps[-1] - self.stamps[0] if len(self.stamps) > 1 else 0.0
        hz = (len(self.stamps) - 1) / elapsed if elapsed > 0.0 else 0.0
        return {
            "count": len(self.stamps),
            "average_hz": float(hz),
            "min_dt_sec": float(np.min(intervals)) if len(intervals) else None,
            "max_dt_sec": float(np.max(intervals)) if len(intervals) else None,
            "duplicate_or_reverse_count": self.duplicate_or_reverse,
            "gaps_100ms_or_more": self.gaps_100ms,
            "estimated_drop_percent": float(max(0.0, 100.0 * (1.0 - hz / self.expected_hz))),
        }


class ImuCalibrationNode(Node):
    """Collect one LEVEL/PITCH/ROLL/YAW validation run from existing topics."""

    def __init__(self):
        super().__init__("imu_calibration_node")
        defaults = {
            "gyro_topic": "/camera/camera/gyro/sample",
            "accel_topic": "/camera/camera/accel/sample",
            "calibration_mode": "LEVEL",
            "duration_sec": 10,
            "result_file": str(
                Path.home() / ".config" / "imu_manager" / "imu_calibration_result.yaml"),
            "sensor_axis_matrix": [],
            "mounting_roll_deg": 0.0,
            "mounting_pitch_deg": 0.0,
            "mounting_yaw_deg": 0.0,
            "gravity_norm_min": 7.0,
            "gravity_norm_max": 12.0,
            "max_stationary_accel_stddev": 0.25,
            "motion_gyro_threshold": 0.08,
            "min_validation_rotation_rate": 0.10,
            "max_cross_axis_ratio": 0.50,
            "allow_inverted_mount": False,
            "plausible_mount_angle_limit_deg": 45.0,
            "level_xy_tolerance_mps2": 0.05,
            "transform_consistency_epsilon": 1.0e-9,
            "allow_axis_reflection": True,
        }
        for name, value in defaults.items():
            if name == "sensor_axis_matrix":
                self.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY)
            else:
                self.declare_parameter(name, value)
        values = list(self.get_parameter("sensor_axis_matrix").value)
        self.axis = axis_matrix_from_row_major(values) if len(values) == 9 else np.array([])
        self.mounting = np.array([
            self.get_parameter("mounting_roll_deg").value,
            self.get_parameter("mounting_pitch_deg").value,
            self.get_parameter("mounting_yaw_deg").value,
        ], dtype=float)
        self.mode = str(self.get_parameter("calibration_mode").value).strip().upper()
        self.duration = max(10.0, float(self.get_parameter("duration_sec").value))
        self.result_file = Path(str(self.get_parameter("result_file").value))
        self.accel_samples = []
        self.gyro_samples = []
        self.bias_gyro_samples = []
        self.gyro_stats = StreamStats(200.0)
        self.accel_stats = StreamStats(100.0)
        self.started = time.monotonic()
        self.resets_for_motion = 0
        self.restart_events = []
        self.max_gyro_norm_at_restart = 0.0
        self.stationary_window_started = None
        self.finished = False
        self.create_subscription(Imu, str(self.get_parameter("gyro_topic").value), self.gyro_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, str(self.get_parameter("accel_topic").value), self.accel_callback, qos_profile_sensor_data)
        self.create_timer(0.1, self.check_complete)
        self.get_logger().info(
            f"{self.mode} calibration: collect for {self.duration:.1f}s; result={self.result_file}"
        )

    @staticmethod
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    @staticmethod
    def vector(field):
        return np.array([field.x, field.y, field.z], dtype=float)

    def gyro_callback(self, msg):
        vector = self.vector(msg.angular_velocity)
        if not np.all(np.isfinite(vector)) or not self.gyro_stats.add(self.stamp(msg)):
            return
        gyro_norm = float(np.linalg.norm(vector))
        threshold = float(self.get_parameter("motion_gyro_threshold").value)
        if self.mode == "LEVEL" and gyro_norm > threshold:
            self.gyro_samples.append(vector)
            self.bias_gyro_samples.clear()
            self.stationary_window_started = None
            self.resets_for_motion += 1
            self.max_gyro_norm_at_restart = max(self.max_gyro_norm_at_restart, gyro_norm)
            self.restart_events.append({
                "sensor_timestamp": self.stamp(msg),
                "gyro_norm": gyro_norm,
                "threshold": threshold,
            })
            if self.resets_for_motion == 1:
                self.get_logger().warning(
                    f"Motion detected: gyro_norm={gyro_norm:.6f}, threshold={threshold:.6f}; "
                    "restarting bias window")
            return
        self.gyro_samples.append(vector)
        if self.mode == "LEVEL":
            if self.stationary_window_started is None:
                self.stationary_window_started = time.monotonic()
            self.bias_gyro_samples.append(vector)

    def accel_callback(self, msg):
        vector = self.vector(msg.linear_acceleration)
        if np.all(np.isfinite(vector)) and self.accel_stats.add(self.stamp(msg)):
            self.accel_samples.append(vector)

    @staticmethod
    def accel_rp(vector):
        x, y, z = vector
        return (
            math.degrees(math.atan2(y, z)),
            math.degrees(math.atan2(-x, math.hypot(y, z))),
        )

    def check_complete(self):
        if not self.finished and time.monotonic() - self.started >= self.duration:
            self.finished = True
            self.finish()

    def finish(self):
        errors = []
        if self.mode not in {"LEVEL", "PITCH", "ROLL", "YAW"}:
            errors.append(f"unsupported calibration_mode {self.mode}")
        matrix_validation = validate_rotation_matrix(
            self.axis,
            allow_reflection=bool(self.get_parameter("allow_axis_reflection").value),
        )
        if not matrix_validation.valid:
            errors.extend(matrix_validation.errors)
        if self.mounting.shape != (3,) or not np.all(np.isfinite(self.mounting)):
            errors.append("mounting RPY must be finite")
        if not self.accel_samples or not self.gyro_samples:
            errors.append("both accel and gyro samples are required")

        raw_mean = np.mean(self.accel_samples, axis=0) if self.accel_samples else np.full(3, np.nan)
        gravity_norm = float(np.linalg.norm(raw_mean))
        accel_std = np.std(self.accel_samples, axis=0) if self.accel_samples else np.full(3, np.nan)
        current_stages = None
        axis_mean = np.full(3, np.nan)
        mounting_rotation = np.full((3, 3), np.nan)
        consistency_error = np.full(3, np.inf)
        consistency_passed = False
        if matrix_validation.valid and np.all(np.isfinite(raw_mean)):
            current_stages = calibration_transform_vector(raw_mean, self.axis, self.mounting)
            axis_mean = current_stages.axis_transformed_vector
            mounting_rotation = current_stages.mounting_rotation_matrix
            # Deliberately recompute from serialized/configured matrix and raw
            # column vector to detect stale/mislabeled transform stages.
            manual_recomputed = self.axis @ raw_mean
            consistency_error = np.abs(manual_recomputed - axis_mean)
            epsilon = float(self.get_parameter("transform_consistency_epsilon").value)
            consistency_passed = bool(np.all(consistency_error <= epsilon))
            if not consistency_passed:
                errors.append("TRANSFORM_CONSISTENCY_ERROR")
        candidate_mounting = self.mounting.copy()
        level_alignment = None
        if self.mode == "LEVEL" and consistency_passed:
            errors.extend(validate_level_samples(
                self.accel_samples,
                self.gyro_samples,
                (float(self.get_parameter("gravity_norm_min").value),
                 float(self.get_parameter("gravity_norm_max").value)),
                float(self.get_parameter("max_stationary_accel_stddev").value),
                float(self.get_parameter("motion_gyro_threshold").value),
            ))
            try:
                level_alignment = compute_level_alignment(
                    axis_mean,
                    allow_inverted_mount=bool(self.get_parameter("allow_inverted_mount").value),
                    plausible_mount_angle_limit_deg=float(
                        self.get_parameter("plausible_mount_angle_limit_deg").value),
                    xy_tolerance_mps2=float(self.get_parameter("level_xy_tolerance_mps2").value),
                    gravity_range=(float(self.get_parameter("gravity_norm_min").value),
                                   float(self.get_parameter("gravity_norm_max").value)),
                )
                candidate_mounting = np.array([
                    level_alignment.correction_roll_deg,
                    level_alignment.correction_pitch_deg,
                    0.0,  # Computational yaw only; LEVEL does not observe mounting yaw.
                ])
                errors.extend(level_alignment.errors)
            except ValueError as error:
                errors.append(str(error))

        current_transformed = (
            current_stages.base_link_vector if current_stages else np.full(3, np.nan))
        candidate_stages = None
        if consistency_passed:
            candidate_stages = calibration_transform_vector(raw_mean, self.axis, candidate_mounting)
        candidate_transformed = (
            candidate_stages.base_link_vector if candidate_stages else np.full(3, np.nan))
        gyro_array = np.asarray(self.gyro_samples) if self.gyro_samples else np.empty((0, 3))
        transformed_gyro = (
            np.asarray([
                calibration_transform_vector(vector, self.axis, candidate_mounting).base_link_vector
                for vector in gyro_array
            ]) if len(gyro_array) and matrix_validation.valid else np.empty((0, 3))
        )
        bias_array = np.asarray(self.bias_gyro_samples) if self.bias_gyro_samples else np.empty((0, 3))
        transformed_bias = (
            np.asarray([
                calibration_transform_vector(vector, self.axis, candidate_mounting).base_link_vector
                for vector in bias_array
            ]) if len(bias_array) and matrix_validation.valid else np.empty((0, 3))
        )
        gyro_bias = np.mean(transformed_bias, axis=0) if len(transformed_bias) else np.full(3, np.nan)
        yaw_integral = float(np.trapz(transformed_gyro[:, 2], dx=1.0 / 200.0)) if len(transformed_gyro) else 0.0
        yaw_sign = 1 if yaw_integral > 0 else (-1 if yaw_integral < 0 else 0)

        if self.mode in {"PITCH", "ROLL", "YAW"} and len(transformed_gyro):
            primary_index = {"ROLL": 0, "PITCH": 1, "YAW": 2}[self.mode]
            primary_peak = float(np.max(np.abs(transformed_gyro[:, primary_index])))
            minimum = float(self.get_parameter("min_validation_rotation_rate").value)
            if primary_peak < minimum:
                errors.append(f"{self.mode} movement too small on expected gyro axis")
            if self.mode in {"PITCH", "ROLL"} and primary_peak > 0.0:
                cross_index = 0 if self.mode == "PITCH" else 1
                cross_peak = float(np.max(np.abs(transformed_gyro[:, cross_index])))
                if cross_peak / primary_peak > float(self.get_parameter("max_cross_axis_ratio").value):
                    errors.append(f"{self.mode} cross-axis coupling too large")

        validation = "PASS" if not errors else "FAIL"
        level_validation = validation if self.mode == "LEVEL" else "NOT_RUN"
        consecutive_stationary_sec = (
            time.monotonic() - self.stationary_window_started
            if self.stationary_window_started is not None else 0.0)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration_mode": self.mode,
            "raw_gravity_mean": raw_mean.tolist(),
            "raw_sensor_vector": raw_mean.tolist(),
            "gravity_norm": gravity_norm,
            "stationary_accel_stddev": accel_std.tolist(),
            "configured_sensor_axis_matrix": self.axis.tolist() if self.axis.shape == (3, 3) else [],
            "selected_sensor_axis_matrix": self.axis.tolist() if self.axis.shape == (3, 3) else [],
            "matrix_convention": "column_vector",
            "expected_axis_transform_expression": "matrix_times_vector",
            "axis_matrix_determinant": matrix_validation.determinant,
            "axis_matrix_reflection": matrix_validation.reflection,
            "axis_reflection_allowed": bool(self.get_parameter("allow_axis_reflection").value),
            "axis_matrix_orthogonality_error": matrix_validation.orthogonality_error,
            "axis_matrix_signed_permutation": matrix_validation.signed_permutation,
            "axis_transformed_gravity": axis_mean.tolist(),
            "mounting_rotation_matrix": (
                candidate_stages.mounting_rotation_matrix.tolist()
                if candidate_stages else mounting_rotation.tolist()),
            "transform_consistency_error": consistency_error.tolist(),
            "transform_consistency_passed": consistency_passed,
            "mounting_roll_deg": float(candidate_mounting[0]),
            "mounting_pitch_deg": float(candidate_mounting[1]),
            "mounting_yaw_deg": None if self.mode == "LEVEL" else float(candidate_mounting[2]),
            "yaw_observable_from_level": False if self.mode == "LEVEL" else None,
            "gravity_before_correction": (
                level_alignment.gravity_before.tolist() if level_alignment else axis_mean.tolist()),
            "gravity_after_correction": (
                level_alignment.gravity_after.tolist() if level_alignment else candidate_transformed.tolist()),
            "measured_level_roll_deg": (
                level_alignment.measured_roll_deg if level_alignment else None),
            "measured_level_pitch_deg": (
                level_alignment.measured_pitch_deg if level_alignment else None),
            "mounting_correction_roll_deg": (
                level_alignment.correction_roll_deg if level_alignment else None),
            "mounting_correction_pitch_deg": (
                level_alignment.correction_pitch_deg if level_alignment else None),
            "correction_rotation_magnitude_deg": (
                level_alignment.correction_rotation_magnitude_deg if level_alignment else None),
            "selected_euler_branch": (
                level_alignment.selected_euler_branch if level_alignment else None),
            "allow_inverted_mount": bool(self.get_parameter("allow_inverted_mount").value),
            "physical_sanity_passed": bool(level_alignment and level_alignment.physical_sanity_passed),
            "level_validation_result": level_validation,
            "current_transformed_accel_mean": current_transformed.tolist(),
            "candidate_transformed_accel_mean": candidate_transformed.tolist(),
            "roll_pitch_before_deg": list(self.accel_rp(current_transformed)),
            "roll_pitch_after_deg": list(self.accel_rp(candidate_transformed)),
            "gyro_bias": gyro_bias.tolist(),
            "yaw_sign": yaw_sign,
            "gyro": self.gyro_stats.result(),
            "accel": self.accel_stats.result(),
            "motion_restart_count": self.resets_for_motion,
            "motion_restart_threshold_rad_s": float(
                self.get_parameter("motion_gyro_threshold").value),
            "motion_restart_max_gyro_norm_rad_s": self.max_gyro_norm_at_restart,
            "motion_restart_events": self.restart_events,
            "bias_consecutive_stationary_sec": consecutive_stationary_sec,
            "validation_result": validation,
            "validation_errors": errors,
        }
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        with self.result_file.open("w", encoding="utf-8") as output:
            yaml.safe_dump(result, output, sort_keys=False)
        self.get_logger().info(yaml.safe_dump(result, sort_keys=False))
        self.get_logger().info("Review the result manually; imu_manager.yaml was not modified")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImuCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
