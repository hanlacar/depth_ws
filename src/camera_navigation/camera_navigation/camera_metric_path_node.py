#!/usr/bin/env python3
"""D456 ground-plane calibration + pixel->metric path projection node.

Consumes:
  - /camera/camera/accel/sample, /camera/camera/gyro/sample (sensor_msgs/Imu)
    ONLY to run a one-time boot-time stationary attitude lock. Never touches
    the RealSense device itself, and never republishes vehicle IMU topics
    (that remains imu_manager's job).
  - /camera/camera_info (sensor_msgs/CameraInfo) for real fx/fy/cx/cy.
  - /camera/image_path_typed (race_interfaces/ImagePath), the existing
    pixel-space path. Never modified/replaced by this node.

Publishes:
  - /camera/path (nav_msgs/Path) in base_link, ONLY while calibration is
    CALIBRATION_VALID. Otherwise no message is published on this topic for
    that frame (fail-safe: no fabricated metric path).
  - /camera/path_valid (std_msgs/Bool), /camera/path_confidence (Float32)
  - /camera/metric_path_status (std_msgs/String, JSON), an exact-stamp
    validity/calibration contract for the path controller.
  - /camera/calibration_valid (std_msgs/Bool), the current calibration gate.
  - /camera/calibration_diagnostics (std_msgs/String, JSON) following the
    existing project convention used by /camera/path_metrics.

This node performs the boot-time IMU calibration lock exactly once (or after
being restarted) and caches the resulting effective rotation matrix; the
per-frame projection path only does a matrix multiply per point, kept off the
hot inference/mask path (section 23).
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Imu
from std_msgs.msg import Bool, Float32, String

from race_interfaces.msg import ImagePath

from .ground_plane_calibration import (
    CALIBRATION_VALID,
    Intrinsics,
    InitConfig,
    NOT_CONFIGURED,
    OPTICAL_TO_MECHANICAL,
    StationaryAttitudeEstimator,
    ValidityConfig,
    compose_effective_orientation,
    evaluate_calibration_state,
    load_camera_mount_config,
    project_pixel_path_to_metric,
)
from .metric_path_quality import (
    MetricPathQualityConfig, analyze_metric_path, path_heading)


class CameraMetricPathNode(Node):
    def __init__(self):
        super().__init__("camera_metric_path_node")
        self.declare_parameter("camera_mount.configured", False)
        self.declare_parameter("camera_mount.position_x_m", 0.0)
        self.declare_parameter("camera_mount.position_y_m", 0.0)
        self.declare_parameter("camera_mount.height_z_m", 0.0)
        self.declare_parameter("camera_mount.reference_roll_deg", 0.0)
        self.declare_parameter("camera_mount.reference_pitch_deg", 0.0)
        self.declare_parameter("camera_mount.reference_yaw_deg", 0.0)
        defaults = {
            "init_window_sec": 1.0, "init_min_samples": 15, "init_max_samples": 400,
            "init_max_accel_stddev_mps2": 0.25, "init_gravity_norm_min_mps2": 8.5,
            "init_gravity_norm_max_mps2": 11.0, "init_max_gyro_norm_rad_s": 0.08,
            "init_low_pass_alpha": 0.2, "init_outlier_mad_k": 3.5,
            "max_runtime_pitch_correction_deg": 5.0, "max_runtime_roll_correction_deg": 5.0,
            "max_calibration_age_sec": 3600.0,
            "accel_topic": "/camera/camera/accel/sample",
            "gyro_topic": "/camera/camera/gyro/sample",
            "camera_info_topic": "/camera/camera_info",
            "pixel_path_topic": "/camera/image_path_typed",
            "metric_path_frame_id": "base_link",
            "metric_path_max_range_m": 30.0,
            "metric_minimum_points": 3,
            "metric_minimum_spacing_m": 0.05,
            "metric_maximum_point_jump_m": 4.0,
            "metric_maximum_reverse_step_m": 0.25,
            "metric_minimum_path_length_m": 1.0,
            "imu_stale_timeout_sec": 2.0,
            # BEV standalone: when True, project the pixel path whenever it is
            # not INVALID (i.e. VALID or DEGRADED) instead of requiring a full
            # VALID flag. Pairs with the planner's steering_only_validity so a
            # steering-feasible-but-imperfect corridor still drives in metric
            # space. Metric-side geometry (analyze_metric_path) still gates.
            "accept_degraded_pixel_path": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.mount = load_camera_mount_config({
            "configured": self.get_parameter("camera_mount.configured").value,
            "position_x_m": self.get_parameter("camera_mount.position_x_m").value,
            "position_y_m": self.get_parameter("camera_mount.position_y_m").value,
            "height_z_m": self.get_parameter("camera_mount.height_z_m").value,
            "reference_roll_deg": self.get_parameter("camera_mount.reference_roll_deg").value,
            "reference_pitch_deg": self.get_parameter("camera_mount.reference_pitch_deg").value,
            "reference_yaw_deg": self.get_parameter("camera_mount.reference_yaw_deg").value,
        })
        self.init_cfg = InitConfig(
            window_sec=float(self.get_parameter("init_window_sec").value),
            min_samples=int(self.get_parameter("init_min_samples").value),
            max_samples=int(self.get_parameter("init_max_samples").value),
            max_accel_stddev_mps2=float(self.get_parameter("init_max_accel_stddev_mps2").value),
            gravity_norm_min_mps2=float(self.get_parameter("init_gravity_norm_min_mps2").value),
            gravity_norm_max_mps2=float(self.get_parameter("init_gravity_norm_max_mps2").value),
            max_gyro_norm_rad_s=float(self.get_parameter("init_max_gyro_norm_rad_s").value),
            low_pass_alpha=float(self.get_parameter("init_low_pass_alpha").value),
            outlier_mad_k=float(self.get_parameter("init_outlier_mad_k").value),
        )
        self.validity_cfg = ValidityConfig(
            max_runtime_pitch_correction_deg=float(self.get_parameter("max_runtime_pitch_correction_deg").value),
            max_runtime_roll_correction_deg=float(self.get_parameter("max_runtime_roll_correction_deg").value),
            max_calibration_age_sec=float(self.get_parameter("max_calibration_age_sec").value),
        )
        self.imu_stale_timeout = float(self.get_parameter("imu_stale_timeout_sec").value)
        self.max_range_m = float(self.get_parameter("metric_path_max_range_m").value)
        self.frame_id = str(self.get_parameter("metric_path_frame_id").value)
        self.accept_degraded_pixel_path = bool(
            self.get_parameter("accept_degraded_pixel_path").value)
        self.path_quality_config = MetricPathQualityConfig(
            minimum_points=int(self.get_parameter("metric_minimum_points").value),
            minimum_spacing_m=float(
                self.get_parameter("metric_minimum_spacing_m").value),
            maximum_point_jump_m=float(
                self.get_parameter("metric_maximum_point_jump_m").value),
            maximum_reverse_step_m=float(
                self.get_parameter("metric_maximum_reverse_step_m").value),
            minimum_path_length_m=float(
                self.get_parameter("metric_minimum_path_length_m").value),
        )
        self.path_quality_config.validate()

        self.estimator = StationaryAttitudeEstimator(self.init_cfg)
        self.init_result = None
        self.last_gyro_stamp = None
        self.last_accel_stamp = None
        self.last_gyro_wall = None
        self.last_accel_wall = None
        self.pending_gyro = None
        self.calibration_locked_wall = None

        self.camera_info = None
        self.intrinsics = None

        self.state = NOT_CONFIGURED
        self.state_reasons = []
        self.effective_roll_deg = float("nan")
        self.effective_pitch_deg = float("nan")
        self.effective_rotation = None
        self.effective_position = None

        self.create_subscription(Imu, str(self.get_parameter("accel_topic").value),
                                 self._on_accel, qos_profile_sensor_data)
        self.create_subscription(Imu, str(self.get_parameter("gyro_topic").value),
                                 self._on_gyro, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.get_parameter("camera_info_topic").value),
                                 self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(ImagePath, str(self.get_parameter("pixel_path_topic").value),
                                 self._on_pixel_path, 10)

        self.path_pub = self.create_publisher(Path, "/camera/path", 10)
        self.valid_pub = self.create_publisher(Bool, "/camera/path_valid", 10)
        self.confidence_pub = self.create_publisher(Float32, "/camera/path_confidence", 10)
        self.metric_valid_pub = self.create_publisher(Bool, "/camera/metric_path_valid", 10)
        self.metric_confidence_pub = self.create_publisher(
            Float32, "/camera/metric_path_confidence", 10)
        self.metric_status_pub = self.create_publisher(
            String, "/camera/metric_path_status", 10)
        self.calibration_valid_pub = self.create_publisher(
            Bool, "/camera/calibration_valid", 10)
        self.diag_pub = self.create_publisher(String, "/camera/calibration_diagnostics", 10)

        self.create_timer(0.2, self._update_state_and_diagnostics)
        self.get_logger().info(
            f"camera_metric_path_node starting; base mount configured={self.mount.configured}")

    # -- IMU intake (boot-window only; not a hot per-frame path) -----------
    @staticmethod
    def _stamp(msg):
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    @staticmethod
    def _vector(field):
        return np.array([field.x, field.y, field.z], dtype=float)

    def _on_gyro(self, msg):
        self.last_gyro_stamp = self._stamp(msg)
        self.last_gyro_wall = time.monotonic()
        self.pending_gyro = self._vector(msg.angular_velocity)

    def _on_accel(self, msg):
        self.last_accel_stamp = self._stamp(msg)
        self.last_accel_wall = time.monotonic()
        if self.pending_gyro is None or self.estimator.locked:
            return
        # RealSense publishes D456 motion samples in the optical sensor axes.
        # The attitude estimator is explicitly defined in camera mechanical
        # axes (x=forward, y=left, z=up), so apply the same mapping used by
        # imu_manager before interpreting gravity as physical roll/pitch.
        accel_mechanical = OPTICAL_TO_MECHANICAL @ self._vector(msg.linear_acceleration)
        gyro_mechanical = OPTICAL_TO_MECHANICAL @ self.pending_gyro
        self.estimator.add_sample(accel_mechanical, gyro_mechanical, self._stamp(msg))

    def _on_camera_info(self, msg):
        self.camera_info = msg
        k = msg.k
        if len(k) == 9 and k[0] > 0 and k[4] > 0:
            self.intrinsics = Intrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def _imu_available(self):
        now = time.monotonic()
        return (self.last_accel_wall is not None and self.last_gyro_wall is not None
                and now - self.last_accel_wall <= self.imu_stale_timeout
                and now - self.last_gyro_wall <= self.imu_stale_timeout)

    def _update_state_and_diagnostics(self):
        imu_available = self._imu_available()
        if self.mount.is_usable() and imu_available and not self.estimator.locked:
            if self.estimator.ready():
                self.init_result = self.estimator.finalize()
                self.calibration_locked_wall = time.monotonic()
                self._recompute_effective_orientation()

        age = (time.monotonic() - self.calibration_locked_wall
               if self.calibration_locked_wall is not None else None)
        self.state, self.state_reasons = evaluate_calibration_state(
            mount=self.mount, imu_available=imu_available, init_result=self.init_result,
            validity=self.validity_cfg, calibration_age_sec=age)

        self._publish_diagnostics(imu_available, age)

    def _recompute_effective_orientation(self):
        if self.init_result is None or not math.isfinite(self.init_result["measured_pitch_deg"]):
            return
        delta_pitch = self.init_result["measured_pitch_deg"] - self.mount.reference_pitch_deg
        delta_roll = self.init_result["measured_roll_deg"] - self.mount.reference_roll_deg
        roll, pitch, yaw, matrix, _quat = compose_effective_orientation(
            (self.mount.reference_roll_deg, self.mount.reference_pitch_deg, self.mount.reference_yaw_deg),
            delta_pitch, delta_roll)
        self.effective_roll_deg = roll
        self.effective_pitch_deg = pitch
        self.effective_rotation = matrix
        self.effective_position = np.array(
            [self.mount.position_x_m, self.mount.position_y_m, self.mount.height_z_m])

    def _on_pixel_path(self, msg: ImagePath):
        valid_msg = Bool(); confidence_msg = Float32()
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if self.state != CALIBRATION_VALID or self.effective_rotation is None or self.intrinsics is None:
            valid_msg.data = False
            confidence_msg.data = 0.0
            self._publish_metric_health(
                stamp_ns, valid_msg, confidence_msg, calibration_valid=False,
                analysis=None)
            return
        pixel_points = [(p.x_px, p.y_px) for p in msg.points]
        metric_points = project_pixel_path_to_metric(
            pixel_points, self.intrinsics, self.effective_rotation, self.effective_position,
            max_range_m=self.max_range_m)
        analysis = analyze_metric_path(metric_points, self.path_quality_config)
        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = self.frame_id
        for index, (x_m, y_m) in enumerate(analysis.points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x_m
            pose.pose.position.y = y_m
            yaw = path_heading(analysis.points, index)
            pose.pose.orientation.z = math.sin(0.5*yaw)
            pose.pose.orientation.w = math.cos(0.5*yaw)
            path.poses.append(pose)
        if self.accept_degraded_pixel_path:
            # Drive on any steering-feasible corridor: accept VALID or
            # DEGRADED (state != INVALID/INACTIVE) rather than the full
            # path_valid flag. Metric geometry (analysis.valid) still gates.
            pixel_drivable = msg.path_state not in (
                ImagePath.STATE_INVALID, ImagePath.STATE_INACTIVE)
        else:
            pixel_drivable = bool(msg.path_valid)
        ok = pixel_drivable and analysis.valid
        valid_msg.data = ok
        confidence_msg.data = float(msg.path_confidence) if ok else 0.0
        # Publish exact-stamp health first. The controller still requires an
        # exact status/path stamp match, so DDS delivery order cannot make an
        # old validity value authorize a new Path.
        self._publish_metric_health(
            stamp_ns, valid_msg, confidence_msg, calibration_valid=True,
            analysis=analysis)
        if ok:
            self.path_pub.publish(path)

    def _publish_metric_health(self, stamp_ns, valid_msg, confidence_msg,
                               calibration_valid, analysis):
        self.valid_pub.publish(valid_msg)
        self.confidence_pub.publish(confidence_msg)
        self.metric_valid_pub.publish(valid_msg)
        self.metric_confidence_pub.publish(confidence_msg)
        status = {
            "stamp_ns": int(stamp_ns),
            "path_valid": bool(valid_msg.data),
            "confidence": float(confidence_msg.data),
            "calibration_valid": bool(calibration_valid),
            "calibration_state": self.state,
            "metric_path_points": analysis.point_count if analysis else 0,
            "metric_minimum_x_m": analysis.minimum_x_m if analysis else None,
            "metric_maximum_x_m": analysis.maximum_x_m if analysis else None,
            "metric_path_length_m": analysis.path_length_m if analysis else 0.0,
            "metric_forward_length_m": (
                analysis.forward_usable_length_m if analysis else 0.0),
            "metric_curvature_per_m": (
                analysis.maximum_curvature_per_m if analysis else 0.0),
            "metric_quality_reason": analysis.reason if analysis else "unavailable",
            "metric_duplicates_removed": (
                analysis.duplicates_removed if analysis else 0),
            "metric_jump_truncated": (
                analysis.jump_truncated if analysis else False),
        }
        self.metric_status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))

    def _publish_diagnostics(self, imu_available, age):
        init = self.init_result or {}
        diag = {
            "state": self.state,
            "reasons": self.state_reasons,
            "imu_available": bool(imu_available),
            "sample_count": int(init.get("sample_count", self.estimator.sample_count)),
            "reference_pitch_deg": self.mount.reference_pitch_deg,
            "reference_roll_deg": self.mount.reference_roll_deg,
            "measured_pitch_deg": init.get("measured_pitch_deg"),
            "measured_roll_deg": init.get("measured_roll_deg"),
            "delta_pitch_deg": (
                init.get("measured_pitch_deg") - self.mount.reference_pitch_deg
                if init.get("measured_pitch_deg") is not None
                and math.isfinite(init.get("measured_pitch_deg", float("nan"))) else None),
            "delta_roll_deg": (
                init.get("measured_roll_deg") - self.mount.reference_roll_deg
                if init.get("measured_roll_deg") is not None
                and math.isfinite(init.get("measured_roll_deg", float("nan"))) else None),
            "effective_pitch_deg": (
                self.effective_pitch_deg if math.isfinite(self.effective_pitch_deg) else None),
            "effective_roll_deg": (
                self.effective_roll_deg if math.isfinite(self.effective_roll_deg) else None),
            "calibration_age_sec": age,
            "valid": self.state == CALIBRATION_VALID,
            "base_mount_configured": self.mount.configured,
        }
        message = String(); message.data = json.dumps(diag)
        self.diag_pub.publish(message)
        self.calibration_valid_pub.publish(
            Bool(data=self.state == CALIBRATION_VALID))


def main(args=None):
    rclpy.init(args=args)
    node = CameraMetricPathNode()
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
