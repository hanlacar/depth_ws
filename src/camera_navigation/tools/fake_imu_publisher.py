#!/usr/bin/env python3
"""Standalone stationary-IMU stand-in for direct_bev_planner_node's boot-time
attitude lock, for use when no physical D456 is attached (mp4 test-video
validation).

Why this exists: direct_bev_planner_node.ground_plane_calibration hard-blocks
on IMU. evaluate_calibration_state() returns CALIBRATION_INVALID with reason
"imu_unavailable" whenever the planner hasn't received *both* an accel and a
gyro sample within imu_stale_timeout_sec (2.0s default) -- there is no
"IMU optional" path. So this publisher must keep running for the whole test,
not just once at startup.

Publishes constant, noise-free samples on the same topics/type the planner
subscribes to (accel_topic/gyro_topic defaults in bev_path.yaml,
sensor_msgs/Imu, qos_profile_sensor_data to match the planner's
subscription):
    /camera/camera/accel/sample  (only linear_acceleration is read)
    /camera/camera/gyro/sample   (only angular_velocity is read)

The accel vector is NOT a plain (0,0,g) "level" reading. It is chosen so
that, after the planner's OPTICAL_TO_MECHANICAL remap and
gravity_roll_pitch_deg() computation, the *measured* pitch/roll come out
matching camera_mount.yaml's reference_pitch_deg/reference_roll_deg
(-5.0/0.0 by default) -- i.e. it simulates the camera sitting still at
exactly its configured mount angle, so calibration locks into
CALIBRATION_VALID with ~0 delta instead of sitting at the
max_runtime_pitch_correction_deg boundary. Passed as ROS parameters
(reference_pitch_deg/reference_roll_deg) so it stays correct if
camera_mount.yaml's measured angles ever change; re-verify against that
file if this script is reused elsewhere.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

# Inverse of ground_plane_calibration.OPTICAL_TO_MECHANICAL (that matrix is
# a pure axis permutation, so its inverse is its transpose).
MECHANICAL_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])
GRAVITY_MPS2 = 9.81


def _optical_accel_for_mount(pitch_deg, roll_deg):
    """Mechanical-frame (x=fwd,y=left,z=up) gravity vector that reads as
    (pitch_deg, roll_deg) through ground_plane_calibration.gravity_roll_pitch_deg,
    converted back into the optical-frame vector the planner expects on the
    accel topic."""
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    mechanical = GRAVITY_MPS2 * np.array([
        math.sin(pitch), -math.sin(roll) * math.cos(pitch), math.cos(roll) * math.cos(pitch),
    ])
    return MECHANICAL_TO_OPTICAL @ mechanical


class FakeImuPublisher(Node):
    def __init__(self):
        super().__init__("fake_imu_publisher")
        self.declare_parameter("accel_topic", "/camera/camera/accel/sample")
        self.declare_parameter("gyro_topic", "/camera/camera/gyro/sample")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("rate_hz", 50.0)
        # Must match camera_mount.yaml's reference_pitch_deg/reference_roll_deg
        # for calibration to lock with ~0 delta instead of near the
        # max_runtime_pitch/roll_correction_deg boundary.
        self.declare_parameter("reference_pitch_deg", -5.0)
        self.declare_parameter("reference_roll_deg", 0.0)

        accel_topic = str(self.get_parameter("accel_topic").value)
        gyro_topic = str(self.get_parameter("gyro_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        pitch = float(self.get_parameter("reference_pitch_deg").value)
        roll = float(self.get_parameter("reference_roll_deg").value)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")

        self.accel_vector = _optical_accel_for_mount(pitch, roll)
        self.get_logger().info(
            f"publishing stationary fake IMU at {rate_hz:.1f} Hz "
            f"(simulated mount pitch={pitch:.1f}deg roll={roll:.1f}deg) on "
            f"{accel_topic} + {gyro_topic}; direct_bev_planner_node needs "
            f"~15 samples within its first ~1s to lock calibration, and a "
            f"fresh sample at least every 2.0s (imu_stale_timeout_sec) after "
            f"that or it reverts to CALIBRATION_INVALID.")

        self.accel_pub = self.create_publisher(Imu, accel_topic, qos_profile_sensor_data)
        self.gyro_pub = self.create_publisher(Imu, gyro_topic, qos_profile_sensor_data)
        self.create_timer(1.0 / rate_hz, self._publish_sample)

    def _publish_sample(self):
        stamp = self.get_clock().now().to_msg()

        gyro_msg = Imu()
        gyro_msg.header.stamp = stamp
        gyro_msg.header.frame_id = self.frame_id
        gyro_msg.angular_velocity.x = 0.0
        gyro_msg.angular_velocity.y = 0.0
        gyro_msg.angular_velocity.z = 0.0
        self.gyro_pub.publish(gyro_msg)

        accel_msg = Imu()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = self.frame_id
        accel_msg.linear_acceleration.x = float(self.accel_vector[0])
        accel_msg.linear_acceleration.y = float(self.accel_vector[1])
        accel_msg.linear_acceleration.z = float(self.accel_vector[2])
        self.accel_pub.publish(accel_msg)


def main():
    rclpy.init()
    node = FakeImuPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
