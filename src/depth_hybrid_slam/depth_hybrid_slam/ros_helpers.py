"""Shared ROS message conversion and diagnostics helpers."""

import math

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

from .models import Pose2D


def stamp_seconds(stamp):
    return float(stamp.sec)+float(stamp.nanosec)*1.0e-9


def yaw_from_quaternion(q):
    return math.atan2(2.0*(q.w*q.z+q.x*q.y),
                      1.0-2.0*(q.y*q.y+q.z*q.z))


def quaternion_from_yaw(yaw):
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))


def pose2d_from_odometry(message):
    return Pose2D(message.pose.pose.position.x, message.pose.pose.position.y,
                  yaw_from_quaternion(message.pose.pose.orientation),
                  stamp_seconds(message.header.stamp),
                  message.twist.twist.linear.x,
                  message.twist.twist.angular.z)


def status(name, level, message, values=()):
    return DiagnosticStatus(
        level=level, name=name, hardware_id="depth_slam",
        message=message,
        values=[KeyValue(key=str(k), value=str(v)) for k, v in values],
    )


def safe_shutdown():
    """Tolerate launch's signal handler winning the rclpy shutdown race."""
    import rclpy
    try:
        rclpy.shutdown()
    except Exception as error:  # rclpy's concrete RCLError is not API-stable.
        if "already called" not in str(error):
            raise
