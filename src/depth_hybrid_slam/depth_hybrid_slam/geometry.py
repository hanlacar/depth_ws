"""Small, dependency-free planar geometry helpers."""

import math

from .models import Pose2D


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def compose(transform, pose):
    """Apply planar transform ``map<-odom`` to an odom-frame pose."""
    tx, ty, tyaw = transform
    c, s = math.cos(tyaw), math.sin(tyaw)
    return Pose2D(
        tx + c * pose.x - s * pose.y,
        ty + s * pose.x + c * pose.y,
        wrap_angle(tyaw + pose.yaw),
        pose.stamp,
        pose.speed,
        pose.yaw_rate,
        pose.confidence,
    )


def interpolate_transform(current, target, alpha):
    alpha = min(1.0, max(0.0, alpha))
    return (
        current[0] + alpha * (target[0] - current[0]),
        current[1] + alpha * (target[1] - current[1]),
        wrap_angle(current[2] + alpha * wrap_angle(target[2] - current[2])),
    )


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
