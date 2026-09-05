"""Fuse high-rate odometry with low-rate map-to-odom corrections."""

import math

from .geometry import compose, interpolate_transform, wrap_angle


class LocalizationFusion:
    def __init__(self, smoothing_alpha=0.15, large_translation_m=0.5,
                 large_rotation_deg=20.0, odom_jump_m=0.75,
                 odom_jump_deg=30.0):
        self.transform = (0.0, 0.0, 0.0)
        self.target = self.transform
        self.pending = None
        self.alpha = float(smoothing_alpha)
        self.large_translation = float(large_translation_m)
        self.large_rotation = math.radians(large_rotation_deg)
        self.odom_jump = float(odom_jump_m)
        self.odom_jump_yaw = math.radians(odom_jump_deg)
        self.previous_odom = None
        self.correction_stamp = None
        self.relocalized = False
        self.pose_jump = False

    def set_correction(self, transform, stamp, safe_to_apply=False):
        delta_xy = math.hypot(transform[0] - self.transform[0],
                              transform[1] - self.transform[1])
        delta_yaw = abs(wrap_angle(transform[2] - self.transform[2]))
        large = (delta_xy > self.large_translation or
                 delta_yaw > self.large_rotation)
        if large and not safe_to_apply:
            self.pending = tuple(transform)
            return False
        self.target = tuple(transform)
        self.pending = None
        self.correction_stamp = float(stamp)
        self.relocalized = True
        return True

    def approve_pending(self, stamp):
        if self.pending is None:
            return False
        self.target = self.pending
        self.pending = None
        self.correction_stamp = float(stamp)
        self.relocalized = True
        return True

    def fuse(self, odom, tracking_valid=True):
        self.pose_jump = False
        if self.previous_odom is not None:
            distance = math.hypot(odom.x - self.previous_odom.x,
                                  odom.y - self.previous_odom.y)
            yaw_delta = abs(wrap_angle(odom.yaw - self.previous_odom.yaw))
            if distance > self.odom_jump or yaw_delta > self.odom_jump_yaw:
                self.pose_jump = True
                return None, "STOP_REQUIRED"
        self.previous_odom = odom
        if not tracking_valid:
            return None, "LOST"
        self.transform = interpolate_transform(
            self.transform, self.target, self.alpha)
        pose = compose(self.transform, odom)
        if self.pending is not None:
            return pose, "DEGRADED"
        if self.correction_stamp is None:
            return pose, "INITIALIZING"
        return pose, "RELOCALIZED" if self.relocalized else "TRACKING"
