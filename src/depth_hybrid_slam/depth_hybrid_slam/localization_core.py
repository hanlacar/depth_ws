"""Fuse high-rate odometry with low-rate map-to-odom corrections."""

import math
from dataclasses import dataclass

from .geometry import compose, interpolate_transform, wrap_angle


@dataclass(frozen=True)
class VslamGateDecision:
    use_vslam: bool
    stop: bool
    state: str
    transform: tuple


class VslamRecoveryGate:
    """Accept map corrections only after visual evidence and jump recovery."""

    def __init__(self, position_jump_threshold_m=0.75,
                 yaw_jump_threshold_deg=25.0, recovery_s=1.0):
        self.position_threshold = float(position_jump_threshold_m)
        self.yaw_threshold = math.radians(float(yaw_jump_threshold_deg))
        self.recovery_s = max(1.0, float(recovery_s))
        self.transform = (0.0, 0.0, 0.0)
        self.initialized = False
        self.recovery_started = None
        self.degraded = False

    def _jump(self, candidate):
        return (math.hypot(candidate[0]-self.transform[0],
                           candidate[1]-self.transform[1]) >
                self.position_threshold or
                abs(wrap_angle(candidate[2]-self.transform[2])) >
                self.yaw_threshold)

    def update(self, candidate, *, visual_consistent, evidence_fresh,
               now):
        now = float(now)
        if not evidence_fresh or not visual_consistent or candidate is None:
            # A visual mismatch rejects VSLAM before pose comparison, so it
            # must never start or prolong jump recovery.
            self.recovery_started = None
            return VslamGateDecision(
                False, False, "RUNNING_ODOM_ONLY", self.transform)
        candidate = tuple(float(value) for value in candidate)
        if not self.initialized:
            self.transform = candidate
            self.initialized = True
            self.degraded = False
            return VslamGateDecision(True, False, "RUNNING", self.transform)
        jump = self._jump(candidate)
        if self.degraded:
            if jump:
                return VslamGateDecision(
                    False, False, "RUNNING_ODOM_ONLY", self.transform)
            self.transform = candidate
            self.degraded = False
            return VslamGateDecision(True, False, "RUNNING", self.transform)
        if not jump:
            self.transform = candidate
            self.recovery_started = None
            return VslamGateDecision(True, False, "RUNNING", self.transform)
        if self.recovery_started is None:
            self.recovery_started = now
            return VslamGateDecision(False, True, "WARNING", self.transform)
        if now-self.recovery_started < self.recovery_s:
            return VslamGateDecision(False, True, "RECOVERING", self.transform)
        # One bounded recovery attempt: a persistent jump is ignored and the
        # vehicle resumes on the last accepted map->odom plus actual odometry.
        self.recovery_started = None
        self.degraded = True
        return VslamGateDecision(
            False, False, "RUNNING_ODOM_ONLY", self.transform)


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
