"""Fail-closed safety gate. It never grants control from a single signal."""

import math

from .models import SafetyDecision


class SafetyGate:
    def __init__(self, pose_timeout_s=0.15, controller_timeout_s=0.10,
                 min_confidence=0.7, max_cross_track_m=1.0,
                 max_heading_deg=80.0):
        self.pose_timeout = float(pose_timeout_s)
        self.controller_timeout = float(controller_timeout_s)
        self.min_confidence = float(min_confidence)
        self.max_cross_track = float(max_cross_track_m)
        self.max_heading = math.radians(max_heading_deg)

    def evaluate(self, value):
        reasons = []
        if not value.tracking_valid:
            reasons.append("CUVSLAM_TRACKING_LOSS")
        if value.localization_state not in ("TRACKING", "RELOCALIZED"):
            reasons.append("LOCALIZATION_NOT_READY")
        if value.localization_confidence < self.min_confidence:
            reasons.append("LOW_LOCALIZATION_CONFIDENCE")
        if value.now-value.pose_stamp > self.pose_timeout:
            reasons.append("STALE_POSE")
        if value.pose_jump:
            reasons.append("POSE_JUMP")
        if not value.map_route_match:
            reasons.append("MAP_ROUTE_MISMATCH")
        if not value.within_map:
            reasons.append("OUTSIDE_MAP")
        if abs(value.cross_track_error) > self.max_cross_track:
            reasons.append("ROUTE_DEVIATION")
        if abs(value.heading_error) > self.max_heading:
            reasons.append("HEADING_ERROR")
        if value.now-value.controller_stamp > self.controller_timeout:
            reasons.append("CONTROLLER_TIMEOUT")
        if value.mission_stop:
            reasons.append(value.mission_reason or "MISSION_STOP")
        if not value.user_approved:
            reasons.append("USER_APPROVAL_REQUIRED")
        if not value.enable_control:
            reasons.append("CONTROL_DISABLED")
        if value.dry_run:
            reasons.append("DRY_RUN")
        return SafetyDecision(not reasons, bool(reasons), tuple(reasons))
