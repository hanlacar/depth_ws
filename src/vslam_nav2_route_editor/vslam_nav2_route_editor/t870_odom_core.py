"""Measured T870 encoder/steering bicycle odometry used only in rviz_* frames."""
from __future__ import annotations

from dataclasses import dataclass
import math


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class T870OdomState:
    rear_x: float = -0.365
    rear_y: float = 0.0
    yaw: float = 0.0
    distance_m: float = 0.0


class T870OdomModel:
    """Reference-release equations with measured constants and no MCU output."""

    def __init__(self, counts_per_meter=797.0, wheelbase_m=0.730,
                 base_from_rear_m=0.365, steer_center_adc=484.0,
                 steer_counts_per_deg=18.0, max_steer_deg=22.0):
        self.counts_per_meter = float(counts_per_meter)
        self.wheelbase_m = float(wheelbase_m)
        self.base_from_rear_m = float(base_from_rear_m)
        self.steer_center_adc = float(steer_center_adc)
        self.steer_counts_per_deg = float(steer_counts_per_deg)
        self.max_steer_deg = float(max_steer_deg)
        self.state = T870OdomState(rear_x=-self.base_from_rear_m)
        self.previous_count = None
        self.steer_deg = 0.0
        self.drive_stage = 0.0
        self.have_drive = False
        self.last_motion_direction = 1

    def set_steering_adc(self, adc: int) -> float:
        raw = (float(adc) - self.steer_center_adc) / self.steer_counts_per_deg
        self.steer_deg = max(-self.max_steer_deg, min(self.max_steer_deg, raw))
        return self.steer_deg

    def set_drive_stage(self, stage: float) -> None:
        self.drive_stage = float(stage)
        self.have_drive = True
        if stage > 0:
            self.last_motion_direction = 1
        elif stage < 0:
            self.last_motion_direction = -1

    def update_encoder(self, count: int) -> float:
        count = int(count)
        if self.previous_count is None:
            self.previous_count = count
            return 0.0
        raw_delta = count - self.previous_count
        self.previous_count = count
        if raw_delta == 0:
            return 0.0
        if self.have_drive:
            direction = (1 if self.drive_stage > 0 else
                         -1 if self.drive_stage < 0 else
                         self.last_motion_direction)
        else:
            direction = 1
        d_front = abs(raw_delta) * direction / self.counts_per_meter
        steer = math.radians(self.steer_deg)
        d_rear = d_front * math.cos(steer)
        dtheta = d_rear * math.tan(steer) / self.wheelbase_m
        old_yaw = self.state.yaw
        if abs(dtheta) > 1e-12:
            radius = d_rear / dtheta
            self.state.rear_x += radius * (
                math.sin(old_yaw + dtheta) - math.sin(old_yaw))
            self.state.rear_y -= radius * (
                math.cos(old_yaw + dtheta) - math.cos(old_yaw))
        else:
            self.state.rear_x += d_rear * math.cos(old_yaw)
            self.state.rear_y += d_rear * math.sin(old_yaw)
        self.state.yaw = normalize_angle(old_yaw + dtheta)
        self.state.distance_m += abs(d_rear)
        return d_rear

    def base_pose(self) -> tuple[float, float, float]:
        return (
            self.state.rear_x + self.base_from_rear_m * math.cos(self.state.yaw),
            self.state.rear_y + self.base_from_rear_m * math.sin(self.state.yaw),
            self.state.yaw,
        )
