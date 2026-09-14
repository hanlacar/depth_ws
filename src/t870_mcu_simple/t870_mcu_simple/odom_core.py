"""Measured encoder/steering Ackermann odometry; no motion model input."""

from dataclasses import dataclass
import math


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass
class OdomState:
    rear_x: float
    rear_y: float = 0.0
    yaw: float = 0.0
    distance_m: float = 0.0


class MeasuredEncoderOdom:
    def __init__(self, counts_per_meter=797.0, wheelbase_m=0.73,
                 base_from_rear_m=0.365, max_steer_deg=22.0,
                 max_encoder_delta_counts=5000):
        self.counts_per_meter = float(counts_per_meter)
        self.wheelbase_m = float(wheelbase_m)
        self.base_from_rear_m = float(base_from_rear_m)
        self.max_steer_deg = float(max_steer_deg)
        self.max_encoder_delta_counts = int(max_encoder_delta_counts)
        if (min(self.counts_per_meter, self.wheelbase_m) <= 0.0 or
                self.max_encoder_delta_counts <= 0):
            raise ValueError("invalid measured odometry calibration")
        self.state = OdomState(-self.base_from_rear_m)
        self.previous_count = None
        self.steer_deg = 0.0
        self.motion_direction = 1
        self.last_discontinuity = False

    def reset_encoder_baseline(self):
        self.previous_count = None
        self.last_discontinuity = False

    def set_steering_deg(self, value):
        self.steer_deg = max(
            -self.max_steer_deg, min(self.max_steer_deg, float(value)))

    def set_direction_from_stage(self, stage):
        value = float(stage)
        if value > 0.0:
            self.motion_direction = 1
        elif value < 0.0:
            self.motion_direction = -1

    @staticmethod
    def _counter_delta(current, previous):
        delta = int(current)-int(previous)
        if delta > 2**31:
            delta -= 2**32
        elif delta < -(2**31):
            delta += 2**32
        return delta

    def update_encoder(self, count):
        count = int(count)
        self.last_discontinuity = False
        if self.previous_count is None:
            self.previous_count = count
            return 0.0, 0.0
        raw_delta = self._counter_delta(count, self.previous_count)
        self.previous_count = count
        # ARM/reboot resets the firmware counter to zero. Never interpret a
        # counter reset (or corrupt serial sample) as physical vehicle travel.
        if abs(raw_delta) > self.max_encoder_delta_counts:
            self.last_discontinuity = True
            return 0.0, 0.0
        if raw_delta == 0:
            return 0.0, 0.0
        front_distance = (
            abs(raw_delta)*self.motion_direction/self.counts_per_meter)
        steering = math.radians(self.steer_deg)
        rear_distance = front_distance*math.cos(steering)
        delta_yaw = rear_distance*math.tan(steering)/self.wheelbase_m
        old_yaw = self.state.yaw
        if abs(delta_yaw) <= 1.0e-12:
            self.state.rear_x += rear_distance*math.cos(old_yaw)
            self.state.rear_y += rear_distance*math.sin(old_yaw)
        else:
            radius = rear_distance/delta_yaw
            self.state.rear_x += radius*(
                math.sin(old_yaw+delta_yaw)-math.sin(old_yaw))
            self.state.rear_y -= radius*(
                math.cos(old_yaw+delta_yaw)-math.cos(old_yaw))
        self.state.yaw = normalize_angle(old_yaw+delta_yaw)
        self.state.distance_m += abs(rear_distance)
        return rear_distance, delta_yaw

    def base_pose(self):
        yaw = self.state.yaw
        return (
            self.state.rear_x+self.base_from_rear_m*math.cos(yaw),
            self.state.rear_y+self.base_from_rear_m*math.sin(yaw), yaw)
