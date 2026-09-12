"""Test-only Ackermann vehicle and fake encoder matching MCU v5 constants."""

from dataclasses import dataclass
import math


COUNTS_PER_METER = 797.0
WHEELBASE_M = 0.730
MAX_STEERING_DEG = 22.0


@dataclass(frozen=True)
class VirtualState:
    x: float
    y: float
    yaw: float
    speed_mps: float
    steer_deg: float
    distance_m: float
    encoder: int
    stopped: bool
    direction_guard_ok: bool


class VirtualAckermannVehicle:
    def __init__(self, stage_speeds_mps, reverse_speed_mps,
                 wheelbase_m=WHEELBASE_M,
                 counts_per_meter=COUNTS_PER_METER,
                 max_steering_deg=MAX_STEERING_DEG,
                 direction_change_hold_s=3.0, encoder_signed=False):
        speeds = {int(stage): float(speed)
                  for stage, speed in dict(stage_speeds_mps).items()}
        if set(speeds) != {1, 2, 3} or not (
                0.0 < speeds[1] < speeds[2] < speeds[3]):
            raise ValueError("forward stage speeds must be positive 1 < 2 < 3")
        reverse = float(reverse_speed_mps)
        if not math.isfinite(reverse) or reverse <= 0.0:
            raise ValueError("reverse_speed_mps must be finite and positive")
        self.stage_mps = {-1: -reverse, 0: 0.0, **speeds}
        self.wheelbase = float(wheelbase_m)
        self.counts_per_meter = float(counts_per_meter)
        self.max_steering = float(max_steering_deg)
        self.direction_change_hold_s = float(direction_change_hold_s)
        self.encoder_signed = bool(encoder_signed)
        if (abs(self.wheelbase-WHEELBASE_M) > 1.0e-9 or
                abs(self.counts_per_meter-COUNTS_PER_METER) > 1.0e-9 or
                abs(self.max_steering-MAX_STEERING_DEG) > 1.0e-9 or
                self.encoder_signed):
            raise ValueError("virtual MCU must use latest measured calibration")
        if self.direction_change_hold_s < 3.0:
            raise ValueError("direction change hold must be at least 3 seconds")
        self.x = self.y = self.yaw = 0.0
        self.distance = 0.0
        self.last_motion_direction = 0
        self.stop_elapsed = 0.0
        self.direction_guard_ok = True

    def reset(self):
        self.x = self.y = self.yaw = 0.0
        self.distance = 0.0
        self.last_motion_direction = 0
        self.stop_elapsed = 0.0
        self.direction_guard_ok = True

    def step(self, drive_stage, steer_deg, stop, dt_s):
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 2.0:
            raise ValueError("virtual integration dt must be in (0, 2]")
        try:
            stage = int(drive_stage)
            steer = float(steer_deg)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid virtual command") from error
        if float(drive_stage) != stage or stage not in self.stage_mps:
            raise ValueError("drive stage must be -1, 0, 1, 2, or 3")
        steer = max(-self.max_steering, min(self.max_steering, steer))
        requested_direction = 1 if stage > 0 else -1 if stage < 0 else 0
        stopped = bool(stop) or stage == 0
        if stopped:
            speed = 0.0
            self.stop_elapsed += dt
        else:
            changing = (self.last_motion_direction != 0 and
                        requested_direction != self.last_motion_direction)
            if changing and self.stop_elapsed+1.0e-9 < self.direction_change_hold_s:
                speed = 0.0
                stopped = True
                self.stop_elapsed += dt
                self.direction_guard_ok = False
            else:
                speed = self.stage_mps[stage]
                self.last_motion_direction = requested_direction
                self.stop_elapsed = 0.0
        distance = speed*dt
        steer_rad = math.radians(steer)
        curvature = math.tan(steer_rad)/self.wheelbase
        dtheta = distance*curvature
        if abs(dtheta) < 1.0e-12:
            self.x += distance*math.cos(self.yaw)
            self.y += distance*math.sin(self.yaw)
        else:
            radius = distance/dtheta
            next_yaw = self.yaw+dtheta
            self.x += radius*(math.sin(next_yaw)-math.sin(self.yaw))
            self.y -= radius*(math.cos(next_yaw)-math.cos(self.yaw))
            self.yaw = math.atan2(math.sin(next_yaw), math.cos(next_yaw))
        self.distance += abs(distance)
        return VirtualState(
            self.x, self.y, self.yaw, speed, steer, self.distance,
            int(round(self.distance*self.counts_per_meter)), stopped,
            self.direction_guard_ok)
