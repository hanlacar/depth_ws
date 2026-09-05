"""ROS-independent steering-only controller for direct BEV paths."""

from dataclasses import dataclass
import math

import numpy as np

from .direct_bev_core import pure_pursuit_unclipped


@dataclass(frozen=True)
class BevControllerConfig:
    wheelbase_m: float = 0.73
    maximum_steering_deg: float = 22.0
    steering_sign: float = -1.0
    lookahead_min_m: float = 0.60
    lookahead_default_m: float = 1.20
    lookahead_max_m: float = 2.20
    curvature_scale_per_m: float = 0.80
    degraded_lookahead_scale: float = 0.85
    steering_rate_deg_per_sec: float = 120.0
    path_timeout_sec: float = 0.20
    lookahead_from_path_start: bool = True
    curvature_adaptive_lookahead: bool = False
    feedforward_weight: float = 0.0
    feedforward_max_delta_deg: float = 4.0
    steering_gain: float = 1.0
    fractional_accumulator: bool = True
    fractional_deadband_deg: float = 0.15
    steering_lineage_enabled: bool = True


class DirectBevController:
    def __init__(self, config=None):
        self.config = config or BevControllerConfig()
        self.previous_steering = 0.0
        self.previous_time = None
        self.fractional_residual = 0.0
        self.fractional_sign = 0

    @staticmethod
    def local_curvature(points, target):
        """Signed metric curvature near the selected target point."""
        points = np.asarray(points, float).reshape(-1, 2)
        if len(points) < 3 or target is None:
            return 0.0
        index = int(np.argmin(np.linalg.norm(points-np.asarray(target), axis=1)))
        sample = points[max(0, index-3):min(len(points), index+4)]
        if len(sample) < 3 or np.ptp(sample[:, 0]) < 1.0e-6:
            return 0.0
        a, b, _ = np.polyfit(sample[:, 0], sample[:, 1], 2)
        slope = 2.0*a*float(points[index, 0])+b
        return float(2.0*a/(1.0+slope*slope)**1.5)

    def lookahead(self, points, confidence, degraded):
        points = np.asarray(points, float).reshape(-1, 2)
        curvature = 0.0
        if len(points) >= 3:
            headings = np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0]))
            ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
            samples = np.abs(np.diff(np.unwrap(headings))) / \
                np.maximum(ds[1:], 1.0e-6)
            if self.config.curvature_adaptive_lookahead:
                # The first observable camera point is joined to the vehicle
                # axis.  Its artificial corner must not collapse preview to
                # the minimum; use robust curvature of the observed path.
                samples = samples[2:] if len(samples) > 3 else samples
                curvature = float(np.percentile(samples, 75)) if len(samples) else 0.0
            else:
                curvature = float(np.max(samples))
        fraction = min(1.0, curvature/max(1.0e-6,
                                          self.config.curvature_scale_per_m))
        value = (self.config.lookahead_max_m-
                 fraction*(self.config.lookahead_max_m-
                           self.config.lookahead_min_m))
        if degraded or confidence < 0.60:
            value *= self.config.degraded_lookahead_scale
        if (self.config.lookahead_from_path_start and
                not self.config.curvature_adaptive_lookahead):
            # A relative lookahead is used only when camera projection leaves
            # the vehicle-near grid unobservable.  Maximum preview prevents a
            # shortened target from landing on the artificial y=0 connector.
            value = self.config.lookahead_max_m
        maximum_path = float(np.max(np.linalg.norm(points, axis=1)))
        return float(np.clip(value, self.config.lookahead_min_m,
                             min(self.config.lookahead_max_m, maximum_path)))

    @staticmethod
    def _round_half_away(value):
        return int(math.copysign(math.floor(abs(value)+0.5), value))

    def _quantize(self, requested):
        if not self.config.fractional_accumulator:
            self.fractional_residual = 0.0
            self.fractional_sign = 0
            return int(round(requested))
        sign = 0 if abs(requested) < self.config.fractional_deadband_deg else \
            (1 if requested > 0.0 else -1)
        if sign == 0:
            self.fractional_residual = 0.0
            self.fractional_sign = 0
            return 0
        if self.fractional_sign not in (0, sign):
            self.fractional_residual = 0.0
        self.fractional_sign = sign
        accumulated = requested+self.fractional_residual
        wheel = self._round_half_away(accumulated)
        wheel = int(np.clip(wheel, -self.config.maximum_steering_deg,
                            self.config.maximum_steering_deg))
        self.fractional_residual = accumulated-wheel
        return wheel

    def command(self, points, confidence, degraded, now):
        points = np.asarray(points, float).reshape(-1, 2)
        lookahead = self.lookahead(points, confidence, degraded)
        pure_pursuit, target = pure_pursuit_unclipped(
            points, self.config.wheelbase_m, lookahead,
            lookahead_from_path_start=self.config.lookahead_from_path_start)
        if pure_pursuit is None or not math.isfinite(pure_pursuit):
            return {"valid": False, "wheel": 0, "reason": "TARGET_UNAVAILABLE"}
        target_index = int(np.argmin(np.linalg.norm(points-target, axis=1)))
        before = points[max(0, target_index-1)]
        after = points[min(len(points)-1, target_index+1)]
        heading_error = math.degrees(math.atan2(
            float(after[1]-before[1]), float(after[0]-before[0])))
        target_squared = float(target@target)
        target_curvature = float(2.0*target[1]/target_squared)
        bicycle_steering = math.degrees(math.atan(
            self.config.wheelbase_m*target_curvature))
        local_curvature = self.local_curvature(points, target)
        feedforward = math.degrees(math.atan(
            self.config.wheelbase_m*local_curvature))
        delta = float(np.clip(feedforward-pure_pursuit,
                              -self.config.feedforward_max_delta_deg,
                              self.config.feedforward_max_delta_deg))
        combined = pure_pursuit+self.config.feedforward_weight*delta
        required = combined*self.config.steering_gain
        if abs(required) > self.config.maximum_steering_deg:
            self.neutral()
            return {"valid": False, "wheel": 0,
                    "reason": "STEERING_LIMIT_EXCEEDED",
                    "required_steering_deg": required,
                    "raw_steering_deg": pure_pursuit,
                    "pure_pursuit_raw_deg": pure_pursuit,
                    "lateral_error_m": float(target[1]),
                    "heading_error_deg": heading_error,
                    "target_curvature_per_m": target_curvature,
                    "bicycle_steering_deg": bicycle_steering,
                    "local_curvature_per_m": local_curvature,
                    "feedforward_steering_deg": feedforward,
                    "lookahead_m": lookahead}
        controller_input = required
        sign_converted = controller_input*self.config.steering_sign
        requested = sign_converted
        dt = 0.0 if self.previous_time is None else max(0.0, now-self.previous_time)
        if self.previous_time is not None:
            limit = self.config.steering_rate_deg_per_sec*dt
            requested = float(np.clip(
                requested, self.previous_steering-limit,
                self.previous_steering+limit))
        requested = float(np.clip(requested,
                                  -self.config.maximum_steering_deg,
                                  self.config.maximum_steering_deg))
        self.previous_steering = requested
        self.previous_time = float(now)
        wheel = self._quantize(requested)
        return {"valid": True, "wheel": wheel,
                "required_steering_deg": required, "steering_deg": requested,
                "raw_steering_deg": pure_pursuit,
                "pure_pursuit_raw_deg": pure_pursuit,
                "lateral_error_m": float(target[1]),
                "heading_error_deg": heading_error,
                "target_curvature_per_m": target_curvature,
                "bicycle_steering_deg": bicycle_steering,
                "local_curvature_per_m": local_curvature,
                "feedforward_steering_deg": feedforward,
                "feedforward_delta_deg": delta,
                "controller_input_steering_deg": controller_input,
                "sign_converted_steering_deg": sign_converted,
                "temporal_filtered_steering_deg": requested,
                "clamped_steering_deg": requested,
                "fractional_residual_deg": self.fractional_residual,
                "rounded_int32_wheel": wheel,
                "lookahead_m": lookahead, "target_point": target.tolist(),
                "reason": "OK"}

    def neutral(self):
        self.previous_steering = 0.0
        self.previous_time = None
        self.fractional_residual = 0.0
        self.fractional_sign = 0
        return {"valid": False, "wheel": 0, "reason": "PATH_INVALID"}
