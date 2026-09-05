"""Dynamic-lookahead controller for original-image paths."""

from dataclasses import dataclass
import math

import numpy as np

from .camera_pixel_controller_node import (
    PixelCommand, PixelController, PixelControllerConfig,
)
from .pixel_lateral_control import lookahead_offset_px, steering_from_offset_deg


@dataclass(frozen=True)
class DynamicLookaheadConfig:
    minimum_ratio: float = 0.25
    maximum_ratio: float = 0.78
    curve_threshold_px: float = 1.5
    low_confidence_threshold: float = 0.60
    low_confidence_reduction: float = 0.12
    maximum_ratio_change: float = 0.12


class AdaptivePixelController(PixelController):
    def __init__(self, config=PixelControllerConfig(), dynamic=None):
        super().__init__(config)
        self.dynamic = dynamic or DynamicLookaheadConfig()
        self.previous_lookahead_ratio = None
        self.selected_lookahead_ratio = config.lookahead_y_ratio

    def _dynamic_ratio(self, path):
        points = np.asarray(path.points, dtype=float).reshape(-1, 2)
        curvature = (float(np.percentile(np.abs(np.diff(points[:, 0], n=2)), 90))
                     if len(points) >= 3 else self.dynamic.curve_threshold_px)
        curve = min(1.0, curvature/max(1e-6, self.dynamic.curve_threshold_px))
        ratio = (self.dynamic.maximum_ratio-
                 curve*(self.dynamic.maximum_ratio-self.dynamic.minimum_ratio))
        if path.confidence < self.dynamic.low_confidence_threshold:
            ratio -= self.dynamic.low_confidence_reduction
        # A short path still uses a point that exists; the ratio is an index
        # fraction, so no extrapolation is possible.
        ratio = max(self.dynamic.minimum_ratio,
                    min(self.dynamic.maximum_ratio, ratio))
        if self.previous_lookahead_ratio is not None:
            delta = max(-self.dynamic.maximum_ratio_change,
                        min(self.dynamic.maximum_ratio_change,
                            ratio - self.previous_lookahead_ratio))
            ratio = self.previous_lookahead_ratio+delta
        return float(ratio)

    def _step(self, now, ros_now_ns):
        # Reuse all fail-closed checks by running the stock step first. Replace
        # only steering after it has authorized this exact path sample.
        prior_offset = self.previous_offset_px
        prior_step = self.last_step_monotonic
        stock = super()._step(now, ros_now_ns)
        if not stock.valid:
            return stock
        path = self.path
        ratio = self._dynamic_ratio(path)
        offset = lookahead_offset_px(path.points, path.image_width, ratio)
        if offset is None or not math.isfinite(offset):
            return self.stop("dynamic_offset_unavailable")
        dt = 0.0 if prior_step is None else max(0.0, now-prior_step)
        steering = steering_from_offset_deg(
            offset, prior_offset, dt, path.image_width,
            self.config.proportional_gain_deg_per_norm,
            self.config.derivative_gain_deg_per_norm_per_s,
            self.config.maximum_steering_deg)*self.config.steering_sign
        if not math.isfinite(steering):
            return self.stop("steering_nonfinite")
        steering = max(-self.config.maximum_steering_deg,
                       min(self.config.maximum_steering_deg, steering))
        self.previous_offset_px = float(offset)
        self.last_step_monotonic = float(now)
        self.previous_lookahead_ratio = ratio
        self.selected_lookahead_ratio = ratio
        wheel = max(-22, min(22, int(round(steering))))
        return PixelCommand(self.drive_for_path(path, wheel), wheel, steering,
                            f"ok_lookahead_{ratio:.2f}", True)
