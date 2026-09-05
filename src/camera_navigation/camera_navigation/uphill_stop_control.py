"""ROS-independent one-shot uphill stop state machine."""

from dataclasses import dataclass
from enum import Enum
import math


class UphillStopState(str, Enum):
    ARMED = "ARMED"
    STOPPING = "STOPPING"
    PASSED = "PASSED"


@dataclass(frozen=True)
class UphillStopConfig:
    uphill_stop_duration_sec: float = 5.0

    def validate(self):
        if (not math.isfinite(self.uphill_stop_duration_sec)
                or self.uphill_stop_duration_sec <= 0.0):
            raise ValueError("uphill_stop_duration_sec must be finite and positive")


class UphillStopController:
    """Stop once per valid False->True slope transition."""

    def __init__(self, config=UphillStopConfig()):
        config.validate()
        self.config = config
        self.state = UphillStopState.ARMED
        self.previous_valid_slope = None
        self.stop_started_at = None

    @property
    def stop_active(self):
        return self.state == UphillStopState.STOPPING

    def update(self, slope, imu_valid, now):
        """Advance state using monotonic ``now`` and return stop-active."""
        slope = bool(slope)
        imu_valid = bool(imu_valid)
        now = float(now)

        if self.state == UphillStopState.STOPPING:
            # Invalid/non-finite time must never end an already-started stop.
            if (not math.isfinite(now) or self.stop_started_at is None or
                    now-self.stop_started_at <
                    self.config.uphill_stop_duration_sec):
                return True

            self.stop_started_at = None
            if not imu_valid:
                # Do not re-arm until a fresh valid flat observation arrives.
                self.state = UphillStopState.PASSED
                self.previous_valid_slope = None
            elif slope:
                self.state = UphillStopState.PASSED
                self.previous_valid_slope = True
            else:
                self.state = UphillStopState.ARMED
                self.previous_valid_slope = False
            return False

        if not imu_valid or not math.isfinite(now):
            # An invalid interval cannot provide either side of an entry edge.
            self.previous_valid_slope = None
            return False

        if self.state == UphillStopState.PASSED:
            self.previous_valid_slope = slope
            if not slope:
                self.state = UphillStopState.ARMED
            return False

        if self.previous_valid_slope is None:
            self.previous_valid_slope = slope
            if slope:
                # Starting/resuming while already uphill is not an entry edge.
                self.state = UphillStopState.PASSED
            return False

        if self.previous_valid_slope is False and slope:
            self.state = UphillStopState.STOPPING
            self.stop_started_at = now
            self.previous_valid_slope = True
            return True

        self.previous_valid_slope = slope
        return False
