"""Pure, ROS-independent streaming metrics used by the monitor and tests."""

from collections import deque
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RateSnapshot:
    fps: float
    samples: int
    dropped: int
    regressions: int


class StampRateTracker:
    """Measure unique source timestamps and estimate missing frames."""

    def __init__(self, window_seconds=2.0, expected_fps=0.0):
        if window_seconds <= 0.0:
            raise ValueError('window_seconds must be positive')
        self.window_seconds = float(window_seconds)
        self.expected_fps = float(expected_fps)
        self._stamps = deque()
        self.dropped = 0
        self.regressions = 0

    def add(self, stamp_seconds):
        stamp = float(stamp_seconds)
        if self._stamps and stamp <= self._stamps[-1]:
            self.regressions += 1
            return False
        if self._stamps and self.expected_fps > 0.0:
            period = 1.0 / self.expected_fps
            missing = max(0, round((stamp - self._stamps[-1]) / period) - 1)
            self.dropped += missing
        self._stamps.append(stamp)
        cutoff = stamp - self.window_seconds
        while len(self._stamps) > 2 and self._stamps[0] < cutoff:
            self._stamps.popleft()
        return True

    @property
    def fps(self):
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / elapsed if elapsed > 0.0 else 0.0

    def snapshot(self):
        return RateSnapshot(self.fps, len(self._stamps), self.dropped,
                            self.regressions)


class LatencyTracker:
    def __init__(self, capacity=600):
        self._values = deque(maxlen=int(capacity))

    def add(self, milliseconds):
        value = float(milliseconds)
        if value >= 0.0:
            self._values.append(value)

    @property
    def average(self):
        return mean(self._values) if self._values else 0.0

    @property
    def p95(self):
        if not self._values:
            return 0.0
        values = sorted(self._values)
        return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


class TrackingWatchdog:
    def __init__(self, timeout_seconds=1.0):
        self.timeout_seconds = float(timeout_seconds)
        self.last_pose_time = None

    def mark(self, monotonic_seconds):
        self.last_pose_time = float(monotonic_seconds)

    def valid(self, monotonic_seconds):
        return (self.last_pose_time is not None and
                float(monotonic_seconds) - self.last_pose_time <= self.timeout_seconds)
