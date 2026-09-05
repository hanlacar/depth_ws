"""Bounded latency statistics for backend and ROS pipeline stages."""
from collections import defaultdict, deque
from threading import Lock
import time
import numpy as np


class LatencyTracker:
    def __init__(self, capacity=300):
        self.capacity = int(capacity)
        self.values = defaultdict(lambda: deque(maxlen=self.capacity))
        self.processed = 0
        self.processed_times = deque(maxlen=self.capacity)

    def add(self, value, stage="total_pipeline_ms"):
        self.values[stage].append(float(value))
        if stage == "total_pipeline_ms":
            self.processed += 1
            self.processed_times.append(time.perf_counter())

    def add_stages(self, values):
        for stage, value in values.items():
            self.add(value, stage)

    def summary(self):
        result = {"processed_frames": self.processed}
        for stage, values in self.values.items():
            array = np.asarray(values, dtype=float)
            result[stage] = {
                "count": int(array.size), "mean": float(array.mean()),
                "median": float(np.median(array)),
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)), "max": float(array.max())}
        total = self.values.get("total_pipeline_ms")
        if total:
            result["processed_fps_equivalent"] = 1000.0 / float(np.mean(total))
        if len(self.processed_times)>1:
            result["processed_fps_observed"]=(len(self.processed_times)-1)/(self.processed_times[-1]-self.processed_times[0])
        return result


class EventRateTracker:
    """Thread-safe bounded event counters and rolling rates."""

    def __init__(self, window_sec=10.0):
        self.window_sec = float(window_sec)
        self.counts = defaultdict(int)
        self.times = defaultdict(deque)
        self._lock = Lock()

    def mark(self, event, now=None):
        timestamp = time.perf_counter() if now is None else float(now)
        with self._lock:
            self.counts[event] += 1
            values = self.times[event]
            values.append(timestamp)
            cutoff = timestamp - self.window_sec
            while values and values[0] < cutoff:
                values.popleft()

    def snapshot(self, now=None):
        timestamp = time.perf_counter() if now is None else float(now)
        cutoff = timestamp - self.window_sec
        result = {}
        with self._lock:
            for event, count in self.counts.items():
                values = self.times[event]
                while values and values[0] < cutoff:
                    values.popleft()
                if len(values) > 1:
                    duration = values[-1] - values[0]
                    rate = (len(values) - 1) / duration if duration > 0.0 else 0.0
                else:
                    rate = 0.0
                result[event] = {"count": count, "fps_10s": rate,
                                 "samples_10s": len(values)}
        return result


class UniqueFrameRateTracker:
    """Per-stage unique header-stamp rates over 1/5/10 second windows."""

    def __init__(self, maximum_window_sec=10.0):
        self.maximum_window_sec = float(maximum_window_sec)
        self.values = defaultdict(deque)
        self.seen = defaultdict(set)
        self._lock = Lock()

    def mark(self, event, stamp_sec, stamp_nanosec, arrival=None):
        key = int(stamp_sec)*1_000_000_000 + int(stamp_nanosec)
        now = time.perf_counter() if arrival is None else float(arrival)
        with self._lock:
            if key in self.seen[event]:
                return False
            self.seen[event].add(key); self.values[event].append((now, key))
            cutoff = now-self.maximum_window_sec-1.0
            while self.values[event] and self.values[event][0][0] < cutoff:
                _, expired = self.values[event].popleft(); self.seen[event].discard(expired)
        return True

    def snapshot(self, event, now=None):
        now = time.perf_counter() if now is None else float(now)
        output = {}
        with self._lock:
            values = tuple(self.values.get(event, ()))
        for window in (1.0, 5.0, 10.0):
            samples = [(arrival, stamp) for arrival, stamp in values
                       if arrival >= now-window]
            if len(samples) > 1:
                header_span = (samples[-1][1]-samples[0][1])/1e9
                arrival_span = samples[-1][0]-samples[0][0]
                header_fps = (len(samples)-1)/header_span if header_span > 0 else 0.0
                arrival_fps = (len(samples)-1)/arrival_span if arrival_span > 0 else 0.0
                arrival_intervals = np.diff([arrival for arrival, _ in samples])*1000.0
                header_intervals = np.diff([stamp for _, stamp in samples])/1e6
                arrival_gap_p95 = float(np.percentile(arrival_intervals, 95))
                arrival_gap_max = float(np.max(arrival_intervals))
                header_gap_p50 = float(np.percentile(header_intervals, 50))
                header_gap_p95 = float(np.percentile(header_intervals, 95))
                header_gap_p99 = float(np.percentile(header_intervals, 99))
                header_gap_max = float(np.max(header_intervals))
            else:
                header_fps = arrival_fps = 0.0
                arrival_gap_p95 = arrival_gap_max = 0.0
                header_gap_p50 = header_gap_p95 = header_gap_p99 = header_gap_max = 0.0
            label = f"{int(window)}s"
            output[label] = {"unique_frames": len(samples),
                             "header_fps": header_fps,
                             "arrival_fps": arrival_fps,
                             "header_gap_p50_ms": header_gap_p50,
                             "header_gap_p95_ms": header_gap_p95,
                             "header_gap_p99_ms": header_gap_p99,
                             "header_gap_max_ms": header_gap_max,
                             "arrival_gap_p95_ms": arrival_gap_p95,
                             "arrival_gap_max_ms": arrival_gap_max}
        return output


class ScalarLatencyTracker:
    """Thread-safe bounded p50/p95/max statistics for a scalar latency."""

    def __init__(self, capacity=600):
        self.values = deque(maxlen=int(capacity))
        self._lock = Lock()

    def add(self, value):
        with self._lock:
            self.values.append(float(value))

    def summary(self):
        with self._lock:
            values = tuple(self.values)
        if not values:
            return {"count":0,"p50":0.0,"p95":0.0,"max":0.0}
        array=np.asarray(values,dtype=float)
        return {"count":int(array.size),"p50":float(np.percentile(array,50)),
                "p95":float(np.percentile(array,95)),"max":float(array.max())}
