"""Thread-safe bounded timestamp matching for optional RGB overlays."""

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


def message_stamp_ns(message):
    return (int(message.header.stamp.sec)*1_000_000_000+
            int(message.header.stamp.nanosec))


@dataclass(frozen=True)
class TimestampMatch:
    message: object
    stamp_ns: int
    delta_ns: int
    exact: bool


class TimestampedMessageCache:
    """Bounded cache with deterministic nearest matching and an age gate."""

    def __init__(self, maximum_frames):
        maximum_frames = int(maximum_frames)
        if maximum_frames <= 0:
            raise ValueError("timestamp cache size must be positive")
        self._frames = deque(maxlen=maximum_frames)
        self._lock = threading.Lock()

    def add(self, message, received_wall=None):
        item = (message_stamp_ns(message),
                time.monotonic() if received_wall is None else float(received_wall),
                message)
        with self._lock:
            self._frames.append(item)

    def nearest_match(self, stamp_ns, tolerance_sec, maximum_age_sec=None,
                      now_wall=None):
        tolerance_ns = int(round(max(0.0, float(tolerance_sec))*1.0e9))
        stamp_ns = int(stamp_ns)
        now_wall = time.monotonic() if now_wall is None else float(now_wall)
        with self._lock:
            if not self._frames:
                return None
            candidate_stamp, received_wall, candidate = min(
                self._frames,
                key=lambda item: (abs(item[0]-stamp_ns), item[0]))
        delta_ns = abs(candidate_stamp-stamp_ns)
        if delta_ns > tolerance_ns:
            return None
        if (maximum_age_sec is not None and
                now_wall-received_wall > max(0.0, float(maximum_age_sec))):
            return None
        return TimestampMatch(candidate, candidate_stamp, delta_ns,
                              delta_ns == 0)

    def nearest(self, stamp_ns, tolerance_sec, maximum_age_sec=None,
                now_wall=None):
        match = self.nearest_match(stamp_ns, tolerance_sec, maximum_age_sec,
                                   now_wall)
        return None if match is None else match.message

    def latest(self, maximum_age_sec=None, now_wall=None):
        now_wall = time.monotonic() if now_wall is None else float(now_wall)
        with self._lock:
            if not self._frames:
                return None
            _stamp, received_wall, message = self._frames[-1]
        if (maximum_age_sec is not None and
                now_wall-received_wall > max(0.0, float(maximum_age_sec))):
            return None
        return message

    def clear(self):
        with self._lock:
            self._frames.clear()

    def __len__(self):
        with self._lock:
            return len(self._frames)


@dataclass(frozen=True)
class ExactStampPair:
    """One path/state result produced from exactly one source frame."""

    stamp_ns: int
    path: object
    state: object
    path_received_ns: int
    state_received_ns: int

    @property
    def arrival_delta_ns(self):
        return int(self.state_received_ns)-int(self.path_received_ns)


class ExactStampPairCache:
    """Bounded, order-independent exact matcher with fail-closed expiry.

    Approximate matching is deliberately unsupported: a path and state from
    different camera frames must never become a controller input pair.
    """

    def __init__(self, maximum_pairs=32, maximum_age_sec=0.25):
        if int(maximum_pairs) <= 0:
            raise ValueError("exact pair cache size must be positive")
        if float(maximum_age_sec) <= 0.0:
            raise ValueError("exact pair cache age must be positive")
        self.maximum_pairs = int(maximum_pairs)
        self.maximum_age_ns = int(float(maximum_age_sec)*1.0e9)
        self._items = OrderedDict()
        self._last_consumed_stamp_ns = -1
        self._lock = threading.Lock()
        self._stats = {"matched": 0, "duplicate": 0, "expired": 0,
                       "capacity_dropped": 0, "old_dropped": 0}

    def _add(self, kind, stamp_ns, value, received_ns):
        stamp_ns, received_ns = int(stamp_ns), int(received_ns)
        with self._lock:
            if stamp_ns <= self._last_consumed_stamp_ns:
                self._stats["old_dropped"] += 1
                return None
            item = self._items.setdefault(stamp_ns, {
                "first_received_ns": received_ns})
            if kind in item:
                self._stats["duplicate"] += 1
                return None
            item[kind] = value
            item[f"{kind}_received_ns"] = received_ns
            self._items.move_to_end(stamp_ns)
            while len(self._items) > self.maximum_pairs:
                self._items.popitem(last=False)
                self._stats["capacity_dropped"] += 1
            if "path" not in item or "state" not in item:
                return None
            pair = ExactStampPair(
                stamp_ns, item["path"], item["state"],
                item["path_received_ns"], item["state_received_ns"])
            del self._items[stamp_ns]
            self._last_consumed_stamp_ns = stamp_ns
            for old_stamp in tuple(self._items):
                if old_stamp <= stamp_ns:
                    del self._items[old_stamp]
                    self._stats["old_dropped"] += 1
            self._stats["matched"] += 1
            return pair

    def add_path(self, stamp_ns, path, received_ns=None):
        return self._add("path", stamp_ns, path,
                         time.monotonic_ns() if received_ns is None else received_ns)

    def add_state(self, stamp_ns, state, received_ns=None):
        return self._add("state", stamp_ns, state,
                         time.monotonic_ns() if received_ns is None else received_ns)

    def discard_through(self, stamp_ns):
        """Invalidate unmatched inputs no newer than an INVALID result."""
        stamp_ns = int(stamp_ns)
        with self._lock:
            self._last_consumed_stamp_ns = max(
                self._last_consumed_stamp_ns, stamp_ns)
            for old_stamp in tuple(self._items):
                if old_stamp <= stamp_ns:
                    del self._items[old_stamp]
                    self._stats["old_dropped"] += 1

    def expire(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        expired = []
        with self._lock:
            for stamp_ns, item in tuple(self._items.items()):
                if now_ns-int(item["first_received_ns"]) > self.maximum_age_ns:
                    expired.append(stamp_ns)
                    del self._items[stamp_ns]
            self._stats["expired"] += len(expired)
        return expired

    def stats(self):
        with self._lock:
            return {**self._stats, "cached": len(self._items),
                    "last_consumed_stamp_ns": self._last_consumed_stamp_ns}

    def __len__(self):
        with self._lock:
            return len(self._items)


def subscription_transition(subscriber_count, active):
    """Return the idempotent action for a subscriber-gated input."""
    wanted = int(subscriber_count) > 0
    if wanted and not bool(active):
        return "CREATE"
    if not wanted and bool(active):
        return "DESTROY"
    return "NONE"
