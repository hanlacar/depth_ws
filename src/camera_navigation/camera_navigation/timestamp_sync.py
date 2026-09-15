"""Exact timestamp pairing for camera path and state messages."""

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class ExactStampPair:
    stamp_ns: int
    path: object
    state: object
    path_received_ns: int
    state_received_ns: int

    @property
    def arrival_delta_ns(self):
        return self.state_received_ns-self.path_received_ns


class ExactStampPairCache:
    """Bounded, order-independent matcher that rejects approximate pairs."""

    def __init__(self, maximum_pairs=32, maximum_age_sec=0.25):
        if int(maximum_pairs) <= 0 or float(maximum_age_sec) <= 0.0:
            raise ValueError("exact pair cache limits must be positive")
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
            item = self._items.setdefault(
                stamp_ns, {"first_received_ns": received_ns})
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
        return self._add(
            "path", stamp_ns, path,
            time.monotonic_ns() if received_ns is None else received_ns)

    def add_state(self, stamp_ns, state, received_ns=None):
        return self._add(
            "state", stamp_ns, state,
            time.monotonic_ns() if received_ns is None else received_ns)

    def discard_through(self, stamp_ns):
        """Invalidate unmatched inputs no newer than an invalid result."""
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
        with self._lock:
            expired = [stamp for stamp, item in self._items.items()
                       if now_ns-item["first_received_ns"] > self.maximum_age_ns]
            for stamp in expired:
                del self._items[stamp]
            self._stats["expired"] += len(expired)
        return expired

    def stats(self):
        with self._lock:
            return {**self._stats, "cached": len(self._items),
                    "last_consumed_stamp_ns": self._last_consumed_stamp_ns}

    def __len__(self):
        with self._lock:
            return len(self._items)
