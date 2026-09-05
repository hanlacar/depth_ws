"""Latest-only, rate-limited support for optional visualization branches."""

import threading


class OverlayRateLimiter:
    def __init__(self, maximum_fps):
        maximum_fps = float(maximum_fps)
        if maximum_fps <= 0.0:
            raise ValueError("overlay maximum FPS must be positive")
        self.minimum_period = 1.0 / maximum_fps
        self.next_submit = None

    def ready(self, now):
        now = float(now)
        if self.next_submit is None:
            self.next_submit = now + self.minimum_period
            return True
        if now < self.next_submit:
            return False
        # Preserve the deadline phase. Resetting it from `now` quantizes a
        # 45 FPS limit on a 60 Hz source down to every second frame (30 FPS).
        self.next_submit += self.minimum_period
        if self.next_submit < now:
            self.next_submit = now + self.minimum_period
        return True

    def reset(self):
        self.next_submit = None


class LatestOnlyWorker:
    """Run optional work off the caller while retaining at most one pending job."""

    def __init__(self, callback, name):
        self._callback = callback
        self._condition = threading.Condition()
        self._pending = None
        self._stopped = False
        self._submitted = 0
        self._replaced = 0
        self._completed = 0
        self._failed = 0
        self._last_error = ""
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, job):
        with self._condition:
            if self._stopped:
                return
            self._submitted += 1
            if self._pending is not None:
                self._replaced += 1
            self._pending = job
            self._condition.notify()

    def snapshot(self):
        with self._condition:
            return {"submitted":self._submitted,"replaced":self._replaced,
                    "completed":self._completed,
                    "failed":self._failed,"last_error":self._last_error,
                    "pending":int(self._pending is not None)}

    def clear(self):
        with self._condition:
            self._pending = None

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopped:
                    self._condition.wait()
                if self._stopped:
                    return
                job = self._pending
                self._pending = None
            try:
                self._callback(job)
            except Exception as error:
                with self._condition:
                    self._failed += 1
                    self._last_error = f"{type(error).__name__}: {error}"
            with self._condition:
                self._completed += 1
