import unittest

from depth_monitor.metrics import LatencyTracker, StampRateTracker, TrackingWatchdog


class MetricsTest(unittest.TestCase):
    def test_fps_uses_unique_source_timestamps(self):
        tracker = StampRateTracker(window_seconds=2.0, expected_fps=10.0)
        for stamp in (1.0, 1.1, 1.2, 1.3):
            self.assertTrue(tracker.add(stamp))
        self.assertAlmostEqual(tracker.fps, 10.0)

    def test_timestamp_regression_and_duplicate_are_rejected(self):
        tracker = StampRateTracker()
        self.assertTrue(tracker.add(2.0))
        self.assertFalse(tracker.add(2.0))
        self.assertFalse(tracker.add(1.9))
        self.assertEqual(tracker.regressions, 2)

    def test_old_window_samples_are_dropped_from_rate(self):
        tracker = StampRateTracker(window_seconds=1.0)
        for stamp in (0.0, 0.5, 1.0, 1.5, 2.0):
            tracker.add(stamp)
        self.assertEqual(tracker.snapshot().samples, 3)
        self.assertAlmostEqual(tracker.fps, 2.0)

    def test_missing_frames_are_counted(self):
        tracker = StampRateTracker(expected_fps=10.0)
        tracker.add(0.0)
        tracker.add(0.1)
        tracker.add(0.4)
        self.assertEqual(tracker.dropped, 2)

    def test_tracking_timeout_state_transition(self):
        watchdog = TrackingWatchdog(timeout_seconds=1.0)
        self.assertFalse(watchdog.valid(4.0))
        watchdog.mark(4.0)
        self.assertTrue(watchdog.valid(5.0))
        self.assertFalse(watchdog.valid(5.001))

    def test_latency_average_and_p95(self):
        tracker = LatencyTracker()
        for value in range(1, 101):
            tracker.add(value)
        self.assertAlmostEqual(tracker.average, 50.5)
        self.assertAlmostEqual(tracker.p95, 95.0)
