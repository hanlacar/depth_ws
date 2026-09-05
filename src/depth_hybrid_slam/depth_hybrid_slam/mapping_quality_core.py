"""ROS-independent mapping input quality and readiness state machine."""

from collections import deque
import math


class StreamTiming:
    def __init__(self, window_seconds=2.0):
        self.window_seconds = float(window_seconds)
        self.stamps = deque()
        self.last_stamp_ns = None
        self.count = 0
        self.duplicates = 0
        self.backwards = 0
        self.periods = RunningMetric()

    def observe(self, stamp_ns):
        stamp_ns = int(stamp_ns)
        if self.last_stamp_ns is not None:
            if stamp_ns == self.last_stamp_ns:
                self.duplicates += 1
            elif stamp_ns < self.last_stamp_ns:
                self.backwards += 1
            else:
                self.periods.observe((stamp_ns-self.last_stamp_ns)/1.0e9)
        self.last_stamp_ns = stamp_ns
        self.count += 1
        seconds = stamp_ns / 1.0e9
        self.stamps.append(seconds)
        while self.stamps and seconds - self.stamps[0] > self.window_seconds:
            self.stamps.popleft()

    @property
    def fps(self):
        if len(self.stamps) < 2:
            return 0.0
        duration = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / duration if duration > 0 else 0.0


class RunningMetric:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.latest = None

    def observe(self, value):
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.latest = value

    def report(self):
        return {
            "count": self.count,
            "latest": self.latest,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


class MappingQuality:
    STREAMS = ("rgb", "depth", "odometry")

    def __init__(self, thresholds):
        self.thresholds = dict(thresholds)
        self.timing = {name: StreamTiming() for name in self.STREAMS}
        self.metrics = {name: RunningMetric() for name in (
            "blur_score", "depth_valid_ratio", "overexposed_ratio",
            "underexposed_ratio", "rtabmap_latency_ms", "rtabmap_inliers",
            "yaw_rate_rad_s", "map_correction_translation_m",
            "map_correction_rotation_rad",
        )}
        self.tracking = False
        self.tracking_since = None
        self.tracking_true = 0
        self.tracking_total = 0
        self.reset_start = None
        self.reset_current = None
        self.invalid_session = False
        self.pose_jump = False
        self.previous_pose = None
        self.total_distance_m = 0.0
        self.rtabmap_seen = False
        self.loop_closure_count = 0
        self._last_loop_pair = None
        self.revisit_match_count = 0
        self._last_revisit_pair = None
        self.previous_map_correction = None
        self.correction_hold_until = 0.0
        self.route_samples = 0
        self.route_valid_samples = 0

    def observe_stream(self, name, stamp_ns):
        self.timing[name].observe(stamp_ns)

    def observe_tracking(self, value, now):
        value = bool(value)
        self.tracking_total += 1
        self.tracking_true += int(value)
        if value and not self.tracking:
            self.tracking_since = float(now)
        elif not value:
            self.tracking_since = None
        self.tracking = value

    def observe_reset(self, value):
        value = int(value)
        if self.reset_start is None:
            self.reset_start = value
        self.reset_current = value
        if value > self.reset_start:
            self.invalid_session = True

    def observe_pose(self, stamp_ns, x, y, yaw):
        previous_stamp = self.timing["odometry"].last_stamp_ns
        self.observe_stream("odometry", stamp_ns)
        current = (float(x), float(y), float(yaw))
        if self.previous_pose is not None:
            distance = math.hypot(current[0] - self.previous_pose[0],
                                  current[1] - self.previous_pose[1])
            rotation = abs(math.atan2(
                math.sin(current[2] - self.previous_pose[2]),
                math.cos(current[2] - self.previous_pose[2])))
            if (distance > float(self.thresholds["pose_jump_translation_m"]) or
                    rotation > float(self.thresholds["pose_jump_rotation_rad"])):
                self.pose_jump = True
            self.total_distance_m += distance
            if previous_stamp is not None and stamp_ns > previous_stamp:
                self.metrics["yaw_rate_rad_s"].observe(
                    rotation/((stamp_ns-previous_stamp)/1.0e9))
        self.previous_pose = current

    def observe_map_correction(self, x, y, yaw, now):
        current = (float(x), float(y), float(yaw))
        if self.previous_map_correction is not None:
            translation = math.hypot(current[0]-self.previous_map_correction[0],
                                     current[1]-self.previous_map_correction[1])
            rotation = abs(math.atan2(
                math.sin(current[2]-self.previous_map_correction[2]),
                math.cos(current[2]-self.previous_map_correction[2])))
            self.metrics["map_correction_translation_m"].observe(translation)
            self.metrics["map_correction_rotation_rad"].observe(rotation)
            if (translation > float(self.thresholds["map_correction_translation_m"]) or
                    rotation > float(self.thresholds["map_correction_rotation_rad"])):
                self.correction_hold_until = max(
                    self.correction_hold_until,
                    float(now)+float(self.thresholds["recording_hold_seconds"]))
        self.previous_map_correction = current

    def observe_route_sample(self, valid):
        self.route_samples += 1
        self.route_valid_samples += int(bool(valid))

    def observe_rtabmap(self, latency_ms, inliers=None, loop_closure_id=0,
                        reference_id=0, proximity_detection_id=0):
        self.rtabmap_seen = True
        if latency_ms is not None:
            self.metrics["rtabmap_latency_ms"].observe(latency_ms)
        if inliers is not None:
            self.metrics["rtabmap_inliers"].observe(inliers)
        pair = (int(reference_id), int(loop_closure_id))
        if pair[1] > 0 and pair != self._last_loop_pair:
            self.loop_closure_count += 1
            self._last_loop_pair = pair
        revisit = (int(reference_id), int(proximity_detection_id))
        if revisit[1] > 0 and revisit != self._last_revisit_pair:
            self.revisit_match_count += 1
            self._last_revisit_pair = revisit

    def recording_status(self, now):
        reasons = []
        if not self.tracking:
            reasons.append("TRACKING_LOST")
        if self.invalid_session:
            reasons.append("ODOMETRY_RESET")
        if self.pose_jump:
            reasons.append("POSE_JUMP")
        if float(now) < self.correction_hold_until:
            reasons.append("MAP_CORRECTION_HOLD")
        yaw_limit = self.thresholds.get("maximum_recording_yaw_rate_rad_s")
        yaw_rate = self.metrics["yaw_rate_rad_s"].latest
        if yaw_limit is not None and yaw_rate is not None and yaw_rate > float(yaw_limit):
            reasons.append("FAST_TURN_HOLD")
        checks = (("blur_score", "minimum_blur_score", lambda value, limit: value < limit,
                   "BLUR_HOLD"),
                  ("depth_valid_ratio", "minimum_depth_valid_ratio",
                   lambda value, limit: value < limit, "DEPTH_HOLD"),
                  ("overexposed_ratio", "maximum_overexposed_ratio",
                   lambda value, limit: value > limit, "OVEREXPOSURE_HOLD"),
                  ("underexposed_ratio", "maximum_underexposed_ratio",
                   lambda value, limit: value > limit, "UNDEREXPOSURE_HOLD"))
        for metric, setting, failed, reason in checks:
            value, limit = self.metrics[metric].latest, self.thresholds.get(setting)
            if value is not None and limit is not None and failed(value, float(limit)):
                reasons.append(reason)
        return not reasons, reasons

    def state(self, now, odometry_publishers, tf_available, database_opened):
        reasons = []
        if self.invalid_session:
            return False, "INVALID_SESSION", ["RESET_COUNT_INCREASED"]
        if self.pose_jump:
            return False, "POSE_JUMP", ["ODOMETRY_POSE_JUMP"]
        if not self.tracking:
            return False, "TRACKING_LOST", ["TRACKING_VALID_FALSE"]
        if int(odometry_publishers) != 1:
            reasons.append(f"ODOMETRY_PUBLISHER_COUNT:{int(odometry_publishers)}")
        for name in self.STREAMS:
            timing = self.timing[name]
            if timing.count == 0:
                reasons.append(name.upper() + "_MISSING")
            if timing.duplicates:
                reasons.append(name.upper() + "_DUPLICATE_TIMESTAMP")
            if timing.backwards:
                reasons.append(name.upper() + "_BACKWARDS_TIMESTAMP")
        minimums = {
            "rgb": "minimum_rgb_fps",
            "depth": "minimum_depth_fps",
            "odometry": "minimum_odometry_fps",
        }
        for name, setting in minimums.items():
            if self.timing[name].fps < float(self.thresholds[setting]):
                reasons.append(name.upper() + "_FPS_LOW")
        if self.thresholds.get("require_reset_zero") and self.reset_current != 0:
            reasons.append("RESET_COUNT_NOT_ZERO")
        if self.tracking_since is None or float(now) - self.tracking_since < float(
                self.thresholds["ready_tracking_seconds"]):
            reasons.append("TRACKING_NOT_STABLE")
        if self.thresholds.get("require_tf_chain") and not tf_available:
            reasons.append("TF_CHAIN_UNAVAILABLE")
        if not self.rtabmap_seen:
            reasons.append("RTABMAP_INFO_MISSING")
        if not database_opened:
            reasons.append("EXPECTED_DB_NOT_OPENED")
        latency = self.metrics["rtabmap_latency_ms"].latest
        if latency is not None and latency > float(
                self.thresholds["maximum_rtabmap_latency_ms"]):
            return False, "OVERLOADED", [f"RTABMAP_LATENCY_MS:{latency:.3f}"] + reasons
        for metric, setting, state in (
                ("blur_score", "minimum_blur_score", "WARN_BLUR"),
                ("depth_valid_ratio", "minimum_depth_valid_ratio", "WARN_DEPTH")):
            limit = self.thresholds.get(setting)
            value = self.metrics[metric].latest
            if limit is not None and value is not None and value < float(limit):
                return False, state, [f"{metric.upper()}:{value:.6f}"] + reasons
        for metric, setting, state in (
                ("overexposed_ratio", "maximum_overexposed_ratio", "WARN_EXPOSURE"),
                ("underexposed_ratio", "maximum_underexposed_ratio", "WARN_EXPOSURE")):
            limit = self.thresholds.get(setting)
            value = self.metrics[metric].latest
            if limit is not None and value is not None and value > float(limit):
                return False, state, [f"{metric.upper()}:{value:.6f}"] + reasons
        inlier_limit = self.thresholds.get("minimum_rtabmap_inliers")
        inliers = self.metrics["rtabmap_inliers"].latest
        if inlier_limit is not None and inliers is not None and inliers < float(inlier_limit):
            return False, "WARN_RTABMAP_INLIERS", [
                f"RTABMAP_INLIERS:{inliers:.3f}"] + reasons
        if reasons:
            if any("TIMESTAMP" in reason for reason in reasons):
                return False, "INVALID_SESSION", reasons
            return False, "INITIALIZING", reasons
        return True, "READY", []

    def report(self):
        reset_increase = 0
        if self.reset_start is not None and self.reset_current is not None:
            reset_increase = self.reset_current - self.reset_start
        return {
            "streams": {name: {
                "count": timing.count,
                "current_fps": timing.fps,
                "duplicate_timestamps": timing.duplicates,
                "backwards_timestamps": timing.backwards,
                "mean_period_s": timing.periods.report()["mean"],
                "minimum_period_s": timing.periods.report()["min"],
            } for name, timing in self.timing.items()},
            "metrics": {name: metric.report() for name, metric in self.metrics.items()},
            "tracking_true_percent": (
                100.0 * self.tracking_true / self.tracking_total
                if self.tracking_total else 0.0),
            "reset_start": self.reset_start,
            "reset_current": self.reset_current,
            "reset_increase": reset_increase,
            "invalid_session": self.invalid_session,
            "pose_jump": self.pose_jump,
            "total_distance_m": self.total_distance_m,
            "valid_route_point_percent": (
                100.0*self.route_valid_samples/self.route_samples
                if self.route_samples else None),
            "route_samples": self.route_samples,
            "loop_closure_count": self.loop_closure_count,
            "revisit_match_count": self.revisit_match_count,
            "rtabmap_processing_seen": self.rtabmap_seen,
            "maximum_pose_correction_m": self.metrics[
                "map_correction_translation_m"].maximum,
            "maximum_pose_correction_rad": self.metrics[
                "map_correction_rotation_rad"].maximum,
            "uncalibrated_metrics": [metric for metric, setting in (
                ("blur_score", "minimum_blur_score"),
                ("depth_valid_ratio", "minimum_depth_valid_ratio"),
                ("overexposed_ratio", "maximum_overexposed_ratio"),
                ("underexposed_ratio", "maximum_underexposed_ratio"),
                ("rtabmap_inliers", "minimum_rtabmap_inliers"),
            ) if self.thresholds.get(setting) is None],
        }

    def verdict(self, now, odometry_publishers, tf_available, database_opened):
        ready, state, reasons = self.state(
            now, odometry_publishers, tf_available, database_opened)
        report = self.report()
        critical = (report["invalid_session"] or report["pose_jump"] or
                    any(value["count"] == 0 for value in report["streams"].values()))
        if critical:
            return "FAIL", state, reasons
        if report["uncalibrated_metrics"] or not ready:
            return "PARTIAL_PASS", state, reasons
        return "PASS", state, reasons
