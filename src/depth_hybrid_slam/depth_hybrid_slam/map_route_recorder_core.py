"""ROS-independent VSLAM map-frame route recording and validation.

The recorder deliberately has no vehicle-command API.  It keeps every received
pose in an atomic raw log, while only TRACKING/RELOCALIZED map poses are used to
build the distance-resampled follower route.
"""

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import yaml

from .geometry import wrap_angle
from .models import Pose2D, RoutePoint
from .route_follower_core import RouteFollower
from .route_recorder_core import file_sha256


GOOD_STATES = frozenset(("TRACKING", "RELOCALIZED"))
SOURCE = "vslam_localization_recording"
COVARIANCE_FIELDS = tuple(f"covariance_{index:02d}" for index in range(36))
RAW_FIELDS = (
    "timestamp_sec", "timestamp_nanosec", "timestamp", "map_x_m", "map_y_m",
    "map_z_m", "map_yaw_rad", "map_yaw_deg", "frame_id",
    "localization_state", "localization_confidence", "cov_xx", "cov_yy",
    "cov_yawyaw",
) + COVARIANCE_FIELDS + (
    "source", "direction", "recording_state", "route_eligible",
    "segment_break",
)
FINAL_FIELDS = (
    "timestamp_sec", "timestamp_nanosec", "route_index", "map_x_m",
    "map_y_m", "map_z_m", "map_yaw_rad", "map_yaw_deg", "yaw_rad",
    "yaw_deg", "cumulative_distance_m",
    "direction", "mode", "drive_level", "event", "segment_id", "source",
    "localization_state", "localization_confidence", "cov_xx", "cov_yy",
    "cov_yawyaw", "valid",
)


def _sync(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _atomic_replace_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        _sync(stream)
    os.replace(temporary, path)


def _atomic_yaml(path, values):
    _atomic_replace_text(
        path, yaml.safe_dump(values, sort_keys=False, allow_unicode=True))


def _atomic_json(path, values):
    _atomic_replace_text(path, json.dumps(values, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class MapPoseSample:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float
    frame_id: str
    localization_state: str
    confidence: float
    covariance: tuple
    direction: int = 1
    segment: int = 1

    def __post_init__(self):
        if len(self.covariance) != 36:
            raise ValueError("PoseWithCovariance covariance must contain 36 values")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be 1 or -1")

    @property
    def eligible(self):
        return (
            self.frame_id == "map" and
            self.localization_state in GOOD_STATES and
            all(math.isfinite(value) for value in
                (self.x, self.y, self.z, self.yaw, self.confidence))
        )


def _interpolate(first, second, ratio):
    stamp = round(first.stamp_ns + ratio * (second.stamp_ns-first.stamp_ns))
    yaw = wrap_angle(first.yaw + ratio * wrap_angle(second.yaw-first.yaw))
    covariance = tuple(
        a + ratio * (b-a) for a, b in zip(first.covariance, second.covariance))
    return replace(
        first,
        stamp_ns=int(stamp),
        x=first.x + ratio*(second.x-first.x),
        y=first.y + ratio*(second.y-first.y),
        z=first.z + ratio*(second.z-first.z),
        yaw=yaw,
        confidence=first.confidence + ratio*(second.confidence-first.confidence),
        covariance=covariance,
    )


def resample_by_distance(samples, spacing_m=0.10, duplicate_distance_m=0.01):
    """Remove stationary duplicates and resample each explicit segment."""
    if spacing_m <= 0.0 or duplicate_distance_m < 0.0:
        raise ValueError("distance thresholds must be positive")
    output = []
    segment_ids = []
    for segment in sorted({sample.segment for sample in samples}):
        source = [sample for sample in samples if sample.segment == segment]
        clean = []
        for sample in source:
            if (not clean or sample.direction != clean[-1].direction or
                    math.hypot(sample.x-clean[-1].x, sample.y-clean[-1].y) >=
                    duplicate_distance_m):
                clean.append(sample)
        if len(clean) < 2:
            continue
        # Direction changes are route discontinuities even without a pose jump.
        runs = []
        run = [clean[0]]
        for sample in clean[1:]:
            if sample.direction != run[-1].direction:
                runs.append(run)
                run = [sample]
            else:
                run.append(sample)
        runs.append(run)
        for run in runs:
            if len(run) < 2:
                continue
            run_id = len(segment_ids) + 1
            segment_ids.append(run_id)
            cumulative = [0.0]
            for first, second in zip(run, run[1:]):
                cumulative.append(cumulative[-1] + math.hypot(
                    second.x-first.x, second.y-first.y))
            if cumulative[-1] <= 0.0:
                continue
            targets = [index*spacing_m for index in
                       range(int(cumulative[-1]/spacing_m)+1)]
            if cumulative[-1]-targets[-1] > 1.0e-9:
                targets.append(cumulative[-1])
            cursor = 0
            for target in targets:
                while cursor+1 < len(cumulative) and cumulative[cursor+1] < target:
                    cursor += 1
                if cursor+1 >= len(run):
                    point = run[-1]
                else:
                    length = cumulative[cursor+1]-cumulative[cursor]
                    ratio = 0.0 if length <= 0.0 else (
                        target-cumulative[cursor])/length
                    point = _interpolate(run[cursor], run[cursor+1], ratio)
                output.append(replace(point, segment=run_id))
    # CSV loader requires strict timestamps.  Preserve acquisition timing when
    # possible and minimally nudge ties caused by interpolation rounding.
    fixed = []
    previous = None
    for sample in output:
        stamp = sample.stamp_ns if previous is None else max(
            sample.stamp_ns, previous+1)
        fixed.append(replace(sample, stamp_ns=stamp))
        previous = stamp
    return fixed


def _within_segment_pairs(points):
    return [(first, second) for first, second in zip(points, points[1:])
            if first.segment == second.segment]


def _curvatures(points, window=5):
    values = []
    for index, point in enumerate(points):
        left = max(0, index-window)
        right = min(len(points)-1, index+window)
        if (points[left].segment != point.segment or
                points[right].segment != point.segment or right-left < 2):
            values.append(0.0)
            continue
        a, b, c = points[left], point, points[right]
        ab = math.hypot(b.x-a.x, b.y-a.y)
        bc = math.hypot(c.x-b.x, c.y-b.y)
        ac = math.hypot(c.x-a.x, c.y-a.y)
        denominator = ab*bc*ac
        cross = (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)
        values.append(0.0 if denominator <= 1.0e-9 else 2.0*cross/denominator)
    return values


def synthetic_follower_validation(points, wheelbase_m=0.73,
                                  max_steering_deg=22.0):
    """Replay exact route poses through the existing follower, per segment."""
    results = []
    for segment in sorted({point.segment for point in points}):
        source = [point for point in points if point.segment == segment]
        if len(source) < 3:
            results.append({"segment": segment, "passed": False,
                            "reason": "TOO_FEW_POINTS"})
            continue
        route = [RoutePoint(
            index, point.x, point.y, point.yaw, point.direction, 1.0,
            "", f"RECORDED_A_{segment:03d}", f"RECORDED_A_{segment:03d}",
            "RECORDED", index, 0.0, 0.0, 0, "NONE")
            for index, point in enumerate(source)]
        follower = RouteFollower(
            wheelbase=wheelbase_m, max_steering_deg=max_steering_deg,
            max_index_backtrack=0)
        nearest = []
        deviations = []
        steerings = []
        final_reason = ""
        for point in source:
            result = follower.compute(
                Pose2D(point.x, point.y, point.yaw, point.stamp_ns/1.0e9),
                route, allow_motion=True)
            nearest.append(result.nearest_index)
            deviations.append(abs(result.cross_track_error))
            steerings.append(abs(result.steering_deg))
            final_reason = result.reason
        backtracks = sum(second < first for first, second in zip(
            nearest, nearest[1:]))
        passed = (
            backtracks == 0 and max(deviations, default=0.0) <= 1.0e-6 and
            max(steerings, default=0.0) <= max_steering_deg+1.0e-9 and
            final_reason == "ROUTE_COMPLETE"
        )
        results.append({
            "segment": segment, "passed": passed,
            "index_backtracks": backtracks,
            "max_deviation_m": max(deviations, default=0.0),
            "max_commanded_steering_deg": max(steerings, default=0.0),
            "final_reason": final_reason,
        })
    return {
        "passed": bool(results) and all(item["passed"] for item in results),
        "segments": results,
        "final_reason": "ROUTE_COMPLETE" if results and all(
            item["final_reason"] == "ROUTE_COMPLETE" for item in results)
        else "SYNTHETIC_TRAVERSAL_FAILED",
    }


def quality_report(points, raw_count, dropout_count, segment_break_count,
                   spacing_m=0.10, wheelbase_m=0.73,
                   max_steering_deg=22.0):
    pairs = _within_segment_pairs(points)
    gaps = [math.hypot(b.x-a.x, b.y-a.y) for a, b in pairs]
    yaw_jumps = [abs(wrap_angle(b.yaw-a.yaw)) for a, b in pairs]
    curvatures = _curvatures(points)
    steering = [abs(math.degrees(math.atan(wheelbase_m*value)))
                for value in curvatures]
    cov_x = [point.covariance[0] for point in points]
    cov_y = [point.covariance[7] for point in points]
    cov_yaw = [point.covariance[35] for point in points]
    positive = [(value, index) for index, value in enumerate(curvatures)
                if value > 0.02]
    negative = [(value, index) for index, value in enumerate(curvatures)
                if value < -0.02]
    first_peak = max(positive, default=(0.0, -1))
    second_peak = min(negative, default=(0.0, -1))
    if first_peak[1] > second_peak[1] >= 0:
        first_peak, second_peak = second_peak, first_peak
    transition = -1
    if first_peak[1] >= 0 and second_peak[1] > first_peak[1]:
        transition = min(
            range(first_peak[1], second_peak[1]+1),
            key=lambda index: abs(curvatures[index]))
    synthetic = synthetic_follower_validation(
        points, wheelbase_m, max_steering_deg)
    finite = all(math.isfinite(value) for point in points for value in
                 (point.x, point.y, point.z, point.yaw))
    indexes_ok = True  # output writer owns a contiguous enumerate() index.
    max_gap = max(gaps, default=0.0)
    max_yaw = max(yaw_jumps, default=0.0)
    max_required = max(steering, default=0.0)
    checks = {
        "route_has_at_least_three_points": len(points) >= 3,
        "no_nan_or_inf": finite,
        "route_index_monotonic": indexes_ok,
        "max_gap_within_0_20_m": max_gap <= 0.20+1.0e-9,
        "max_yaw_jump_within_30_deg": max_yaw <= math.radians(30.0),
        "required_steering_within_limit": max_required <= max_steering_deg+1.0e-9,
        "synthetic_traversal": synthetic["passed"],
        "route_complete": synthetic["final_reason"] == "ROUTE_COMPLETE",
    }
    total = sum(gaps)
    return {
        "automated_pass": all(checks.values()),
        "checks": checks,
        "raw_sample_count": int(raw_count),
        "route_waypoint_count": len(points),
        "total_distance_m": total,
        "spacing_m": {"requested": spacing_m,
                      "mean": sum(gaps)/len(gaps) if gaps else 0.0,
                      "maximum": max_gap},
        "maximum_yaw_jump_rad": max_yaw,
        "maximum_yaw_jump_deg": math.degrees(max_yaw),
        "localization_dropout_count": int(dropout_count),
        "segment_break_count": int(segment_break_count),
        "covariance": {
            "xx_mean": sum(cov_x)/len(cov_x) if cov_x else None,
            "xx_max": max(cov_x, default=None),
            "yy_mean": sum(cov_y)/len(cov_y) if cov_y else None,
            "yy_max": max(cov_y, default=None),
            "yawyaw_mean": sum(cov_yaw)/len(cov_yaw) if cov_yaw else None,
            "yawyaw_max": max(cov_yaw, default=None),
        },
        "vehicle_geometry": {
            "wheelbase_m": wheelbase_m,
            "max_steering_deg": max_steering_deg,
            "maximum_required_steering_deg": max_required,
        },
        "s_curve": {
            "detected": first_peak[1] >= 0 and second_peak[1] >= 0,
            "entry_index": max(0, first_peak[1]-5) if first_peak[1] >= 0 else -1,
            "first_curvature_peak_index": first_peak[1],
            "curvature_transition_index": transition,
            "second_curvature_peak_index": second_peak[1],
            "exit_index": min(len(points)-1, second_peak[1]+5)
            if second_peak[1] >= 0 else -1,
        },
        "synthetic_follower": synthetic,
    }


class MapRouteRecordingSession:
    """One explicitly started, paused and finished recording session."""

    def __init__(self, output_directory, map_path, *, timestamp_tag=None,
                 spacing_m=0.10, duplicate_distance_m=0.01,
                 recovery_jump_m=0.75, wheelbase_m=0.73,
                 max_steering_deg=22.0, gps_reference_path=""):
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.map_path = Path(map_path).expanduser().resolve()
        if not self.map_path.is_file():
            raise FileNotFoundError(f"map database does not exist: {self.map_path}")
        tag = timestamp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_path = self.output_directory/f"map_route_raw_{tag}.csv"
        self.route_path = self.output_directory/f"map_route_A_{tag}.csv"
        self.metadata_path = self.output_directory/f"map_route_A_{tag}.metadata.yaml"
        self.report_path = self.output_directory/f"map_route_A_{tag}.quality.json"
        self.overview_plot_path = self.output_directory/f"map_route_A_{tag}_overview.png"
        self.s_curve_plot_path = self.output_directory/f"map_route_A_{tag}_s_curve.png"
        self.gps_comparison_plot_path = self.output_directory/f"map_route_A_{tag}_gps_reference.png"
        self.gps_reference_path = Path(gps_reference_path).expanduser().resolve() \
            if gps_reference_path else None
        self.spacing_m = float(spacing_m)
        self.duplicate_distance_m = float(duplicate_distance_m)
        self.recovery_jump_m = float(recovery_jump_m)
        self.wheelbase_m = float(wheelbase_m)
        self.max_steering_deg = float(max_steering_deg)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        planned = (self.raw_path, self.route_path, self.metadata_path,
                   self.report_path, self.overview_plot_path,
                   self.s_curve_plot_path, self.gps_comparison_plot_path)
        collision = next((path for path in planned
                          if path.exists() or path.with_suffix(path.suffix+".partial").exists()), None)
        if collision:
            raise FileExistsError(f"refusing to overwrite recording output: {collision}")
        self._raw_partial = self.raw_path.with_suffix(".csv.partial")
        self._raw_stream = self._raw_partial.open("x", newline="", encoding="utf-8")
        self._raw_writer = csv.DictWriter(self._raw_stream, fieldnames=RAW_FIELDS)
        self._raw_writer.writeheader()
        _sync(self._raw_stream)
        self.state = "RECORDING"
        self.direction = 1
        self.samples = []
        self.raw_count = 0
        self.dropout_count = 0
        self.segment_break_count = 0
        self.segment = 1
        self._dropout_active = False
        self._pending_break = False
        self._last_eligible = None
        self.report = None

    def pause(self):
        if self.state != "RECORDING":
            raise RuntimeError("recording is not active")
        self.state = "PAUSED"

    def resume(self):
        if self.state != "PAUSED":
            raise RuntimeError("recording is not paused")
        self.state = "RECORDING"

    def set_direction(self, direction):
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("direction must be 1 or -1")
        if direction != self.direction and self.samples:
            self._pending_break = True
        self.direction = direction

    def notify_pose_timeout(self):
        if self.state == "RECORDING" and not self._dropout_active:
            self.dropout_count += 1
            self._dropout_active = True

    def append(self, sample):
        if self.state not in ("RECORDING", "PAUSED"):
            raise RuntimeError("recording has finished")
        sample = replace(sample, direction=self.direction, segment=self.segment)
        eligible = self.state == "RECORDING" and sample.eligible
        segment_break = False
        if not eligible:
            if self.state == "RECORDING" and not self._dropout_active:
                self.dropout_count += 1
                self._dropout_active = True
        elif self._last_eligible is not None:
            jump = math.hypot(
                sample.x-self._last_eligible.x, sample.y-self._last_eligible.y)
            if self._pending_break or (self._dropout_active and jump > self.recovery_jump_m):
                self.segment += 1
                self.segment_break_count += 1
                segment_break = True
                sample = replace(sample, segment=self.segment)
            self._pending_break = False
            self._dropout_active = False
        elif eligible:
            self._dropout_active = False
        row = {
            "timestamp_sec": sample.stamp_ns//1_000_000_000,
            "timestamp_nanosec": sample.stamp_ns % 1_000_000_000,
            "timestamp": sample.stamp_ns/1.0e9,
            "map_x_m": sample.x, "map_y_m": sample.y, "map_z_m": sample.z,
            "map_yaw_rad": sample.yaw, "map_yaw_deg": math.degrees(sample.yaw),
            "frame_id": sample.frame_id,
            "localization_state": sample.localization_state,
            "localization_confidence": sample.confidence,
            "cov_xx": sample.covariance[0], "cov_yy": sample.covariance[7],
            "cov_yawyaw": sample.covariance[35], "source": SOURCE,
            "direction": sample.direction, "recording_state": self.state,
            "route_eligible": eligible, "segment_break": segment_break,
        }
        row.update({name: sample.covariance[index]
                    for index, name in enumerate(COVARIANCE_FIELDS)})
        self._raw_writer.writerow(row)
        _sync(self._raw_stream)
        self.raw_count += 1
        if eligible:
            self.samples.append(sample)
            self._last_eligible = sample
        return eligible, segment_break

    def _write_route(self, points):
        partial = self.route_path.with_suffix(".csv.partial")
        cumulative = 0.0
        previous = None
        with partial.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FINAL_FIELDS)
            writer.writeheader()
            for index, point in enumerate(points):
                if previous is not None and previous.segment == point.segment:
                    cumulative += math.hypot(point.x-previous.x, point.y-previous.y)
                writer.writerow({
                    "timestamp_sec": point.stamp_ns//1_000_000_000,
                    "timestamp_nanosec": point.stamp_ns % 1_000_000_000,
                    "route_index": index, "map_x_m": point.x,
                    "map_y_m": point.y, "map_z_m": point.z,
                    "map_yaw_rad": point.yaw,
                    "map_yaw_deg": math.degrees(point.yaw),
                    "yaw_rad": point.yaw, "yaw_deg": math.degrees(point.yaw),
                    "cumulative_distance_m": cumulative,
                    "direction": point.direction, "mode": 0,
                    "drive_level": 1.0, "event": "NONE",
                    "segment_id": f"RECORDED_A_{point.segment:03d}",
                    "source": SOURCE,
                    "localization_state": point.localization_state,
                    "localization_confidence": point.confidence,
                    "cov_xx": point.covariance[0], "cov_yy": point.covariance[7],
                    "cov_yawyaw": point.covariance[35], "valid": True,
                })
                previous = point
            _sync(stream)
        os.replace(partial, self.route_path)

    def _write_plots(self, points, report):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"generated": False, "reason": "MATPLOTLIB_NOT_AVAILABLE"}
        x, y = [point.x for point in points], [point.y for point in points]
        figure, axis = plt.subplots(figsize=(9, 7))
        for segment in sorted({point.segment for point in points}):
            selected = [point for point in points if point.segment == segment]
            axis.plot([point.x for point in selected], [point.y for point in selected],
                      label=f"RECORDED_A_{segment:03d}")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("VSLAM map-frame recorded A route")
        axis.set_xlabel("map x [m]"); axis.set_ylabel("map y [m]")
        axis.grid(True); axis.legend()
        figure.tight_layout(); figure.savefig(self.overview_plot_path, dpi=150)
        plt.close(figure)
        curvature = _curvatures(points)
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.plot(range(len(points)), curvature, label="signed curvature [1/m]")
        names = (("entry", "entry_index"), ("peak 1", "first_curvature_peak_index"),
                 ("transition", "curvature_transition_index"),
                 ("peak 2", "second_curvature_peak_index"), ("exit", "exit_index"))
        for label, key in names:
            index = report["s_curve"][key]
            if index >= 0:
                axis.axvline(index, linestyle="--", label=label)
        axis.set_title("Recorded-route S-curve continuity/features (no GPS fit)")
        axis.set_xlabel("route index"); axis.grid(True); axis.legend(ncol=3)
        figure.tight_layout(); figure.savefig(self.s_curve_plot_path, dpi=150)
        plt.close(figure)
        gps_generated = False
        if self.gps_reference_path and self.gps_reference_path.is_file():
            with self.gps_reference_path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            gps_x = [float(row["x_m"]) for row in rows]
            gps_y = [float(row["y_m"]) for row in rows]
            figure, axes = plt.subplots(1, 2, figsize=(13, 6))
            axes[0].plot(x, y); axes[0].set_title("Recorded VSLAM map route")
            axes[1].plot(gps_x, gps_y); axes[1].set_title(
                "Legacy GPS route (reference only; native coordinates)")
            for axis in axes:
                axis.set_aspect("equal", adjustable="box"); axis.grid(True)
            figure.suptitle("Structure-only comparison; not an alignment/failure metric")
            figure.tight_layout(); figure.savefig(
                self.gps_comparison_plot_path, dpi=150)
            plt.close(figure)
            gps_generated = True
        return {"generated": True, "overview": str(self.overview_plot_path),
                "s_curve": str(self.s_curve_plot_path),
                "gps_reference_generated": gps_generated,
                "gps_reference": str(self.gps_comparison_plot_path)
                if gps_generated else ""}

    def finish(self):
        if self.state not in ("RECORDING", "PAUSED"):
            raise RuntimeError("recording has already finished")
        _sync(self._raw_stream)
        self._raw_stream.close()
        os.replace(self._raw_partial, self.raw_path)
        points = resample_by_distance(
            self.samples, self.spacing_m, self.duplicate_distance_m)
        self._write_route(points)
        report = quality_report(
            points, self.raw_count, self.dropout_count, self.segment_break_count,
            self.spacing_m, self.wheelbase_m, self.max_steering_deg)
        plots = self._write_plots(points, report)
        report["plots"] = plots
        report["gps_reference_is_failure_metric"] = False
        _atomic_json(self.report_path, report)
        route_hash = file_sha256(self.route_path)
        map_hash = file_sha256(self.map_path)
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "finalized": True,
            "frame_id": "map",
            "route_coordinate_frame": "map",
            "source": SOURCE,
            "pose_topic": "/depth_slam/localization/pose",
            "pose_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
            "raw_route_path": str(self.raw_path),
            "route_path": str(self.route_path),
            "quality_report_path": str(self.report_path),
            "map_db_path": str(self.map_path),
            "rtabmap_db_sha256": map_hash,
            "route_csv_sha256": route_hash,
            "raw_route_csv_sha256": file_sha256(self.raw_path),
            "alignment": {"method": "none", "required": False,
                          "required_for_runtime": False, "validated": False},
            "route_validation": {
                "automated_pass": report["automated_pass"],
                "rviz_reviewed": False,
                "validated": False,
                "reason": "PENDING_RVIZ_REVIEW" if report["automated_pass"]
                else "AUTOMATED_QUALITY_FAILED",
            },
            "resampling": {"method": "distance", "spacing_m": self.spacing_m,
                           "stationary_duplicate_threshold_m": self.duplicate_distance_m},
            "segment_break_jump_threshold_m": self.recovery_jump_m,
            "vehicle_geometry": {"wheelbase_m": self.wheelbase_m,
                                 "max_steering_deg": self.max_steering_deg},
        }
        _atomic_yaml(self.metadata_path, metadata)
        self.state = "FINISHED"
        self.report = report
        return {"raw_path": str(self.raw_path), "route_path": str(self.route_path),
                "metadata_path": str(self.metadata_path),
                "report_path": str(self.report_path), **report}

    def close(self):
        if not self._raw_stream.closed:
            _sync(self._raw_stream)
            self._raw_stream.close()


def approve_rviz_review(metadata_path, route_path="", map_path=""):
    """Explicitly approve a visually reviewed route; never infers RViz success."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    with metadata_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    route = Path(route_path or values.get("route_path", "")).expanduser().resolve()
    database = Path(map_path or values.get("map_db_path", "")).expanduser().resolve()
    validation = values.get("route_validation", {}) or {}
    if not validation.get("automated_pass", False):
        raise ValueError("automated route quality has not passed")
    if file_sha256(route) != str(values.get("route_csv_sha256", "")):
        raise ValueError("route checksum mismatch")
    if file_sha256(database) != str(values.get("rtabmap_db_sha256", "")):
        raise ValueError("map checksum mismatch")
    validation.update({"rviz_reviewed": True, "validated": True, "reason": "PASS"})
    values["route_validation"] = validation
    alignment = values.get("alignment", {}) or {}
    alignment.update({"method": "none", "required": False,
                      "required_for_runtime": False, "validated": True})
    values["alignment"] = alignment
    _atomic_yaml(metadata_path, values)
    return values
