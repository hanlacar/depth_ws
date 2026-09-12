import ast
import csv
import math
from pathlib import Path

from geometry_msgs.msg import PoseWithCovarianceStamped
import pytest
import yaml

from depth_hybrid_slam.map_route_recorder_core import (
    COVARIANCE_FIELDS, FINAL_FIELDS, GOOD_STATES, RAW_FIELDS,
    MapPoseSample, MapRouteRecordingSession, approve_rviz_review,
    quality_report, resample_by_distance, synthetic_follower_validation,
)
from depth_hybrid_slam.map_route_recorder_node import map_pose_sample
from depth_hybrid_slam.route_io import load_map_route, verify_route_binding


def sample(index, x, y=0.0, yaw=0.0, state="TRACKING", frame="map"):
    covariance = [0.0] * 36
    covariance[0], covariance[7], covariance[35] = 0.01, 0.02, 0.03
    return MapPoseSample(
        1_000_000_000 + index*100_000_000, x, y, 0.0, yaw, frame,
        state, 0.9, tuple(covariance))


def session(tmp_path, **kwargs):
    database = tmp_path/"rtabmap.db"
    database.write_bytes(b"immutable-test-map")
    return MapRouteRecordingSession(
        tmp_path/"output", database, timestamp_tag="20260911_120000",
        gps_reference_path="", **kwargs), database


def test_actual_pose_message_parser_preserves_map_yaw_and_covariance():
    message = PoseWithCovarianceStamped()
    message.header.frame_id = "map"
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 34
    message.pose.pose.position.x = 1.25
    message.pose.pose.position.y = -2.5
    yaw = 1.1
    message.pose.pose.orientation.z = math.sin(yaw/2.0)
    message.pose.pose.orientation.w = math.cos(yaw/2.0)
    message.pose.covariance[0] = 0.1
    parsed = map_pose_sample(message, "tracking", 0.8)
    assert parsed.stamp_ns == 12_000_000_034
    assert parsed.frame_id == "map"
    assert parsed.yaw == pytest.approx(yaw)
    assert parsed.covariance[0] == pytest.approx(0.1)
    assert parsed.eligible


def test_only_tracking_and_relocalized_are_route_eligible():
    assert GOOD_STATES == {"TRACKING", "RELOCALIZED"}
    assert sample(0, 0.0, state="TRACKING").eligible
    assert sample(0, 0.0, state="RELOCALIZED").eligible
    assert not sample(0, 0.0, state="LOST").eligible
    assert not sample(0, 0.0, state="DEGRADED").eligible
    assert not sample(0, 0.0, frame="odom").eligible


def test_distance_resampling_and_stationary_duplicate_rejection():
    source = [sample(index, index*0.02) for index in range(31)]
    source.insert(10, source[9])
    route = resample_by_distance(source, spacing_m=0.10)
    gaps = [math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(route, route[1:])]
    assert len(route) == 7
    assert gaps == pytest.approx([0.1]*6)


def test_short_final_remainder_is_kept_without_large_gap():
    route = resample_by_distance([sample(0, 0.0), sample(1, 0.26)])
    assert [point.x for point in route] == pytest.approx([0.0, 0.1, 0.2, 0.26])


def test_lost_samples_stay_in_raw_but_not_final(tmp_path):
    recorder, _ = session(tmp_path)
    recorder.append(sample(0, 0.0))
    eligible, _ = recorder.append(sample(1, 0.1, state="LOST"))
    recorder.append(sample(2, 0.2))
    assert not eligible
    assert recorder.raw_count == 3
    assert len(recorder.samples) == 2
    recorder.close()


def test_pose_timeout_counts_one_dropout_until_recovery(tmp_path):
    recorder, _ = session(tmp_path)
    recorder.append(sample(0, 0.0))
    recorder.notify_pose_timeout()
    recorder.notify_pose_timeout()
    recorder.append(sample(1, 0.1))
    assert recorder.dropout_count == 1
    recorder.close()


def test_large_recovery_jump_creates_segment_break(tmp_path):
    recorder, _ = session(tmp_path, recovery_jump_m=0.5)
    recorder.append(sample(0, 0.0))
    recorder.append(sample(1, 0.1, state="LOST"))
    accepted, segment_break = recorder.append(sample(2, 1.0))
    assert accepted and segment_break
    assert recorder.segment_break_count == 1
    assert recorder.samples[-1].segment == 2
    recorder.close()


def test_small_recovery_jump_continues_same_segment(tmp_path):
    recorder, _ = session(tmp_path, recovery_jump_m=0.5)
    recorder.append(sample(0, 0.0))
    recorder.append(sample(1, 0.1, state="LOST"))
    _, segment_break = recorder.append(sample(2, 0.2))
    assert not segment_break
    assert recorder.segment_break_count == 0
    recorder.close()


def test_direction_change_creates_distinct_final_segment(tmp_path):
    recorder, _ = session(tmp_path)
    recorder.append(sample(0, 0.0))
    recorder.append(sample(1, 0.2))
    recorder.set_direction(-1)
    recorder.append(sample(2, 0.4, yaw=math.pi))
    recorder.append(sample(3, 0.2, yaw=math.pi))
    assert recorder.segment_break_count == 1
    assert recorder.samples[-1].direction == -1
    recorder.close()


def test_pause_resume_excludes_paused_route_point_but_keeps_raw(tmp_path):
    recorder, _ = session(tmp_path)
    recorder.append(sample(0, 0.0))
    recorder.pause()
    accepted, _ = recorder.append(sample(1, 0.1))
    recorder.resume()
    recorder.append(sample(2, 0.2))
    assert not accepted
    assert recorder.raw_count == 3
    assert len(recorder.samples) == 2
    recorder.close()


def test_finish_csv_schemas_round_trip_and_metadata_gate(tmp_path):
    recorder, database = session(tmp_path)
    for index in range(21):
        recorder.append(sample(index, index*0.05, yaw=0.2))
    summary = recorder.finish()
    with Path(summary["raw_path"]).open(newline="") as stream:
        raw_reader = csv.DictReader(stream)
        assert tuple(raw_reader.fieldnames) == RAW_FIELDS
        raw_rows = list(raw_reader)
    assert len(raw_rows) == 21
    assert tuple(name for name in COVARIANCE_FIELDS if name in raw_rows[0]) == COVARIANCE_FIELDS
    with Path(summary["route_path"]).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == FINAL_FIELDS
        rows = list(reader)
    assert [int(row["route_index"]) for row in rows] == list(range(len(rows)))
    assert all(row["segment_id"] == "RECORDED_A_001" for row in rows)
    assert all(row["mode"] == "0" and row["event"] == "NONE" for row in rows)
    assert all(float(row["map_yaw_rad"]) == pytest.approx(0.2) for row in rows)
    loaded = load_map_route(summary["route_path"])
    assert len(loaded) == len(rows)
    assert all(point["yaw"] == pytest.approx(0.2) for point in loaded)
    valid, reason = verify_route_binding(
        summary["route_path"], database, summary["metadata_path"])
    assert not valid and reason == "MAP_ROUTE_NOT_VALIDATED"
    approve_rviz_review(summary["metadata_path"])
    valid, reason = verify_route_binding(
        summary["route_path"], database, summary["metadata_path"])
    assert valid and reason == "VERIFIED"
    with Path(summary["metadata_path"]).open() as stream:
        metadata = yaml.safe_load(stream)
    assert metadata["frame_id"] == "map"
    assert metadata["alignment"] == {
        "method": "none", "required": False,
        "required_for_runtime": False, "validated": True}


def test_output_files_are_never_overwritten(tmp_path):
    recorder, database = session(tmp_path)
    recorder.close()
    with pytest.raises(FileExistsError):
        MapRouteRecordingSession(
            tmp_path/"output", database, timestamp_tag="20260911_120000",
            gps_reference_path="")


def test_vehicle_geometry_rejects_too_tight_recorded_curve():
    radius = 1.0
    points = [sample(index, radius*math.cos(index*0.04),
                     radius*math.sin(index*0.04), index*0.04+math.pi/2)
              for index in range(80)]
    report = quality_report(points, len(points), 0, 0)
    assert report["vehicle_geometry"]["maximum_required_steering_deg"] > 22.0
    assert not report["checks"]["required_steering_within_limit"]


def test_s_curve_features_and_synthetic_traversal():
    points = []
    for index in range(121):
        x = index*0.1
        y = 0.6*math.sin(x*0.55)
        dydx = 0.33*math.cos(x*0.55)
        points.append(sample(index, x, y, math.atan2(dydx, 1.0)))
    report = quality_report(points, len(points), 0, 0)
    assert report["s_curve"]["detected"]
    assert synthetic_follower_validation(points)["passed"]
    assert report["synthetic_follower"]["final_reason"] == "ROUTE_COMPLETE"


def test_recorder_node_has_no_vehicle_command_publishers():
    source_path = Path(__file__).parents[1]/"depth_hybrid_slam"/"map_route_recorder_node.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    published_literals = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == "create_publisher"):
            published_literals.extend(
                value.value for value in ast.walk(node)
                if isinstance(value, ast.Constant) and isinstance(value.value, str))
    forbidden = ("/slam_drive", "/slam_wheel", "/camera_drive", "/camera_wheel",
                 "/gps_drive", "/gps_wheel")
    assert not set(forbidden) & set(published_literals)
    assert "/depth_slam/recorded_route/raw_path" in published_literals
    assert "/depth_slam/recorded_route/resampled_path" in published_literals


def test_recording_launch_defaults_to_manual_direction_and_no_control():
    launch = (Path(__file__).parents[1]/"launch"/
              "map_route_recording.launch.py").read_text()
    assert 'DeclareLaunchArgument("use_drive_command_direction", default_value="false")' in launch
    assert '"wheelbase_m": 0.73, "max_steering_deg": 22.0' in launch
    assert "enable_control" not in launch
