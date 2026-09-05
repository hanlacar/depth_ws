import csv
from pathlib import Path

from depth_hybrid_slam.route_io import load_map_route
from depth_hybrid_slam.route_recorder_core import RouteRecorder


def sample(stamp, x=0.0, yaw=0.0, marker="", state="TRACKING"):
    return {"timestamp": stamp, "x": x, "y": 0, "z": 0, "roll": 0,
            "pitch": 0, "yaw": yaw, "direction": 1, "mode": "MANUAL",
            "drive_level": 1, "steering_command": 0, "measured_steering": 0,
            "localization_confidence": .9, "tracking_state": state,
            "mission_marker": marker, "stop_line_id": "", "section_id": "a"}


def test_dedup_flush_finalize_and_invalid_segment(tmp_path):
    path = tmp_path/"route.csv"
    recorder = RouteRecorder(path)
    assert recorder.append(sample(1.0))
    assert not recorder.append(sample(1.01))
    assert recorder.append(sample(1.02, marker="stop"))
    assert recorder.append(sample(1.03, state="LOST"))
    assert recorder.partial.exists()
    recorder.finalize()
    assert path.exists() and not recorder.partial.exists()
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and rows[-1]["record_valid"] == "False"
    assert rows[0]["map_x_m"] == "0.0" and rows[0]["route_index"] == "0"


def test_map_frame_sampling_distance_rotation_and_invariants(tmp_path):
    path = tmp_path / "route.csv"
    recorder = RouteRecorder(path, min_distance_m=0.05, min_angle_rad=0.03)
    assert recorder.append(sample(1.0))
    assert not recorder.append(sample(1.1, x=0.049, yaw=0.029))
    assert recorder.append(sample(1.2, x=0.05))
    assert recorder.append(sample(1.3, x=0.05, yaw=0.03))
    result = recorder.finalize()
    assert result["point_count"] == 3 and result["total_distance_m"] == 0.05


def test_non_map_invalid_pose_is_not_marked_valid(tmp_path):
    recorder = RouteRecorder(tmp_path / "route.csv")
    value = sample(1.0)
    value.update(frame_id="odom", tracking_valid=True)
    assert recorder.append(value)
    recorder.finalize()
    with open(tmp_path / "route.csv", newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["valid"] == "False"


def test_nan_duplicate_and_backwards_are_rejected(tmp_path):
    recorder = RouteRecorder(tmp_path / "route.csv")
    assert recorder.append(sample(1.0))
    for bad in (sample(1.0, x=1.0), sample(0.9, x=1.0), sample(1.1, x=float("nan"))):
        try:
            recorder.append(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid route sample accepted")
    recorder.close()


def test_required_columns_future_blanks_and_offline_loader(tmp_path):
    path = tmp_path / "route.csv"
    recorder = RouteRecorder(path)
    assert recorder.append(sample(1.0))
    recorder.finalize()
    with open(path, newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    required = {"timestamp_sec", "timestamp_nanosec", "route_index",
                "map_x_m", "map_y_m", "map_z_m", "yaw_rad", "yaw_deg",
                "cumulative_distance_m", "direction", "localization_state",
                "localization_confidence", "tracking_valid", "reset_count",
                "nearest_rtabmap_node_id", "pose_source", "valid"}
    assert required.issubset(row)
    assert all(row[name] == "" for name in
               ("speed_mps", "encoder", "wheel_odom_x", "imu_roll"))
    points = load_map_route(path)
    assert len(points) == 1 and points[0]["x"] == 0.0


def test_slam_capture_and_vehicle_route_topics_are_separate():
    root = Path(__file__).resolve().parents[1]
    rviz = (root / "config" / "map_route_high_density.rviz").read_text()
    assert "/rtabmap/mapPath" in rviz
    assert "/depth_slam/route/path" in rviz
    assert "SLAM Capture Trajectory" in rviz
    assert "Vehicle Reference Route" in rviz
