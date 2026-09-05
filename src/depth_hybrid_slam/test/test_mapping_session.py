from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

from depth_hybrid_slam.mapping_session import (
    create_mapping_session, finalize_mapping_session, load_mapping_profile,
    mapping_graph_conflicts, mapping_target_conflicts, PROTECTED_MAP_DIRS,
    rtabmap_argument_string, sha256, write_session_metadata,
)
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def session_roots(tmp_path):
    return {
        "maps_root": tmp_path / "maps",
        "routes_root": tmp_path / "routes",
        "bags_root": tmp_path / "bags",
        "reports_root": tmp_path / "reports",
    }


def fixed_local_time():
    return datetime(2026, 9, 5, 2, 45, 30,
                    tzinfo=timezone(timedelta(hours=9), "KST"))


def test_automatic_session_name_paths_and_metadata(tmp_path):
    session = create_mapping_session(now=fixed_local_time(), **session_roots(tmp_path))
    assert session.session_id == "map_20260905_024530"
    assert re.fullmatch(r"map_\d{8}_\d{6}", session.session_id)
    assert not any(character in session.session_id for character in " :/")
    assert all(session.session_id in str(path) for path in (
        session.map_path, session.route_path, session.bag_path, session.report_path))
    metadata = session.metadata()
    assert metadata["created_at_local"] == "2026-09-05T02:45:30+09:00"
    assert metadata["created_at_utc"] == "2026-09-04T17:45:30Z"
    assert metadata["timezone"] == "KST" and metadata["utc_offset"] == "+0900"
    assert metadata["automatic_name"] is True
    output = write_session_metadata(session.metadata_path, metadata)
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == metadata
    try:
        write_session_metadata(session.metadata_path, dict(metadata, session_id="changed"))
    except FileExistsError:
        pass
    else:
        raise AssertionError("session metadata was overwritten")
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["session_id"] == session.session_id


def test_explicit_map_path_then_explicit_session_id_priority(tmp_path):
    roots = session_roots(tmp_path)
    target = roots["maps_root"] / "explicit_map" / "rtabmap.db"
    session = create_mapping_session(
        map_path=target, session_id="different_outputs", now=fixed_local_time(), **roots)
    assert session.map_path == target.resolve()
    assert session.session_id == "explicit_map"
    assert session.route_path == (roots["routes_root"] /
                                  "explicit_map" / "route.csv").resolve()
    manual = create_mapping_session(
        session_id="manual_session", now=fixed_local_time(), **roots)
    assert manual.map_path == (roots["maps_root"] /
                               "manual_session" / "rtabmap.db").resolve()
    assert not manual.automatic_name


def test_automatic_collision_suffix_and_existing_output_nonoverwrite(tmp_path):
    roots = session_roots(tmp_path)
    first = create_mapping_session(now=fixed_local_time(), **roots)
    marker = first.map_path.parent / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    second = create_mapping_session(now=fixed_local_time(), **roots)
    assert second.session_id == "map_20260905_024530_01"
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_automatic_name_skips_existing_route_output(tmp_path):
    roots = session_roots(tmp_path)
    route = roots["routes_root"] / "map_20260905_024530" / "route.csv"
    route.parent.mkdir(parents=True)
    route.write_text("existing route", encoding="utf-8")
    session = create_mapping_session(now=fixed_local_time(), **roots)
    assert session.session_id == "map_20260905_024530_01"
    assert route.read_text(encoding="utf-8") == "existing route"


def test_auto_disabled_without_inputs_fails(tmp_path):
    try:
        create_mapping_session(
            auto_session_name=False, now=fixed_local_time(), **session_roots(tmp_path))
    except ValueError as error:
        assert "auto_session_name=false" in str(error)
    else:
        raise AssertionError("empty session inputs were accepted")


def test_automatic_name_reports_exhausted_collision_attempts(tmp_path):
    roots = session_roots(tmp_path)
    for name in ("map_20260905_024530", "map_20260905_024530_01"):
        (roots["maps_root"] / name).mkdir(parents=True)
    try:
        create_mapping_session(
            now=fixed_local_time(), max_attempts=2, **roots)
    except RuntimeError as error:
        assert "after 2 attempts" in str(error)
    else:
        raise AssertionError("exhausted automatic name attempts did not fail")


def test_concurrent_automatic_session_reservations_are_unique(tmp_path):
    roots = session_roots(tmp_path)

    def reserve(_index):
        return create_mapping_session(now=fixed_local_time(), **roots).session_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        names = list(executor.map(reserve, range(8)))
    assert len(names) == len(set(names)) == 8
    assert "map_20260905_024530" in names


def test_protected_map_identity_is_unchanged_by_rejection():
    before = []
    for directory in PROTECTED_MAP_DIRS:
        database = directory / "rtabmap.db"
        before.append((database.exists(), database.stat() if database.exists() else None,
                       sha256(database) if database.exists() else None))
        assert "PROTECTED_MAP_PATH" in mapping_target_conflicts(database)
    after = []
    for directory in PROTECTED_MAP_DIRS:
        database = directory / "rtabmap.db"
        after.append((database.exists(), database.stat() if database.exists() else None,
                      sha256(database) if database.exists() else None))
    assert before == after


def test_default_and_high_density_profiles_are_exact_strings():
    config = ROOT / "config" / "high_density_mapping.yaml"
    normal = load_mapping_profile(config, False)
    dense = load_mapping_profile(config, True)
    assert normal == {
        "Rtabmap/DetectionRate": "5", "RGBD/LinearUpdate": "0.05",
        "RGBD/AngularUpdate": "0.03", "Vis/MinInliers": "15",
    }
    assert dense == {
        "Rtabmap/DetectionRate": "10", "RGBD/LinearUpdate": "0.03",
        "RGBD/AngularUpdate": "0.02", "Vis/MinInliers": "15",
    }
    assert all(isinstance(value, str) for value in dense.values())
    assert "--Rtabmap/DetectionRate 10" in rtabmap_argument_string(dense)


def test_new_mapping_target_rejects_existing_db_sidecars_and_checksum(tmp_path):
    root = tmp_path / "maps"
    target = root / "new" / "rtabmap.db"
    target.parent.mkdir(parents=True)
    assert mapping_target_conflicts(target, root) == []
    target.touch()
    assert "TARGET_EXISTS:rtabmap.db" in mapping_target_conflicts(target, root)
    target.unlink()
    Path(str(target) + "-wal").touch()
    assert "TARGET_EXISTS:rtabmap.db-wal" in mapping_target_conflicts(target, root)
    Path(str(target) + "-wal").unlink()
    (target.parent / "checksums.sha256").touch()
    assert "TARGET_EXISTS:checksums.sha256" in mapping_target_conflicts(target, root)


def test_protected_and_outside_map_paths_are_rejected():
    assert "PROTECTED_MAP_PATH" in mapping_target_conflicts(
        "/home/qor/depth_ws/maps/classroom_test/rtabmap.db")
    assert "PROTECTED_MAP_PATH" in mapping_target_conflicts(
        "/home/qor/depth_ws/maps/corridor_hand_test/rtabmap.db")
    assert "MAP_PATH_OUTSIDE_ALLOWED_ROOT" in mapping_target_conflicts(
        "/tmp/session/rtabmap.db")


def test_duplicate_mapping_graph_is_rejected():
    assert mapping_graph_conflicts(["/rtabmap/rtabmap"], 2) == [
        "RTABMAP_NODE_ALREADY_RUNNING", "ODOMETRY_PUBLISHER_COUNT:2"]
    assert mapping_graph_conflicts(["/localization_fusion"], 1) == [
        "LOCALIZATION_FUSION_ALREADY_RUNNING"]


def _route(path):
    fields = ["timestamp_sec", "timestamp_nanosec", "route_index", "map_x_m",
              "map_y_m", "map_z_m", "yaw_rad", "yaw_deg",
              "cumulative_distance_m", "valid"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, [1, 0, 0, 0, 0, 0, 0, 0, 0, True])))


def test_mapping_finalize_checks_sqlite_route_and_nonoverwrite(tmp_path):
    map_dir = tmp_path / "maps" / "s1"
    route_dir = tmp_path / "routes" / "s1"
    map_dir.mkdir(parents=True)
    route_dir.mkdir(parents=True)
    database = map_dir / "rtabmap.db"
    connection = sqlite3.connect(database)
    connection.execute("create table test(value integer)")
    connection.commit()
    connection.close()
    route = route_dir / "route.csv"
    _route(route)
    metadata = route_dir / "route_metadata.yaml"
    metadata.write_text(yaml.safe_dump({"session_id": "s1"}), encoding="utf-8")
    quality = tmp_path / "reports" / "s1" / "mapping_quality.json"
    quality.parent.mkdir(parents=True)
    quality.write_text(json.dumps({
        "tracking_true_percent": 98.5, "reset_increase": 0,
        "invalid_session": False, "final_state": "READY",
    }), encoding="utf-8")
    checksum = finalize_mapping_session(
        database, route, metadata, "s1", str(quality))
    assert checksum.is_file()
    values = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    assert values["finalized"] and values["point_count"] == 1
    assert values["tracking_true_percent"] == 98.5
    assert values["reset_increase"] == 0
    assert values["quality_verdict"] == "READY"
    assert values["mapping_quality_report_sha256"]
    try:
        finalize_mapping_session(database, route, metadata, "s1", str(quality))
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing checksum was overwritten")


def test_mapping_finalize_rejects_wrong_graph_before_creating_final_route(tmp_path):
    map_dir = tmp_path / "maps" / "s1"
    route_dir = tmp_path / "routes" / "s1"
    map_dir.mkdir(parents=True)
    route_dir.mkdir(parents=True)
    database = map_dir / "rtabmap.db"
    connection = sqlite3.connect(database)
    connection.execute("create table test(value integer)")
    connection.commit()
    connection.close()
    route = route_dir / "route.csv"
    _route(route)
    metadata = route_dir / "route_metadata.yaml"
    metadata.write_text(yaml.safe_dump({"session_id": "s1"}), encoding="utf-8")
    graph = route_dir / "rtabmap_graph.json"
    graph.write_text(json.dumps({"session_id": "another_session", "poses": {}}),
                     encoding="utf-8")
    final_route = route_dir / "route_final.csv"

    with pytest.raises(ValueError, match="graph snapshot session ID"):
        finalize_mapping_session(
            database, route, metadata, "s1", graph_path=str(graph),
            final_route_path=str(final_route))
    assert not final_route.exists()


def test_mapping_launch_keeps_old_arguments_and_adds_profile_switch():
    source = (ROOT / "launch" / "hybrid_mapping.launch.py").read_text()
    common = (ROOT / "depth_hybrid_slam" / "launch_common.py").read_text()
    assert '"high_density": "false"' in common
    assert '"auto_session_name": "true"' in common
    assert "load_mapping_profile" in source
    assert "create_mapping_session" in source
    assert "mapping_session_info" in source
    assert "database_path=str(path)" in source
    assert '"mapping_mode": "true", "localization_mode": "false"' in source


def test_offline_launch_has_no_camera_and_uses_temporary_copy():
    path = ROOT / "launch" / "map_route_view.launch.py"
    spec = importlib.util.spec_from_file_location("map_route_view", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    assert "tempfile.mkdtemp" in source and "shutil.copy2" in source
    assert "realsense2_camera" not in source and "cuvslam" not in source.lower()
    assert "offline_map_requester" in source and "ros2\", \"service" not in source
    assert '"use_temporary_db_copy": "true"' in source
