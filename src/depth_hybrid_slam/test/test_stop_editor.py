import csv
import hashlib
import importlib.util
from pathlib import Path
import shlex
from types import SimpleNamespace

from depth_hybrid_slam.stop_editor_core import STOP_EVENT, StopRouteEditor
import yaml


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT / "routes/network/route_network_segmented.csv"
METADATA = ROOT / "routes/network/route_network_segmented.metadata.yaml"
ACTIVE_ALIGNED = ROOT / "routes/network/route_network_segmented_aligned.csv"
ALIGNED = ROOT / "routes/network/route_network_segmented_all_branches_display_aligned.csv"
ALIGNED_METADATA = ROOT / "routes/network/route_network_segmented_all_branches_display_aligned.metadata.yaml"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def editor():
    return StopRouteEditor(ROUTE, METADATA)


def overlay_editor():
    return StopRouteEditor(
        ROUTE, METADATA, visualization_route_path=ALIGNED,
        visualization_metadata_path=ALIGNED_METADATA)


def load_stop_editor_launch():
    path = ROOT / "src/depth_hybrid_slam/launch/stop_editor_vslam.launch.py"
    spec = importlib.util.spec_from_file_location("stop_editor_vslam_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordinary_empty(e):
    return next(
        e.key(row) for row in e.rows
        if e.key(row) in e.visualization_keys and
        row["segment_id"] != "AAA_BASE" and
        e.key(row) not in e.required_transitions and
        row["event"] != STOP_EVENT)


def test_source_alignment_remains_unvalidated_and_required_stops_are_added():
    value = overlay_editor()
    assert not value.alignment_validated
    assert value.visualization_metadata["alignment"]["validated"] is False
    assert value.visualization_branch == "ALL"
    assert len(value.visualization_rows) == len(value.rows) == 7463
    assert len(value.route_network_keys) == 4318
    assert len(value.reference_keys) == 3145
    assert len(value.segments) == 13
    assert set(value.segments) == {
        "AAA_BASE", "START_A", "START_B", "COMMON_1", "T_foword",
        "T_A", "T_B", "COMMON_2", "V_A", "V_B", "END_common",
        "END_AA", "END_AB"}
    assert list(value.route_cases) == [
        "AAAA", "AAAB", "AABA", "AABB", "ABAA", "ABAB", "ABBA", "ABBB",
        "BAAA", "BAAB", "BABA", "BABB", "BBAA", "BBAB", "BBBA", "BBBB"]
    assert {int(row["mode"]) for row in value.visualization_rows} == set(
        range(1, 12))
    assert all(value.key(row) in value._index for row in value.visualization_rows)
    with ALIGNED.open(newline="") as stream:
        full_overlay = list(csv.DictReader(stream))
    first_aligned = full_overlay[0]
    first_key = value.key(first_aligned)
    assert value.map_xy[first_key] == (
        float(first_aligned["x_m"]), float(first_aligned["y_m"]))
    full_by_key = {value.key(row): row for row in full_overlay}
    with ACTIVE_ALIGNED.open(newline="") as stream:
        for active in csv.DictReader(stream):
            generated = full_by_key[value.key(active)]
            assert generated["x_m"] == active["x_m"]
            assert generated["y_m"] == active["y_m"]
    network = value.visualization_metadata["network"]
    assert (network["duplicate_keys"], network["missing_keys"],
            network["ambiguous_keys"], network["route_case_count"]) == (0, 0, 0, 16)
    assert value.source_stop_count == 8
    assert value.stop_count == 16
    assert len(value.required_transitions) == 8
    for key in value.required_transitions:
        assert value.rows[value._index[key]]["event"] == STOP_EVENT


def test_click_distance_add_remove_and_undo():
    value = overlay_editor()
    assert value.add(10000.0, 10000.0).state == "CLICK_TOO_FAR_FROM_ROUTE"
    key = ordinary_empty(value)
    result = value.add(*value.map_xy[key])
    assert result.accepted and result.key == key and value.stop_count == 17
    assert value.rows[value._index[key]]["event"] == STOP_EVENT
    assert value.undo().accepted and value.stop_count == 16
    assert value.add(*value.map_xy[key]).accepted
    assert value.remove(*value.map_xy[key]).accepted
    assert value.undo().accepted
    assert value.rows[value._index[key]]["event"] == STOP_EVENT
    assert value.set_case("AAAA")
    for mode in (2, 10):
        mode_key = next(
            value.key(row) for row in value.visualization_rows
            if int(row["mode"]) == mode and
            row["segment_id"] != "AAA_BASE" and
            value.key(row) not in value.required_transitions and
            row["event"] != STOP_EVENT)
        assert value.add(*value.map_xy[mode_key]).accepted
        assert value.rows[value._index[mode_key]]["event"] == STOP_EVENT
        assert value.undo().accepted
    assert value.set_case("BBBB")
    b_key = next(value.key(row) for row in value.case_rows("BBBB")
                 if row["segment_id"] == "V_B" and
                 value.key(row) not in value.required_transitions and
                 row["event"] != STOP_EVENT)
    assert value.add(*value.map_xy[b_key]).accepted
    assert value.undo().accepted
    assert not value.set_case("INVALID")
    assert value.set_case("ALL")
    ambiguous_key = ("T_A", 0)
    ambiguous = value.add(*value.map_xy[ambiguous_key])
    assert not ambiguous.accepted
    assert ambiguous.state == "AMBIGUOUS_BRANCH_CLICK:SELECT_CASE"


def test_required_direction_change_stop_cannot_be_removed():
    value = editor()
    key = next(iter(value.required_transitions))
    result = value.remove(*value.map_xy[key])
    assert not result.accepted
    assert result.state == "REMOVE_REJECTED:REQUIRED_DIRECTION_CHANGE_STOP"
    assert value.rows[value._index[key]]["event"] == STOP_EVENT


def test_save_changes_only_event_and_records_audit_metadata(tmp_path):
    value = overlay_editor()
    source_hash = digest(ROUTE)
    metadata_hash = digest(METADATA)
    aligned_hash = digest(ALIGNED)
    aligned_metadata_hash = digest(ALIGNED_METADATA)
    key = ordinary_empty(value)
    assert value.add(*value.map_xy[key]).accepted
    output = tmp_path / "route_network_segmented_stop_edited.csv"
    csv_path, metadata_path, output_hash = value.save(output)
    assert digest(ROUTE) == source_hash
    assert digest(METADATA) == metadata_hash
    assert digest(ALIGNED) == aligned_hash
    assert digest(ALIGNED_METADATA) == aligned_metadata_hash
    assert digest(csv_path) == output_hash
    with ROUTE.open(newline="") as before_stream, csv_path.open(newline="") as after_stream:
        before = list(csv.DictReader(before_stream))
        after = list(csv.DictReader(after_stream))
    assert len(before) == len(after)
    for first, second in zip(before, after):
        assert {k: v for k, v in first.items() if k != "event"} == \
            {k: v for k, v in second.items() if k != "event"}
    metadata = yaml.safe_load(metadata_path.read_text())
    audit = metadata["stop_edit"]
    assert not metadata["alignment"]["validated"]
    assert audit["source_sha256"] == source_hash
    assert audit["output_sha256"] == output_hash
    assert audit["geometry_unchanged"] is True
    assert audit["stop_before_count"] == 8
    assert audit["stop_after_count"] == 17
    overlay = audit["visualization_only_overlay"]
    assert overlay["route_sha256"] == aligned_hash
    assert overlay["metadata_sha256"] == aligned_metadata_hash
    assert overlay["frame_id"] == "map"
    assert overlay["alignment_validated"] is False
    assert overlay["waypoint_count"] == 7463
    output = tmp_path / "edited.csv"
    metadata_output = tmp_path / "edited.metadata.yaml"
    for csv_path, yaml_path in (
            (ROUTE, metadata_output),
            (METADATA, metadata_output),
            (ALIGNED, metadata_output),
            (output, ROUTE),
            (output, METADATA),
            (output, ALIGNED_METADATA),
            (output, output)):
        try:
            value.save(csv_path, yaml_path)
        except ValueError:
            pass
        else:
            raise AssertionError("protected/colliding output path was accepted")
    assert not list(tmp_path.glob("*.tmp"))


def test_every_a_and_b_direction_transition_has_required_stop():
    value = editor()
    expected = {
        ("T_A", 4, 1, -1), ("T_A", 80, -1, 1),
        ("V_A", 90, 1, -1), ("V_A", 134, -1, 1),
        ("T_B", 5, 1, -1), ("T_B", 66, -1, 1),
        ("V_B", 84, 1, -1), ("V_B", 154, -1, 1),
    }
    actual = {(item["segment"], item["point_index"],
               item["from_direction"], item["to_direction"])
              for item in value.required_transitions.values()}
    assert actual == expected
    for name, keys in value.case_keys.items():
        segments = {key[0] for key in keys}
        assert len(name) == 4
        assert ("START_A" if name[0] == "A" else "START_B") in segments
        assert ("T_A" if name[1] == "A" else "T_B") in segments
        assert ("V_A" if name[2] == "A" else "V_B") in segments
        assert ("END_AA" if name[3] == "A" else "END_AB") in segments
        assert not ({"START_A", "START_B"} - {
            "START_A" if name[0] == "A" else "START_B"}) & segments


def test_mode11_source_stops_are_preserved_for_review():
    value = overlay_editor()
    for branch in ("A", "B"):
        result = value.mode11_stop_analysis(branch)
        assert [(item["segment"], item["point_index"])
                for item in result] == [("END_common", 3), ("END_common", 145)]
        assert all(item["classification"] == "REVIEW_REQUIRED" for item in result)
        assert all(item["mission_meaning"] == "STOP_LINE_MISSION_MARKER"
                   for item in result)
    key = ("END_common", 3)
    assert value.remove(*value.map_xy[key]).accepted
    assert value.rows[value._index[key]]["event"] == "NONE"
    assert value.undo().accepted


def test_rviz_and_launch_contracts_display_map_routes_stops_and_click_tool(
        tmp_path, monkeypatch):
    config_path = ROOT / "src/depth_hybrid_slam/config/stop_editor_vslam.rviz"
    config = config_path.read_text()
    launch = (ROOT / "src/depth_hybrid_slam/launch/stop_editor_vslam.launch.py").read_text()
    node = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam/stop_editor_node.py").read_text()
    parsed = yaml.safe_load(config)
    assert parsed["Visualization Manager"]["Global Options"]["Fixed Frame"] == "map"
    for token in ("RTAB-Map Occupancy Map", "/rtabmap/map",
                  "Full 7463-Row Route Network", "/depth_slam/stop_editor/network",
                  "Mode 1-11 Route Overlay", "mode_markers",
                  "STOP and Selected Waypoint Markers", "PublishPoint",
                  "/clicked_point"):
        assert token in config
    for token in ("CSV OVERLAY: VISUALIZATION ONLY", "ALIGNMENT:",
                  "MODE 2 SLOPE", "MODE 10 PARALLEL PARK",
                  "DIRECTION-CHANGE STOP", 'header.frame_id = "map"',
                  "selected_case"):
        assert token in node
    for token in ("offline_rtabmap_include", "stop_editor", "_read_only_prefix",
                  "--ro-bind", "maximum_click_distance_m",
                  "visualization_route_path", "_all_branches_display_aligned.csv",
                  'default_value="ALL"', "READ_ONLY_BACKEND=", "TEMP_COPY",
                  "INSUFFICIENT_TEMP_SPACE", "_probe_bwrap",
                  "rtabmap_startup_timeout_s", "rtabmap_map_delivery_timeout_s",
                  "failure_only=True", "OnShutdown"):
        assert token in launch
    module = load_stop_editor_launch()
    database = tmp_path / "saved_map" / "rtabmap.db"
    database.parent.mkdir()
    database.touch()
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/bwrap")
    parts = shlex.split(module._read_only_prefix(database))
    assert parts[:5] == ["/usr/bin/bwrap", "--bind", "/", "/", "--ro-bind"]
    assert parts[5:7] == [str(database.parent), str(database.parent)]
    assert parts[-2:] == ["--die-with-parent", "--"]
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr=
                                        "bwrap: setting up uid map: Permission denied"))
    available, reason, executable = module._probe_bwrap(
        apparmor_gate=tmp_path / "no_apparmor_gate")
    assert not available and "Permission denied" in reason
    assert executable == "/usr/bin/bwrap"

    source = tmp_path / "small.db"
    source.write_bytes(b"sqlite test data")
    usage = SimpleNamespace(free=10_000, total=20_000, used=10_000)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _path: usage)
    copied, temporary, size, free, required = module._temporary_database(
        source, tmp_path)
    assert copied.read_bytes() == source.read_bytes()
    assert size == source.stat().st_size and free == usage.free
    assert required == int(__import__("math").ceil(size * 1.5))
    assert copied.parent.name.startswith("depth_stop_editor.")
    module._cleanup(None, temporary)
    assert not copied.parent.exists()

    usage.free = 1
    before = set(tmp_path.iterdir())
    try:
        module._temporary_database(source, tmp_path)
    except RuntimeError as error:
        assert "INSUFFICIENT_TEMP_SPACE" in str(error)
    else:
        raise AssertionError("insufficient temp space was accepted")
    assert set(tmp_path.iterdir()) == before
    usage.free = 10_000
    monkeypatch.setattr(module.shutil, "copy2", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(OSError("copy failed")))
    try:
        module._temporary_database(source, tmp_path)
    except OSError as error:
        assert "copy failed" in str(error)
    else:
        raise AssertionError("copy failure was hidden")
    assert not list(tmp_path.glob("depth_stop_editor.*"))
    requester = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam/"
                 "offline_map_requester.py").read_text()
    for token in ("startup_timeout_s", "map_delivery_timeout_s",
                  "get_publishers_info_by_topic", "RTABMAP_START_FAILED",
                  '"/rtabmap/map"', "service_complete", "map_received"):
        assert token in requester
