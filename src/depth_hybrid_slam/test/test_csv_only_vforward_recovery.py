import csv
import hashlib
import math
from pathlib import Path

import pytest
import yaml

from depth_hybrid_slam.csv_only_branching import (
    IndependentRouteCaseSelector, StartBranchClassifier,
    csv_only_network_segments, load_csv_only_route_case,
    parking_branch_geometry)
from depth_hybrid_slam.csv_only_recovery import (
    lateral_offset_xy, simulate_lateral_recovery, virtual_vehicle)
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.route_follower_core import RouteFollower
from depth_hybrid_slam.route_io import load_segmented_route
from depth_hybrid_slam.stop_editor_network import route_case_segments


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "routes/network/route_network_segmented_stop_edited.csv"
CANDIDATE = ROOT / "routes/network/route_network_segmented_stop_edited_vforward.csv"
METADATA = CANDIDATE.with_suffix(".metadata.yaml")
DISPLAY = ROOT / (
    "routes/network/route_network_segmented_all_branches_display_aligned_vforward.csv")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _segments():
    return csv_only_network_segments(CANDIDATE, METADATA)


def _selector():
    segments = _segments()
    classifier = StartBranchClassifier(
        segments["START_A"], segments["START_B"])
    point = segments["START_A"][0]
    selector = IndependentRouteCaseSelector(
        parking_branch_geometry(segments))
    selector.set_start(classifier.classify(
        Pose2D(point.x, point.y, point.yaw, 0.0)))
    return selector


def test_vforward_candidate_preserves_every_non_mode10_source_row():
    source, candidate = _rows(SOURCE), _rows(CANDIDATE)
    source_other = [row for row in source
                    if row["segment_id"] not in ("V_A", "V_B")]
    candidate_other = [row for row in candidate if row["segment_id"] not in
                       ("V_foword", "V_A", "V_B")]
    assert source_other == candidate_other
    counts = {name: sum(row["segment_id"] == name for row in candidate)
              for name in ("V_foword", "V_A", "V_B")}
    assert counts == {"V_foword": 108, "V_A": 132, "V_B": 154}
    assert len(candidate) == 7419
    metadata = yaml.safe_load(METADATA.read_text())
    provenance = metadata["vforward_integration"]
    assert provenance["reference_commit"] == (
        "fc535402100c19159fc87474143b9a874bd0848e")
    assert provenance["geometry_coordinates_modified"] is False
    assert provenance["explicit_stop_points"]["V_foword"] == [105]
    digest = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert metadata["route_sha256"] == digest


def test_vforward_is_shared_in_all_cases_and_display_has_exact_keys():
    for case in ("AAAA", "AABA", "ABAA", "ABBA", "BBBB"):
        route = load_csv_only_route_case(CANDIDATE, METADATA, case)
        order = tuple(dict.fromkeys(point.segment_id for point in route))
        assert order == route_case_segments(case, {"V_foword"})
        assert order[order.index("COMMON_2")+1:order.index("COMMON_2")+3] == (
            "V_foword", f"V_{case[2]}")
    source_keys = {(row["segment_id"], row["point_index"])
                   for row in _rows(CANDIDATE)}
    display_keys = {(row["segment_id"], row["point_index"])
                    for row in _rows(DISPLAY)}
    assert source_keys == display_keys


def test_vforward_geometry_and_safe_early_commit_are_explicit():
    geometry = parking_branch_geometry(_segments())["V"]
    assert geometry["verdict"] == "PASS"
    assert geometry["common_forward"]["points"] == 108
    assert geometry["common_forward"]["end"][0] == 107
    assert geometry["common_forward"]["commit"][0] == 105
    assert geometry["common_forward"]["commit_event"] == "STOP_LINE"
    for branch in "AB":
        transition = geometry["transitions"][branch]
        assert transition["continuous"]
        assert transition["gap_m"] <= 1.0
        assert transition["required_steering_deg"] <= 22.0
    topology = geometry["topology_transitions"]
    assert topology["COMMON_2_TO_V_foword"]["gap_m"] == pytest.approx(
        0.25834134607723475)
    assert topology["V_foword_TO_V_A"]["gap_m"] == pytest.approx(
        0.16645074684422567)
    assert topology["V_foword_TO_V_B"]["gap_m"] == pytest.approx(
        0.17256211058630574)
    # The pinned reference's 8 mm p106->p107 tail makes an end-only V_B
    # steering estimate exceed the physical limit.  The explicit p105 commit
    # lets the bounded controller start that turn while still on common path.
    assert topology["V_foword_TO_V_B"]["required_steering_deg"] > 22.0
    assert geometry["transitions"]["B"]["required_steering_deg"] == \
        pytest.approx(12.80999501323265)


@pytest.mark.parametrize("requested,expected", [(False, "A"), (True, "B")])
def test_vforward_pending_requested_commit_hold_reverse_and_late_command(
        requested, expected):
    selector = _selector()
    assert selector.enter_segment("V_foword", 10.0)
    assert selector.lifecycle_status()["V"] == "PENDING"
    if requested:
        assert selector.request("V:B", 12.0)
        assert selector.lifecycle_status()["V"] == "B_REQUESTED"
    selector.observe_point(
        "V_foword", 105, 1, "V_foword:105", "MINIMUM_3S_HOLD", 20.0)
    assert selector.choices["V"] is None and selector.stop
    selector.observe_point(
        "V_foword", 105, 1, "V_foword:105",
        "WAIT_TRAFFIC_RELEASE", 23.0)
    assert selector.choices["V"] == expected
    assert selector.lifecycle_status()["V"] == f"COMMITTED_{expected}"
    assert selector.stop
    selector.advance(25.999)
    assert selector.stop
    selector.advance(26.0)
    assert not selector.stop
    selector.observe_point(f"V_{expected}", 1 if expected == "B" else 12,
                           -1, "", "IDLE", 26.1)
    assert selector.lifecycle_status()["V"] == "REVERSE_STARTED"
    assert not selector.request("V:B", 26.2)
    assert selector.last_event == "LATE_BRANCH_COMMAND_IGNORED"


def test_candidate_loader_retains_stop_and_physical_limits():
    a = load_segmented_route(CANDIDATE, METADATA, branch="A")
    b = load_segmented_route(CANDIDATE, METADATA, branch="B")
    assert "V_foword" in a.common_segments and "V_foword" in b.common_segments
    assert a.maximum_connection_m < 0.91
    assert b.maximum_connection_m < 0.91
    for route in (a.points, b.points):
        vf_stop = [point for point in route if point.segment_id == "V_foword"
                   and point.event == "STOP_LINE"]
        assert [(point.point_index, point.direction) for point in vf_stop] == [(105, 1)]
        assert any(point.direction < 0 for point in route
                   if point.segment_id in ("V_A", "V_B"))


@pytest.mark.parametrize("offset", (
    0.20, -0.20, 0.50, -0.50, 0.80, -0.80,
    1.20, -1.20, 1.80, -1.80, 2.10, -2.10))
def test_closed_loop_disturbance_matrix_uses_actual_one_meter_policy(offset):
    route = load_csv_only_route_case(CANDIDATE, METADATA, "AAAA")
    start = next(point.index for point in route
                 if point.segment_id == "COMMON_1" and point.point_index == 300)
    report = simulate_lateral_recovery(route, start, offset)
    assert report.result == "PASS"
    assert not report.wrong_segment_jump
    assert not report.direction_jump
    assert not report.cursor_backtrack
    assert report.maximum_steering_deg <= 22.0
    if abs(offset) <= 1.0:
        assert not report.stop_occurred
        assert report.decrease_started_s is not None
        assert report.recovered_025_s is not None
        assert report.recovered_010
    else:
        assert report.stop_occurred
        assert report.stop_reason == "CORRIDOR_VIOLATION"


@pytest.mark.parametrize("case,segment,point_index", (
    ("AAAA", "COMMON_1", 300),
    ("AAAA", "COMMON_1", 600),
    ("AAAA", "COMMON_2", 20),
    ("ABAA", "COMMON_2", 20),
    ("AAAA", "V_foword", 30),
    ("AAAA", "END_common", 50),
))
@pytest.mark.parametrize("offset", (0.50, -0.50))
def test_half_meter_recovery_at_required_route_locations(
        case, segment, point_index, offset):
    route = load_csv_only_route_case(CANDIDATE, METADATA, case)
    start = next(point.index for point in route
                 if point.segment_id == segment and
                 point.point_index == point_index)
    report = simulate_lateral_recovery(route, start, offset)
    assert report.result == "PASS"
    assert report.recovered_025_s is not None
    assert report.recovered_010
    assert not report.stop_occurred


def test_t_b_to_common2_default_rate_has_no_deviation_or_cursor_jump():
    route = load_csv_only_route_case(CANDIDATE, METADATA, "ABAA")
    start = next(point.index for point in route
                 if point.segment_id == "T_B" and point.point_index == 100)
    point = route[start]
    follower = RouteFollower(corridor_m=1.0, max_index_backtrack=0)
    follower.last_index = start
    vehicle = virtual_vehicle()
    vehicle.x, vehicle.y, vehicle.yaw = point.x, point.y, point.yaw
    maximum_cte = 0.0
    last_index = start
    reached = False
    segments = []
    for step in range(1500):
        result = follower.compute(
            Pose2D(vehicle.x, vehicle.y, vehicle.yaw, step*0.05), route,
            dt=0.05, allow_motion=True, global_search=False)
        active = route[result.nearest_index]
        if not segments or segments[-1] != active.segment_id:
            segments.append(active.segment_id)
        maximum_cte = max(maximum_cte, abs(result.cross_track_error))
        assert not result.stop_required
        assert result.nearest_index >= last_index
        last_index = result.nearest_index
        vehicle.step(result.drive, result.steering_deg, False, 0.05)
        if active.segment_id == "COMMON_2" and active.point_index >= 30:
            reached = True
            break
    assert reached
    assert segments == ["T_B", "COMMON_2"]
    assert maximum_cte < 0.16
    t_end = next(point for point in route
                 if point.segment_id == "T_B" and point.point_index == 151)
    common_start = next(point for point in route
                        if point.segment_id == "COMMON_2" and
                        point.point_index == 0)
    assert math.hypot(t_end.x-common_start.x,
                      t_end.y-common_start.y) == pytest.approx(
                          0.9060908788002431)


def test_test_only_injection_uses_route_normal_and_is_not_production_code():
    assert lateral_offset_xy(3.0, 4.0, 0.0, 0.5) == (3.0, 4.5)
    assert lateral_offset_xy(3.0, 4.0, math.pi/2.0, -0.5) == pytest.approx(
        (3.5, 4.0))
    source = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam/"
              "csv_only_virtual_vehicle_node.py").read_text()
    assert "/depth_slam/test/inject_lateral_offset_m" in source
    assert "TEST ONLY lateral disturbance" in source
    production = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam/"
                  "route_follower_node.py").read_text()
    assert "inject_lateral_offset" not in production


def test_live_probe_allows_only_t_pending_a_not_v_pending_a():
    probe = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam/"
             "csv_only_closed_loop_probe.py").read_text()
    assert '{"T_A"} if self.expected_case[1] == "B" else set()' in probe
    assert 'seeing V_A in a V:B run remains a hard failure' in probe
