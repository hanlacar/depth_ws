import copy
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.piecewise_alignment import (
    AlignmentZone,
    apply_piecewise,
    interpolate_angle,
    parse_alignment_zones,
    smoothstep,
    transform_at_distance,
)
from depth_hybrid_slam.route_follower_core import RouteFollower
from depth_hybrid_slam.route_io import (
    B_EXCLUSIVE_SEGMENTS, load_segmented_route, verify_route_binding,
)


ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "routes/network/route_network_segmented_aligned.csv"
METADATA = ROOT / "routes/network/route_network_segmented_aligned.metadata.yaml"
MAP = ROOT / "maps/merged_competition_level_aligned_v10/rtabmap.db"
REPORT = ROOT / "analysis/csv_vslam_piecewise_alignment/validation_report.json"


def metadata():
    return yaml.safe_load(METADATA.read_text(encoding="utf-8"))


def test_piecewise_zone_parser_accepts_complete_ordered_schedule():
    values = metadata()
    zones = parse_alignment_zones(values)
    assert len(zones) == 6
    assert zones[0].zone_id == "ZONE_01_START"
    assert zones[-1].zone_id == "ZONE_06_END"
    assert zones[0].start_s_m == 0.0
    assert zones[-1].end_s_m == pytest.approx(
        values["alignment"]["route_length_m"])
    assert all(left.end_s_m == pytest.approx(right.start_s_m)
               for left, right in zip(zones, zones[1:]))


@pytest.mark.parametrize("offset,word", [(0.1, "gap"), (-0.1, "overlap")])
def test_piecewise_zone_parser_rejects_gap_and_overlap(offset, word):
    values = copy.deepcopy(metadata())
    values["alignment"]["zones"][1]["start_s_m"] += offset
    with pytest.raises(ValueError, match=word):
        parse_alignment_zones(values)


def test_smoothstep_and_shortest_angle_interpolation():
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(2.0) == 1.0
    midpoint = interpolate_angle(math.radians(179), math.radians(-179), 0.5)
    assert abs(abs(math.degrees(midpoint)) - 180.0) < 1.0e-9


def test_transform_interpolation_is_continuous_at_zone_boundary():
    zones = (
        AlignmentZone("A", 0.0, 10.0, math.radians(179), 0.0, 1.0),
        AlignmentZone("B", 10.0, 20.0, math.radians(-179), 2.0, 3.0),
    )
    before = transform_at_distance(9.999999, zones, 4.0)
    after = transform_at_distance(10.000001, zones, 4.0)
    assert abs(before[0] - after[0]) < 1.0e-6
    assert abs(before[1] - after[1]) < 1.0e-5
    assert abs(before[2] - after[2]) < 1.0e-5
    points, schedule = apply_piecewise(
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        np.asarray([9.999999, 10.000001]), zones, 4.0)
    assert np.linalg.norm(points[1] - points[0]) < 1.0e-5
    assert all("A->B" == item[3] for item in schedule)


def test_aligned_csv_is_map_frame_lossless_active_a_candidate():
    route = load_segmented_route(CSV, METADATA)
    assert route.raw_row_count == route.active_row_count == 3202
    assert route.source_frame == route.frame_id == "map"
    assert not {point.segment_id for point in route.points}.intersection(
        B_EXCLUSIVE_SEGMENTS)
    assert any(point.direction < 0 for point in route.points)
    assert any(point.event == "STOP_LINE" for point in route.points)
    assert route.maximum_step_m < 1.09
    assert route.maximum_connection_m < 0.91
    assert route.source_points[0].x == pytest.approx(1.081892, abs=1e-9)
    with CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "routes/network/route_network_segmented.csv").open(
            newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream)
                       if row["segment_id"] in route.segment_order]
    preserved = ("segment_id", "segment_type", "point_index", "latitude",
                 "longitude", "direction", "mode", "drive_level", "event",
                 "from_node", "to_node")
    assert [[row[name] for name in preserved] for row in rows] == [
        [row[name] for name in preserved] for row in source_rows]
    assert rows[0]["alignment_zone"] == "ZONE_01_START"
    assert any("->" in row["alignment_zone"] for row in rows)


def test_unapproved_aligned_candidate_still_fails_closed():
    values = metadata()
    assert values["alignment"]["required_for_runtime"] is True
    assert values["alignment"]["validated"] is False
    valid, reason = verify_route_binding(CSV, MAP, METADATA)
    assert not valid
    assert reason == "CSV_MAP_ALIGNMENT_NOT_VALIDATED"


def test_piecewise_continuity_and_vehicle_geometry_report_pass():
    import json
    values = json.loads(REPORT.read_text(encoding="utf-8"))
    continuity = values["piecewise"]["continuity"]
    geometry = values["piecewise"]["vehicle_geometry"]
    assert continuity["max_waypoint_gap_m"] <= 1.25
    assert continuity["max_transform_boundary_gap_m"] < 0.35
    assert continuity["max_transform_boundary_yaw_jump_deg"] < 6.0
    assert geometry["max_abs_required_steering_deg"] <= 22.0
    assert geometry["passed"] is True


def test_piecewise_synthetic_traversal_preserves_monotonic_a_route():
    route = load_segmented_route(CSV, METADATA)
    follower = RouteFollower(
        search_ahead_points=180, search_behind_points=8,
        max_index_backtrack=0, steering_rate_deg_s=1000.0)
    previous = 0
    samples = list(range(0, len(route.points), 5))
    if samples[-1] != len(route.points) - 1:
        samples.append(len(route.points) - 1)
    for sample_number, index in enumerate(samples):
        point = route.points[index]
        result = follower.compute(
            Pose2D(point.x, point.y, point.yaw, float(index)), route.points,
            allow_motion=True, global_search=sample_number == 0)
        assert result.nearest_index >= previous
        assert route.points[result.nearest_index].segment_id not in B_EXCLUSIVE_SEGMENTS
        assert abs(result.steering_deg) <= 22.0
        previous = result.nearest_index
    assert previous == len(route.points) - 2
    assert result.reason == "ROUTE_COMPLETE"
    assert result.progress == 1.0
