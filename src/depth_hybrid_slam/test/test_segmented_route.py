import math
from pathlib import Path

from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.route_follower_core import RouteFollower
from depth_hybrid_slam.route_io import (
    A_EXCLUSIVE_SEGMENTS,
    B_EXCLUSIVE_SEGMENTS,
    DEFAULT_BRANCH,
    load_segmented_route,
    SEGMENTED_COLUMNS,
    verify_route_binding,
)
import pytest


ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "routes/network/route_network_segmented.csv"
METADATA = ROOT / "routes/network/route_network_segmented.metadata.yaml"
MAP = ROOT / "maps/merged_competition_level_aligned_v10/rtabmap.db"


def loaded():
    return load_segmented_route(CSV, METADATA)


def test_full_csv_schema_and_parser_preserve_future_fields():
    route = loaded()
    assert SEGMENTED_COLUMNS == (
        "segment_id", "segment_type", "point_index", "latitude", "longitude",
        "x_m", "y_m", "direction", "mode", "drive_level", "event",
        "from_node", "to_node")
    assert route.raw_row_count == 7463
    assert route.active_row_count == 3202
    assert any(point.direction < 0 for point in route.points)
    assert any(point.event == "STOP_LINE" for point in route.points)
    assert {point.mode for point in route.points} == set(range(1, 12))
    assert all(point.segment_id and point.point_index >= 0
               for point in route.points)


def test_default_a_excludes_all_explicit_b_segments():
    route = loaded()
    names = {point.segment_id for point in route.points}
    assert DEFAULT_BRANCH == "A"
    assert set(A_EXCLUSIVE_SEGMENTS) <= names
    assert not names.intersection(B_EXCLUSIVE_SEGMENTS)
    assert route.excluded_b_segments == B_EXCLUSIVE_SEGMENTS
    assert route.excluded_b_row_count == 1116


def test_common_segments_and_order_are_complete():
    route = loaded()
    assert route.segment_order == (
        "START_A", "COMMON_1", "T_foword", "T_A", "COMMON_2",
        "V_A", "END_common", "END_AA")
    assert route.common_segments == (
        "COMMON_1", "T_foword", "COMMON_2", "END_common")


def test_active_route_indexes_yaw_and_connections_are_continuous():
    route = loaded()
    assert [point.index for point in route.points] == list(
        range(route.active_row_count))
    assert all(math.isfinite(value) for point in route.points
               for value in (point.x, point.y, point.yaw))
    assert route.maximum_step_m < 1.09
    assert route.maximum_connection_m < 0.91


def test_output_is_map_frame_but_unapproved_alignment_fails_closed():
    route = loaded()
    assert route.source_frame == "gps_local_enu"
    assert route.frame_id == "map"
    assert route.coordinate_transform["scale"] == 1.0
    assert len(route.source_points) == len(route.points)
    assert route.source_points[0].x == pytest.approx(1.081892, abs=1e-9)
    assert route.source_points[0].y == pytest.approx(-0.506308, abs=1e-9)
    assert route.points[0].x == pytest.approx(-3.037103887, abs=1e-6)
    assert route.points[0].y == pytest.approx(-1.505063193, abs=1e-6)
    assert route.metadata["alignment"]["validated"] is False
    valid, reason = verify_route_binding(CSV, MAP, METADATA)
    assert not valid
    assert reason == "CSV_MAP_ALIGNMENT_NOT_VALIDATED"


def test_synthetic_pose_progress_is_monotonic_and_never_enters_b():
    route = loaded()
    follower = RouteFollower(
        search_ahead_points=180, search_behind_points=8,
        max_index_backtrack=0, steering_rate_deg_s=1000.0)
    previous = 0
    sampled = list(range(0, len(route.points), 5))
    if sampled[-1] != len(route.points) - 1:
        sampled.append(len(route.points) - 1)
    for sample_number, index in enumerate(sampled):
        point = route.points[index]
        result = follower.compute(
            Pose2D(point.x, point.y, point.yaw, float(index)),
            route.points, allow_motion=True,
            global_search=sample_number == 0)
        assert result.nearest_index >= previous
        assert route.points[result.nearest_index].segment_id not in (
            B_EXCLUSIVE_SEGMENTS)
        assert abs(result.steering_deg) <= 22.0
        previous = result.nearest_index
    assert previous == len(route.points) - 2
    assert result.reason == "ROUTE_COMPLETE"
    assert result.progress == 1.0
