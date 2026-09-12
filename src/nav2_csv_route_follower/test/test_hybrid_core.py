import math

import pytest

from nav2_csv_route_follower.hybrid_core import (
    BranchSelector, DeviationHysteresis, HybridFollower, Pose2D,
    SegmentProgressTracker, StopController, is_csv_only_point,
    is_parking_point, merge_stop_events,
    commanded_synthetic_speed, select_drive, special_section,
    steering_command,
)
from nav2_csv_route_follower.route_network import RouteNetwork, RoutePoint


CSV = "/home/qor/depth_ws/routes/network/route_network_segmented.csv"
YAML = "/home/qor/depth_ws/routes/network/route_network_segmented.yaml"


def point(index, x, y=0.0, direction="F", drive=2.0, event="NONE",
          mode=1, segment="TEST"):
    return RoutePoint(index, segment, index, x, y, 0.0, direction, mode,
                      drive, event)


def line(offset_y=0.0, direction="F", drive=2.0):
    return [point(i, i * 0.2, offset_y, direction, drive) for i in range(30)]


def test_workspace_segmented_route_is_strictly_parsed_and_analysed():
    network = RouteNetwork.load(CSV, YAML)
    analysis = network.analysis()
    assert analysis["waypoint_count"] == 7463
    assert list(network.segments) == [
        "AAA_BASE", "START_A", "START_B", "COMMON_1", "T_foword",
        "T_A", "T_B", "COMMON_2", "V_A", "V_B", "END_common",
        "END_AA", "END_AB"]
    assert analysis["drive_3_ranges"] == [{
        "segment_id": "COMMON_2", "start_index": 800,
        "end_index": 1059, "start_xy": [-94.017204, -45.003089],
        "end_xy": [-35.534738, -69.697572]}]
    assert len(analysis["stop_lines"]) == 8
    assert len(analysis["drive_level_3_points"]) == 260
    assert {item["segment_id"] for item in analysis["drive_level_3_points"]} == {
        "COMMON_2"}
    assert {item["mode"] for item in analysis["drive_level_3_points"]} == {9}
    assert len(analysis["direction_changes"]) == 10
    assert analysis["metadata"]["origin_lat"] == pytest.approx(37.288977)
    assert analysis["metadata"]["origin_lon"] == pytest.approx(127.1076499809)
    assert analysis["metadata"]["segmented"] is True
    all_points = [point for points in network.segments.values() for point in points]
    assert {point.direction for point in all_points} == {"F", "R"}
    assert {point.drive_level for point in all_points} == {1.0, 2.0, 3.0}


def test_default_and_valid_b_branch_selection():
    branches = BranchSelector()
    assert branches.values == {
        "START": "A", "T": "A", "PARALLEL": "A", "END": "A"}
    assert branches.request_or_default("T", "UNKNOWN")
    assert branches.values["T"] == "A"
    assert branches.request("PARALLEL", "B")
    branches.latch("PARALLEL")
    assert not branches.request("PARALLEL", "A")
    network = RouteNetwork.load(CSV, YAML)
    names = {p.segment_id for p in network.assemble(
        branches.values["START"], branches.values["T"],
        branches.values["PARALLEL"], branches.values["END"])}
    assert "START_A" in names and "T_A" in names
    assert "V_B" in names and "END_common" in names and "END_AA" in names


@pytest.mark.parametrize("value", ["", "TIMEOUT", "UNKNOWN", "INVALID", "A"])
def test_missing_or_invalid_parking_decision_defaults_to_a(value):
    branches = BranchSelector()
    assert branches.request_or_default("T", value)
    assert branches.values["T"] == "A"


def test_latest_parking_modes_and_default_a_routes():
    network = RouteNetwork.load(CSV, YAML)
    assert {point.mode for point in network.segments["T_A"]} == {7}
    assert {point.mode for point in network.segments["T_B"]} == {7}
    assert {point.mode for point in network.segments["V_A"]} == {10}
    assert {point.mode for point in network.segments["V_B"]} == {10}
    assembled = network.assemble()
    names = {point.segment_id for point in assembled}
    assert "T_A" in names and "T_B" not in names
    assert "V_A" in names and "V_B" not in names


def test_nav2_priority_fallback_recovery_and_hysteresis():
    selector = DeviationHysteresis(
        1.5, 0.75, 20.0, 10.0, nav_to_csv_samples=3,
        csv_to_nav_samples=4)
    assert selector.update(0.1, 1.0) == "NAV2"
    assert selector.update(2.0, 1.0) == "NAV2"
    assert selector.update(0.1, 1.0) == "NAV2"  # bad counter reset
    assert [selector.update(2.0, 1.0) for _ in range(3)] == [
        "NAV2", "NAV2", "CSV"]
    assert [selector.update(0.2, 2.0) for _ in range(4)] == [
        "CSV", "CSV", "CSV", "NAV2"]


def test_csv_nearest_is_monotonic_and_segment_constrained_at_overlap():
    first = [point(i, float(i), segment="FIRST") for i in range(5)]
    second = [point(i + 5, float(i), segment="SECOND") for i in range(5)]
    points = first + second
    tracker = SegmentProgressTracker(0, transition_window=1)
    assert tracker.nearest(points, Pose2D(1.0, 0.0, 0.0)) == 1
    # Identical geometry in the later segment must not steal current progress.
    assert tracker.nearest(points, Pose2D(2.0, 0.0, 0.0)) == 2
    assert tracker.nearest(points, Pose2D(4.0, 0.0, 0.0)) == 4
    # A completed segment may then move to the next contiguous segment.
    assert tracker.nearest(points, Pose2D(0.0, 0.0, 0.0)) == 5


@pytest.mark.parametrize("speed,wheel,expected", [
    (1.0, 0, 1.0), (2.0, 2, 2.0), (3.0, 2, 3.0),
    (2.0, 12, 1.0), (3.0, -14, 1.0),
])
def test_csv_speed_and_steering_safety_priority(speed, wheel, expected):
    assert select_drive("F", speed, wheel) == expected
    assert select_drive("F", speed, wheel, stopping=True) == 0.0
    assert select_drive("R", speed, wheel) == -1.0


def test_steering_sign_and_clamp():
    pose = Pose2D(0.0, 0.0, 0.0)
    left, left_sat = steering_command(pose, point(1, 0.2, 2.0), "F")
    right, right_sat = steering_command(pose, point(1, 0.2, -2.0), "F")
    assert left == 22 and left_sat
    assert right == -22 and right_sat


def test_near_nav_and_csv_stop_are_merged_and_latched_once():
    csv_points = [point(0, 0.0), point(1, 1.0, event="STOP_LINE")]
    nav_points = [point(0, 0.0), point(1, 1.1)]
    events = merge_stop_events(csv_points, nav_points, {1}, 0.75)
    assert len(events) == 1
    assert "CSV_STOP_LINE" in events[0].source
    assert "NAV2_STOP" in events[0].source
    controller = StopController(1.0, 0.35)
    pose = Pose2D(1.0, 0.0, 0.0)
    assert controller.update(0.0, pose, events, 1, 1) == "HOLD_STOP"
    assert controller.update(0.5, pose, events, 1, 1) == "HOLD_STOP"
    assert controller.update(1.01, pose, events, 1, 1) == "RELEASE"
    assert controller.update(2.0, pose, events, 1, 1) == "RUNNING"


def test_far_nav_and_csv_stops_remain_separate():
    csv_points = [point(0, 0.0, event="STOP_LINE"), point(1, 2.0)]
    nav_points = [point(0, 0.0), point(1, 2.0)]
    events = merge_stop_events(csv_points, nav_points, {1}, 0.75)
    assert len(events) == 2


def test_stop_line_and_direction_change_at_same_place_are_one_stop():
    csv_points = [point(0, 0.0, direction="F"),
                  point(1, 1.0, direction="R", event="STOP_LINE")]
    events = merge_stop_events(csv_points, [], set(), 0.75)
    assert len(events) == 1
    assert "CSV_STOP_LINE" in events[0].source
    assert "F_TO_R" in events[0].source


def test_f_to_r_and_r_to_f_each_stop_then_resume():
    nav = line()
    csv = [point(0, 0.0, direction="F"),
           point(1, 0.2, direction="R"),
           point(2, 0.4, direction="R"),
           point(3, 0.6, direction="F"),
           point(4, 0.8, direction="F")]
    follower = HybridFollower(
        nav_to_csv_samples=100, stop_trigger_distance_m=0.1,
        stop_hold_seconds=1.0)
    first = follower.command(0.0, Pose2D(0.2, 0.0, 0.0), nav, csv)
    assert first.drive == 0.0 and "F_TO_R" in first.stop_source
    reverse = follower.command(1.01, Pose2D(0.2, 0.0, 0.0), nav, csv)
    assert reverse.drive == -1.0
    second = follower.command(2.0, Pose2D(0.6, 0.0, 0.0), nav, csv)
    assert second.drive == 0.0 and "R_TO_F" in second.stop_source
    forward = follower.command(3.01, Pose2D(0.6, 0.0, 0.0), nav, csv)
    assert forward.drive in {1.0, 2.0}


def test_hybrid_geometry_uses_nav2_then_csv_after_sustained_offset():
    nav = line(0.0)
    csv = line(0.0)
    follower = HybridFollower(
        nav_to_csv_samples=2, csv_to_nav_samples=2,
        nav_to_csv_position_m=1.5, csv_to_nav_position_m=0.75)
    pose = Pose2D(0.0, 0.0, 0.0)
    assert follower.command(0.0, pose, nav, csv).source == "NAV2"
    offset = line(3.0)
    assert follower.command(0.1, pose, offset, csv).source == "NAV2"
    assert follower.command(0.2, pose, offset, csv).source == "CSV_FALLBACK"
    assert follower.command(0.3, pose, nav, csv).source == "CSV_FALLBACK"
    assert follower.command(0.4, pose, nav, csv).source == "NAV2"


@pytest.mark.parametrize("mode,segment", [
    (7, "T_A"), (7, "T_foword"), (10, "V_A"), (1, "T_A"),
])
def test_parking_forces_csv_only_regardless_of_nav_deviation(mode, segment):
    nav = line(4.0)
    csv = [point(i, i * 0.2, mode=mode, segment=segment) for i in range(30)]
    follower = HybridFollower(nav_to_csv_samples=100)
    result = follower.command(0.0, Pose2D(0.0, 0.0, 0.0), nav, csv)
    assert is_parking_point(csv[result.csv_nearest])
    assert result.source == "CSV_ONLY"


def test_non_parking_mode_uses_nav2_normally():
    nav = line()
    csv = line()
    result = HybridFollower().command(
        0.0, Pose2D(0.0, 0.0, 0.0), nav, csv)
    assert result.source == "NAV2"


@pytest.mark.parametrize("mode,section", [
    (4, "INTERSECTION"), (6, "INTERSECTION"),
    (7, "T_PARK"), (9, "ACCELERATION"),
    (10, "PARALLEL_PARK"),
])
def test_actual_special_modes_force_csv_only(mode, section):
    nav = line(4.0)
    csv = [point(i, i * 0.2, mode=mode, segment="COMMON")
           for i in range(30)]
    result = HybridFollower(nav_to_csv_samples=100).command(
        0.0, Pose2D(0.0, 0.0, 0.0), nav, csv)
    assert special_section(csv[result.csv_nearest]) == section
    assert is_csv_only_point(csv[result.csv_nearest])
    assert result.source == "CSV_ONLY"


def test_latest_default_route_csv_only_ranges():
    points = RouteNetwork.load(CSV, YAML).assemble()
    ranges = []
    start = None
    section = None
    for index, route_point in enumerate(points + [None]):
        current = "NORMAL" if route_point is None else special_section(route_point)
        if current != section:
            if section not in (None, "NORMAL"):
                ranges.append((section, start, index - 1))
            start = index if current != "NORMAL" else None
            section = current
    assert ranges == [
        ("INTERSECTION", 601, 809),
        ("INTERSECTION", 1063, 1312),
        ("T_PARK", 1313, 1559),
        ("ACCELERATION", 2268, 2776),
        ("PARALLEL_PARK", 2777, 2976),
    ]


def test_stop_state_machine_holds_exactly_twenty_20hz_frames_before_release():
    event = merge_stop_events(
        [point(0, 1.0, event="STOP_LINE")], [], set())[0]
    controller = StopController(
        hold_seconds=1.0, trigger_distance_m=0.30,
        approach_distance_m=1.0)
    assert controller.update(
        0.0, Pose2D(0.5, 0.0, 0.0), [event], 0, 0) == "APPROACH_STOP"
    # Trigger 0.29 m before the stop, never after crossing it.
    pose = Pose2D(0.71, 0.0, 0.0)
    states = [controller.update(i * 0.05, pose, [event], 0, 0)
              for i in range(20)]
    assert states == ["HOLD_STOP"] * 20
    assert controller.update(1.0, pose, [event], 0, 0) == "RELEASE"
    assert controller.update(1.05, pose, [event], 0, 0) == "RUNNING"
    assert event.key in controller.consumed


def test_synthetic_motion_is_gated_and_capped_independently_from_drive_level():
    assert commanded_synthetic_speed(0.0, 0.25) == 0.0
    assert commanded_synthetic_speed(1.0, 0.25) == pytest.approx(0.125)
    assert commanded_synthetic_speed(2.0, 0.25) == pytest.approx(0.25)
    assert commanded_synthetic_speed(3.0, 0.25) == pytest.approx(0.25)
    assert commanded_synthetic_speed(-1.0, 0.25) == pytest.approx(0.25)
