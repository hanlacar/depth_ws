from pathlib import Path

import pytest

from depth_hybrid_slam.command_arbiter_core import CommandCandidate, arbitrate
from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.intersection_route import (
    cumulative_route_distance, intersection_progress, progress_from_json,
    progress_json)
from depth_hybrid_slam.traffic_gate import (
    APPROACH, INTERSECTION_COMMITTED, INTERSECTION_EXITED,
    RELEASE_PENDING, STOP_LINE_HOLD, IntersectionProgress,
    IntersectionTrafficGate)


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
METADATA = ROUTE.with_suffix(".metadata.json")


def _progress(mode=4, route_index=100, progress_m=10.0,
              segment="COMMON_1", direction=1, stop_route_index=100,
              stop_index=25, stop_progress_m=10.0, exit_route_index=130,
              exit_progress_m=20.0):
    return IntersectionProgress(
        valid=True, mode=mode, segment_id=segment, direction=direction,
        route_index=route_index, progress_m=progress_m,
        stop_segment_id="COMMON_1", stop_direction=1,
        stop_route_index=stop_route_index, stop_index=stop_index,
        stop_progress_m=stop_progress_m,
        exit_route_index=exit_route_index, exit_progress_m=exit_progress_m)


def test_mode4_red_holds_indefinitely_and_green_releases_pending():
    gate = IntersectionTrafficGate(
        unknown_hold_s=3.0, traffic_timeout_s=0.3,
        commit_margin_m=0.35)
    assert gate.evaluate(_progress(), False, "R", 0.0, 0.0).state == APPROACH
    first = gate.evaluate(_progress(), True, "R", 0.0, 1.0)
    assert first.state == STOP_LINE_HOLD and first.stop
    assert "RED_HOLD" in first.event
    assert gate.evaluate(_progress(), True, "R", 0.0, 6.1).stop
    released = gate.evaluate(_progress(), True, "G", 0.0, 6.2)
    assert released.state == RELEASE_PENDING and not released.stop


def test_mode4_red_reholds_before_line_but_is_ignored_after_commit():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    gate.evaluate(_progress(), True, "G", 0.0, 0.0)
    rehold = gate.evaluate(
        _progress(route_index=101, progress_m=10.30), False, "R", 0.0, 0.1)
    assert rehold.state == STOP_LINE_HOLD and rehold.stop
    assert "RED_REHOLD_BEFORE_LINE" in rehold.event
    gate.evaluate(_progress(route_index=101, progress_m=10.30), False,
                  "G", 0.0, 0.2)
    committed = gate.evaluate(
        _progress(route_index=102, progress_m=10.36), False, "G", 0.0, 0.3)
    assert committed.state == INTERSECTION_COMMITTED
    assert committed.committed and not committed.stop
    red = gate.evaluate(
        _progress(route_index=115, progress_m=15.0), False, "R", 0.0, 0.4)
    assert red.state == INTERSECTION_COMMITTED
    assert not red.stop and not red.traffic_stop_allowed
    assert "RED_IGNORED_AFTER_COMMIT" in red.event
    csv = CommandCandidate(2.0, -22, True, True)
    selected = arbitrate(csv, CommandCandidate(), mission_hold=red.stop)
    assert selected.state == "CSV_TRACKING" and selected.drive == 2.0
    emergency = arbitrate(csv, CommandCandidate(), hard_emergency=True,
                          mission_hold=red.stop)
    assert emergency.state == "HARD_EMERGENCY_STOP" and emergency.drive == 0.0


def test_unknown_three_seconds_then_red_rehold_or_commit_and_ignore():
    gate = IntersectionTrafficGate(unknown_hold_s=3.0)
    assert gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 0.0).stop
    assert gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 2.999).stop
    release = gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 3.0)
    assert release.state == RELEASE_PENDING and not release.stop
    assert gate.evaluate(_progress(), False, "R", 0.0, 3.1).stop

    gate = IntersectionTrafficGate(unknown_hold_s=3.0)
    gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 0.0)
    gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 3.0)
    committed = gate.evaluate(
        _progress(route_index=102, progress_m=10.5), False,
        "UNKNOWN", 0.0, 3.1)
    assert committed.state == INTERSECTION_COMMITTED
    assert not gate.evaluate(
        _progress(route_index=110, progress_m=14.0), False,
        "R", 0.0, 3.2).stop


def test_valid_red_cancels_unknown_timer_and_valid_green_releases_immediately():
    gate = IntersectionTrafficGate(unknown_hold_s=3.0)
    gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 0.0)
    assert gate.evaluate(_progress(), True, "R", 0.0, 2.9).stop
    assert gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 3.0).stop
    assert gate.evaluate(_progress(), True, "UNKNOWN", 0.0, 5.99).stop
    green = gate.evaluate(_progress(), True, "G", 0.0, 6.0)
    assert green.state == RELEASE_PENDING and not green.stop


@pytest.mark.parametrize("signal", ("R+G", "Y+G"))
def test_valid_green_has_priority_over_red_or_yellow(signal):
    gate = IntersectionTrafficGate()
    decision = gate.evaluate(_progress(), True, signal, 0.0, 0.0)
    assert decision.state == RELEASE_PENDING and not decision.stop
    assert "RED_PLUS_GREEN_GO" in decision.event


def test_crossing_and_red_in_same_update_commits_before_traffic_rehold():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    gate.evaluate(_progress(), True, "G", 0.0, 0.0)
    decision = gate.evaluate(
        _progress(route_index=102, progress_m=10.5), False,
        "R", 0.0, 0.1)
    assert decision.state == INTERSECTION_COMMITTED
    assert not decision.stop and not decision.traffic_stop_allowed
    assert "RED_IGNORED_AFTER_COMMIT" in decision.event


def test_commit_latch_is_immutable_across_g_r_g_r_sequence():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    gate.evaluate(_progress(), True, "G", 0.0, 0.0)
    gate.evaluate(_progress(route_index=102, progress_m=10.5), False,
                  "G", 0.0, 0.1)
    for sequence, now in zip(("R", "G", "R"), (0.2, 0.3, 0.4)):
        decision = gate.evaluate(
            _progress(route_index=110, progress_m=14.0), False,
            sequence, 0.0, now)
        assert decision.state == INTERSECTION_COMMITTED
        assert decision.committed and not decision.stop


@pytest.mark.parametrize("signal,event", (
    ("GREEN_LEFT", "MODE8_GREEN_LEFT_RELEASE"),
    ("R+GREEN_LEFT", "MODE8_RED_PLUS_GREEN_LEFT_GO"),
    ("Y+GREEN_LEFT", "MODE8_RED_PLUS_GREEN_LEFT_GO"),
))
def test_mode8_green_left_is_permitted_over_red_yellow(signal, event):
    gate = IntersectionTrafficGate()
    progress = _progress(mode=8)
    decision = gate.evaluate(progress, True, signal, 0.0, 0.0)
    assert decision.state == RELEASE_PENDING and not decision.stop
    assert event in decision.event


def test_green_left_is_permitted_only_for_mode8():
    progress = _progress(mode=4)
    red_plus_left = IntersectionTrafficGate().evaluate(
        progress, True, "R+GREEN_LEFT", 0.0, 0.0)
    assert red_plus_left.state == STOP_LINE_HOLD and red_plus_left.stop
    left_only = IntersectionTrafficGate().evaluate(
        progress, True, "GREEN_LEFT", 0.0, 0.0)
    assert left_only.state == STOP_LINE_HOLD and left_only.stop
    assert left_only.aspect == "UNKNOWN"


def test_mode8_red_only_and_unknown_hold_but_unknown_releases_at_three_seconds():
    progress = _progress(mode=8)
    red = IntersectionTrafficGate()
    assert red.evaluate(progress, True, "R", 0.0, 0.0).stop
    unknown = IntersectionTrafficGate()
    assert unknown.evaluate(progress, True, "UNKNOWN", 0.0, 0.0).stop
    assert unknown.evaluate(progress, True, "UNKNOWN", 0.0, 2.99).stop
    assert not unknown.evaluate(progress, True, "UNKNOWN", 0.0, 3.0).stop


def test_mode8_red_plus_left_commits_then_red_only_cannot_stop():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    line = _progress(mode=8)
    assert not gate.evaluate(
        line, True, "R+GREEN_LEFT", 0.0, 0.0).stop
    crossed = _progress(mode=8, route_index=102, progress_m=10.5)
    assert gate.evaluate(
        crossed, False, "R+GREEN_LEFT", 0.0, 0.1).committed
    red = gate.evaluate(
        _progress(mode=8, route_index=120, progress_m=16.0),
        False, "R", 0.0, 0.2)
    assert not red.stop and not red.traffic_stop_allowed
    command = arbitrate(
        CommandCandidate(2.0, 22, True, True), CommandCandidate(),
        mission_hold=red.stop)
    assert command.state == "CSV_TRACKING" and command.drive == 2.0


def test_stale_red_becomes_unknown_and_does_not_latch_forever():
    gate = IntersectionTrafficGate(unknown_hold_s=3.0, traffic_timeout_s=0.3)
    assert gate.evaluate(_progress(), True, "R", 0.0, 0.0).stop
    stale = gate.evaluate(_progress(), True, "R", 0.31, 5.0)
    assert stale.aspect == "UNKNOWN" and stale.stop
    assert gate.evaluate(_progress(), True, "R", 0.31, 7.99).stop
    release = gate.evaluate(_progress(), True, "R", 0.31, 8.0)
    assert release.state == RELEASE_PENDING and not release.stop


def test_wrong_segment_direction_and_backtrack_cannot_commit():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    gate.evaluate(_progress(), True, "G", 0.0, 0.0)
    assert gate.evaluate(
        _progress(route_index=101, progress_m=10.2), False,
        "G", 0.0, 0.1).state == RELEASE_PENDING
    assert gate.evaluate(
        _progress(route_index=100, progress_m=10.8), False,
        "G", 0.0, 0.2).state == RELEASE_PENDING
    assert gate.evaluate(
        _progress(route_index=102, progress_m=10.8, segment="OTHER"), False,
        "G", 0.0, 0.3).state == RELEASE_PENDING
    assert gate.evaluate(
        _progress(route_index=102, progress_m=10.8, direction=-1), False,
        "G", 0.0, 0.4).state == RELEASE_PENDING
    assert gate.evaluate(
        _progress(route_index=102, progress_m=10.8), False,
        "G", 0.0, 0.5).state == INTERSECTION_COMMITTED


def test_exit_resets_latch_and_mode6_gets_an_independent_gate():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    gate.evaluate(_progress(), True, "G", 0.0, 0.0)
    gate.evaluate(_progress(route_index=102, progress_m=10.5), False,
                  "G", 0.0, 0.1)
    exited = gate.evaluate(
        _progress(mode=5, route_index=131, progress_m=20.1), False,
        "R", 0.0, 0.2)
    assert exited.state == INTERSECTION_EXITED and not exited.stop
    mode6 = _progress(
        mode=6, route_index=200, progress_m=30.0,
        stop_route_index=200, stop_index=554, stop_progress_m=30.0,
        exit_route_index=240, exit_progress_m=40.0)
    hold = gate.evaluate(mode6, True, "R", 0.0, 0.3)
    assert hold.state == STOP_LINE_HOLD and hold.stop
    assert not hold.committed


def test_mode6_right_turn_keeps_csv_command_after_committed_red():
    gate = IntersectionTrafficGate(commit_margin_m=0.35)
    line = _progress(
        mode=6, route_index=200, progress_m=30.0,
        stop_route_index=200, stop_index=554, stop_progress_m=30.0,
        exit_route_index=250, exit_progress_m=45.0)
    assert gate.evaluate(line, True, "G", 0.0, 0.0).state == RELEASE_PENDING
    crossed = _progress(
        mode=6, route_index=204, progress_m=30.5,
        stop_route_index=200, stop_index=554, stop_progress_m=30.0,
        exit_route_index=250, exit_progress_m=45.0)
    assert gate.evaluate(crossed, False, "G", 0.0, 0.1).committed
    turn = _progress(
        mode=6, route_index=225, progress_m=38.0,
        stop_route_index=200, stop_index=554, stop_progress_m=30.0,
        exit_route_index=250, exit_progress_m=45.0)
    decision = gate.evaluate(turn, False, "R", 0.0, 0.2)
    command = arbitrate(
        CommandCandidate(1.0, -22, True, True), CommandCandidate(),
        mission_hold=decision.stop)
    assert command.state == "CSV_TRACKING"
    assert command.drive == 1.0 and command.wheel == -22


def test_actual_csv_context_has_exact_mode4_and_mode6_stop_indexes():
    for case, expected_route_indexes in (
            ("AAAA", (678, 1207, 1957)),
            ("BAAA", (660, 1189, 1939))):
        route = load_csv_only_route_case(ROUTE, METADATA, case)
        cumulative = cumulative_route_distance(route)
        for mode, stop_index, route_index in (
                (4, 25, expected_route_indexes[0]),
                (6, 554, expected_route_indexes[1]),
                (8, 397, expected_route_indexes[2])):
            value = intersection_progress(
                route, cumulative, route_index,
                route_index/max(1, len(route)-1))
            assert value.valid and value.mode == mode
            expected_segment = "COMMON_2" if mode == 8 else "COMMON_1"
            assert value.segment_id == value.stop_segment_id == expected_segment
            assert value.direction == value.stop_direction == 1
            assert value.stop_route_index == route_index
            assert value.stop_index == stop_index
            assert value.exit_route_index > value.stop_route_index
            assert progress_from_json(progress_json(value)) == value
