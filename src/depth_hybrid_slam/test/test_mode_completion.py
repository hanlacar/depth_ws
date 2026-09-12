import json
from pathlib import Path

import pytest

from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.mode_completion import RouteModeCompletionTracker
from depth_hybrid_slam.models import RoutePoint


def point(index, mode, segment="COMMON", point_index=None):
    return RoutePoint(
        index, float(index), 0.0, 0.0, segment_id=segment,
        point_index=index if point_index is None else point_index, mode=mode)


def route_three_modes():
    # Mode 2 changes inside COMMON and Mode 3 spans a segment boundary.  This
    # proves neither segment start nor segment end drives completion.
    return [
        point(0, 1, "START"), point(1, 1, "START"),
        point(2, 2, "COMMON"), point(3, 2, "COMMON"),
        point(4, 3, "COMMON"), point(5, 3, "END"),
    ]


def observe(tracker, index, *, healthy=True, stopped=False,
            failure_reason="", reason="OK"):
    return tracker.observe(
        index, index/5.0, healthy=healthy, stopped=stopped,
        failure_reason=failure_reason, controller_reason=reason)


def test_route_only_lifecycle_is_ordered_exactly_once_and_course_is_terminal():
    tracker = RouteModeCompletionTracker(route_three_modes())
    events = []
    for index in (0, 0, 1, 2, 2, 3, 4, 4):
        events.extend(observe(tracker, index))
    # Near the final waypoint alone is not enough; existing ROUTE_COMPLETE is
    # required for Mode 3 and course completion.
    assert "[MODE 3 COMPLETE] criterion=ROUTE_ONLY" not in events
    events.extend(observe(tracker, 4, reason="ROUTE_COMPLETE"))
    events.extend(observe(tracker, 4, reason="ROUTE_COMPLETE"))
    assert events == [
        "[MODE 1 START]",
        "[MODE 1 COMPLETE] criterion=ROUTE_ONLY", "[MODE 2 START]",
        "[MODE 2 COMPLETE] criterion=ROUTE_ONLY", "[MODE 3 START]",
        "[MODE 3 COMPLETE] criterion=ROUTE_ONLY", "[COURSE COMPLETE]",
    ]
    assert tracker.started_modes == {1, 2, 3}
    assert tracker.completed_modes == {1, 2, 3}
    assert tracker.course_complete


def test_route_mission_and_mode_completion_remain_separate():
    tracker = RouteModeCompletionTracker([point(0, 1), point(1, 1)])
    observe(tracker, 0)
    events = observe(tracker, 0, reason="ROUTE_COMPLETE")
    state = json.loads(tracker.status_json())
    assert events == (
        "[MODE 1 COMPLETE] criterion=ROUTE_ONLY", "[COURSE COMPLETE]")
    assert state["route_complete"] is True
    assert state["mission_complete"] is False
    assert state["mode_complete"] is True
    tracker.mark_mission_complete(1)
    assert tracker.status()["mission_complete"] is True


@pytest.mark.parametrize("blocked", ["stop", "failure", "backtrack"])
def test_stop_failure_and_cursor_backtrack_cannot_complete_a_mode(blocked):
    tracker = RouteModeCompletionTracker(route_three_modes())
    observe(tracker, 0)
    observe(tracker, 1)
    if blocked == "stop":
        events = observe(tracker, 2, stopped=True)
    elif blocked == "failure":
        events = observe(
            tracker, 2, healthy=False,
            failure_reason="CORRIDOR_VIOLATION")
    else:
        observe(tracker, 0)  # invalidates Mode 1 before its transition
        events = observe(tracker, 2)
    assert "[MODE 1 COMPLETE] criterion=ROUTE_ONLY" not in events
    assert 1 not in tracker.completed_modes
    assert 1 in tracker.invalid_modes


def test_skipping_last_waypoint_cannot_complete_even_when_mode_changes():
    tracker = RouteModeCompletionTracker([
        point(0, 1), point(1, 1), point(2, 1),
        point(3, 2), point(4, 2)])
    observe(tracker, 0)
    events = observe(tracker, 3)
    assert events == ("[MODE 2 START]",)
    assert tracker.last_error == "WAYPOINT_PROGRESS_INCOMPLETE"
    assert tracker.completed_modes == set()


def test_one_sample_boundary_crossing_proves_last_waypoint_was_passed():
    tracker = RouteModeCompletionTracker(route_three_modes())
    observe(tracker, 0)
    events = observe(tracker, 2)
    assert events == (
        "[MODE 1 COMPLETE] criterion=ROUTE_ONLY", "[MODE 2 START]")
    assert tracker.completed_modes == {1}


def test_branch_route_rebind_preserves_history_and_maps_current_progress():
    route_a = route_three_modes()
    route_b = [
        point(0, 1, "START_B"), point(1, 1, "START_B"),
        point(2, 2, "COMMON"), point(3, 2, "T_B"),
        point(4, 2, "T_B"), point(5, 3, "END"), point(6, 3, "END"),
    ]
    tracker = RouteModeCompletionTracker(route_a)
    observe(tracker, 0)
    observe(tracker, 1)
    observe(tracker, 2)
    assert tracker.completed_modes == {1}
    tracker.bind_route(route_b, resume_index=2)
    assert tracker.completed_modes == {1}
    assert tracker.current_mode == 2
    observe(tracker, 3)
    observe(tracker, 4)
    events = observe(tracker, 5)
    assert events == (
        "[MODE 2 COMPLETE] criterion=ROUTE_ONLY", "[MODE 3 START]")


def test_rebind_to_a_different_mode_is_invalid_not_complete():
    tracker = RouteModeCompletionTracker(route_three_modes())
    observe(tracker, 0)
    tracker.bind_route(route_three_modes(), resume_index=2)
    assert tracker.invalid_modes == {1}
    assert tracker.last_error == "BRANCH_REMAP_MODE_JUMP"


@pytest.mark.parametrize("failure_reason", [
    "ROUTE_DEVIATION_STOP", "CORRIDOR_VIOLATION", "OFF_ROUTE_STOP",
    "WRONG_BRANCH", "FATAL_ROUTE_ERROR", "EMERGENCY_STOP",
    "FOLLOWER_FAILURE",
])
def test_named_failure_states_never_emit_complete(failure_reason):
    tracker = RouteModeCompletionTracker(route_three_modes())
    observe(tracker, 0)
    observe(tracker, 1)
    events = observe(
        tracker, 2, healthy=False, failure_reason=failure_reason)
    assert not any(" COMPLETE]" in event for event in events)
    assert tracker.completed_modes == set()
    assert tracker.invalid_modes == {1}


def test_route_modes_must_be_contiguous_and_consecutive():
    with pytest.raises(ValueError, match="contiguous"):
        RouteModeCompletionTracker([
            point(0, 1), point(1, 2), point(2, 1)])
    with pytest.raises(ValueError, match="consecutively"):
        RouteModeCompletionTracker([point(0, 1), point(1, 3)])


@pytest.mark.parametrize("route_case", ["AAAA", "BAAA"])
def test_actual_candidate_route_emits_modes_1_through_11_once(route_case):
    root = Path(__file__).resolve().parents[3]
    route_path = (
        root / "routes/network/route_network_segmented_stop_edited_vforward.csv")
    route = load_csv_only_route_case(
        route_path, route_path.with_suffix(".metadata.yaml"), route_case)
    tracker = RouteModeCompletionTracker(route)
    events = []
    # RouteFollower reports completion on the final segment (len-2) at >0.9
    # projection, so this mirrors the real cursor contract exactly.
    for index in range(len(route)-1):
        events.extend(tracker.observe(
            index, index/max(1, len(route)-1), healthy=True,
            controller_reason="OK"))
    events.extend(tracker.observe(
        len(route)-2, 1.0, healthy=True,
        controller_reason="ROUTE_COMPLETE"))
    assert [event for event in events if event.endswith("START]")] == [
        f"[MODE {mode} START]" for mode in range(1, 12)]
    assert [event for event in events
            if event.startswith("[MODE ") and " COMPLETE]" in event] == [
        f"[MODE {mode} COMPLETE] criterion=ROUTE_ONLY"
        for mode in range(1, 12)]
    assert events[-1] == "[COURSE COMPLETE]"
