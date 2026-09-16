import ast
import itertools
import math
from pathlib import Path

import cv2
from depth_hybrid_slam.command_arbiter_core import arbitrate, CommandCandidate
from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.lidar_local_planner import plan_detour, plan_parking
from depth_hybrid_slam.lidar_mission_core import Mode11ExitGate
from depth_hybrid_slam.lidar_roi_core import (
    ackermann_centerline, assess_curved_roi, Cluster, DYNAMIC, STATIC)
from depth_hybrid_slam.mission_completion import MissionCompletionTracker
from depth_hybrid_slam.mode_completion import RouteModeCompletionTracker
from depth_hybrid_slam.signal_exit_core import (
    classify_exit_triplet, DetectorConfig, HSVRange, NormalizedROI, ObservationState,
    SelectedRoute, SignalDetection, SignalExitDetector, SignalState,
    SignalVoteWindow)
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
METADATA = ROUTE.with_suffix(".metadata.yaml")


def _detector():
    return SignalExitDetector(DetectorConfig(
        red_ranges=(HSVRange((0, 100, 100), (10, 255, 255)),
                    HSVRange((170, 100, 100), (179, 255, 255))),
        core_green=HSVRange((40, 100, 100), (85, 255, 255)),
        extended_green=HSVRange((86, 60, 50), (110, 255, 125)),
        minimum_contour_area=2.0, minimum_color_pixel_ratio=0.003,
        red_dark_ratio=0.30, core_green_dark_ratio=0.30,
        extended_green_dark_ratio=0.50,
        candidate_bounds=(0.0, 1.0, 0.0, 1.0)),
        NormalizedROI(0.1, 0.1, 0.9, 0.9))


def _frame(states):
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    for center, state in zip((25, 50, 75), states):
        hue = 60 if state == "G" else 2
        hsv[40:60, center-8:center+8] = (0, 0, 30)
        hsv[46:54, center-4:center+4] = (hue, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _cluster(distance, motion=STATIC):
    return Cluster(((distance, -0.01), (distance, 0.01)),
                   (distance, 0.0), motion=motion)


def test_mode11_detector_uses_position_specific_a_b_patterns():
    detector = _detector()
    assert detector.detect(_frame("GRR")).state == SignalState.GREEN
    assert detector.detect(_frame("RGR")).state == SignalState.RED
    for states in ("RRG", "RRR"):
        assert detector.detect(_frame(states)).state == SignalState.UNKNOWN
    assert detector.detect(_frame("GR")).state == SignalState.RED


@pytest.mark.parametrize("states,expected", (
    (("G", "R", "R"), SignalState.GREEN),
    (("R", "G", "R"), SignalState.RED),
    (("R", "R", "G"), SignalState.UNKNOWN),
    (("R", "R", "R"), SignalState.UNKNOWN),
    (("UNKNOWN", "R", "R"), SignalState.GREEN),
))
def test_mode11_triplet_contract(states, expected):
    candidates = tuple(
        ((index+1)*0.2, state, 20) for index, state in enumerate(states))
    assert classify_exit_triplet(candidates)[0] == expected


def test_merged_five_second_vote_camera_and_default_are_latched():
    camera = SignalVoteWindow(5.0, 4, 0.75)
    assert camera.start(0.0)
    for now in (0.0, 1.0, 2.0, 3.0):
        camera.observe(SignalState.RED, now)
    assert camera.evaluate(4.99).state == ObservationState.OBSERVING
    result = camera.evaluate(5.0)
    assert result.state == ObservationState.LATCHED
    assert result.route == SelectedRoute.B
    assert camera.observe(SignalState.GREEN, 6.0).route == SelectedRoute.B

    default = SignalVoteWindow(5.0, 4, 0.75)
    default.start(0.0)
    result = default.evaluate(5.0)
    assert result.state == ObservationState.DEFAULTED
    assert result.route == SelectedRoute.A


@pytest.mark.parametrize("signal,expected", (
    ("A", "A"), ("1", "A"), ("B", "B"), ("2", "B")))
def test_mode11_actual_hold_mapping_and_late_opposite(signal, expected):
    gate = Mode11ExitGate(confirmations=2)
    gate.enter(0.0)
    gate.observe(signal, 4.8)
    gate.observe(signal, 4.9)
    assert gate.evaluate(4.999).stop
    selected = gate.evaluate(5.0)
    assert selected.branch == expected and not selected.stop
    opposite = "B" if expected == "A" else "A"
    gate.observe(opposite, 5.1)
    assert gate.evaluate(8.0).branch == expected


def test_mode11_stale_unknown_and_divided_votes_default_a():
    for observations in (("UNKNOWN", "UNKNOWN"), ("A", "B")):
        gate = Mode11ExitGate(confirmations=2)
        gate.enter(0.0)
        for index, signal in enumerate(observations):
            gate.observe(signal, 0.1+index*0.1)
        assert gate.evaluate(5.0).branch == "A"
        assert gate.commit_source == "DEFAULT_A"


def _exit_detection(lamps):
    state = SignalState.UNKNOWN
    if lamps[0] == SignalState.GREEN:
        state = SignalState.GREEN
    elif lamps[1] == SignalState.GREEN:
        state = SignalState.RED
    return SignalDetection(state, 0, 0, tuple(lamps), (), .9)


@pytest.mark.parametrize("lamps,expected,reason", (
    ((SignalState.GREEN, SignalState.RED, SignalState.RED),
     SelectedRoute.A, "LEFT_GREEN"),
    ((SignalState.RED, SignalState.GREEN, SignalState.RED),
     SelectedRoute.B, "CENTER_GREEN"),
    ((SignalState.UNKNOWN, SignalState.RED, SignalState.RED),
     SelectedRoute.A, "CENTER_RED+RIGHT_RED"),
    ((SignalState.RED, SignalState.UNKNOWN, SignalState.RED),
     SelectedRoute.B, "LEFT_RED+RIGHT_RED"),
))
def test_mode11_position_evidence_selects_a_or_b_after_five_seconds(
        lamps, expected, reason):
    window = SignalVoteWindow(5.0, 1, .75)
    window.start(0.0)
    window.observe_detection(_exit_detection(lamps), 1.0)
    window.observe_detection(_exit_detection(lamps), 2.0)
    assert window.evaluate(4.99).route == SelectedRoute.UNKNOWN
    result = window.evaluate(5.0)
    assert result.route == expected
    assert result.reason == reason


def test_mode11_missing_green_frames_do_not_erase_direct_green_history():
    window = SignalVoteWindow(5.0, 1, .75, green_weight=3.0)
    window.start(0.0)
    window.observe_detection(_exit_detection((
        SignalState.GREEN, SignalState.RED, SignalState.RED)), .1)
    window.observe_detection(_exit_detection((
        SignalState.GREEN, SignalState.RED, SignalState.RED)), .2)
    for now in (1.0, 2.0, 3.0, 4.0):
        window.observe_detection(_exit_detection((
            SignalState.UNKNOWN, SignalState.RED, SignalState.RED)), now)
    result = window.evaluate(5.0)
    assert result.route == SelectedRoute.A
    assert result.position_counts[0][1] == 2


def test_mode11_inconclusive_or_multi_red_always_defaults_a_not_unknown():
    window = SignalVoteWindow(5.0, 2, .75)
    window.start(0.0)
    window.observe_detection(_exit_detection((
        SignalState.RED, SignalState.RED, SignalState.RED)), 1.0)
    result = window.evaluate(5.0)
    assert result.route == SelectedRoute.A
    assert result.state == ObservationState.DEFAULTED


def test_mode11_weighted_final_decision_can_commit_actual_gate():
    gate = Mode11ExitGate(confirmations=60)
    gate.enter(0.0)
    assert gate.commit_external("B", "CAMERA_WEIGHTED:CENTER_GREEN")
    assert gate.evaluate(4.99).stop
    result = gate.evaluate(5.0)
    assert result.branch == "B" and not result.stop
    assert gate.commit_source == "CAMERA_WEIGHTED:CENTER_GREEN"


def _satisfy_all_missions(tracker, end_branch):
    tracker.set_mode(2)
    tracker.tick(0.0, 0.0, stop_waypoint_active=True,
                 pitch_deg=6.0, pitch_valid=True)
    tracker.tick(4.0, 0.0, pitch_deg=6.0, pitch_valid=True)
    for mode in (4, 6, 8):
        tracker.observe_intersection_event(f"EXITED mode={mode}")
    tracker.set_mode(5)
    tracker.observe_maneuver({"mode": 5, "event": "AVOIDANCE_REJOINED"})
    tracker.observe_maneuver({"mode": 5, "event": "AVOIDANCE_REJOINED"})
    for mode in (7, 10):
        tracker.set_mode(mode)
        tracker.observe_lidar_safety({"mode": mode, "slot_a": True})
        tracker.observe_maneuver({
            "mode": mode, "event": "PARKING_CSV_REJOINED",
            "source": "LIDAR"})
    tracker.set_mode(9)
    tracker.observe_lidar_safety({
        "mode": 9, "hard_stop": True, "distance_m": 0.49})
    tracker.tick(9.0, 0.0)
    tracker.set_mode(11)
    tracker.tick(11.0, 0.0, maneuver_state="MODE11_5S_HOLD")
    tracker.tick(16.0, 0.0, maneuver_state="MODE11_COMMITTED")
    tracker.observe_maneuver({
        "mode": 11, "event": "MODE11_BRANCH_COMMITTED",
        "branch": end_branch, "source": "CAMERA"})


@pytest.mark.parametrize(
    "route_case", ["".join(value) for value in itertools.product("AB", repeat=4)])
def test_all_16_routes_complete_only_expected_topology_and_all_missions(
        route_case):
    route = load_csv_only_route_case(ROUTE, METADATA, route_case)
    segment_ids = {point.segment_id for point in route}
    expected = {
        "START_"+route_case[0], "T_"+route_case[1],
        "V_"+route_case[2], "END_A"+route_case[3]}
    opposite = {
        "START_"+("B" if route_case[0] == "A" else "A"),
        "T_"+("B" if route_case[1] == "A" else "A"),
        "V_"+("B" if route_case[2] == "A" else "A"),
        "END_A"+("B" if route_case[3] == "A" else "A")}
    assert expected <= segment_ids
    assert not opposite & segment_ids
    assert [*dict.fromkeys(point.mode for point in route)] == list(range(1, 12))
    assert [point.index for point in route] == list(range(len(route)))

    route_tracker = RouteModeCompletionTracker(route)
    for index in range(len(route)-1):
        route_tracker.observe(
            index, index/max(1, len(route)-1),
            healthy=True, controller_reason="OK")
    events = route_tracker.observe(
        len(route)-2, 1.0, healthy=True,
        controller_reason="ROUTE_COMPLETE")
    assert "[COURSE COMPLETE]" in events
    assert not route_tracker.invalid_modes

    mission = MissionCompletionTracker()
    _satisfy_all_missions(mission, route_case[3])
    mission.observe_route_status({"route_complete_modes": list(range(1, 12))})
    mission.tick(17.0, 0.0)
    assert mission.status()["course_complete"]
    assert not mission.status()["invalid_modes"]


@pytest.mark.parametrize("distance,hard,slow", (
    (0.49, True, False), (0.50, True, False),
    (0.51, False, True), (0.99, False, True), (1.00, False, True),
    (1.01, False, False), (1.49, False, False),
    (1.50, False, False), (1.51, False, False)))
def test_front_roi_exact_static_boundaries(distance, hard, slow):
    value = assess_curved_roi((_cluster(distance),), 0.0)
    assert value.hard_stop is hard
    assert value.slowdown is slow


@pytest.mark.parametrize("distance,slow", (
    (1.49, False), (1.50, False), (1.51, False)))
def test_front_roi_dynamic_zone3_boundaries(distance, slow):
    assert assess_curved_roi(
        (_cluster(distance, DYNAMIC),), 0.0).slowdown is slow


@pytest.mark.parametrize("steering", (-22, -20, -10, -5, 0, 5, 10, 20, 22))
def test_roi_origin_and_steering_sign_for_full_contract(steering):
    centerline, _, _ = ackermann_centerline(steering)
    assert centerline[0] == (0.0, 0.0, 0.0)
    assert math.isclose(centerline[-1][0], 1.5)
    assert math.copysign(1.0, centerline[-1][2]) == math.copysign(
        1.0, steering) if steering else abs(centerline[-1][2]) < 1.0e-9


def test_mode5_obstacle_orders_and_all_generated_paths_are_below_22():
    maximum = 0.0
    for order in ((0.3, -0.3), (-0.3, 0.3)):
        laterals = []
        for obstacle_y in order:
            plan = plan_detour(obstacle_y)
            assert plan.valid and plan.max_steering_deg < 22.0
            maximum = max(maximum, plan.max_steering_deg)
            laterals.append(max(plan.points, key=lambda p: abs(p[1]))[1])
        assert laterals[0]*order[0] < 0.0
        assert laterals[1]*order[1] < 0.0
    for mode in (7, 10):
        for branch in ("A", "B"):
            plan = plan_parking(mode, branch)
            assert plan.valid and plan.max_steering_deg < 22.0
            maximum = max(maximum, plan.max_steering_deg)
    assert maximum <= 21.0+1.0e-9


def test_arbiter_all_priority_combinations_and_stale_source_guard_exists():
    csv = CommandCandidate(3.0, 10, True, True)
    lidar = CommandCandidate(1.0, -10, True, True)
    assert arbitrate(csv, CommandCandidate()).owner == "CSV"
    assert arbitrate(csv, CommandCandidate(), lidar_slowdown=True).drive == 1.0
    assert arbitrate(csv, CommandCandidate(), hard_emergency=True).owner == "SAFETY"
    assert arbitrate(csv, lidar).owner == "LIDAR"
    assert arbitrate(csv, lidar, hard_emergency=True).owner == "SAFETY"
    assert arbitrate(csv, lidar, mission_hold=True).owner == "MISSION"
    assert arbitrate(CommandCandidate(), CommandCandidate()).owner == "NONE"
    source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/command_arbiter_node.py") \
        .read_text(encoding="utf-8")
    for key in ("mission", "branch", "lidar_hold"):
        assert f'not self._fresh(("{key}",), now)' in source
    assert "lidar_safety_fresh = self._fresh(" in source
    for key in ("hard", "front_scan_fresh", "distance_slowdown",
                "steering_slowdown"):
        assert f'"{key}"' in source
    assert "front_hard_emergency_applies(" in source
    assert 'steering_fresh = self._fresh(("steering",), now)' in source
    assert "safety_stop_reasons(" in source


@pytest.mark.parametrize("stop_s,pitch,route_complete,expected", (
    (4.0, 6.0, True, True),
    (4.0, 4.5, True, True),
    (4.0, -4.5, True, True),
    (3.9, 6.0, True, False),
    (4.0, 4.49, True, False),
    (0.0, 6.0, True, False),
    (4.0, 6.0, False, False),
))
def test_mode2_all_stop_pitch_route_combinations(
        stop_s, pitch, route_complete, expected):
    tracker = MissionCompletionTracker()
    tracker.set_mode(2)
    tracker.tick(0.0, 0.0, stop_waypoint_active=True,
                 pitch_deg=pitch, pitch_valid=True)
    tracker.tick(stop_s, 0.0, pitch_deg=pitch, pitch_valid=True)
    if route_complete:
        tracker.observe_route_status({"route_complete_modes": [2]})
    assert tracker.mode_complete(2) is expected


def test_mode2_pitch_requires_half_second_continuity():
    tracker = MissionCompletionTracker()
    tracker.set_mode(2)
    tracker.tick(0.0, 0.0, stop_waypoint_active=True,
                 pitch_deg=4.5, pitch_valid=True)
    tracker.tick(0.49, 0.0, pitch_deg=4.5, pitch_valid=True)
    assert not tracker.mode2_slope_seen
    tracker.tick(0.50, 0.0, pitch_deg=4.49, pitch_valid=True)
    tracker.tick(0.51, 0.0, pitch_deg=4.5, pitch_valid=True)
    tracker.tick(1.00, 0.0, pitch_deg=4.5, pitch_valid=True)
    assert not tracker.mode2_slope_seen
    tracker.tick(1.01, 0.0, pitch_deg=4.5, pitch_valid=True)
    assert tracker.mode2_slope_seen


@pytest.mark.parametrize("mode", (7, 10))
@pytest.mark.parametrize("slot,source,expected", (
    ("A", "LIDAR", True), ("B", "LIDAR", True),
    ("A", "CSV_FALLBACK", True), ("B", "CSV_FALLBACK", True),
    ("", "CSV_FALLBACK", False), ("", "LIDAR", False)))
def test_parking_slot_source_matrix(mode, slot, source, expected):
    tracker = MissionCompletionTracker()
    tracker.set_mode(mode)
    tracker.observe_route_status({"route_complete_modes": [mode]})
    if slot:
        tracker.observe_lidar_safety({
            "mode": mode, "slot_a": slot == "A", "slot_b": slot == "B"})
    if source == "LIDAR":
        tracker.observe_maneuver({
            "mode": mode, "event": "PARKING_CSV_REJOINED",
            "source": "LIDAR"})
    else:
        tracker.observe_maneuver({
            "mode": mode, "event": "PARKING_CSV_FALLBACK"})
        tracker.observe_csv_parking_complete(mode)
    assert tracker.mode_complete(mode) is expected


def test_mode5_completion_count_matrix_and_mode11_requires_route():
    for count, expected in ((0, False), (1, False), (2, True), (3, True)):
        tracker = MissionCompletionTracker()
        tracker.set_mode(5)
        tracker.observe_route_status({"route_complete_modes": [5]})
        for _ in range(count):
            tracker.observe_maneuver({
                "mode": 5, "event": "AVOIDANCE_REJOINED"})
        assert tracker.mode_complete(5) is expected

    mode11 = MissionCompletionTracker()
    mode11.set_mode(11)
    mode11.tick(0.0, 0.0, maneuver_state="MODE11_5S_HOLD")
    mode11.tick(5.0, 0.0, maneuver_state="MODE11_COMMITTED")
    mode11.observe_maneuver({
        "mode": 11, "event": "MODE11_BRANCH_COMMITTED",
        "branch": "A", "source": "DEFAULT"})
    assert mode11.mission_complete(11)
    assert not mode11.mode_complete(11)


def test_only_arbiter_publishes_final_commands_and_signal_exit_never_does():
    owners = {"/cmd_drive": [], "/cmd_wheel": []}
    for path in (ROOT/"src").rglob("*.py"):
        if any(part in ("build", "install", "__pycache__") for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and
                    node.func.attr == "create_publisher" and len(node.args) > 1 and
                    isinstance(node.args[1], ast.Constant) and
                    node.args[1].value in owners):
                owners[node.args[1].value].append(path.name)
    assert owners == {"/cmd_drive": ["command_arbiter_node.py"],
                      "/cmd_wheel": ["command_arbiter_node.py"]}
    signal_source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/" /
                     "signal_exit_node.py").read_text(encoding="utf-8")
    assert "/cmd_drive" not in signal_source and "/cmd_wheel" not in signal_source


def test_mode11_republishes_same_branch_used_by_mode10():
    source = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/" /
              "maneuver_manager_node.py").read_text(encoding="utf-8")
    transition = source.split("def _transition_mode(self):", 1)[1].split(
        "def _decision(self):", 1)[0]
    assert "self.last_branch = None" in transition
