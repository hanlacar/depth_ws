from pathlib import Path

from depth_hybrid_slam.command_arbiter_core import (
    arbitrate, CommandCandidate, front_hard_emergency_applies,
    parking_reverse_requested)
from depth_hybrid_slam.csv_only_branching import (
    csv_only_network_segments, StartBranchClassifier)
from depth_hybrid_slam.lidar_mission_core import (
    Mode9EmergencyLatch, parking_decision, SteeringSlowdownLatch)
from depth_hybrid_slam.mission_completion import MissionCompletionTracker
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.segment_result import SegmentResultLatch
from depth_hybrid_slam.start_validation import StartDoubleValidator


ROOT = Path(__file__).resolve().parents[3]
ROUTE = (ROOT / "routes/network" /
         "route_network_segmented_stop_edited_vforward.csv")
METADATA = ROUTE.with_suffix(".metadata.yaml")


def _classifier():
    segments = csv_only_network_segments(ROUTE, METADATA)
    classifier = StartBranchClassifier(
        segments["START_A"], segments["START_B"])
    return classifier, segments


def _pose(point):
    return Pose2D(point.x, point.y, point.yaw, 0.0)


def test_start_a_a_double_validation():
    classifier, segments = _classifier()
    result = StartDoubleValidator("A", classifier).update(
        _pose(segments["START_A"][0]), True, 0.0)
    assert result.final and result.passed and not result.stop
    assert result.detected_branch == "A"


def test_start_b_b_double_validation():
    classifier, segments = _classifier()
    result = StartDoubleValidator("B", classifier).update(
        _pose(segments["START_B"][0]), True, 0.0)
    assert result.final and result.passed and not result.stop
    assert result.detected_branch == "B"


def test_start_a_b_mismatch_latches_fail_stop():
    classifier, segments = _classifier()
    validator = StartDoubleValidator("A", classifier)
    first = validator.update(_pose(segments["START_B"][0]), True, 0.0)
    second = validator.update(_pose(segments["START_A"][0]), True, 1.0)
    assert first.final and not first.passed and first.stop
    assert second == first and "VSLAM=B USER=A" in first.reason


def test_segment_result_is_exactly_one_shot():
    latch = SegmentResultLatch()
    assert latch.report(5, True, "avoidance count=2") == \
        "[SEGMENT 5] COMPLETE - avoidance count=2"
    assert latch.report(5, False, "late failure") is None
    assert len(latch.results) == 1


def test_selected_end_mode_failure_is_advisory_and_does_not_stop_motion():
    latch = SegmentResultLatch(end_mode=5)
    assert not latch.terminal
    assert latch.report(4, True, "") == "[SEGMENT 4] COMPLETE"
    assert not latch.terminal
    assert latch.report(5, False, "blocked") == \
        "[SEGMENT 5] FAIL - blocked"
    assert not latch.terminal


def test_failure_is_advisory_for_every_mode_and_never_latches_stop():
    for mode in range(1, 12):
        latch = SegmentResultLatch(end_mode=mode)
        assert latch.report(mode, False, "validation only") == \
            f"[SEGMENT {mode}] FAIL - validation only"
        assert not latch.terminal


def test_selected_end_mode_complete_latches_terminal_stop():
    latch = SegmentResultLatch(end_mode=5)
    assert latch.report(5, True, "") == "[SEGMENT 5] COMPLETE"
    assert latch.terminal


def test_mode2_failed_validation_releases_four_second_hold_and_keeps_driving():
    completion = MissionCompletionTracker()
    completion.set_mode(2)
    results = SegmentResultLatch(end_mode=2)

    completion.tick(
        0.0, 0.0, stop_waypoint_active=True, pitch_valid=False)
    assert completion.force_mode2_stop()
    completion.tick(
        3.999, 0.0, stop_waypoint_active=True, pitch_valid=False)
    assert completion.force_mode2_stop()
    completion.tick(
        4.0, 0.0, stop_waypoint_active=True, pitch_valid=False)
    assert not completion.force_mode2_stop()

    events = completion.drain_events()
    assert any(event["event"] == "MODE2_STOP_EVALUATED" and
               not event["passed"] for event in events)
    assert any(event["event"] == "MODE2_SLOPE_INVALID" for event in events)
    assert results.report(2, False, "") == "[SEGMENT 2] FAIL"
    assert not results.terminal


def test_parking_defaults_a_unless_b_is_explicitly_detected():
    for mode, prefix in ((7, "T"), (10, "V")):
        decision = parking_decision(mode, False, False, False)
        assert not decision.stop and decision.branch == "A"
        assert decision.state == prefix + "_SELECT_A"
        both = parking_decision(mode, True, True, False)
        assert both.branch == "B"


def test_parking_reverse_ignores_only_fresh_front_obstacle_hard_stop():
    reverse_csv = CommandCandidate(-1.0, 5, True, True)
    reverse_lidar = CommandCandidate(-1.0, -5, True, True)
    forward = CommandCandidate(1.0, 0, True, True)
    empty = CommandCandidate()

    for mode in (7, 10):
        assert parking_reverse_requested(mode, reverse_csv, empty)
        assert parking_reverse_requested(mode, forward, reverse_lidar)
        assert not front_hard_emergency_applies(
            True, True, mode, reverse_csv, empty)
        assert not front_hard_emergency_applies(
            True, True, mode, forward, reverse_lidar)
        assert front_hard_emergency_applies(
            True, True, mode, forward, empty)
        assert front_hard_emergency_applies(
            True, False, mode, reverse_csv, empty)

    assert front_hard_emergency_applies(
        True, True, 9, reverse_csv, empty)
    stale_reverse = CommandCandidate(-1.0, 0, True, False)
    assert front_hard_emergency_applies(
        True, True, 7, stale_reverse, empty)


def test_mode9_hard_stop_clear_requires_over_1_5m_for_full_second():
    latch = Mode9EmergencyLatch(1.5, 1.0)
    assert latch.update(True, 0.5, True, True, 0.0)
    assert latch.update(False, 1.5, True, True, 1.0)
    assert latch.update(False, 1.51, True, True, 2.0)
    assert latch.update(False, 1.51, True, True, 2.99)
    assert not latch.update(False, 1.51, True, True, 3.0)


def test_steering_exact_ten_one_second_and_clear_one_second():
    latch = SteeringSlowdownLatch(10.0, 1.0, 1.0)
    assert not latch.update(10.0, 0.0)
    assert not latch.update(10.0, 0.99)
    assert latch.update(10.0, 1.0)
    assert latch.update(9.99, 1.01)
    assert latch.update(9.99, 2.0)
    assert not latch.update(9.99, 2.02)


def test_command_priority_stop_distance_steering_then_mode9_speed():
    csv = CommandCandidate(2.0, 10, True, True)
    empty = CommandCandidate()
    assert arbitrate(csv, empty, mode=9).drive == 3.0
    assert arbitrate(csv, empty, mode=9,
                     steering_slowdown=True).drive == 3.0
    assert arbitrate(csv, empty, mode=9, lidar_slowdown=True,
                     steering_slowdown=True).state == "LIDAR_DISTANCE_SLOWDOWN"
    assert arbitrate(csv, empty, mission_hold=True, mode=9,
                     lidar_slowdown=True).drive == 0.0
    assert arbitrate(csv, empty, hard_emergency=True,
                     mission_hold=True, mode=9,
                     transient_obstacle_wait=True).state == \
        "MODE9_OBSTACLE_WAIT"


def test_mode9_repeated_obstacles_always_resume_after_qualified_clear():
    csv = CommandCandidate(2.0, 4, True, True)
    empty = CommandCandidate()
    latch = Mode9EmergencyLatch(1.5, 1.0)
    now = 0.0
    for _ in range(5):
        assert latch.update(True, 0.8, True, True, now)
        waiting = arbitrate(
            csv, empty, hard_emergency=latch.latched, mode=9,
            transient_obstacle_wait=True)
        assert waiting.drive == 0.0
        assert waiting.state == "MODE9_OBSTACLE_WAIT"
        assert latch.update(False, 1.51, True, True, now+0.1)
        assert latch.update(False, 1.51, True, True, now+1.09)
        assert not latch.update(False, 1.51, True, True, now+1.1)
        resumed = arbitrate(csv, empty, hard_emergency=False, mode=9)
        assert resumed.drive == 3.0 and resumed.state == "MODE9_FIXED_SPEED"
        now += 2.0


def test_route_deviation_terminal_banner_is_present():
    source = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam" /
              "route_follower_node.py").read_text(encoding="utf-8")
    assert "경로이탈" in source
    assert "ROUTE_DEVIATION_STOP:" in source


def test_production_launch_has_real_vslam_odom_and_no_fake_source():
    path = ROOT / "src/depth_hybrid_slam/launch/depth_real_vehicle.launch.py"
    launch = path.read_text(encoding="utf-8")
    assert 'executable="odom_localization"' in launch
    assert '"hybrid_localization.launch.py"' in launch
    assert '"use_vehicle_odom": "true"' in launch
    assert '"cuvslam_only.launch.py"' not in launch
    assert 'executable="start_validation"' not in launch
    assert '"initial_branch": branch' in launch
    for forbidden in ("test_odom_publisher", "synthetic_odom", "fake_odom",
                      "gazebo"):
        assert forbidden not in launch.lower()
