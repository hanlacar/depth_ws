from pathlib import Path

from depth_hybrid_slam.command_arbiter_core import (
    arbitrate, CommandCandidate)
from depth_hybrid_slam.csv_only_branching import (
    csv_only_network_segments, StartBranchClassifier)
from depth_hybrid_slam.lidar_mission_core import (
    Mode9EmergencyLatch, parking_decision, SteeringSlowdownLatch)
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


def test_parking_never_defaults_a_without_measured_slot():
    for mode, prefix in ((7, "T"), (10, "V")):
        decision = parking_decision(mode, False, False, False)
        assert decision.stop and not decision.branch
        assert decision.state == prefix + "_WAIT_LIDAR_SLOT"


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
                     mission_hold=True, mode=9).state == "HARD_EMERGENCY_STOP"


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
