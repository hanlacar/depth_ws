import math

from depth_hybrid_slam.mapping_quality_core import MappingQuality, StreamTiming
import pytest


def thresholds():
    return {
        "ready_tracking_seconds": 1.0,
        "minimum_rgb_fps": 1.0,
        "minimum_depth_fps": 1.0,
        "minimum_odometry_fps": 1.0,
        "maximum_rtabmap_latency_ms": 50.0,
        "pose_jump_translation_m": 0.75,
        "pose_jump_rotation_rad": math.radians(30),
        "map_correction_translation_m": 0.5,
        "map_correction_rotation_rad": math.radians(20),
        "recording_hold_seconds": 2.0,
        "maximum_recording_yaw_rate_rad_s": 0.4,
        "require_reset_zero": True,
        "require_tf_chain": True,
        "minimum_blur_score": None,
        "minimum_depth_valid_ratio": None,
    }


def ready_quality():
    quality = MappingQuality(thresholds())
    for name in quality.STREAMS:
        quality.observe_stream(name, 1_000_000_000)
        quality.observe_stream(name, 1_100_000_000)
    quality.observe_tracking(True, 1.0)
    quality.observe_reset(0)
    quality.observe_rtabmap(10.0)
    return quality


def test_tracking_false_prevents_ready():
    quality = ready_quality()
    quality.observe_tracking(False, 2.0)
    assert quality.state(3.0, 1, True, True)[1] == "TRACKING_LOST"


def test_reset_increase_is_sticky_invalid_session():
    quality = ready_quality()
    quality.observe_reset(1)
    assert quality.state(3.0, 1, True, True)[1] == "INVALID_SESSION"
    quality.observe_reset(0)
    assert quality.state(4.0, 1, True, True)[1] == "INVALID_SESSION"


def test_duplicate_and_backwards_timestamps_are_counted():
    timing = StreamTiming()
    for stamp in (10, 10, 9):
        timing.observe(stamp)
    assert timing.duplicates == 1 and timing.backwards == 1


def test_pose_jump_is_detected():
    quality = ready_quality()
    quality.previous_pose = (0.0, 0.0, 0.0)
    quality.observe_pose(1_200_000_000, 1.0, 0.0, 0.0)
    assert quality.state(3.0, 1, True, True)[1] == "POSE_JUMP"


def test_all_ready_conditions_pass():
    assert ready_quality().state(3.0, 1, True, True) == (True, "READY", [])


def test_uncalibrated_metrics_force_partial_pass_not_fabricated_pass():
    quality = ready_quality()
    verdict, state, reasons = quality.verdict(3.0, 1, True, True)
    assert verdict == "PARTIAL_PASS" and state == "READY" and reasons == []
    assert set(quality.report()["uncalibrated_metrics"]) == {
        "blur_score", "depth_valid_ratio", "overexposed_ratio",
        "underexposed_ratio", "rtabmap_inliers"}


def test_large_map_correction_tracking_loss_and_fast_turn_hold_recording():
    quality = ready_quality()
    quality.observe_map_correction(0.0, 0.0, 0.0, 3.0)
    quality.observe_map_correction(1.0, 0.0, 0.0, 3.1)
    allowed, reasons = quality.recording_status(3.2)
    assert not allowed and "MAP_CORRECTION_HOLD" in reasons
    quality.observe_tracking(False, 3.3)
    assert "TRACKING_LOST" in quality.recording_status(3.3)[1]


def test_loop_closures_distance_periods_and_valid_route_ratio_are_reported():
    quality = ready_quality()
    quality.observe_pose(2_000_000_000, 0.0, 0.0, 0.0)
    quality.observe_pose(2_100_000_000, 0.1, 0.0, 0.01)
    quality.observe_rtabmap(12.0, 20, 4, 10)
    quality.observe_rtabmap(11.0, 21, 4, 10)
    quality.observe_route_sample(True)
    quality.observe_route_sample(False)
    report = quality.report()
    assert report["total_distance_m"] == pytest.approx(0.1)
    assert report["loop_closure_count"] == 1
    assert report["valid_route_point_percent"] == 50.0
    assert report["streams"]["odometry"]["mean_period_s"] is not None
