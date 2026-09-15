from depth_hybrid_slam.command_arbiter_core import (
    CommandCandidate, arbitrate)
from depth_hybrid_slam.models import SafetyInputs
from depth_hybrid_slam.safety_core import SafetyGate


def ready(now=10.0, **kwargs):
    data = {"now": now, "tracking_valid": True, "localization_state": "TRACKING",
            "localization_confidence": .9, "pose_stamp": now-.01,
            "map_route_match": True, "within_map": True,
            "controller_stamp": now-.01, "user_approved": True,
            "enable_control": True, "dry_run": False}
    data.update(kwargs)
    return SafetyInputs(**data)


def test_all_gates_are_required():
    assert SafetyGate().evaluate(ready()).ready


def test_default_is_fail_closed():
    result = SafetyGate().evaluate(SafetyInputs(now=10.0))
    assert result.stop_required and "CONTROL_DISABLED" in result.reasons


def test_stale_jump_mismatch_and_mission_stop():
    result = SafetyGate().evaluate(ready(
        pose_stamp=9.0, pose_jump=True, map_route_match=False,
        mission_stop=True, mission_reason="RED_AT_STOP_LINE"))
    assert {"STALE_POSE", "POSE_JUMP", "MAP_ROUTE_MISMATCH",
            "RED_AT_STOP_LINE"}.issubset(result.reasons)


def test_odom_only_fresh_tracking_is_ready_without_vslam_confidence():
    result = SafetyGate(localization_mode="ODOM_ONLY").evaluate(ready(
        localization_confidence=0.6))
    assert result.ready
    assert "LOW_LOCALIZATION_CONFIDENCE" not in result.reasons


def test_odom_only_stale_odom_or_pose_and_pose_jump_still_stop():
    gate = SafetyGate(localization_mode="ODOM_ONLY")
    odom_stale = gate.evaluate(ready(
        tracking_valid=False, localization_state="STALE", pose_stamp=9.0))
    assert odom_stale.stop_required
    assert {"LOCALIZATION_TRACKING_LOSS", "LOCALIZATION_NOT_READY",
            "STALE_POSE"}.issubset(odom_stale.reasons)
    pose_stale = gate.evaluate(ready(pose_stamp=9.0))
    assert pose_stale.stop_required and "STALE_POSE" in pose_stale.reasons
    jump = gate.evaluate(ready(pose_jump=True))
    assert jump.stop_required and "POSE_JUMP" in jump.reasons


def test_vslam_mode_retains_confidence_threshold():
    gate = SafetyGate(localization_mode="VSLAM")
    low = gate.evaluate(ready(localization_confidence=0.69))
    assert low.stop_required and "LOW_LOCALIZATION_CONFIDENCE" in low.reasons
    assert gate.evaluate(ready(localization_confidence=0.7)).ready


def test_camera_stale_falls_back_to_csv_when_safety_is_ready():
    safety = SafetyGate(localization_mode="ODOM_ONLY").evaluate(ready(
        localization_confidence=0.6))
    command = arbitrate(
        CommandCandidate(2.0, 0, True, True), CommandCandidate(),
        hard_emergency=not safety.ready, mission_hold=False,
        camera=CommandCandidate(0.0, 0, False, False))
    assert command.owner == "CSV"
    assert command.state == "CSV_TRACKING"
    assert command.drive == 2.0
