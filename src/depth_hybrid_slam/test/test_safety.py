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
