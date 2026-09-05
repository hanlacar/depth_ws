import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "analyze_camera_mission_bag.py"
SPEC = importlib.util.spec_from_file_location("mission_bag_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def empty_data():
    return {topic: [] for topic in MODULE.TOPICS}


def test_analysis_flags_neither_valid_stages_nor_safe_zero():
    data = empty_data()
    data["/camera/mission/decision_state"] = [(0.0, "ACCEL_HIGH_SPEED")]
    data["/camera/mission/drive_override"] = [(0.0, 3.0), (1.0, 4.0)]
    data["/camera/mission/drive_override_active"] = [(0.0, True)]
    data["/camera/mission/decision_diagnostics"] = [
        (0.0, {"safety_blocked": True,
               "effective_drive_if_connected": 0.0,
               "failure_reason": "INPUT_TIMEOUT"})]
    report = MODULE.analyze(data)
    assert report["invalid_drive_stage_count"] == 1
    assert report["safety_blocked_nonzero_effective_count"] == 0
    assert report["failure_reasons"] == {"INPUT_TIMEOUT": 1}


def test_measurement_errors_and_unsafe_effective_are_reported():
    data = empty_data()
    data["/camera/mission/stop_line_distance_m"] = [(10.0, 1.1)]
    data["/camera/mission/decision_diagnostics"] = [
        (10.0, {"safety_blocked": True,
                "effective_drive_if_connected": 3.0})]
    report = MODULE.analyze(data, [{"timestamp_sec": "10.0",
                                    "measured_distance_m": "1.0",
                                    "traffic_truth": ""}])
    assert report["stop_line_distance_error_m"]["max"] == \
        pytest.approx(.1)
    assert report["safety_blocked_nonzero_effective_count"] == 1


import pytest
