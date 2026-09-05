from depth_hybrid_slam.mission_core import MissionMachine
from depth_hybrid_slam.models import MissionInputs


def value(now=0.0, **kwargs):
    return MissionInputs(now=now, **kwargs)


def test_red_green_and_no_line_policy():
    core = MissionMachine()
    red = core.update(value(traffic_state="R", traffic_confidence=.9,
                            traffic_age=.1, stop_line_detected=True))
    green = core.update(value(traffic_state="G", traffic_confidence=.9,
                              traffic_age=.1, stop_line_detected=True))
    no_line = core.update(value(traffic_state="R", traffic_confidence=.9,
                                traffic_age=.1, stop_line_detected=False))
    assert red.stop_required and not green.stop_required
    assert not no_line.stop_required


def test_unknown_or_stale_at_line_stops():
    core = MissionMachine()
    assert core.update(value(stop_line_detected=True)).stop_required
    assert core.update(value(traffic_state="G", traffic_confidence=.9,
                             traffic_age=.31, stop_line_detected=True)).stop_required


def test_red_approach_slows_before_stop_distance():
    core = MissionMachine()
    decision = core.update(value(traffic_state="R", traffic_confidence=.9,
                                 traffic_age=.1, stop_line_detected=True,
                                 stop_line_distance_m=1.5))
    assert not decision.stop_required and decision.speed_limit == 1.0


def test_ramp_requires_section_duration_and_holds_once():
    core = MissionMachine(ramp_sections=("ramp",), ramp_duration_s=.5,
                          ramp_stop_s=4.0)
    assert not core.update(value(0, section_id="other", pitch_deg=20)).stop_required
    assert not core.update(value(0, section_id="ramp", pitch_deg=14.9)).stop_required
    assert not core.update(value(1, section_id="ramp", pitch_deg=16,
                                 uphill_detected=True)).stop_required
    assert core.update(value(1.5, section_id="ramp", pitch_deg=16,
                             uphill_detected=True)).state == "RAMP_HOLD"
    assert core.update(value(5.4, section_id="ramp", pitch_deg=-5)).stop_required
    assert not core.update(value(5.6, section_id="ramp", pitch_deg=-5)).stop_required


def test_acceleration_latches_slow_after_steering():
    core = MissionMachine(acceleration_sections=("accel",), sign_confirmations=2)
    assert core.update(value(0, section_id="accel", sign_detected=True)).speed_limit == 2
    assert core.update(value(.1, section_id="accel", sign_detected=True,
                             steering_deg=5)).speed_limit == 3
    assert core.update(value(.2, section_id="accel", sign_detected=True,
                             steering_deg=5.1)).speed_limit == 2
    assert core.update(value(.3, section_id="accel", sign_detected=True,
                             steering_deg=0)).speed_limit == 2


def test_finish_red_x_green_down_and_unknown():
    core = MissionMachine(finish_sections=("finish",))
    common = {"section_id": "finish", "stop_line_detected": True,
              "traffic_confidence": .9, "traffic_age": .1}
    assert core.update(value(traffic_aspect="RED_X", **common)).stop_required
    assert not core.update(value(traffic_aspect="GREEN_DOWN", **common)).stop_required
    assert core.update(value(traffic_aspect="UNKNOWN", **common)).stop_required
