from depth_hybrid_slam.mission_core import MissionMachine
from depth_hybrid_slam.models import MissionInputs
from depth_hybrid_slam.traffic_gate import IntersectionProgress


def value(now=0.0, **kwargs):
    return MissionInputs(now=now, **kwargs)


def progress(mode=4, route_index=100, progress_m=10.0):
    return IntersectionProgress(
        valid=True, mode=mode, segment_id="COMMON_1", direction=1,
        route_index=route_index, progress_m=progress_m,
        stop_segment_id="COMMON_1", stop_direction=1,
        stop_route_index=100, stop_index=25, stop_progress_m=10.0,
        exit_route_index=130, exit_progress_m=20.0)


def test_mode4_red_green_and_no_csv_line_policy():
    core = MissionMachine()
    red = core.update(value(traffic_state="R", traffic_confidence=.9,
                            traffic_age=.1, csv_stop_line_active=True,
                            intersection_progress=progress()))
    green = core.update(value(3.0, traffic_state="G", traffic_confidence=.9,
                              traffic_age=.1, csv_stop_line_active=True,
                              intersection_progress=progress()))
    other = MissionMachine()
    no_line = other.update(value(
        traffic_state="R", traffic_confidence=.9, traffic_age=.1,
        intersection_progress=progress()))
    assert red.stop_required and not green.stop_required
    assert not no_line.stop_required


def test_unknown_or_stale_at_mode4_csv_line_stops():
    core = MissionMachine()
    common = {"csv_stop_line_active": True,
              "intersection_progress": progress()}
    assert core.update(value(**common)).stop_required
    assert core.update(value(traffic_state="G", traffic_confidence=.9,
                             traffic_age=.31, **common)).stop_required


def test_red_at_mode4_csv_stop_line_holds():
    core = MissionMachine()
    decision = core.update(value(traffic_state="R", traffic_confidence=.9,
                                 traffic_age=.1, csv_stop_line_active=True,
                                 intersection_progress=progress(),
                                 stop_line_distance_m=1.5))
    assert decision.stop_required and decision.speed_limit == 0.0
    assert decision.state == "MINIMUM_3S_HOLD"


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


def test_mode11_stop_is_not_mixed_with_intersection_gate():
    core = MissionMachine()
    mode11 = progress(mode=11)
    decision = core.update(value(
        section_id="finish", csv_stop_line_active=True,
        intersection_progress=mode11, traffic_aspect="RED_X",
        traffic_confidence=.9, traffic_age=.1))
    assert not decision.stop_required
    assert decision.state == "CRUISE"


def test_mission_fresh_red_presence_overrides_simultaneous_green():
    core = MissionMachine()
    decision = core.update(value(
        traffic_state="G", traffic_aspect="GREEN_CIRCLE",
        traffic_confidence=.9, traffic_age=.1,
        traffic_red_present=True, traffic_green_present=True,
        traffic_diagnostics_age=.1, csv_stop_line_active=True,
        intersection_progress=progress(mode=4)))
    assert decision.state == "MINIMUM_3S_HOLD"
    assert decision.stop_required
    assert core.last_traffic_decision.aspect == "R"


def test_mission_mode8_red_presence_overrides_green_left():
    core = MissionMachine()
    decision = core.update(value(
        traffic_state="G", traffic_aspect="GREEN_LEFT",
        traffic_confidence=.9, traffic_age=.1,
        traffic_red_present=True, traffic_green_present=True,
        traffic_diagnostics_age=.1, csv_stop_line_active=True,
        intersection_progress=progress(mode=8)))
    assert decision.state == "MINIMUM_3S_HOLD"
    assert decision.stop_required
    assert core.last_traffic_decision.aspect == "R"


def test_fresh_red_source_clears_to_green_without_restarting_hold():
    for mode in (4, 6, 8):
        core = MissionMachine()
        common = {
            "traffic_state": "G", "traffic_aspect": "GREEN_CIRCLE",
            "traffic_confidence": .9, "traffic_age": .1,
            "traffic_green_present": True, "traffic_diagnostics_age": .1,
            "csv_stop_line_active": True,
            "intersection_progress": progress(mode=mode),
        }
        assert core.update(value(
            0.0, traffic_red_present=True, **common)).stop_required
        waiting = core.update(value(
            3.0, traffic_red_present=True, **common))
        assert waiting.state == "WAIT_TRAFFIC_RELEASE"
        assert waiting.stop_required
        released = core.update(value(
            5.2, traffic_red_present=False, **common))
        assert released.state == "RELEASED"
        assert not released.stop_required


def test_direct_red_topic_holds_regardless_of_fusion_confidence():
    core = MissionMachine()
    common = {
        "csv_stop_line_active": True,
        "intersection_progress": progress(),
    }
    red = core.update(value(
        traffic_state="UNKNOWN", traffic_confidence=0.0,
        traffic_red_override=True, **common))
    assert red.stop_required
    assert core.last_traffic_decision.aspect == "R"
    green = core.update(value(
        now=3.0, traffic_green_override=True, **common))
    assert not green.stop_required
    assert core.last_traffic_decision.aspect == "G"
