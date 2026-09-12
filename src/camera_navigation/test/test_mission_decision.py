"""Real-vehicle mission and final command acceptance tests."""

from pathlib import Path

import pytest

from camera_navigation.camera_command_selector_node import CameraCommandSelector
from camera_navigation.mission_decision_core import (
    MissionDecisionConfig, MissionDecisionMachine, MissionInputs,
    ROUTE_MODE_SECTIONS, map_route_section)


def sample(now=0.0, section="1", **kwargs):
    values = dict(
        now=now, stamp=now + 100.0, section=section, input_fresh=True,
        planner_valid=True, planner_state="VALID", planner_drive=2.0,
        required_steering_deg=0.0, stop_line_tf_valid=True,
        imu_valid=True, speed_valid=True, current_speed_mps=0.0)
    values.update(kwargs)
    return MissionInputs(**values)


def test_exact_route_mode_mapping():
    assert ROUTE_MODE_SECTIONS == {
        1: "NORMAL_1", 2: "SLOPE", 3: "D_COURSE",
        4: "INTERSECTION_4", 5: "S_COURSE", 6: "INTERSECTION_6",
        7: "T_PARK", 8: "NORMAL_8", 9: "ACCELERATION",
        10: "PARALLEL_PARK", 11: "NORMAL_11"}
    for mode, state in ROUTE_MODE_SECTIONS.items():
        assert map_route_section(str(mode)) == (mode, state)


def test_intersection_4_and_6_have_independent_latches():
    machine = MissionDecisionMachine()
    stopped = machine.update(sample(section="4", traffic_light="R",
        stop_line_detected=True, stop_line_distances_m=(0.5,)))
    assert stopped.override_active and stopped.drive_override == 0.0
    assert machine.red_stop_latched
    mode6 = machine.update(sample(.1, section="6", traffic_light="G",
        stop_line_detected=True, stop_line_distances_m=(0.5,)))
    assert mode6.section == "INTERSECTION_6" and not mode6.override_active
    assert not machine.red_stop_latched


def test_4_to_5_to_6_does_not_leak_latches():
    machine = MissionDecisionMachine()
    machine.update(sample(section="4", traffic_light="R",
        stop_line_distances_m=(0.5,)))
    machine.update(sample(.1, section="5"))
    result = machine.update(sample(.2, section="6", traffic_light="G",
        stop_line_distances_m=(0.5,)))
    assert not machine.red_stop_latched and not machine.line_crossed_latched
    assert result.state == "GREEN_PROCEED"


@pytest.mark.parametrize("light", ("R", "G", "UNKNOWN"))
def test_signal_without_stop_line_never_controls(light):
    result = MissionDecisionMachine().update(sample(
        section="4", traffic_light=light, stop_line_detected=False,
        stop_line_distances_m=()))
    assert result.state == "INTERSECTION_NO_STOP_LINE"
    assert not result.override_active


@pytest.mark.parametrize("mode", ("4", "6"))
def test_red_stops_green_relinquishes_to_gps(mode):
    red = MissionDecisionMachine().update(sample(
        section=mode, traffic_light="R", stop_line_detected=True,
        stop_line_distances_m=(0.5,)))
    green = MissionDecisionMachine().update(sample(
        section=mode, traffic_light="G", stop_line_detected=True,
        stop_line_distances_m=(0.5,)))
    assert red.override_active and red.drive_override == 0.0
    assert not green.override_active


def test_unknown_with_stop_line_is_safe_stop():
    result = MissionDecisionMachine().update(sample(
        section="4", traffic_light="UNKNOWN", stop_line_detected=True,
        stop_line_distances_m=(1.8,)))
    assert result.override_active and result.drive_override == 0.0


def cross_first_line(machine):
    machine.update(sample(0.0, "2", stop_line_distances_m=(0.2, 2.0),
                          imu_pitch_deg=15.0))
    machine.update(sample(0.1, "2", stop_line_distances_m=(-0.01, 1.9),
                          imu_pitch_deg=15.0))


def arm_and_track_second_line(machine):
    cross_first_line(machine)
    machine.update(sample(.2, "2", stop_line_distances_m=(1.5,),
                          uphill_detected=True, imu_pitch_deg=15.0))
    machine.update(sample(.5, "2", stop_line_distances_m=(1.4,),
                          uphill_detected=True, imu_pitch_deg=15.0))
    machine.update(sample(.55, "2", stop_line_distances_m=(1.3,),
                          uphill_detected=True, imu_pitch_deg=15.0))
    return machine.update(sample(.6, "2", stop_line_distances_m=(.6,),
        uphill_detected=True, imu_pitch_deg=15.0, current_speed_mps=0.0))


def test_pitch_before_first_line_and_downhill_do_not_stop():
    before = MissionDecisionMachine().update(sample(
        section="2", stop_line_distances_m=(1.0, 3.0),
        uphill_detected=True, imu_pitch_deg=16.0))
    assert not before.override_active
    machine = MissionDecisionMachine(); cross_first_line(machine)
    downhill = machine.update(sample(.5, "2", stop_line_distances_m=(1.5,),
        uphill_detected=False, imu_pitch_deg=-16.0))
    assert not downhill.override_active and not machine.slope_armed


def test_between_lines_positive_pitch_stops_and_holds_four_seconds():
    machine = MissionDecisionMachine(MissionDecisionConfig(
        slope_stationary_confirm_sec=0.0))
    stopped = arm_and_track_second_line(machine)
    assert stopped.override_active and stopped.drive_override == 0.0
    holding = machine.update(sample(4.59, "2", stop_line_distances_m=(.6,),
        uphill_detected=True, imu_pitch_deg=15.0, current_speed_mps=0.0))
    assert holding.state == "SLOPE_HOLD" and holding.override_active
    released = machine.update(sample(4.61, "2", stop_line_distances_m=(.6,),
        uphill_detected=True, imu_pitch_deg=15.0, current_speed_mps=0.0))
    assert released.state == "SLOPE_COMPLETE" and not released.override_active


def test_missing_exact_tf_fails_closed_without_odometry_guess():
    machine = MissionDecisionMachine(); arm_and_track_second_line(machine)
    result = machine.update(sample(.7, "2", stop_line_distances_m=(.5,),
        uphill_detected=True, imu_pitch_deg=15.0, stop_line_tf_valid=False,
        distance_valid=True, distance_m=100.0))
    assert result.override_active and result.drive_override == 0.0
    assert result.failure_reason == "SLOPE_FEEDBACK_STALE"


def test_downhill_after_slope_arm_releases_stop_condition():
    machine = MissionDecisionMachine(); arm_and_track_second_line(machine)
    result = machine.update(sample(.7, "2", stop_line_distances_m=(.5,),
        uphill_detected=False, imu_pitch_deg=-1.0, current_speed_mps=0.0))
    assert not result.override_active
    assert result.state == "BETWEEN_LINES_WAIT_UPHILL"


def test_acceleration_three_states_and_mode_reset():
    machine = MissionDecisionMachine()
    waiting = machine.update(sample(section="9", planner_drive=3.0,
                                    sign_detected=False))
    assert waiting.state == "WAIT_SIGN" and waiting.effective_drive == 2.0
    allowed = machine.update(sample(.1, "9", sign_detected=True,
                                    required_steering_deg=5.0))
    assert allowed.state == "HIGH_ACCEL_ALLOWED" and allowed.effective_drive == 3.0
    locked = machine.update(sample(.2, "9", sign_detected=True,
                                   required_steering_deg=5.01))
    assert locked.state == "HIGH_ACCEL_LOCKED_OUT" and locked.effective_drive == 2.0
    still_locked = machine.update(sample(.3, "9", sign_detected=True,
                                         required_steering_deg=0.0))
    assert still_locked.effective_drive == 2.0
    machine.update(sample(.4, "1"))
    assert not machine.acceleration_armed and not machine.turn_detected


def test_mode_11_disabled_by_default_and_opt_in_available():
    default = MissionDecisionMachine().update(sample(
        section="11", traffic_aspect="RED_X", traffic_aspect_fresh=True))
    assert default.section == "NORMAL_11" and not default.override_active
    enabled = MissionDecisionMachine(MissionDecisionConfig(
        mode_11_exit_signal_enabled=True)).update(sample(
            section="11", traffic_aspect="RED_X", traffic_aspect_fresh=True))
    assert enabled.override_active and enabled.drive_override == 0.0


def test_selector_authority_avoidance_parking_and_limit():
    policy = CameraCommandSelector()
    normal = policy.select(1, 2.0, 99, True)
    assert normal.authority and normal.wheel == 22
    stop = policy.select(4, 2.0, 10, True, True, 0.0, True)
    assert stop.authority and stop.drive == 0.0 and stop.stop
    stop_without_path = policy.select(4, 0.0, 0, False, True, 0.0, True)
    assert stop_without_path.authority and stop_without_path.drive == 0.0
    go = policy.select(4, 2.0, 10, True, False, 0.0, True)
    assert not go.authority and go.source == "GPS_DR"
    avoid = policy.select(5, 2.0, 10, True, avoidance_active=True)
    assert not avoid.authority and avoid.source == "LIDAR"
    assert not policy.select(7, 2.0, 10, True).authority


def test_only_selector_publishes_final_topics_and_no_direct_arduino_bridge():
    workspace = Path(__file__).parents[2]
    sources = [path for path in workspace.rglob("*.py")
               if "test" not in path.parts and "build" not in path.parts and
               "install" not in path.parts]
    drive = []; wheel = []; stop = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if 'create_publisher(Float32, "/camera_drive"' in text:
            drive.append(path)
        if 'create_publisher(Int32, "/camera_wheel"' in text:
            wheel.append(path)
        if 'create_publisher(Bool, "/camera_stop"' in text:
            stop.append(path)
    selector = workspace / "camera_navigation" / "camera_navigation" / \
        "camera_command_selector_node.py"
    assert drive == [selector] and wheel == [selector] and stop == [selector]
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    vehicle = workspace / "race_vehicle_interface"
    assert not (vehicle / "race_vehicle_interface" /
                "arduino_serial_bridge_node.py").exists()
    assert "arduino_serial_bridge_node" not in all_text
