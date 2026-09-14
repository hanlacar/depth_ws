import math

from t870_mcu_simple.odom_core import MeasuredEncoderOdom


def test_bridge_exposes_command_and_firmware_confirmation_contract():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]/"t870_mcu_simple"/
              "bridge_node.py").read_text(encoding="utf-8")
    for topic in (
            "/mcu/command_diagnostics", "/mcu/firmware_drive_pwm",
            "/mcu/firmware_armed"):
        assert topic in source
    for key in (
            "commanded_stage", "applied_stage", "applied_pwm",
            "firmware_drive_pwm", "firmware_armed"):
        assert key in source

    arm = source.split("def configure_and_arm", 1)[1].split(
        "def io_tick", 1)[0]
    assert "self.odom_model.reset_encoder_baseline()" in arm
    assert "self.last_encoder = None" in arm


def test_stationary_encoder_never_moves_odom():
    odom = MeasuredEncoderOdom()
    odom.update_encoder(100)
    before = odom.base_pose()
    for _ in range(20):
        assert odom.update_encoder(100) == (0.0, 0.0)
    assert odom.base_pose() == before == (0.0, 0.0, 0.0)


def test_measured_ticks_and_measured_steering_drive_pose():
    odom = MeasuredEncoderOdom(counts_per_meter=100.0)
    odom.set_direction_from_stage(1)
    odom.set_steering_deg(22.0)
    odom.update_encoder(0)
    distance, delta_yaw = odom.update_encoder(100)
    assert distance > 0.0 and delta_yaw > 0.0
    assert odom.state.distance_m == abs(distance)
    assert all(math.isfinite(value) for value in odom.base_pose())


def test_reverse_direction_uses_ticks_not_commanded_distance():
    odom = MeasuredEncoderOdom(counts_per_meter=100.0)
    odom.update_encoder(0)
    odom.set_direction_from_stage(-1)
    distance, _ = odom.update_encoder(50)
    assert distance < 0.0
    pose = odom.base_pose()
    odom.update_encoder(50)
    assert odom.base_pose() == pose


def test_arm_counter_reset_is_rebaselined_without_vehicle_motion():
    odom = MeasuredEncoderOdom()
    odom.update_encoder(153048)
    odom.reset_encoder_baseline()
    assert odom.update_encoder(0) == (0.0, 0.0)
    assert odom.base_pose() == (0.0, 0.0, 0.0)
    assert odom.state.distance_m == 0.0


def test_impossible_encoder_jump_is_ignored_and_next_sample_recovers():
    odom = MeasuredEncoderOdom(
        counts_per_meter=100.0, max_encoder_delta_counts=5000)
    odom.update_encoder(153048)
    assert odom.update_encoder(0) == (0.0, 0.0)
    assert odom.last_discontinuity
    assert odom.base_pose() == (0.0, 0.0, 0.0)
    distance, _ = odom.update_encoder(100)
    assert not odom.last_discontinuity
    assert distance == 1.0
