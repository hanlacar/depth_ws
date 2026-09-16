import math

from t870_mcu_simple.odom_core import MeasuredEncoderOdom, steering_from_adc


def test_bridge_exposes_command_and_firmware_confirmation_contract():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]/"t870_mcu_simple"/
              "bridge_node.py").read_text(encoding="utf-8")
    for topic in (
            'Float32, "/cmd_drive"', 'Int32, "/cmd_wheel"',
            'Bool, "/mcu/ready"', 'Float32, "/mcu/steer_deg"',
            'Int32, "/mcu/encoder"',
            "/mcu/command_diagnostics", "/mcu/firmware_drive_pwm",
            "/mcu/firmware_armed", "/mcu/speed_mps", "/mcu/distance_m",
            'Odometry, "/odom"'):
        assert topic in source
    for key in (
            "commanded_stage", "applied_stage", "applied_pwm",
            "firmware_drive_pwm", "firmware_armed"):
        assert key in source

    arm = source.split("def configure_and_arm", 1)[1].split(
        "def io_tick", 1)[0]
    assert "self.odom_model.reset_encoder_baseline()" in arm
    assert "self.last_enc = None" in arm


def test_launch_keeps_port_auto_argument_contract():
    from pathlib import Path

    launch = (Path(__file__).resolve().parents[1]/"launch"/
              "mcu.launch.py").read_text(encoding="utf-8")
    assert "DeclareLaunchArgument('port', default_value='auto')" in launch
    assert "'port': LaunchConfiguration('port')" in launch


def test_rear_forward_fix_is_preserved_in_runtime_configuration():
    from pathlib import Path

    package = Path(__file__).resolve().parents[1]
    config = (package / "config" / "mcu.yaml").read_text(encoding="utf-8")
    source = (package / "t870_mcu_simple" / "bridge_node.py").read_text(
        encoding="utf-8")

    assert "front_forward_level: 0" in config
    assert "rear_forward_level: 1" in config
    assert 'self.declare_parameter("rear_forward_level", 1)' in source
    assert 'f"CFG,REAR_FWD,{self.rear_fwd}"' in source


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


def test_left_positive_axis_reaches_mcu_and_feedback_without_inversion():
    source = (__import__("pathlib").Path(__file__).resolve().parents[1] /
              "t870_mcu_simple" / "bridge_node.py").read_text()
    assert 'self.send(f"W,{deg}")' in source
    assert steering_from_adc(484+18*10, 484, 18, 22) == 10
    assert steering_from_adc(484-18*10, 484, 18, 22) == -10


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
