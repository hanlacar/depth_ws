from pathlib import Path

from depth_hybrid_slam.mcu_source_adapter_core import adapt_slam_command


def test_valid_stage_and_physical_left_are_converted_for_gps_boundary():
    result = adapt_slam_command(2.0, 12, False, (0.1, 0.1, 0.1))
    assert result.valid and not result.stop
    assert result.drive == 2.0
    assert result.wheel == -12  # GPS team contract: +RIGHT / -LEFT


def test_right_turn_sign_is_converted_for_gps_boundary():
    result = adapt_slam_command(1.0, -8, False, (0.0, 0.0, 0.0))
    assert result.wheel == 8


def test_all_current_drive_stages_are_accepted():
    for stage in (-1, 0, 1, 2, 3):
        assert adapt_slam_command(stage, 0, False, (0, 0, 0)).valid


def test_non_stage_drive_is_rejected_with_stop():
    result = adapt_slam_command(0.5, 0, False, (0, 0, 0))
    assert not result.valid and result.stop and result.drive == 0.0
    assert result.state == "INVALID_DRIVE_STAGE"


def test_wheel_outside_measured_limit_is_rejected():
    result = adapt_slam_command(1, 23, False, (0, 0, 0))
    assert not result.valid and result.stop and result.wheel == 0


def test_source_stop_has_priority():
    result = adapt_slam_command(3, 22, True, (0, 0, 0))
    assert result.valid and result.stop
    assert result.drive == 0.0 and result.wheel == 0


def test_stale_triplet_forces_stop():
    result = adapt_slam_command(2, 5, False, (0.1, 0.6, 0.1))
    assert not result.valid and result.stop and result.state == "STALE_COMMAND"


def test_adapter_uses_topics_only_and_never_publishes_final_mcu_command():
    root = Path(__file__).parents[1]/"depth_hybrid_slam"
    source = (root/"mcu_source_adapter_node.py").read_text()
    assert "t870_mcu" not in source
    assert '"/gps_drive"' in source and '"/gps_wheel"' in source
    assert '"/gps_stop"' in source
    assert '"/mcu/cmd_drive"' not in source
    assert '"/mcu/cmd_wheel"' not in source


def test_follower_candidate_reaches_single_internal_command_owner():
    root = Path(__file__).parents[1]/"depth_hybrid_slam"
    follower = (root/"route_follower_node.py").read_text()
    selector = (root/"behavior_selector_node.py").read_text()
    for topic in ("candidate_drive", "candidate_wheel", "candidate_stop"):
        assert f'"/depth_slam/follower/{topic}"' in follower
    for topic in ("/slam_drive", "/slam_wheel", "/slam_stop"):
        assert topic not in follower
        assert topic in selector
    assert '"/drive_mode"' in follower
