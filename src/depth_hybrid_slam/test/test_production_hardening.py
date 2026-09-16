from datetime import datetime
from pathlib import Path

import numpy as np

from depth_hybrid_slam.camera_correction_core import (
    CameraCorrectionMachine, camera_correction_allowed)
from depth_hybrid_slam.command_arbiter_core import OwnershipHandshake
from depth_hybrid_slam.csv_road_validator_core import (
    CameraRiskGate, predict_path_horizon)
from depth_hybrid_slam.lidar_local_planner import swept_footprint_clear
from depth_hybrid_slam.lidar_mission_core import (
    Mode5Avoidance, Mode9EmergencyLatch, Mode9Safety, Mode11ExitGate,
    select_parking_fallback)
from depth_hybrid_slam.localization_core import VslamRecoveryGate
from depth_hybrid_slam.mission_completion import MissionCompletionTracker
from depth_hybrid_slam.rosbag_rotation import (
    completed_bags, prepare_bag_path, run_name)
from depth_hybrid_slam.signal_exit_core import (
    ExitSignalTrack, SignalDetection, SignalState)


ROOT = Path(__file__).resolve().parents[3]


def test_mode9_fixed_boundaries_clear_hold_and_dynamic_safety():
    policy = Mode9Safety()
    assert not policy.evaluate(1.51, 0.0).slowdown
    assert policy.evaluate(1.50, 0.0).slowdown
    assert policy.evaluate(1.01, 0.0).slowdown
    assert policy.evaluate(1.00, 0.0).stop
    assert policy.evaluate(0.99, 0.0).stop
    assert policy.evaluate(2.0, 3.0).stop
    early_slow = Mode9Safety(
        reaction_time_s=0.0, deceleration_mps2=100.0,
        braking_margin_m=0.0, ttc_stop_s=0.5, ttc_slow_s=2.1)
    assert early_slow.evaluate(2.0, 1.0).state == "MODE9_DYNAMIC_SLOW"
    ttc = Mode9Safety(reaction_time_s=0.0, deceleration_mps2=100.0,
                      braking_margin_m=0.0, ttc_stop_s=1.1,
                      ttc_slow_s=1.5)
    assert ttc.evaluate(2.0, 2.0).state == "MODE9_DYNAMIC_STOP"
    latch = Mode9EmergencyLatch()
    assert latch.update(True, 1.0, True, True, 0.0)
    assert latch.update(False, 1.51, True, True, 1.0)
    assert latch.update(False, 1.51, True, True, 1.9)
    assert not latch.update(False, 1.51, True, True, 2.0)


def _mode11(signal=None, signal_at=4.9):
    gate = Mode11ExitGate()
    gate.enter(0.0)
    if signal is not None:
        gate.observe(signal, signal_at)
    return gate.evaluate(5.0).branch


def test_mode11_explicit_fresh_b_else_a():
    assert _mode11("B") == "B"
    assert _mode11("A") == "A"
    assert _mode11("UNKNOWN") == "A"
    assert _mode11() == "A"
    assert _mode11("B", 0.0) == "A"
    assert _mode11("PARTIAL") == "A"


def test_mode11_track_continuity_bbox_confidence_and_short_occlusion():
    track = ExitSignalTrack(confirmations=3, missing_hold_s=0.25)
    detection = SignalDetection(
        SignalState.RED, 30, 0, (), (10, 10, 40, 30), 0.9)
    assert not track.update(detection, 0.0)["valid"]
    assert not track.update(detection, 0.1)["valid"]
    confirmed = track.update(detection, 0.2)
    assert confirmed["valid"] and confirmed["bbox"] == (10, 10, 40, 30)
    hidden = SignalDetection(SignalState.UNKNOWN, 0, 0)
    held = track.update(hidden, 0.4)
    assert held["valid"] and 0.19 <= held["missing_duration"] <= 0.21
    assert not track.update(hidden, 0.46)["valid"]


def test_vslam_visual_gate_jump_and_bounded_recovery():
    gate = VslamRecoveryGate(0.5, 10.0, 1.0)
    assert gate.update((0, 0, 0), visual_consistent=True,
                       evidence_fresh=True, now=0.0).use_vslam
    mismatch = gate.update((4, 0, 0), visual_consistent=False,
                           evidence_fresh=True, now=1.0)
    assert not mismatch.stop and not mismatch.use_vslam
    jump = gate.update((4, 0, 0), visual_consistent=True,
                       evidence_fresh=True, now=2.0)
    assert jump.stop and jump.state == "WARNING"
    assert gate.update((4, 0, 0), visual_consistent=True,
                       evidence_fresh=True, now=2.9).stop
    recovered = gate.update((0.1, 0, 0), visual_consistent=True,
                            evidence_fresh=True, now=3.0)
    assert recovered.use_vslam and not recovered.stop
    gate.update((4, 0, 0), visual_consistent=True,
                evidence_fresh=True, now=4.0)
    fallback = gate.update((4, 0, 0), visual_consistent=True,
                           evidence_fresh=True, now=5.0)
    assert not fallback.stop and not fallback.use_vslam
    assert fallback.state == "RUNNING_ODOM_ONLY"


def test_camera_prediction_horizon_persistence_and_mode_gate():
    path = np.asarray([(index*0.1, 0.0, 0.0) for index in range(80)])
    predicted = predict_path_horizon(path, 2.0, 10.0, 1.5)
    assert 2.9 <= predicted[-1, 0] <= 3.01
    assert np.max(np.abs(predicted[:, 2])) > 0.0
    machine = CameraCorrectionMachine()
    assert not machine.update("FAIL", True, now=0.0,
                              enabled=False).active
    persistent = CameraRiskGate(0.4)
    assert persistent.update("FAIL", 0.0) == "UNKNOWN"
    assert persistent.update("TRUE", 0.2) == "TRUE"
    assert persistent.update("FAIL", 1.0) == "UNKNOWN"
    assert persistent.update("FAIL", 1.401) == "FAIL"
    for mode in (1, 2, 3):
        assert camera_correction_allowed(mode)
    for mode in (4, 6, 8):
        assert camera_correction_allowed(mode, "APPROACH")
        for state in ("MINIMUM_3S_HOLD", "WAIT_TRAFFIC_RELEASE",
                      "RELEASED", "INTERSECTION_COMMITTED"):
            assert not camera_correction_allowed(mode, state)
        assert camera_correction_allowed(mode, "INTERSECTION_EXITED")
    for mode in (5, 7, 9, 10, 11):
        assert not camera_correction_allowed(mode, "APPROACH")


def test_owner_handoff_stop_same_owner_and_late_command_rejection():
    gate = OwnershipHandshake(stop_duration_s=1.0)
    assert gate.update("CSV", speed_mps=0.0, odom_fresh=True,
                       source_received_at=0.0, now=0.0) is None
    assert gate.update("CAMERA", speed_mps=0.0, odom_fresh=True,
                       source_received_at=0.0, now=1.0).drive == 0.0
    assert gate.update("CAMERA", speed_mps=0.0, odom_fresh=True,
                       source_received_at=1.1, now=1.9).drive == 0.0
    assert gate.update("CAMERA", speed_mps=0.0, odom_fresh=True,
                       source_received_at=2.0, now=2.0) is None
    assert gate.owner == "CAMERA"
    assert gate.update("CAMERA", speed_mps=1.0, odom_fresh=True,
                       source_received_at=2.1, now=2.1) is None
    assert gate.update("LIDAR", speed_mps=0.0, odom_fresh=True,
                       source_received_at=2.0, now=3.0).drive == 0.0
    assert gate.update("LIDAR", speed_mps=0.0, odom_fresh=True,
                       source_received_at=2.0, now=4.1).drive == 0.0
    for initial, target in (("CSV", "CAMERA"), ("CSV", "LIDAR"),
                            ("CSV", "PARKING"), ("LIDAR", "CSV")):
        handoff = OwnershipHandshake(initial_owner=initial)
        assert handoff.update(
            target, speed_mps=0.0, odom_fresh=True,
            source_received_at=0.0, now=0.0).drive == 0.0
        assert handoff.update(
            target, speed_mps=0.0, odom_fresh=True,
            source_received_at=1.0, now=1.0) is None
        assert handoff.owner == target


def test_mode2_average_boundary_and_invalid():
    for pitch, expected in ((4.49, False), (4.50, True), (5.0, True)):
        tracker = MissionCompletionTracker()
        tracker.set_mode(2)
        tracker.tick(0.0, 0.0, stop_waypoint_active=True,
                     pitch_deg=pitch, pitch_valid=True)
        tracker.tick(4.0, 0.0, stop_waypoint_active=True,
                     pitch_deg=pitch, pitch_valid=True)
        assert tracker.mission_complete(2) is expected
    invalid = MissionCompletionTracker()
    invalid.set_mode(2)
    invalid.tick(0.0, 0.0, stop_waypoint_active=True,
                 pitch_deg=9.0, pitch_valid=False)
    invalid.tick(4.0, 0.0, stop_waypoint_active=True,
                 pitch_deg=9.0, pitch_valid=False)
    assert not invalid.mission_complete(2)


def test_mode5_path_lock_and_hard_stop_without_replan():
    machine = Mode5Avoidance()
    assert machine.update(avoidance_required=True).stop
    machine.update(avoidance_required=True, path_valid=True)
    assert machine.path_locked
    decision = machine.update(avoidance_required=True, path_valid=True,
                              local_wheel=5)
    assert decision.state == "LIDAR_PATH_TRACKING"
    emergency = machine.update(avoidance_required=True, hard_obstacle=True,
                               path_valid=True, local_wheel=5)
    # Global arbiter supplies the emergency stop; the locked planner is not
    # reset or regenerated by a changing scan.
    assert machine.path_locked and emergency.state == "LIDAR_PATH_TRACKING"


def test_full_footprint_and_parking_fallback_rules():
    path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert not swept_footprint_clear(path, ((0.0, 0.35),))
    assert swept_footprint_clear(path, ((0.0, 2.0),))
    common = {"lidar_fresh": True, "scan_valid": True,
              "slot_a": None, "slot_b": None}
    assert select_parking_fallback(**common, path_a_safe=True,
                                   path_b_safe=False) == "A"
    assert select_parking_fallback(**common, path_a_safe=False,
                                   path_b_safe=True) == "A"
    assert select_parking_fallback(**common, path_a_safe=True,
                                   path_b_safe=True) == "A"
    assert select_parking_fallback(**common, path_a_safe=False,
                                   path_b_safe=False) == "A"
    assert select_parking_fallback(
        lidar_fresh=False, scan_valid=False, path_a_safe=True,
        path_b_safe=True) == "A"
    assert select_parking_fallback(
        lidar_fresh=True, scan_valid=True, slot_a=False, slot_b=True,
        path_a_safe=False, path_b_safe=False) == "B"


def _bag(root, name, active=False):
    path = root/name
    path.mkdir()
    (path/"metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    if active:
        (path/".active").write_text("")
    return path


def test_rosbag_timestamp_and_max_three_rotation(tmp_path):
    assert run_name(datetime(2026, 9, 15, 14, 35, 22)) == \
        "run_20260915_143522"
    for count in range(3):
        _bag(tmp_path, f"run_20260915_00000{count}")
    active = _bag(tmp_path, "run_20260915_000009", active=True)
    output = prepare_bag_path(
        tmp_path, datetime(2026, 9, 15, 0, 0, 4), max_bags=3)
    assert output.name == "run_20260915_000004"
    assert active.exists()
    assert [path.name for path in completed_bags(tmp_path)] == [
        "run_20260915_000001", "run_20260915_000002"]
    output.mkdir()
    (output/"metadata.yaml").write_text("done")
    assert len(completed_bags(tmp_path)) == 3
    for initial in range(3):
        root = tmp_path/f"case_{initial}"
        root.mkdir()
        for index in range(initial):
            _bag(root, f"run_20260914_00000{index}")
        new_path = prepare_bag_path(
            root, datetime(2026, 9, 15, 0, 0, initial), max_bags=3)
        _bag(root, new_path.name)
        assert len(completed_bags(root)) == initial+1


def test_readme_simplified_a_vslam_off_command_matches_production_launch():
    readme = (ROOT/"README.md").read_text()
    launch = (ROOT/"src/depth_hybrid_slam/launch/"
              "depth_csv_camera_lidar.launch.py").read_text()
    assert readme.count("\n## ") == 1
    assert readme.count("\n### ") == 1
    for value in ("cd ~/depth_ws", "source setup_depth.sh",
                  "ros2 launch depth_hybrid_slam "
                  "depth_csv_camera_lidar.launch.py"):
        assert value in readme
    for name, value in (
            ("start_branch", "A"), ("start_mode", "1"),
            ("end_mode", "11"), ("enable_vslam", "false"),
            ("front_serial_port", "/dev/ttyUSB0"),
            ("enable_control", "true"), ("user_approved", "true"),
            ("enable_rosbag", "true")):
        assert f"{name}:={value}" in readme
        assert f'DeclareLaunchArgument("{name}"' in launch
    runtime = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/"
               "runtime_monitor_node.py").read_text()
    mission = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/"
               "mission_node.py").read_text()
    arbiter = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam/"
               "command_arbiter_node.py").read_text()
    assert "[RUN] SEG=" in runtime
    assert "[SEGMENT {mode}] ENTER" in mission
    assert "[OWNER]" in arbiter
