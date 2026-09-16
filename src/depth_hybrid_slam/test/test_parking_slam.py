from pathlib import Path
import math

import pytest

from depth_hybrid_slam.command_arbiter_core import (
    arbitrate, CommandCandidate, OwnershipHandshake)
from depth_hybrid_slam.parking_planner_core import (
    assess_parking_path, map_to_odom_from_base_poses, parking_tf_owner,
    parking_csv_fallback_segment, path_direction_profile,
    ParkingPathAssessment, ParkingRuntimeCoordinator,
    ParkingVehicleGeometry, resolve_parking_plan, select_csv_parking_path,
    select_explicit_b_slot, select_slam_slot, validate_nav2_parking_path)


ROOT = Path(__file__).resolve().parents[1]


def assessment(safe, clearance):
    return ParkingPathAssessment(
        safe, clearance, clearance, clearance, 10,
        "CLEAR" if safe else "FOOTPRINT_COLLISION")


def test_csv_slot_selection_matrix_and_default_a():
    safe = assessment(True, 0.4)
    unsafe_a = assessment(False, -0.2)
    unsafe_b = assessment(False, -0.1)
    assert select_csv_parking_path(safe, unsafe_b).slot == "A"
    assert select_csv_parking_path(unsafe_a, safe).slot == "B"
    assert select_csv_parking_path(safe, safe).slot == "A"
    assert select_csv_parking_path(None, safe).slot == "A"
    assert select_csv_parking_path(safe, safe, evidence_valid=False).slot == "A"
    assert select_csv_parking_path(unsafe_a, unsafe_b).slot == "B"
    assert select_csv_parking_path(unsafe_b, unsafe_b).slot == "A"


def test_body_and_four_wheel_footprints_are_swept():
    geometry = ParkingVehicleGeometry().validate()
    clear = assess_parking_path(
        ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
        ((4.0, 4.0),), geometry)
    collision = assess_parking_path(
        ((0.0, 0.0, 0.0),),
        ((geometry.wheelbase_m, geometry.wheel_track_m/2.0),), geometry)
    assert clear.safe
    assert math.isfinite(clear.body_clearance_m)
    assert math.isfinite(clear.wheel_clearance_m)
    assert not collision.safe
    assert collision.wheel_clearance_m <= 0.0
    assert geometry.planner_max_steering_deg == 20.0


def test_direction_profile_rejects_mixed_nav2_execution():
    reverse = ((0.0, 0.0, 0.0), (-0.5, 0.0, 0.0),
               (-1.0, 0.0, 0.0))
    mixed = reverse+((-0.5, 0.0, 0.0),)
    assert path_direction_profile(reverse) == (-1,)
    assert path_direction_profile(mixed) == (-1, 1)


def test_slam_slot_policy_and_both_occupied_clearance():
    occupied_a = assessment(False, -0.3)
    occupied_b = assessment(False, -0.1)
    assert select_slam_slot("FREE", "OCCUPIED").slot == "A"
    assert select_slam_slot("OCCUPIED", "FREE").slot == "B"
    assert select_slam_slot("FREE", "FREE").slot == "A"
    assert select_slam_slot("UNKNOWN", "UNKNOWN").slot == "A"
    assert select_slam_slot(
        "OCCUPIED", "OCCUPIED", occupied_a, occupied_b).slot == "B"


def test_nav2_retry_and_csv_fallback_chain():
    assert resolve_parking_plan(
        slam_ready=True, a_state="FREE", b_state="OCCUPIED",
        nav2_a_valid=True).source == "NAV2"
    retry = resolve_parking_plan(
        slam_ready=True, a_state="FREE", b_state="FREE",
        nav2_a_valid=False, nav2_b_valid=True)
    assert (retry.slot, retry.source, retry.attempted_slots) == (
        "B", "NAV2", ("A", "B"))
    fallback = resolve_parking_plan(
        slam_ready=True, a_state="OCCUPIED", b_state="FREE",
        nav2_b_valid=False)
    assert (fallback.slot, fallback.source) == ("B", "CSV_FALLBACK")
    slam_failure = resolve_parking_plan(
        slam_ready=False, a_state="UNKNOWN", b_state="UNKNOWN")
    assert (slam_failure.slot, slam_failure.source) == (
        "A", "CSV_FALLBACK")


def test_map_to_odom_has_exactly_one_logical_owner():
    assert parking_tf_owner(False, 7, False) == "DEPTH_ODOM"
    assert parking_tf_owner(True, 6, False) == "DEPTH_ODOM"
    assert parking_tf_owner(True, 7, True) == "SLAM_TOOLBOX"
    assert parking_tf_owner(True, 10, True) == "SLAM_TOOLBOX"
    assert parking_tf_owner(True, 7, False) == "DEPTH_ODOM"
    transform = map_to_odom_from_base_poses(
        (12.0, 3.0, 0.0), (2.0, 1.0, 0.0))
    assert transform == (10.0, 2.0, 0.0)


def test_launch_uses_front_only_slam_nav2_and_real_action():
    wrapper = (ROOT/"launch"/"depth_csv_camera_lidar.launch.py").read_text()
    launch = (ROOT/"launch"/"depth_real_vehicle.launch.py").read_text()
    manager = (ROOT/"depth_hybrid_slam"/"maneuver_manager_node.py").read_text()
    slam = (ROOT/"config"/"parking_slam.yaml").read_text()
    nav2 = (ROOT/"config"/"parking_nav2.yaml").read_text()
    assert 'DeclareLaunchArgument("enable_parking_slam", default_value="false")' in wrapper
    assert 'DeclareLaunchArgument("enable_parking_slam", default_value="false")' in launch
    assert 'scan_topic: /front/scan' in slam
    assert '/rear/scan' not in slam
    assert 'motion_model_for_search: "REEDS_SHEPP"' in nav2
    assert 'minimum_turning_radius: 2.01' in nav2
    assert 'ComputePathThroughPoses' in manager
    assert 'planner_max_steering_deg", 20.0' in manager


def test_combined_vslam_parking_slam_uses_disjoint_tf_frames():
    launch = (ROOT/"launch"/"depth_real_vehicle.launch.py").read_text()
    localization = (
        ROOT/"depth_hybrid_slam"/"odom_localization_node.py").read_text()
    common = (ROOT/"depth_hybrid_slam"/"launch_common.py").read_text()
    assert '"centralize_vslam_map_tf"' in launch
    assert '"publish_rtabmap_tf"' in launch
    assert '"false" if parking_slam_enabled else "true"' in launch
    assert 'LaunchConfiguration("rtabmap_map_frame")' in common
    assert 'LaunchConfiguration("publish_rtabmap_tf")' in common
    assert '"/rtabmap/localization_pose"' in localization
    assert "not self.parking_slam_active" in localization


@pytest.mark.parametrize("mode", (7, 10))
def test_parking_slam_remains_active_until_reverse_commit(mode):
    runtime = ParkingRuntimeCoordinator(minimum_observation_s=3.0)
    entered = runtime.update(mode=mode, observation_elapsed_s=0.0)
    assert entered.slam_should_run and entered.csv_approach
    planning = runtime.update(
        mode=mode, observation_elapsed_s=2.9, nav2_planning=True)
    assert planning.state == "NAV2_PLANNING"
    assert planning.slam_should_run and planning.csv_approach
    committed = runtime.update(
        mode=mode, observation_elapsed_s=4.0,
        reverse_decision_point=True)
    assert committed.state == "CSV_SLOT_COMMITTED"
    assert not committed.slam_should_run and committed.csv_fallback
    reverse = runtime.update(mode=mode, reverse_started=True)
    assert reverse.state == "CSV_PARKING"
    assert not reverse.slam_should_run


def test_nav2_pending_never_blocks_csv_and_early_success_rules():
    runtime = ParkingRuntimeCoordinator(minimum_observation_s=3.0)
    pending = runtime.update(
        mode=7, observation_elapsed_s=1.0, nav2_planning=True)
    assert pending.csv_approach and not pending.branch_locked
    early_a = runtime.update(
        mode=7, observation_elapsed_s=1.0,
        nav2_slot="A", nav2_path_valid=True)
    assert early_a.csv_approach and not early_a.nav2_path_ready
    accepted_a = runtime.update(
        mode=7, observation_elapsed_s=3.0,
        nav2_slot="A", nav2_path_valid=True)
    assert accepted_a.state == "NAV2_READY"
    assert accepted_a.nav2_path_ready and accepted_a.branch_locked
    assert not accepted_a.slam_should_run


def test_fresh_explicit_b_commits_b_nav2_and_branch_is_immutable():
    runtime = ParkingRuntimeCoordinator(minimum_observation_s=3.0)
    accepted = runtime.update(
        mode=10, explicit_b=True, observation_elapsed_s=0.2,
        nav2_slot="B", nav2_path_valid=True)
    assert accepted.selected_slot == "B"
    assert accepted.nav2_path_ready and accepted.branch_locked
    runtime.update(mode=10, explicit_b=False, reverse_started=True)
    late = runtime.update(mode=10, explicit_b=True)
    assert late.selected_slot == "B" and late.branch_locked


def test_default_a_branch_cannot_change_from_late_b_after_reverse():
    runtime = ParkingRuntimeCoordinator()
    committed = runtime.update(
        mode=7, explicit_b=False, reverse_decision_point=True)
    assert committed.selected_slot == "A" and committed.branch_locked
    runtime.update(mode=7, reverse_started=True)
    late_b = runtime.update(mode=7, explicit_b=True)
    assert late_b.selected_slot == "A"
    assert late_b.state == "CSV_PARKING"


@pytest.mark.parametrize(
    "mode,explicit_b,slot,segment",
    ((7, False, "A", "T_A"), (7, True, "B", "T_B"),
     (10, False, "A", "V_A"), (10, True, "B", "V_B")))
def test_reverse_decision_never_waits_for_slot_and_uses_matching_csv(
        mode, explicit_b, slot, segment):
    runtime = ParkingRuntimeCoordinator()
    result = runtime.update(
        mode=mode, explicit_b=explicit_b,
        reverse_decision_point=True, nav2_path_valid=False)
    assert result.selected_slot == slot
    assert result.csv_fallback and result.branch_locked
    assert parking_csv_fallback_segment(mode, slot) == segment


def test_explicit_b_only_selection_defaults_everything_else_to_a():
    assert select_explicit_b_slot(True).slot == "B"
    assert select_explicit_b_slot(False).slot == "A"
    assert select_explicit_b_slot(None).slot == "A"


def test_nav2_ready_gate_checks_full_contract():
    footprint = assessment(True, 0.2)
    points = ((0.0, 0.0, 0.0), (-0.3, 0.0, 0.0),
              (-0.6, 0.0, 0.0))
    ready = validate_nav2_parking_path(
        points, planning_success=True, audit_feasible=True,
        max_required_steering_deg=19.9, footprint=footprint,
        selected_slot="B", target_slot="B")
    assert ready.valid and ready.reason == "READY"
    assert ready.direction_profile == (-1,)
    too_short = validate_nav2_parking_path(
        points[:2], planning_success=True, audit_feasible=True,
        max_required_steering_deg=0.0, footprint=footprint,
        selected_slot="A", target_slot="A")
    assert not too_short.valid and too_short.reason == "PATH_TOO_SHORT"
    wrong_slot = validate_nav2_parking_path(
        points, planning_success=True, audit_feasible=True,
        max_required_steering_deg=0.0, footprint=footprint,
        selected_slot="A", target_slot="B")
    assert not wrong_slot.valid and wrong_slot.reason == "TARGET_SLOT_INVALID"
    mixed = validate_nav2_parking_path(
        points+((-0.2, 0.0, 0.0),), planning_success=True,
        audit_feasible=True, max_required_steering_deg=0.0,
        footprint=footprint, selected_slot="A", target_slot="A")
    assert not mixed.valid and mixed.reason == "DIRECTION_SEQUENCE_INVALID"


@pytest.mark.parametrize(
    "overrides,reason",
    (({"planning_success": False}, "PLANNING_NOT_SUCCESS"),
     ({"audit_feasible": False}, "ACKERMANN_PATH_INFEASIBLE"),
     ({"max_required_steering_deg": 20.1}, "STEERING_LIMIT"),
     ({"footprint": assessment(False, -0.01)}, "FOOTPRINT_COLLISION"),
     ({"points": ((0.8, 0.0, 0.0), (0.4, 0.0, 0.0),
                   (0.0, 0.0, 0.0))}, "CURRENT_POSE_DISCONNECTED"),
     ({"points": ((0.0, 0.0, math.radians(46.0)),
                   (-0.3, -0.31, math.radians(46.0)),
                   (-0.6, -0.62, math.radians(46.0)))},
      "CURRENT_HEADING_DISCONNECTED")))
def test_nav2_ready_gate_rejects_each_safety_contract(overrides, reason):
    arguments = {
        "points": ((0.0, 0.0, 0.0), (-0.3, 0.0, 0.0),
                   (-0.6, 0.0, 0.0)),
        "planning_success": True,
        "audit_feasible": True,
        "max_required_steering_deg": 19.9,
        "footprint": assessment(True, 0.2),
        "selected_slot": "A",
        "target_slot": "A",
    }
    arguments.update(overrides)
    result = validate_nav2_parking_path(**arguments)
    assert not result.valid and result.reason == reason


def test_runtime_wiring_keeps_csv_during_planning_and_freezes_on_lock():
    manager = (ROOT/"depth_hybrid_slam"/
               "maneuver_manager_node.py").read_text()
    lifecycle = (ROOT/"depth_hybrid_slam"/
                 "parking_slam_manager_node.py").read_text()
    perception = (ROOT/"depth_hybrid_slam"/
                  "lidar_perception_node.py").read_text()
    assert 'prefix+"_"+runtime.state, "CSV", False' in manager
    assert 'prefix+"_SLAM_FREEZE"' in manager
    assert '"branch_locked": self.parking_runtime.branch_locked' in manager
    assert 'if branch_locked:' in lifecycle
    assert 'parking_fresh and (parking_use_front or rear_active)' in perception


def test_parking_diagnostics_expose_runtime_and_handoff_contract():
    manager = (ROOT/"depth_hybrid_slam"/
               "maneuver_manager_node.py").read_text()
    lifecycle = (ROOT/"depth_hybrid_slam"/
                 "parking_slam_manager_node.py").read_text()
    required = (
        "mode", "parking_slam_enabled", "parking_slam_state",
        "slam_active", "slam_stop_reason", "slot_observation_state",
        "explicit_b_received", "selected_slot", "selection_source",
        "nav2_planning", "nav2_path_ready", "nav2_plan_valid",
        "nav2_failure_reason", "reverse_decision_point_reached",
        "slot_latched", "branch_locked", "owner",
        "owner_handoff_active", "owner_handoff_elapsed_s", "csv_fallback",
        "fallback_branch", "drive", "wheel", "parking_reverse")
    for key in required:
        assert f'"{key}"' in manager or f'"{key}"' in lifecycle
    assert '"duplicate_map_odom_publishers": 0' in lifecycle


def test_csv_to_parking_owner_handoff_keeps_exact_one_second_stop():
    csv = CommandCandidate(2.0, 0, True, True, 0.0)
    parking = CommandCandidate(0.0, 0, True, True, 0.0)
    requested = arbitrate(csv, parking, mode=7)
    assert requested.owner == "PARKING" and requested.drive == 0.0
    handoff = OwnershipHandshake(initial_owner="CSV", stop_duration_s=1.0)
    first = handoff.update(
        requested.owner, speed_mps=0.0, odom_fresh=True,
        source_received_at=0.0, now=0.0)
    assert first.owner == "STOP" and first.drive == 0.0
    still = handoff.update(
        requested.owner, speed_mps=0.0, odom_fresh=True,
        source_received_at=0.5, now=0.99)
    assert still.owner == "STOP"
    assert handoff.update(
        requested.owner, speed_mps=0.0, odom_fresh=True,
        source_received_at=1.0, now=1.0) is None
    assert handoff.owner == "PARKING"
