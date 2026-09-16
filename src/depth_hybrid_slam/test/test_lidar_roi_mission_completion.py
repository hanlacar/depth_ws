import math

from depth_hybrid_slam.command_arbiter_core import arbitrate, CommandCandidate
from depth_hybrid_slam.lidar_local_planner import plan_detour
from depth_hybrid_slam.lidar_roi_core import (
    ackermann_centerline, assess_curved_roi, assess_mode5_broad, Cluster,
    clusters_in_centerline_corridor, DYNAMIC, DynamicClusterTracker, mode_gates,
    speed_bump_suppressed, STATIC, UNKNOWN)
from depth_hybrid_slam.lidar_scan_core import optional_rear_hard_stop
from depth_hybrid_slam.mission_completion import MissionCompletionTracker


def obstacle(x, y, motion=STATIC):
    points = ((x, y-0.015), (x, y+0.015))
    return Cluster(points, (x, y), motion=motion)


def test_a_ackermann_roi_uses_left_positive_right_negative():
    left, _, _ = ackermann_centerline(20)
    straight, _, radius = ackermann_centerline(0)
    right, _, _ = ackermann_centerline(-20)
    assert left[-1][2] > 0.1
    assert all(abs(point[2]) < 1.0e-9 for point in straight)
    assert math.isinf(radius)
    assert right[-1][2] < -0.1


def test_b_distance_zones_static_and_dynamic():
    assert assess_curved_roi((obstacle(0.49, 0),), 0).hard_stop
    slow = assess_curved_roi((obstacle(0.75, 0),), 0)
    assert slow.slowdown and not slow.hard_stop
    assert not assess_curved_roi((obstacle(1.25, 0),), 0).slowdown
    assert not assess_curved_roi((obstacle(1.25, 0, DYNAMIC),), 0).slowdown


def test_c_front_rear_mode_gating():
    assert mode_gates(2) == (True, False)
    assert mode_gates(7) == (True, True)
    assert mode_gates(10) == (True, True)
    for mode in (1, 3, 4, 5, 6, 8, 9, 11):
        assert mode_gates(mode) == (True, False)


def test_front_roi_has_exact_half_width_and_distance_contract():
    assert not assess_curved_roi((obstacle(0.5, 0.32),), 0).hard_stop
    assert assess_curved_roi((obstacle(0.49, 0.29),), 0).hard_stop
    assert assess_curved_roi((obstacle(1.0, 0.0),), 0).slowdown
    assert not assess_curved_roi((obstacle(1.01, 0.0),), 0).slowdown
    assert not assess_curved_roi((obstacle(1.49, 0.0, DYNAMIC),), 0).slowdown
    assert not assess_curved_roi((obstacle(1.51, 0.0, DYNAMIC),), 0).slowdown


def test_mode5_selection_uses_only_visible_three_zone_corridor():
    centerline, _, _ = ackermann_centerline(0)
    inside = obstacle(1.25, 0.0)
    outside_width = obstacle(0.75, 0.32)
    outside_length = obstacle(1.51, 0.0)
    assert clusters_in_centerline_corridor(
        (inside, outside_width, outside_length), centerline) == (inside,)


def test_fixed_frame_tracking_does_not_call_ego_motion_dynamic():
    tracker = DynamicClusterTracker(0.25, 3, 1.0, 0.6)
    result = tracker.update((obstacle(2.0, 0, UNKNOWN),), 0.0, (0, 0, 0))
    assert result[0].motion == UNKNOWN
    for index in range(1, 5):
        result = tracker.update(
            (obstacle(2.0-0.1*index, 0, UNKNOWN),), index*0.1,
            (0.1*index, 0, 0))
    assert result[0].motion == STATIC


def test_dynamic_requires_n_fixed_frame_observations():
    tracker = DynamicClusterTracker(0.25, 3, 1.0, 0.6)
    motions = []
    for index in range(4):
        result = tracker.update((obstacle(1.0+0.04*index, 0, UNKNOWN),),
                                index*0.1, (0, 0, 0))
        motions.append(result[0].motion)
    assert motions[:3] == [UNKNOWN, UNKNOWN, UNKNOWN]
    assert motions[3] == DYNAMIC
    assert tracker.update((obstacle(1, 0),), 3.0, None)[0].motion == UNKNOWN


def test_d_mode5_collision_trigger_and_curb_roles():
    route = tuple((x/10.0, 0.0) for x in range(21))
    left = assess_mode5_broad((obstacle(1.0, 0.9),), route)
    right = assess_mode5_broad((obstacle(1.0, -0.9),), route)
    blocked = assess_mode5_broad((obstacle(1.0, 0.2),), route)
    curb = Cluster(((0.3, 0.8), (0.45, 0.81), (0.65, 0.8), (0.8, 0.8)),
                   (0.55, 0.8025), motion=STATIC)
    curb_only = assess_mode5_broad((curb,), route)
    assert left.obstacles and not left.path_blocked
    assert right.obstacles and not right.path_blocked
    assert blocked.path_blocked and blocked.collision_obstacles
    assert curb_only.curbs and not curb_only.obstacles
    assert not curb_only.path_blocked


def test_mode5_csv_collision_roi_is_capped_at_one_point_five_metres():
    route = tuple((x/10.0, 0.0) for x in range(21))
    outside_roi = obstacle(2.0, 0.0)
    beside_csv = obstacle(1.0, 0.9)
    on_csv = obstacle(1.4, 0.0)
    result = assess_mode5_broad(
        (outside_roi, beside_csv, on_csv), route, range_m=1.5)
    assert outside_roi not in result.obstacles
    assert beside_csv in result.obstacles
    assert beside_csv not in result.collision_obstacles
    assert result.collision_obstacles == (on_csv,)
    assert result.path_blocked


def test_mode5_broad_sensing_origin_is_front_laser_but_sweep_is_base_link():
    sensor_pose = (0.73, 0.0, 0.0)
    route_in_base = ((0.90, 0.0),)
    ahead = obstacle(0.90, 0.0)
    behind_sensor = obstacle(0.50, 0.0)
    result = assess_mode5_broad(
        (ahead, behind_sensor), route_in_base, sensor_pose=sensor_pose)
    assert result.obstacles == (ahead,)
    assert result.collision_obstacles == (ahead,)
    assert result.path_blocked


def test_d_mode5_obstacle_and_curbs_generate_sub_22_degree_path():
    plan = plan_detour(0.2, obstacles=((1.7, 0.2),),
                       left_boundary_m=1.0, right_boundary_m=-1.0)
    assert plan.valid
    assert plan.max_steering_deg < 22.0
    assert all(-1.0 < point[1] < 1.0 for point in plan.points)


def test_e_speed_bump_fusion_is_fail_closed():
    evidence = {"camera_fresh": True, "drivable_road": True,
                "object_evidence_valid": True, "object_detected": False,
                "motion": STATIC}
    assert speed_bump_suppressed(8, 105, ((8, 100, 110),), **evidence)
    assert speed_bump_suppressed(9, 205, ((9, 200, 210),), **evidence)
    assert not speed_bump_suppressed(
        8, 105, ((8, 100, 110),), **{**evidence, "object_detected": True})
    assert not speed_bump_suppressed(
        8, 105, ((8, 100, 110),), **{**evidence, "camera_fresh": False})
    assert not speed_bump_suppressed(
        8, 105, ((8, 100, 110),), **{**evidence, "motion": DYNAMIC})


def test_canonical_low_speed_stage_is_used_without_path_takeover():
    csv = CommandCandidate(3.0, 12, True, True)
    decision = arbitrate(csv, CommandCandidate(), lidar_slowdown=True)
    assert (decision.drive, decision.wheel, decision.owner) == (1.0, 12, "CSV")


def test_f_mode2_completion_requires_actual_four_seconds_and_pitch():
    value = MissionCompletionTracker()
    value.set_mode(2)
    value.observe_route_status({"route_complete_modes": [2]})
    value.tick(0.0, 0.0, stop_waypoint_active=True,
               pitch_deg=5.1, pitch_valid=True)
    value.tick(0.5, 0.0, stop_waypoint_active=False,
               pitch_deg=5.1, pitch_valid=True)
    value.tick(3.99, 0.0, stop_waypoint_active=False,
               pitch_deg=5.1, pitch_valid=True)
    assert not value.mode_complete(2)
    value.tick(4.01, 0.0, pitch_deg=5.1, pitch_valid=True)
    assert value.mode_complete(2)


def test_f_intersections_complete_only_after_exited_event():
    value = MissionCompletionTracker()
    value.observe_route_status({"route_complete_modes": [4, 6, 8]})
    for mode in (4, 6, 8):
        assert not value.mode_complete(mode)
        value.observe_intersection_event(f"EXITED mode={mode}")
        assert value.mode_complete(mode)


def test_f_mode5_two_rejoins_pass_one_rejoin_fails_at_exit():
    passed = MissionCompletionTracker()
    passed.set_mode(5)
    passed.observe_route_status({"route_complete_modes": [5]})
    passed.observe_maneuver({"mode": 5, "event": "AVOIDANCE_REJOINED"})
    passed.observe_maneuver({"mode": 5, "event": "AVOIDANCE_REJOINED"})
    assert passed.mode_complete(5)
    failed = MissionCompletionTracker()
    failed.set_mode(5)
    failed.observe_route_status({"route_complete_modes": [5]})
    failed.observe_maneuver({"mode": 5, "event": "AVOIDANCE_REJOINED"})
    failed.set_mode(6)
    assert 5 in failed.invalid_modes and not failed.mode_complete(5)


def _parking(source, mode):
    value = MissionCompletionTracker()
    value.set_mode(mode)
    value.observe_rear_lidar(True)
    value.observe_route_status({"route_complete_modes": [mode]})
    value.observe_lidar_safety({"mode": mode, "slot_a": True})
    if source == "LIDAR":
        value.observe_maneuver({"mode": mode, "event": "PARKING_CSV_REJOINED",
                                "source": "LIDAR"})
    else:
        value.observe_maneuver({"mode": mode, "event": "PARKING_CSV_FALLBACK"})
        value.observe_csv_parking_complete(mode)
    return value


def test_f_mode7_and_10_lidar_and_csv_fallback_parking():
    for mode in (7, 10):
        assert _parking("LIDAR", mode).mode_complete(mode)
        assert _parking("CSV_FALLBACK", mode).mode_complete(mode)
        front_only = MissionCompletionTracker()
        front_only.set_mode(mode)
        front_only.observe_route_status({"route_complete_modes": [mode]})
        front_only.observe_maneuver({
            "mode": mode, "event": "PARKING_CSV_FALLBACK"})
        front_only.observe_csv_parking_complete(mode)
        assert not front_only.mode_complete(mode)
        missing = MissionCompletionTracker()
        missing.observe_route_status({"route_complete_modes": [mode]})
        missing.observe_maneuver({"mode": mode, "event": "PARKING_CSV_REJOINED"})
        assert not missing.mode_complete(mode)


def test_optional_rear_lidar_never_stops_when_missing_but_blocks_completion():
    assert not optional_rear_hard_stop(True, False, True)
    assert not optional_rear_hard_stop(False, True, True)
    assert optional_rear_hard_stop(True, True, True)

    value = MissionCompletionTracker()
    value.set_mode(7)
    value.observe_route_status({"route_complete_modes": [7]})
    value.observe_lidar_safety({"mode": 7, "slot_a": True})
    value.observe_maneuver({
        "mode": 7, "event": "PARKING_CSV_REJOINED", "source": "LIDAR"})
    assert not value.mode_complete(7)
    assert not value.status()["parking_rear_verified"][7]
    value.observe_rear_lidar(True)
    assert value.mode_complete(7)


def test_f_mode9_requires_detection_and_applied_zero_command():
    value = MissionCompletionTracker()
    value.set_mode(9)
    value.observe_route_status({"route_complete_modes": [9]})
    value.observe_lidar_safety({"mode": 9, "slowdown": True,
                                "hard_stop": False, "distance_m": 0.8})
    value.tick(0, 1.0)
    assert not value.mode_complete(9)
    value.observe_lidar_safety({"mode": 9, "hard_stop": True,
                                "distance_m": 0.43})
    value.tick(1, 1.0)
    assert not value.mode_complete(9)
    value.tick(2, 0.0)
    assert value.mode_complete(9)


def test_f_mode11_camera_and_default_a_complete_after_actual_stop():
    for source, branch in (("CAMERA", "B"), ("DEFAULT", "A")):
        value = MissionCompletionTracker()
        value.set_mode(11)
        value.observe_route_status({"route_complete_modes": [11]})
        value.tick(0.0, 0.0, maneuver_state="MODE11_5S_HOLD")
        value.tick(5.01, 0.0, maneuver_state="MODE11_COMMITTED")
        value.observe_maneuver({"mode": 11,
                                "event": "MODE11_BRANCH_COMMITTED",
                                "source": source, "branch": branch})
        assert value.mode_complete(11)
