import math
import csv

import pytest
from geometry_msgs.msg import PoseStamped

from vslam_nav2_route_editor.route_model import (
    RoutePoint, VehiclePolicy, apply_vehicle_policy, insert_required_stops,
    load_route, nearest_route_point, save_route, set_direction_range,
    set_event_point, set_stop_point,
    straight_points, validate_route,
)
from vslam_nav2_route_editor.route_follower_core import Pose2D, RouteFollower
from vslam_nav2_route_editor.map_odom_alignment_node import map_to_odom_alignment
from vslam_nav2_route_editor.t870_odom_core import T870OdomModel
from vslam_nav2_route_editor.synthetic_route_odom_node import (
    interpolate_pose, map_path_to_local,
)


def test_exact_straight_generation():
    points = straight_points(0.0, 0.0, 1.0, 0.0)
    assert len(points) == 6
    assert all(abs(point.y) < 1e-12 for point in points)
    assert all(abs(point.required_steering_deg) < 1e-12 for point in points)
    assert all(point.speed == 2.0 for point in points)


def test_turning_speed_and_invalid_limit():
    radius = 3.0
    points = [RoutePoint(i, 1, radius * math.cos(a), radius * math.sin(a), 0.0)
              for i, a in enumerate((0.0, 0.1, 0.2))]
    apply_vehicle_policy(points)
    assert all(abs(point.required_steering_deg) >= 10.0 for point in points)
    assert all(point.speed == 1.0 for point in points)
    tight = [RoutePoint(i, 1, math.cos(a), math.sin(a), 0.0)
             for i, a in enumerate((0.0, 0.2, 0.4))]
    apply_vehicle_policy(tight)
    assert validate_route(tight)["invalid_steering_indices"] == [0, 1, 2]


def test_reverse_keeps_vehicle_yaw_and_speed():
    points = [RoutePoint(i, 1, -0.2 * i, 0.0, 0.0, "R") for i in range(4)]
    apply_vehicle_policy(points)
    assert all(point.yaw_deg == 0.0 for point in points)
    assert all(point.speed == -1.0 for point in points)


def test_reverse_line_requires_explicit_vehicle_heading():
    try:
        straight_points(0.0, 0.0, -1.0, 0.0, direction="R")
        assert False, "reverse line without heading should fail"
    except ValueError as error:
        assert "vehicle-yaw" in str(error)
    points = straight_points(0.0, 0.0, -1.0, 0.0, direction="R",
                             vehicle_yaw_deg=0.0)
    assert all(point.yaw_deg == 0.0 for point in points)


def test_direction_change_gets_stop():
    points = [RoutePoint(0, 1, 0, 0, 0, "F"),
              RoutePoint(1, 1, .2, 0, 0, "R")]
    points = insert_required_stops(points)
    assert [point.direction for point in points] == ["F", "F", "R"]
    assert [point.event for point in points] == ["NONE", "STOP", "NONE"]
    assert validate_route(points)["direction_changes_without_stop"] == []


def test_csv_metadata_roundtrip(tmp_path):
    path = tmp_path / "START_A.csv"
    points = straight_points(1.0, 2.0, 2.0, 2.0)
    metadata = save_route(path, points, "START_A", VehiclePolicy(),
                          "/tmp/map.db", "abc")
    loaded = load_route(path)
    assert len(loaded) == len(points)
    assert loaded[-1].x == 2.0
    assert metadata["frame_id"] == "map"
    assert path.with_suffix(".metadata.yaml").is_file()


def test_live_direction_sequence_and_roundtrip(tmp_path):
    points = straight_points(0.0, 0.0, 4.0, 0.0, spacing=0.2)
    assert len(points) == 21
    yaws = [point.yaw_deg for point in points]
    points = set_direction_range(points, 0, 10, "F")
    points = set_stop_point(points, 11)
    points = set_direction_range(points, 12, 20, "R")
    assert [point.direction for point in points] == (
        ["F"] * 12 + ["R"] * 9)
    assert [point.event for point in points] == (
        ["NONE"] * 11 + ["STOP"] + ["NONE"] * 9)
    assert [point.segment_id for point in points] == (
        [1] * 11 + [2] + [3] * 9)
    assert [point.yaw_deg for point in points] == yaws
    assert all(point.speed == 2.0 for point in points[:11])
    assert points[11].speed == 0.0
    assert all(point.speed == -1.0 for point in points[12:])
    assert validate_route(points)["valid"]

    path = tmp_path / "START_A.csv"
    save_route(path, points, "START_A", VehiclePolicy())
    loaded = load_route(path)
    assert [(p.direction, p.event, p.segment_id, p.speed, p.yaw_deg)
            for p in loaded] == [
        (p.direction, p.event, p.segment_id, p.speed, p.yaw_deg)
        for p in points]
    with path.open(newline="", encoding="utf-8") as stream:
        assert next(csv.reader(stream)) == [
            "index", "segment_id", "x", "y", "yaw_deg", "direction",
            "speed", "required_steering_deg", "event"]


def test_direct_direction_change_is_rejected_without_stop():
    points = straight_points(0.0, 0.0, 4.0, 0.0, spacing=0.2)
    with pytest.raises(ValueError, match="requires STOP"):
        set_direction_range(points, 12, 20, "R")
    assert all(point.direction == "F" for point in points)


def test_legacy_csv_stop_migrates_to_event(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text(
        "index,segment_id,x,y,yaw_deg,direction,speed,required_steering_deg\n"
        "0,1,0,0,0,F,2,0\n"
        "1,2,0.2,0,0,STOP,0,0\n"
        "2,3,0.2,0,0,R,-1,0\n", encoding="utf-8")
    points = load_route(path)
    assert [p.direction for p in points] == ["F", "F", "R"]
    assert [p.event for p in points] == ["NONE", "STOP", "NONE"]
    assert validate_route(points)["valid"]


def test_event_and_nearest_selection():
    points = straight_points(0.0, 0.0, 1.0, 0.0)
    index, distance = nearest_route_point(points, 0.41, 0.02)
    assert index == 2
    assert distance < 0.03
    with pytest.raises(ValueError, match="maximum"):
        nearest_route_point(points, 0.4, 1.0)
    points = set_event_point(points, 2, "ACCEL")
    assert points[2].event == "ACCEL"
    points = set_event_point(points, 2, "STOP")
    assert points[2].event == "STOP" and points[2].speed == 0.0


def test_t870_measured_odom_equations_and_limits():
    model = T870OdomModel()
    assert model.set_steering_adc(484) == 0.0
    assert model.set_steering_adc(880) == 22.0
    assert model.set_steering_adc(88) == -22.0
    model.set_steering_adc(484)
    model.set_drive_stage(1.0)
    model.update_encoder(1000)
    model.update_encoder(1797)
    x, y, yaw = model.base_pose()
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
    model.set_drive_stage(-1.0)
    model.update_encoder(2594)
    assert model.base_pose()[0] == pytest.approx(0.0)


def test_follower_straight_turn_stop_reverse_and_clamp():
    straight = straight_points(0.0, 0.0, 4.0, 0.0)
    command = RouteFollower().command(straight, Pose2D(0.0, 0.0, 0.0))
    assert command.wheel_deg == pytest.approx(0.0)
    assert command.drive == 2.0

    turn = straight_points(0.0, 0.0, 4.0, 0.0)
    turn[0].required_steering_deg = 12.0
    command = RouteFollower().command(turn, Pose2D(0.0, 0.0, 0.0))
    # Stored CSV steering does not drive the follower; /nav2_route/path
    # geometry is authoritative.
    assert command.drive == 2.0

    corner = [RoutePoint(0, 1, 0.0, 0.0, 0.0),
              RoutePoint(1, 1, 0.8, 0.8, 45.0),
              RoutePoint(2, 1, 1.0, 1.5, 90.0)]
    command = RouteFollower(lookahead_m=0.5).command(
        corner, Pose2D(0.0, 0.0, 0.0))
    assert command.drive == 1.0

    stopped = set_event_point(straight, 0, "STOP")
    command = RouteFollower().command(stopped, Pose2D(0.0, 0.0, 0.0))
    assert command.drive == 0.0

    reverse = straight_points(0.0, 0.0, -4.0, 0.0, direction="R",
                              vehicle_yaw_deg=0.0)
    command = RouteFollower().command(reverse, Pose2D(0.0, 0.0, 0.0))
    assert command.drive == -1.0
    assert command.wheel_deg == pytest.approx(0.0)

    sharp = [RoutePoint(0, 1, 0, 0, 0), RoutePoint(1, 1, 0, 1, 0)]
    command = RouteFollower(lookahead_m=0.2).command(
        sharp, Pose2D(0.0, 0.0, 0.0))
    assert command.wheel_deg == 22.0 and command.saturated

    right = [RoutePoint(0, 1, 0, 0, 0), RoutePoint(1, 1, 0, -1, 0)]
    command = RouteFollower(lookahead_m=0.2).command(
        right, Pose2D(0.0, 0.0, 0.0))
    assert command.wheel_deg == -22.0 and command.saturated


def test_integer_wheel_and_speed_threshold_share_same_value():
    # Choose a target yielding about 9.6 degrees. It must publish wheel=10
    # and use the slow forward policy, rather than comparing an unseen float.
    alpha = math.asin(math.tan(math.radians(9.6)) / (2.0 * 0.73))
    points = [RoutePoint(0, 1, 0.0, 0.0, 0.0),
              RoutePoint(1, 1, math.cos(alpha), math.sin(alpha), 0.0)]
    command = RouteFollower(lookahead_m=0.2).command(
        points, Pose2D(0.0, 0.0, 0.0))
    assert command.wheel_deg == 10
    assert command.drive == 1.0


def test_follower_start_index_is_a_hard_lower_bound():
    points = straight_points(0.0, 0.0, 2.0, 0.0, spacing=0.2)
    follower = RouteFollower(start_index=4)
    command = follower.command(points, Pose2D(0.0, 0.0, 0.0))
    assert command.nearest_index == 4
    assert command.target_index >= 4

    # Even if the pose later lies closest to an earlier point, skipped points
    # are never reconsidered.
    command = follower.command(points, Pose2D(0.2, 0.0, 0.0))
    assert command.nearest_index >= 4

    with pytest.raises(ValueError, match="non-negative"):
        RouteFollower(start_index=-1)
    with pytest.raises(ValueError, match="outside route"):
        RouteFollower(start_index=len(points)).command(
            points, Pose2D(0.0, 0.0, 0.0))


def test_map_to_rviz_odom_alignment_matches_route_start():
    x, y, yaw = map_to_odom_alignment(
        (10.0, 20.0, math.radians(30.0)), (1.0, 2.0, math.radians(5.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    assert x + c * 1.0 - s * 2.0 == pytest.approx(10.0)
    assert y + s * 1.0 + c * 2.0 == pytest.approx(20.0)
    assert yaw + math.radians(5.0) == pytest.approx(math.radians(30.0))


def test_synthetic_path_starts_at_identity_and_interpolates():
    poses = []
    for x, y, yaw_deg in ((10.0, 20.0, 90.0),
                          (10.0, 21.0, 90.0),
                          (9.0, 21.0, 180.0)):
        stamped = PoseStamped()
        stamped.pose.position.x = x; stamped.pose.position.y = y
        yaw = math.radians(yaw_deg)
        stamped.pose.orientation.z = math.sin(yaw / 2.0)
        stamped.pose.orientation.w = math.cos(yaw / 2.0)
        poses.append(stamped)
    local = map_path_to_local(poses, 1)
    assert local[1] == pytest.approx((0.0, 0.0, 0.0))
    assert local[2] == pytest.approx((0.0, 1.0, math.pi / 2.0))
    assert interpolate_pose(local[1], local[2], 0.5) == pytest.approx(
        (0.0, 0.5, math.pi / 4.0))
    with pytest.raises(ValueError, match="outside Path"):
        map_path_to_local(poses, len(poses))
