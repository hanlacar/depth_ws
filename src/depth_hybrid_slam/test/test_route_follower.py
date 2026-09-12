import math

from depth_hybrid_slam.models import Pose2D, RoutePoint
from depth_hybrid_slam.route_follower_core import RouteFollower


def route(points):
    return [RoutePoint(i, x, y, yaw) for i, (x, y, yaw) in enumerate(points)]


def test_straight_and_unapproved_is_zero_drive():
    result = RouteFollower().compute(Pose2D(0, 0, 0, 1),
                                     route([(0, 0, 0), (2, 0, 0)]))
    assert result.drive == 0.0 and result.reason == "CONTROL_NOT_APPROVED"
    assert result.steering_deg == 0.0


def test_right_turn_is_negative_and_left_is_positive():
    right = RouteFollower(steering_rate_deg_s=1000).compute(
        Pose2D(0, 0, 0, 1), route([(0, 0, 0), (1, -1, -math.pi/4)]),
        allow_motion=True)
    left = RouteFollower(steering_rate_deg_s=1000).compute(
        Pose2D(0, 0, 0, 1), route([(0, 0, 0), (1, 1, math.pi/4)]),
        allow_motion=True)
    assert right.steering_deg < 0 and left.steering_deg > 0


def test_reverse_uses_body_yaw_and_preserves_left_positive_sign():
    points = [RoutePoint(0, 0, 0, math.pi, direction=-1),
              RoutePoint(1, 1, 1, -3*math.pi/4, direction=-1)]
    result = RouteFollower(steering_rate_deg_s=1000).compute(
        Pose2D(0, 0, math.pi, 1), points, allow_motion=True)
    assert result.drive < 0
    assert result.steering_deg < 0


def test_steering_limit_and_slew_rate():
    follower = RouteFollower(lookahead_m=0.1, steering_rate_deg_s=30)
    result = follower.compute(Pose2D(0, 0, 0, 1),
                              route([(0, 0, 0), (0, -1, -math.pi/2)]),
                              dt=1/30, allow_motion=True)
    assert abs(result.steering_deg) <= 1.000001
    for _ in range(100):
        result = follower.compute(Pose2D(0, 0, 0, 1),
                                  route([(0, 0, 0), (0, -1, -math.pi/2)]),
                                  dt=1/30, allow_motion=True)
    assert abs(result.steering_deg) <= 22.0


def test_corridor_violation_stops():
    result = RouteFollower(corridor_m=0.5).compute(
        Pose2D(0, 1, 0, 1), route([(0, 0, 0), (2, 0, 0)]), allow_motion=True)
    assert result.stop_required and result.reason == "CORRIDOR_VIOLATION"


def test_s_curve_is_continuous_and_respects_sign_and_limit():
    points = route([(0, 0, 0), (0.5, 0.3, .5), (1.0, 0, -.5),
                    (1.5, -.3, -.5), (2.0, 0, 0)])
    follower = RouteFollower(steering_rate_deg_s=90)
    previous = 0.0
    for pose in (Pose2D(0, 0, 0, 1), Pose2D(.5, .3, .5, 2),
                 Pose2D(1, 0, -.5, 3)):
        result = follower.compute(pose, points, dt=1/30, allow_motion=True)
        assert abs(result.steering_deg) <= 22.0
        assert abs(result.steering_deg-previous) <= 3.000001
        previous = result.steering_deg


def test_route_end_and_explicit_stop_point_stop():
    points = route([(0, 0, 0), (1, 0, 0)])
    assert RouteFollower().compute(Pose2D(1, 0, 0, 1), points,
                                   allow_motion=True).reason == "ROUTE_COMPLETE"
    marked = [RoutePoint(0, 0, 0, 0),
              RoutePoint(1, 1, 0, 0, mission_marker="STOP")]
    assert RouteFollower(lookahead_m=.5).compute(
        Pose2D(.75, 0, 0, 1), marked, allow_motion=True).reason == "ROUTE_STOP_POINT"
