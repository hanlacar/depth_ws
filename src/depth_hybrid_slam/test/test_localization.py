import math

from depth_hybrid_slam.localization_core import (
    LocalizationFusion, align_odom_pose_to_route_entry, odom_only_map_edge)
from depth_hybrid_slam.models import Pose2D


def pose(x=0.0, y=0.0, yaw=0.0, stamp=1.0):
    return Pose2D(x, y, yaw, stamp)


def test_odom_only_adds_map_to_odom_without_competing_with_vslam():
    assert odom_only_map_edge(False, "odom", "map") == ("map", "odom")
    assert odom_only_map_edge(False, "/odom", "/map") == ("map", "odom")
    assert odom_only_map_edge(True, "odom", "map") is None
    assert odom_only_map_edge(False, "map", "map") is None


def test_odom_only_aligns_current_pose_to_selected_route_entry():
    odom = (2.0, -1.0, math.radians(20.0))
    entry = (-3.0, 4.0, math.radians(70.0))
    tx, ty, yaw = align_odom_pose_to_route_entry(*odom, *entry)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    mapped_x = tx+cosine*odom[0]-sine*odom[1]
    mapped_y = ty+sine*odom[0]+cosine*odom[1]
    assert math.isclose(mapped_x, entry[0], abs_tol=1.0e-9)
    assert math.isclose(mapped_y, entry[1], abs_tol=1.0e-9)
    assert math.isclose(odom[2]+yaw, entry[2], abs_tol=1.0e-9)


def test_map_correction_is_applied_smoothly():
    core = LocalizationFusion(smoothing_alpha=0.5)
    assert core.set_correction((0.2, 0.0, 0.0), 1.0)
    output, state = core.fuse(pose(1.0))
    assert output.x == 1.1 and state == "RELOCALIZED"


def test_large_correction_waits_for_safe_approval():
    core = LocalizationFusion()
    assert not core.set_correction((2.0, 0.0, 0.0), 1.0)
    _, state = core.fuse(pose())
    assert state == "DEGRADED" and core.pending == (2.0, 0.0, 0.0)
    assert core.approve_pending(2.0)


def test_pose_jump_is_rejected():
    core = LocalizationFusion(odom_jump_m=0.5)
    assert core.fuse(pose(0.0, stamp=1.0))[0] is not None
    output, state = core.fuse(pose(1.0, stamp=1.1))
    assert output is None and state == "STOP_REQUIRED" and core.pose_jump


def test_tracking_loss_does_not_publish_pose():
    output, state = LocalizationFusion().fuse(pose(), tracking_valid=False)
    assert output is None and state == "LOST"
