from depth_hybrid_slam.localization_core import LocalizationFusion
from depth_hybrid_slam.models import Pose2D


def pose(x=0.0, y=0.0, yaw=0.0, stamp=1.0):
    return Pose2D(x, y, yaw, stamp)


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
