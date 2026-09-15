"""Commissioned D456 mount geometry used by metric BEV projection."""

from dataclasses import dataclass
import math

import numpy as np


# Camera optical (right, down, forward) -> REP-103 mechanical
# (forward, left, up).
OPTICAL_TO_MECHANICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


@dataclass
class CameraMountConfig:
    configured: bool = False
    position_x_m: float = 0.0
    position_y_m: float = 0.0
    height_z_m: float = 0.0
    reference_roll_deg: float = 0.0
    reference_pitch_deg: float = 0.0
    reference_yaw_deg: float = 0.0

    def is_usable(self):
        values = (
            self.position_x_m, self.position_y_m, self.height_z_m,
            self.reference_roll_deg, self.reference_pitch_deg,
            self.reference_yaw_deg)
        return (self.configured and self.height_z_m > 0.0 and
                all(math.isfinite(value) for value in values))


def rotation_matrix_rpy(roll_deg, pitch_deg, yaw_deg):
    """Convert the physical camera mount RPY to a base-frame rotation."""
    roll = math.radians(roll_deg)
    # The mount file describes downward camera pitch as negative.
    pitch = -math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    return rz @ ry @ rx


def load_camera_mount_config(params):
    """Build mount geometry from resolved flat ROS parameters."""
    return CameraMountConfig(
        configured=bool(params.get("configured", False)),
        position_x_m=float(params.get("position_x_m", 0.0)),
        position_y_m=float(params.get("position_y_m", 0.0)),
        height_z_m=float(params.get("height_z_m", 0.0)),
        reference_roll_deg=float(params.get("reference_roll_deg", 0.0)),
        reference_pitch_deg=float(params.get("reference_pitch_deg", 0.0)),
        reference_yaw_deg=float(params.get("reference_yaw_deg", 0.0)),
    )
