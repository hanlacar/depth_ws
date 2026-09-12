"""Audited 0911 MCU mode ownership reference, with no MCU code dependency."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeOwnership:
    mode: int
    drive_owner: str
    wheel_owner: str
    stop_owner: str = "ANY_FRESH_SOURCE_OR_ESTOP"
    override: str = "MANUAL_FRESH_PAIR"


MODE_OWNERSHIP = {
    **{mode: ModeOwnership(mode, "CAMERA_IF_AUTH_ELSE_GPS",
                           "CAMERA_IF_AUTH_ELSE_GPS",
                           override="MANUAL; OBSTACLE_REQUIRES_SAFE_GPS")
       for mode in (0, 1, 3, 8, 11)},
    2: ModeOwnership(2, "CAMERA_THEN_GPS", "CAMERA_THEN_GPS"),
    4: ModeOwnership(4, "CAMERA_STOP_OR_GPS", "GPS",
                     override="MANUAL; CAMERA_STOP_REQUIRES_MISSION_AUTH"),
    5: ModeOwnership(5, "LIDAR_IF_ACTIVE_ELSE_CAMERA_OR_GPS",
                     "LIDAR_IF_ACTIVE_ELSE_CAMERA_OR_GPS"),
    6: ModeOwnership(6, "CAMERA_STOP_OR_GPS", "GPS",
                     override="MANUAL; CAMERA_STOP_REQUIRES_MISSION_AUTH"),
    7: ModeOwnership(7, "LIDAR", "LIDAR"),
    9: ModeOwnership(9, "LIDAR_WITH_OPTIONAL_CAMERA_LIMIT",
                     "CAMERA_THEN_GPS",
                     override="MANUAL; LIDAR_STATUS_AND_STOP_HARD_GATE"),
    10: ModeOwnership(10, "LIDAR", "LIDAR"),
}


def ownership_for_mode(mode):
    numeric = int(str(mode).strip())
    if numeric not in MODE_OWNERSHIP:
        raise ValueError("mode must be 0..11")
    return MODE_OWNERSHIP[numeric]
