"""Pure depth_ws-to-MCU GPS source contract adapter.

No MCU Python module is imported.  This is the single sign/unit boundary from
the route follower's physical ``+LEFT/-RIGHT`` convention to the current MCU
team GPS topic convention ``+RIGHT/-LEFT``.
"""

from dataclasses import dataclass
import math


ALLOWED_DRIVE_STAGES = frozenset((-1.0, 0.0, 1.0, 2.0, 3.0))


@dataclass(frozen=True)
class SourceCommand:
    drive: float
    wheel: int
    stop: bool
    valid: bool
    state: str


def adapt_slam_command(drive, wheel_left_positive, stop, ages_s,
                       timeout_s=0.5, max_wheel_deg=22):
    """Validate a complete fresh triplet and produce the MCU GPS contract."""
    try:
        drive = float(drive)
        wheel = int(wheel_left_positive)
        ages = tuple(float(value) for value in ages_s)
    except (TypeError, ValueError, OverflowError):
        return SourceCommand(0.0, 0, True, False, "INVALID_TYPE")
    if (len(ages) != 3 or not all(math.isfinite(value) for value in ages) or
            any(value < 0.0 or value > timeout_s for value in ages)):
        return SourceCommand(0.0, 0, True, False, "STALE_COMMAND")
    if not math.isfinite(drive) or drive not in ALLOWED_DRIVE_STAGES:
        return SourceCommand(0.0, 0, True, False, "INVALID_DRIVE_STAGE")
    if wheel < -max_wheel_deg or wheel > max_wheel_deg:
        return SourceCommand(0.0, 0, True, False, "INVALID_WHEEL_RANGE")
    if bool(stop):
        return SourceCommand(0.0, 0, True, True, "SOURCE_STOP")
    # MCU manager negates GPS/team steering once more on entry, restoring the
    # physical convention for /mcu/cmd_wheel and bridge odometry.
    return SourceCommand(drive, -wheel, False, True, "READY")
