"""Final vehicle command contract and priority arbitration."""

from dataclasses import dataclass


DRIVE_STAGES = frozenset((-1.0, 0.0, 1.0, 2.0, 3.0))


@dataclass(frozen=True)
class CommandCandidate:
    drive: float = 0.0
    wheel: int = 0
    valid: bool = False
    fresh: bool = False


@dataclass(frozen=True)
class ArbiterDecision:
    drive: float
    wheel: int
    owner: str
    state: str


def _valid(candidate):
    try:
        return (candidate.valid and candidate.fresh and
                float(candidate.drive) in DRIVE_STAGES and
                abs(int(candidate.wheel)) <= 22)
    except (TypeError, ValueError):
        return False


def arbitrate(csv, lidar, hard_emergency=False, mission_hold=False,
              lidar_slowdown=False, mode=-1, steering_deg=None,
              steering_slowdown=False):
    """Priority: stop, mission hold, distance slow, steering slow, speed."""
    if hard_emergency:
        return ArbiterDecision(0.0, 0, "SAFETY", "HARD_EMERGENCY_STOP")
    if mission_hold:
        return ArbiterDecision(0.0, 0, "MISSION", "MISSION_STOP_HOLD")
    if _valid(lidar):
        decision = ArbiterDecision(float(lidar.drive), int(lidar.wheel),
                                   "LIDAR", "LIDAR_ACTIVE")
    elif _valid(csv):
        decision = ArbiterDecision(float(csv.drive), int(csv.wheel),
                                   "CSV", "CSV_TRACKING")
    else:
        return ArbiterDecision(0.0, 0, "NONE", "NO_VALID_SOURCE")
    if lidar_slowdown and decision.drive > 1.0:
        return ArbiterDecision(1.0, decision.wheel, decision.owner,
                               "LIDAR_DISTANCE_SLOWDOWN")
    if steering_slowdown and decision.drive > 1.0:
        return ArbiterDecision(1.0, decision.wheel, decision.owner,
                               "STEERING_SLOWDOWN")
    if int(mode) == 9 and decision.drive > 0.0:
        return ArbiterDecision(3.0, decision.wheel, decision.owner,
                               "MODE9_FIXED_SPEED")
    return decision
