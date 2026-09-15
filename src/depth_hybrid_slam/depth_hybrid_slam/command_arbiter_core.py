"""Final vehicle command contract and priority arbitration."""

from dataclasses import dataclass
import time


DRIVE_STAGES = frozenset((-1.0, 0.0, 1.0, 2.0, 3.0))


@dataclass(frozen=True)
class CommandCandidate:
    drive: float = 0.0
    wheel: int = 0
    valid: bool = False
    fresh: bool = False
    received_at: float = 0.0


@dataclass(frozen=True)
class ArbiterDecision:
    drive: float
    wheel: int
    owner: str
    state: str


class OwnershipHandshake:
    """Exclusive owner handoff with measured standstill confirmation."""

    SOURCE_OWNERS = frozenset(("CSV", "CAMERA", "LIDAR", "PARKING"))

    def __init__(self, initial_owner="CSV", stop_duration_s=1.0,
                 standstill_speed_mps=0.03):
        self.owner = str(initial_owner)
        self.stop_duration_s = float(stop_duration_s)
        self.standstill_speed_mps = float(standstill_speed_mps)
        if self.stop_duration_s < 1.0 or self.standstill_speed_mps < 0.0:
            raise ValueError("invalid ownership handoff policy")
        self.pending_owner = None
        self.requested_at = None
        self.stopped_since = None

    @property
    def transitioning(self):
        return self.pending_owner is not None

    def update(self, requested_owner, *, speed_mps, odom_fresh,
               source_received_at=None, now=None):
        now = time.monotonic() if now is None else float(now)
        requested = str(requested_owner)
        if requested not in self.SOURCE_OWNERS:
            return ArbiterDecision(0.0, 0, "STOP", "SAFETY_STOP")
        if requested == self.owner:
            self.pending_owner = None
            self.requested_at = None
            self.stopped_since = None
            return None
        if self.pending_owner != requested:
            self.pending_owner = requested
            self.requested_at = now
            self.stopped_since = None
        # A command sampled before the ownership request is a late command
        # from the old episode and cannot complete the handoff.
        target_fresh = (source_received_at is not None and
                        float(source_received_at) >= self.requested_at)
        stopped = (bool(odom_fresh) and speed_mps is not None and
                   abs(float(speed_mps)) <= self.standstill_speed_mps)
        if not stopped:
            self.stopped_since = None
        elif self.stopped_since is None:
            self.stopped_since = now
        if (self.stopped_since is not None and target_fresh and
                now-self.stopped_since >= self.stop_duration_s):
            self.owner = requested
            self.pending_owner = None
            self.requested_at = None
            self.stopped_since = None
            return None
        return ArbiterDecision(
            0.0, 0, "STOP",
            f"OWNER_HANDOFF_{self.owner}_TO_{requested}")


def _valid(candidate):
    try:
        return (candidate.valid and candidate.fresh and
                float(candidate.drive) in DRIVE_STAGES and
                abs(int(candidate.wheel)) <= 22)
    except (TypeError, ValueError):
        return False


def arbitrate(csv, lidar, hard_emergency=False, mission_hold=False,
              lidar_slowdown=False, mode=-1, steering_slowdown=False,
              camera=None):
    """Choose exactly one owner: STOP > LiDAR/parking > camera > CSV."""
    if hard_emergency:
        return ArbiterDecision(0.0, 0, "SAFETY", "HARD_EMERGENCY_STOP")
    if mission_hold:
        return ArbiterDecision(0.0, 0, "MISSION", "MISSION_STOP_HOLD")
    if _valid(lidar):
        owner = "PARKING" if int(mode) in (7, 10) else "LIDAR"
        decision = ArbiterDecision(float(lidar.drive), int(lidar.wheel),
                                   owner, owner+"_ACTIVE")
    elif int(mode) not in (7, 10) and camera is not None and _valid(camera):
        decision = ArbiterDecision(float(camera.drive), int(camera.wheel),
                                   "CAMERA", "CAMERA_CORRECTION_ACTIVE")
    elif _valid(csv):
        decision = ArbiterDecision(float(csv.drive), int(csv.wheel),
                                   "CSV", "CSV_TRACKING")
    else:
        return ArbiterDecision(0.0, 0, "NONE", "NO_VALID_SOURCE")
    if lidar_slowdown and decision.drive > 1.0:
        return ArbiterDecision(1.0, decision.wheel, decision.owner,
                               "LIDAR_DISTANCE_SLOWDOWN")
    if steering_slowdown and int(mode) != 9 and decision.drive > 1.0:
        return ArbiterDecision(1.0, decision.wheel, decision.owner,
                               "STEERING_SLOWDOWN")
    if int(mode) == 9 and decision.drive > 0.0:
        return ArbiterDecision(3.0, decision.wheel, decision.owner,
                               "MODE9_FIXED_SPEED")
    return decision
