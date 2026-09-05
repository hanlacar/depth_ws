"""ROS-independent data contracts used by nodes and deterministic tests."""

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float
    speed: float = 0.0
    yaw_rate: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class RoutePoint:
    index: int
    x: float
    y: float
    yaw: float
    direction: int = 1
    drive_level: float = 1.0
    mission_marker: str = ""
    section_id: str = ""


@dataclass(frozen=True)
class ControllerResult:
    drive: float
    steering_deg: float
    nearest_index: int
    target_index: int
    cross_track_error: float
    heading_error: float
    progress: float
    stop_required: bool
    reason: str


@dataclass(frozen=True)
class PathPose:
    x: float
    y: float
    yaw: float
    direction: int = 1


@dataclass(frozen=True)
class RejoinPlan:
    feasible: bool
    path: Sequence[PathPose] = field(default_factory=tuple)
    route_index: int = -1
    length_m: float = 0.0
    maximum_curvature: float = 0.0
    direction: int = 1
    reason: str = "REJOIN_NO_FEASIBLE_PATH"
    candidates_checked: int = 0
    collision_free: bool = False


@dataclass
class MissionInputs:
    now: float
    traffic_state: str = "UNKNOWN"
    traffic_aspect: str = "UNKNOWN"
    traffic_confidence: float = 0.0
    traffic_age: float = float("inf")
    stop_line_detected: bool = False
    stop_line_distance_m: float = float("inf")
    sign_detected: bool = False
    sign_confidence: float = 1.0
    pitch_deg: float = 0.0
    uphill_detected: bool = False
    steering_deg: float = 0.0
    section_id: str = ""


@dataclass(frozen=True)
class MissionDecision:
    state: str
    stop_required: bool
    stop_reason: str
    speed_limit: float
    traffic_permission: str


@dataclass
class SafetyInputs:
    now: float
    tracking_valid: bool = False
    localization_state: str = "INITIALIZING"
    localization_confidence: float = 0.0
    pose_stamp: float = 0.0
    pose_jump: bool = False
    map_route_match: bool = False
    within_map: bool = False
    cross_track_error: float = 0.0
    heading_error: float = 0.0
    controller_stamp: float = 0.0
    mission_stop: bool = False
    mission_reason: str = ""
    user_approved: bool = False
    enable_control: bool = False
    dry_run: bool = True


@dataclass(frozen=True)
class SafetyDecision:
    ready: bool
    stop_required: bool
    reasons: Sequence[str] = field(default_factory=tuple)
