"""Conservative camera-only local correction over an immutable CSV route."""

from dataclasses import dataclass
import math

from .lidar_local_planner import audit_path, swept_footprint_clear
from .vehicle_kinematics import planner_steering_feasible


@dataclass(frozen=True)
class CameraCorrectionDecision:
    state: str
    stop: bool
    active: bool
    need_plan: bool = False
    discard_path: bool = False


@dataclass(frozen=True)
class CameraPlan:
    valid: bool
    state: str
    points: tuple = ()
    max_steering_deg: float = math.inf
    max_curvature_jump: float = math.inf


class CameraCorrectionMachine:
    """Stop, observe for three seconds, plan, follow, and rejoin CSV."""

    def __init__(self, validation_s=3.0):
        self.validation_s = max(3.0, float(validation_s))
        self.state = "CSV_FOLLOW"
        self.wait_started = None

    def reset(self):
        self.state = "CSV_FOLLOW"
        self.wait_started = None

    def update(self, validation, lane_confident, *, now,
               vehicle_stopped=False, lidar_active=False, emergency=False,
               plan_valid=None, path_complete=False, rejoin_valid=False):
        validation = str(validation).strip().upper()
        now = float(now)
        if emergency:
            self.reset()
            return CameraCorrectionDecision(
                "EMERGENCY_STOP", True, False, discard_path=True)
        if lidar_active:
            self.reset()
            return CameraCorrectionDecision(
                "CAMERA_DISABLED_LIDAR_ACTIVE", False, False,
                discard_path=True)
        if self.state == "CSV_FOLLOW":
            if validation != "FAIL":
                return CameraCorrectionDecision("CSV_FOLLOW", False, False)
            self.state = "CAMERA_STOP"
        if self.state == "CAMERA_STOP":
            if validation != "FAIL":
                self.reset()
                return CameraCorrectionDecision("CSV_FOLLOW", False, False,
                                                discard_path=True)
            if vehicle_stopped:
                self.state = "CAMERA_WAIT_3S"
                self.wait_started = now
            return CameraCorrectionDecision(self.state, True, False)
        if self.state == "CAMERA_WAIT_3S":
            if validation != "FAIL":
                self.reset()
                return CameraCorrectionDecision("CSV_FOLLOW", False, False,
                                                discard_path=True)
            if not vehicle_stopped:
                self.state = "CAMERA_STOP"
                self.wait_started = None
                return CameraCorrectionDecision(self.state, True, False)
            elapsed = now-float(self.wait_started)
            if elapsed < self.validation_s:
                return CameraCorrectionDecision(self.state, True, False)
            if not lane_confident:
                self.state = "CAMERA_NO_VALID_GEOMETRY"
                return CameraCorrectionDecision(self.state, True, False)
            self.state = "CAMERA_PLAN"
        if self.state in ("CAMERA_PLAN", "CAMERA_NO_VALID_GEOMETRY"):
            if validation != "FAIL":
                self.reset()
                return CameraCorrectionDecision("CSV_FOLLOW", False, False,
                                                discard_path=True)
            if not lane_confident:
                self.state = "CAMERA_NO_VALID_GEOMETRY"
                return CameraCorrectionDecision(self.state, True, False)
            if plan_valid is None:
                self.state = "CAMERA_PLAN"
                return CameraCorrectionDecision(
                    self.state, True, False, need_plan=True)
            if not bool(plan_valid):
                self.state = "CAMERA_NO_VALID_PATH"
                return CameraCorrectionDecision(
                    self.state, True, False, discard_path=True)
            self.state = "CAMERA_FOLLOW"
            return CameraCorrectionDecision(self.state, False, True)
        if self.state == "CAMERA_NO_VALID_PATH":
            if validation != "FAIL":
                self.reset()
                return CameraCorrectionDecision("CSV_FOLLOW", False, False,
                                                discard_path=True)
            return CameraCorrectionDecision(self.state, True, False)
        if self.state == "CAMERA_FOLLOW":
            if path_complete:
                self.state = "CAMERA_CSV_REJOIN"
                return CameraCorrectionDecision(self.state, True, False)
            return CameraCorrectionDecision(self.state, False, True)
        if self.state == "CAMERA_CSV_REJOIN":
            if rejoin_valid:
                self.reset()
                return CameraCorrectionDecision(
                    "CSV_FOLLOW", False, False, discard_path=True)
            return CameraCorrectionDecision(self.state, True, False)
        self.reset()
        return CameraCorrectionDecision("CSV_FOLLOW", False, False,
                                        discard_path=True)


def _curvature_jumps(points):
    curvatures = []
    for first, second in zip(points, points[1:]):
        distance = math.hypot(second[0]-first[0], second[1]-first[1])
        if distance <= 1.0e-6:
            return math.inf
        heading = math.atan2(
            math.sin(second[2]-first[2]), math.cos(second[2]-first[2]))
        curvatures.append(heading/distance)
    return max((abs(second-first) for first, second in
                zip(curvatures, curvatures[1:])), default=0.0)


def plan_camera_correction(csv_path, left_boundary_m, right_boundary_m, *,
                           lane_points=(), curbs=(), vehicle_width_m=0.80,
                           planner_limit_deg=20.0,
                           minimum_length_m=3.0, maximum_length_m=5.0,
                           curvature_jump_limit=0.75):
    """Build a short quintic-like lane-safe shift that rejoins the CSV."""
    source = tuple((float(x), float(y), float(yaw))
                   for x, y, yaw in (csv_path or ())
                   if math.isfinite(float(x)) and math.isfinite(float(y)) and
                   math.isfinite(float(yaw)) and float(x) >= 0.0)
    if len(source) < 2:
        return CameraPlan(False, "CAMERA_CSV_PATH_UNAVAILABLE")
    source = ((0.0, 0.0, 0.0),)+source
    cumulative = [0.0]
    for first, second in zip(source, source[1:]):
        cumulative.append(cumulative[-1]+math.hypot(
            second[0]-first[0], second[1]-first[1]))
    usable = [index for index, distance in enumerate(cumulative)
              if distance <= float(maximum_length_m)+1.0e-9]
    source = source[:max(2, usable[-1]+1)]
    cumulative = cumulative[:len(source)]
    length = cumulative[-1]
    if length < float(minimum_length_m):
        return CameraPlan(False, "CAMERA_REJOIN_TOO_SHORT")
    left = math.inf if left_boundary_m is None else float(left_boundary_m)
    right = math.inf if right_boundary_m is None else float(right_boundary_m)
    required = float(vehicle_width_m)*0.5+0.10
    left_risk, right_risk = left < required, right < required
    if left_risk == right_risk:
        return CameraPlan(False, "CAMERA_LANE_SIDE_AMBIGUOUS")
    direction = -1.0 if left_risk else 1.0
    near = left if left_risk else right
    offset = direction*max(0.18, required-near+0.08)
    opposite = right if left_risk else left
    if opposite < required+abs(offset):
        return CameraPlan(False, "CAMERA_LANE_CORRIDOR_TOO_NARROW")
    raw = []
    for point, distance in zip(source, cumulative):
        phase = min(1.0, max(0.0, distance/length))
        lateral = offset*math.sin(math.pi*phase)**2
        x = point[0]-lateral*math.sin(point[2])
        y = point[1]+lateral*math.cos(point[2])
        raw.append([x, y, point[2]])
    for index, point in enumerate(raw):
        before = raw[max(0, index-1)]
        after = raw[min(len(raw)-1, index+1)]
        point[2] = math.atan2(after[1]-before[1], after[0]-before[0])
    raw[0] = [0.0, 0.0, 0.0]
    raw[-1] = list(source[-1])
    points = tuple(tuple(point) for point in raw)
    audit = audit_path(points)
    jump = _curvature_jumps(points)
    if (not audit.feasible or not planner_steering_feasible(
            audit.max_required_steering_deg, planner_limit_deg)):
        return CameraPlan(False, "CAMERA_STEERING_LIMIT", points,
                          audit.max_required_steering_deg, jump)
    if jump > float(curvature_jump_limit):
        return CameraPlan(False, "CAMERA_CURVATURE_JUMP", points,
                          audit.max_required_steering_deg, jump)
    if not swept_footprint_clear(points, curbs):
        return CameraPlan(False, "CAMERA_CURB_COLLISION", points,
                          audit.max_required_steering_deg, jump)
    if not swept_footprint_clear(
            points, lane_points, vehicle_width_m=vehicle_width_m,
            margin_m=0.02):
        return CameraPlan(False, "CAMERA_LANE_COLLISION", points,
                          audit.max_required_steering_deg, jump)
    return CameraPlan(True, "CAMERA_PATH_VALID", points,
                      audit.max_required_steering_deg, jump)
