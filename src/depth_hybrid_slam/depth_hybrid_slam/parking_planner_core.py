"""ROS-independent parking selection, footprint and fallback contracts."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ParkingVehicleGeometry:
    wheelbase_m: float = 0.73
    vehicle_length_m: float = 1.40
    vehicle_width_m: float = 0.78
    front_overhang_m: float = 0.47
    rear_overhang_m: float = 0.20
    wheel_track_m: float = 0.62
    wheel_length_m: float = 0.20
    wheel_width_m: float = 0.08
    safety_margin_m: float = 0.15
    planner_max_steering_deg: float = 20.0

    def validate(self):
        positive = (
            self.wheelbase_m, self.vehicle_length_m, self.vehicle_width_m,
            self.wheel_track_m, self.wheel_length_m, self.wheel_width_m)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("parking vehicle dimensions must be positive")
        reconstructed = (self.front_overhang_m+self.wheelbase_m+
                         self.rear_overhang_m)
        if abs(reconstructed-self.vehicle_length_m) > 1.0e-6:
            raise ValueError("parking overhangs do not reconstruct vehicle length")
        if not 0.0 < self.planner_max_steering_deg <= 20.0:
            raise ValueError("parking planner steering must be in (0,20] deg")
        if self.wheel_track_m+self.wheel_width_m > self.vehicle_width_m+1.0e-9:
            raise ValueError("wheel footprints exceed overall vehicle width")
        return self


@dataclass(frozen=True)
class ParkingPathAssessment:
    safe: bool
    minimum_clearance_m: float
    body_clearance_m: float
    wheel_clearance_m: float
    checked_poses: int
    reason: str


@dataclass(frozen=True)
class ParkingSelection:
    slot: str
    source: str
    reason: str


@dataclass(frozen=True)
class ParkingPlanResolution:
    slot: str
    source: str
    reason: str
    attempted_slots: tuple


@dataclass(frozen=True)
class ParkingRuntimeDecision:
    state: str
    selected_slot: str
    selection_source: str
    slam_should_run: bool
    csv_approach: bool
    nav2_path_ready: bool
    csv_fallback: bool
    branch_locked: bool


@dataclass(frozen=True)
class ParkingNav2Validation:
    valid: bool
    reason: str
    direction_profile: tuple
    path_point_count: int


class ParkingRuntimeCoordinator:
    """Latch explicit-B-only parking choice at Nav2/reverse commitment."""

    def __init__(self, minimum_observation_s=3.0):
        self.minimum_observation_s = float(minimum_observation_s)
        if self.minimum_observation_s < 0.0:
            raise ValueError("parking observation time cannot be negative")
        self.reset()

    def reset(self, mode=-1):
        self.mode = int(mode)
        self.explicit_b_received = False
        self.selected_slot = ""
        self.selection_source = ""
        self.reverse_decision_point_reached = False
        self.branch_locked = False
        self.nav2_committed = False
        self.csv_fallback = False
        self.state = "INACTIVE"

    @property
    def candidate_slot(self):
        return "B" if self.explicit_b_received else "A"

    def update(self, *, mode, explicit_b=False, observation_elapsed_s=0.0,
               nav2_planning=False, nav2_slot="", nav2_path_valid=False,
               reverse_decision_point=False, reverse_started=False):
        mode = int(mode)
        if mode not in (7, 10):
            self.reset(mode)
            return self.decision()
        if self.mode != mode:
            self.reset(mode)
        if not self.branch_locked and bool(explicit_b):
            self.explicit_b_received = True
            self.selected_slot = "B"
            self.selection_source = "EXPLICIT_B"
        if bool(reverse_decision_point):
            self.reverse_decision_point_reached = True

        if not self.branch_locked and bool(nav2_path_valid):
            slot = str(nav2_slot).strip().upper()
            b_allowed = slot == "B" and self.explicit_b_received
            a_allowed = (
                slot == "A" and not self.explicit_b_received and
                (float(observation_elapsed_s) >= self.minimum_observation_s or
                 self.reverse_decision_point_reached))
            if a_allowed or b_allowed:
                self.selected_slot = slot
                self.selection_source = (
                    "NAV2_EXPLICIT_B" if slot == "B" else
                    "NAV2_DEFAULT_A_AFTER_OBSERVATION")
                self.branch_locked = True
                self.nav2_committed = True
                self.state = "NAV2_READY"

        if not self.branch_locked and self.reverse_decision_point_reached:
            self.selected_slot = self.candidate_slot
            self.selection_source = (
                "REVERSE_DECISION_EXPLICIT_B" if
                self.explicit_b_received else
                "REVERSE_DECISION_DEFAULT_A")
            self.branch_locked = True
            self.csv_fallback = True
            self.state = "CSV_SLOT_COMMITTED"

        if self.branch_locked:
            if reverse_started:
                self.state = (
                    "NAV2_TRACKING" if self.nav2_committed else "CSV_PARKING")
        else:
            self.selected_slot = self.candidate_slot
            self.selection_source = (
                "EXPLICIT_B_CANDIDATE" if self.explicit_b_received else
                "DEFAULT_A_CANDIDATE")
            self.state = "NAV2_PLANNING" if nav2_planning else "SLAM_OBSERVING"
        return self.decision()

    def decision(self):
        return ParkingRuntimeDecision(
            self.state, self.selected_slot, self.selection_source,
            not self.branch_locked, not self.branch_locked,
            self.nav2_committed, self.csv_fallback, self.branch_locked)


def fresh_explicit_b(b_free, reported_fresh, observed_at, now,
                     timeout_s=0.5):
    """Accept B only from a fresh, finite slot observation."""
    try:
        age = float(now)-float(observed_at)
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        return False
    return bool(b_free and reported_fresh and math.isfinite(age) and
                math.isfinite(timeout) and timeout >= 0.0 and
                0.0 <= age <= timeout)


def parking_slot_observation_state(b_free, reported_fresh, observed_at, now,
                                   timeout_s=0.5):
    """Describe explicit-B evidence without treating stale data as B."""
    try:
        age = float(now)-float(observed_at)
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        return "STALE"
    if (not reported_fresh or not math.isfinite(age) or
            not math.isfinite(timeout) or timeout < 0.0 or
            age < 0.0 or age > timeout):
        return "STALE"
    return "FRESH_EXPLICIT_B" if bool(b_free) else "FRESH_NO_B"


def select_explicit_b_slot(explicit_b, source="CSV"):
    """Select B only from a fresh explicit B indication; default to A."""
    return ParkingSelection(
        "B" if bool(explicit_b) else "A", str(source),
        "EXPLICIT_B" if bool(explicit_b) else "DEFAULT_A_NO_EXPLICIT_B")


def _point_rectangle_clearance(point, center, yaw, half_length, half_width):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = float(point[0])-center[0], float(point[1])-center[1]
    longitudinal = cosine*dx+sine*dy
    lateral = -sine*dx+cosine*dy
    outside_x = max(abs(longitudinal)-half_length, 0.0)
    outside_y = max(abs(lateral)-half_width, 0.0)
    if outside_x == 0.0 and outside_y == 0.0:
        return -min(half_length-abs(longitudinal),
                    half_width-abs(lateral))
    return math.hypot(outside_x, outside_y)


def _wheel_centers(x, y, yaw, geometry):
    half_track = geometry.wheel_track_m*0.5
    cosine, sine = math.cos(yaw), math.sin(yaw)
    output = []
    for longitudinal in (0.0, geometry.wheelbase_m):
        for lateral in (-half_track, half_track):
            output.append((x+cosine*longitudinal-sine*lateral,
                           y+sine*longitudinal+cosine*lateral))
    return tuple(output)


def assess_parking_path(points, obstacles,
                        geometry=ParkingVehicleGeometry()):
    """Sweep the body and four wheel rectangles over every path pose."""
    geometry = geometry.validate()
    poses = tuple(points or ())
    objects = tuple((float(x), float(y)) for x, y in (obstacles or ()))
    if not poses:
        return ParkingPathAssessment(
            False, -math.inf, -math.inf, -math.inf, 0, "NO_PATH")
    if not objects:
        return ParkingPathAssessment(
            True, math.inf, math.inf, math.inf, len(poses), "CLEAR")
    body_best = wheel_best = math.inf
    body_half_length = geometry.vehicle_length_m*0.5 + \
        geometry.safety_margin_m
    body_half_width = geometry.vehicle_width_m*0.5 + \
        geometry.safety_margin_m
    wheel_half_length = geometry.wheel_length_m*0.5 + \
        geometry.safety_margin_m
    wheel_half_width = geometry.wheel_width_m*0.5 + \
        geometry.safety_margin_m
    # base_link/route poses are treated as rear-axle poses. Shift the body
    # rectangle so the asymmetric front/rear overhangs are represented.
    body_center_offset = (
        geometry.wheelbase_m+geometry.front_overhang_m-
        geometry.rear_overhang_m)*0.5
    for raw_pose in poses:
        x, y, yaw = (float(value) for value in raw_pose[:3])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        body_center = (x+cosine*body_center_offset,
                       y+sine*body_center_offset)
        wheels = _wheel_centers(x, y, yaw, geometry)
        for obstacle in objects:
            body_best = min(body_best, _point_rectangle_clearance(
                obstacle, body_center, yaw,
                body_half_length, body_half_width))
            for center in wheels:
                wheel_best = min(wheel_best, _point_rectangle_clearance(
                    obstacle, center, yaw,
                    wheel_half_length, wheel_half_width))
    clearance = min(body_best, wheel_best)
    return ParkingPathAssessment(
        clearance > 0.0, clearance, body_best, wheel_best, len(poses),
        "CLEAR" if clearance > 0.0 else "FOOTPRINT_COLLISION")


def select_csv_parking_path(path_a, path_b, *, evidence_valid=True):
    """Select a stored path; unsafe pairs are ranked by total clearance."""
    if not evidence_valid or path_a is None or path_b is None:
        return ParkingSelection("A", "CSV", "DEFAULT_A_INSUFFICIENT_EVIDENCE")
    if path_a.safe and not path_b.safe:
        return ParkingSelection("A", "CSV", "A_SAFE_B_UNSAFE")
    if path_b.safe and not path_a.safe:
        return ParkingSelection("B", "CSV", "B_SAFE_A_UNSAFE")
    if path_a.safe and path_b.safe:
        return ParkingSelection("A", "CSV", "BOTH_SAFE_DEFAULT_A")
    a_clearance = float(path_a.minimum_clearance_m)
    b_clearance = float(path_b.minimum_clearance_m)
    if math.isfinite(b_clearance) and (
            not math.isfinite(a_clearance) or b_clearance > a_clearance):
        return ParkingSelection("B", "CSV", "B_GREATER_CLEARANCE")
    return ParkingSelection("A", "CSV", "A_GREATER_OR_EQUAL_CLEARANCE")


def select_slam_slot(a_state, b_state, path_a=None, path_b=None):
    """Select a SLAM slot, ranking clearance when both are occupied."""
    a = str(a_state).strip().upper()
    b = str(b_state).strip().upper()
    if a == "FREE":
        return ParkingSelection("A", "SLAM", "A_FREE")
    if b == "FREE":
        return ParkingSelection("B", "SLAM", "B_FREE_A_NOT_FREE")
    if a == "OCCUPIED" and b == "OCCUPIED":
        ranked = select_csv_parking_path(path_a, path_b)
        return ParkingSelection(
            ranked.slot, "SLAM", "BOTH_OCCUPIED_"+ranked.reason)
    return ParkingSelection("A", "SLAM", "DEFAULT_A_SLOT_UNKNOWN_OR_OCCUPIED")


def resolve_parking_plan(*, slam_ready, a_state, b_state,
                         nav2_a_valid=False, nav2_b_valid=False):
    """Resolve SLAM→Nav2→matching CSV with one permitted free-slot retry."""
    selected = select_slam_slot(a_state, b_state) if slam_ready else \
        ParkingSelection("A", "SLAM", "SLAM_FAILURE_DEFAULT_A")
    attempted = [selected.slot]
    selected_valid = nav2_a_valid if selected.slot == "A" else nav2_b_valid
    if selected_valid:
        return ParkingPlanResolution(
            selected.slot, "NAV2", "SELECTED_NAV2_VALID", tuple(attempted))
    both_free = (str(a_state).upper() == "FREE" and
                 str(b_state).upper() == "FREE")
    if selected.slot == "A" and both_free:
        attempted.append("B")
        if nav2_b_valid:
            return ParkingPlanResolution(
                "B", "NAV2", "A_NAV2_FAILED_B_RETRY_VALID", tuple(attempted))
    return ParkingPlanResolution(
        selected.slot, "CSV_FALLBACK", "NAV2_FAILED_MATCHING_CSV",
        tuple(attempted))


def parking_tf_owner(enable_parking_slam, mode, slam_active):
    """Exactly one map->odom owner is allowed at a time."""
    parking = bool(enable_parking_slam) and int(mode) in (7, 10)
    if parking and bool(slam_active):
        return "SLAM_TOOLBOX"
    return "DEPTH_ODOM"


def path_direction_profile(points, tolerance_m=1.0e-4):
    """Return signed Ackermann travel phases relative to body heading."""
    phases = []
    values = tuple(points or ())
    for first, second in zip(values, values[1:]):
        x0, y0, yaw = (float(value) for value in first[:3])
        dx, dy = float(second[0])-x0, float(second[1])-y0
        projection = math.cos(yaw)*dx+math.sin(yaw)*dy
        if abs(projection) <= float(tolerance_m):
            continue
        direction = 1 if projection > 0.0 else -1
        if not phases or phases[-1] != direction:
            phases.append(direction)
    return tuple(phases)


def validate_nav2_parking_path(
        points, *, planning_success, audit_feasible,
        max_required_steering_deg, footprint, selected_slot, target_slot,
        steering_limit_deg=20.0, minimum_points=3,
        maximum_connection_distance_m=0.75,
        maximum_connection_heading_deg=45.0):
    """Apply the complete gate before a Nav2 path may request ownership."""
    poses = tuple(points or ())
    profile = path_direction_profile(poses)
    if not bool(planning_success):
        reason = "PLANNING_NOT_SUCCESS"
    elif len(poses) < int(minimum_points):
        reason = "PATH_TOO_SHORT"
    elif str(selected_slot).upper() not in ("A", "B") or \
            str(selected_slot).upper() != str(target_slot).upper():
        reason = "TARGET_SLOT_INVALID"
    elif not bool(audit_feasible):
        reason = "ACKERMANN_PATH_INFEASIBLE"
    elif (not math.isfinite(float(max_required_steering_deg)) or
          abs(float(max_required_steering_deg)) >
          float(steering_limit_deg)+1.0e-6):
        reason = "STEERING_LIMIT"
    elif footprint is None or not bool(footprint.safe):
        reason = "FOOTPRINT_COLLISION"
    elif math.hypot(float(poses[0][0]), float(poses[0][1])) > \
            float(maximum_connection_distance_m):
        reason = "CURRENT_POSE_DISCONNECTED"
    elif abs(math.degrees(math.atan2(
            math.sin(float(poses[0][2])),
            math.cos(float(poses[0][2]))))) > \
            float(maximum_connection_heading_deg):
        reason = "CURRENT_HEADING_DISCONNECTED"
    elif profile != (-1,):
        # Current tracker executes one immutable reverse phase. Reject mixed
        # or forward-only results instead of silently applying the wrong sign.
        reason = "DIRECTION_SEQUENCE_INVALID"
    else:
        reason = "READY"
    return ParkingNav2Validation(
        reason == "READY", reason, profile, len(poses))


def parking_csv_fallback_segment(mode, slot):
    prefix = "T" if int(mode) == 7 else "V" if int(mode) == 10 else ""
    branch = str(slot).strip().upper()
    if not prefix or branch not in ("A", "B"):
        raise ValueError("parking fallback requires Mode 7/10 and slot A/B")
    return prefix+"_"+branch


def map_to_odom_from_base_poses(map_base, odom_base):
    """Solve T_map_odom from synchronized map/base and odom/base poses."""
    mx, my, myaw = (float(value) for value in map_base)
    ox, oy, oyaw = (float(value) for value in odom_base)
    yaw = math.atan2(math.sin(myaw-oyaw), math.cos(myaw-oyaw))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (mx-cosine*ox+sine*oy,
            my-sine*ox-cosine*oy, yaw)
