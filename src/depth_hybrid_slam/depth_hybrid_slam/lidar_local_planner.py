"""Steering-constrained temporary paths; immutable CSV stays the reference."""

from dataclasses import dataclass
import math


HARD_STEERING_LIMIT_DEG = 22.0


@dataclass(frozen=True)
class PathAudit:
    path_point_count: int
    max_required_steering_deg: float
    max_curvature: float
    min_turn_radius_m: float
    feasible: bool


@dataclass(frozen=True)
class LocalPlan:
    valid: bool
    state: str
    points: tuple
    max_steering_deg: float
    max_curvature: float = math.inf
    min_turn_radius_m: float = 0.0
    replans: int = 0
    replan_reason: str = ""
    initial_steering_deg: float = 0.0

    @property
    def path_point_count(self):
        return len(self.points)


def steering_geometry(wheelbase_m=0.73, planner_max_steering_deg=21.0):
    """Return derived curvature/radius from one canonical steering setting."""
    wheelbase = float(wheelbase_m)
    steering = float(planner_max_steering_deg)
    if (abs(wheelbase-0.73) > 1.0e-9 or not math.isfinite(steering) or
            not 0.0 < steering < HARD_STEERING_LIMIT_DEG):
        raise ValueError("invalid commissioned LiDAR planner geometry")
    curvature = math.tan(math.radians(steering))/wheelbase
    return curvature, 1.0/curvature


def audit_path(points, wheelbase_m=0.73,
               hard_steering_limit_deg=HARD_STEERING_LIMIT_DEG):
    """Independently validate every generated segment without a clamp."""
    values = tuple(points)
    maximum_curvature = 0.0
    valid = len(values) >= 2
    for first, second in zip(values, values[1:]):
        try:
            distance = math.hypot(
                float(second[0])-float(first[0]),
                float(second[1])-float(first[1]))
            delta = math.atan2(
                math.sin(float(second[2])-float(first[2])),
                math.cos(float(second[2])-float(first[2])))
        except (TypeError, ValueError, IndexError):
            valid = False
            continue
        if not math.isfinite(distance) or distance <= 1.0e-7:
            valid = False
            continue
        curvature = abs(delta)/distance
        if not math.isfinite(curvature):
            valid = False
            continue
        maximum_curvature = max(maximum_curvature, curvature)
    required = math.degrees(math.atan(float(wheelbase_m)*maximum_curvature))
    radius = (math.inf if maximum_curvature <= 1.0e-12 else
              1.0/maximum_curvature)
    valid = valid and math.isfinite(required) and \
        required < float(hard_steering_limit_deg)
    return PathAudit(len(values), required, maximum_curvature, radius, valid)


def path_collision_free(points, obstacles, clearance_m=0.35):
    """Swept centerline clearance for the vehicle-width envelope."""
    clearance = float(clearance_m)
    if clearance <= 0.0:
        return False
    centers = tuple((float(point[0]), float(point[1])) for point in points)
    for ox, oy in (obstacles or ()):
        obstacle = (float(ox), float(oy))
        for first, second in zip(centers, centers[1:]):
            dx, dy = second[0]-first[0], second[1]-first[1]
            length_sq = dx*dx+dy*dy
            projection = 0.0 if length_sq <= 1.0e-12 else max(
                0.0, min(1.0, ((obstacle[0]-first[0])*dx+
                               (obstacle[1]-first[1])*dy)/length_sq))
            closest = (first[0]+projection*dx, first[1]+projection*dy)
            if math.hypot(
                    closest[0]-obstacle[0], closest[1]-obstacle[1]) < clearance:
                return False
        if len(centers) == 1 and math.hypot(
                centers[0][0]-obstacle[0],
                centers[0][1]-obstacle[1]) < clearance:
            return False
    return True


def path_within_boundaries(points, left_boundary_m=None,
                           right_boundary_m=None):
    """Treat detected curbs as planning bounds, never collision triggers."""
    for point in points:
        lateral = float(point[1])
        if left_boundary_m is not None and lateral >= float(left_boundary_m):
            return False
        if right_boundary_m is not None and lateral <= float(right_boundary_m):
            return False
    return True


def _blend(value):
    return 10.0*value**3-15.0*value**4+6.0*value**5


def _blend_derivative(value):
    return 30.0*value**2-60.0*value**3+30.0*value**4


def _detour_candidate(length_m, target_y, spacing_m):
    count = max(12, int(math.ceil(length_m/spacing_m))+1)
    points = []
    for index in range(count):
        x = length_m*index/(count-1)
        phase = x/length_m
        blend_phase = phase*2.0 if phase <= 0.5 else (phase-0.5)*2.0
        if phase <= 0.5:
            y = target_y*_blend(blend_phase)
            slope = target_y*_blend_derivative(blend_phase)*2.0/length_m
        else:
            y = target_y*(1.0-_blend(blend_phase))
            slope = -target_y*_blend_derivative(blend_phase)*2.0/length_m
        # The analytic tangent gives exact start/rejoin heading continuity;
        # finite-difference headings leave a small artificial endpoint kink.
        points.append((x, y, math.atan2(slope, 1.0)))
    return tuple(points)


def _plan(valid, state, points, audit, replans=0, replan_reason="",
          initial_steering_deg=0.0):
    return LocalPlan(
        bool(valid), str(state), tuple(points),
        float(audit.max_required_steering_deg), float(audit.max_curvature),
        float(audit.min_turn_radius_m), int(replans), str(replan_reason),
        float(initial_steering_deg))


def plan_detour(obstacle_y, length_m=5.5, lateral_m=0.65,
                wheelbase_m=0.73, planner_max_steering_deg=21.0,
                spacing_m=0.10, obstacles=(), clearance_m=None,
                vehicle_width_m=0.80, obstacle_margin_m=0.15,
                maximum_replans=24, left_boundary_m=None,
                right_boundary_m=None):
    """Search wider/longer quintic detours under the Ackermann limit."""
    try:
        planner_curvature, _ = steering_geometry(
            wheelbase_m, planner_max_steering_deg)
    except ValueError:
        empty = PathAudit(0, math.inf, math.inf, 0.0, False)
        return _plan(False, "INVALID_GEOMETRY", (), empty)
    if (length_m <= 1.0 or lateral_m <= 0.0 or spacing_m <= 0.0 or
            vehicle_width_m <= 0.0 or obstacle_margin_m < 0.0 or
            int(maximum_replans) < 0):
        empty = PathAudit(0, math.inf, math.inf, 0.0, False)
        return _plan(False, "INVALID_GEOMETRY", (), empty)

    clearance = (float(clearance_m) if clearance_m is not None else
                 float(vehicle_width_m)/2.0+float(obstacle_margin_m))
    side = -1.0 if float(obstacle_y) >= 0.0 else 1.0
    offsets = [float(lateral_m)+0.15*index for index in range(7)]
    lengths = [float(length_m)*(1.0+0.35*index) for index in range(9)]
    attempts = 0
    last_points = ()
    last_audit = PathAudit(0, math.inf, math.inf, 0.0, False)
    reasons = []
    initial_steering = 0.0
    for length in lengths:
        for offset in offsets:
            if attempts > int(maximum_replans):
                break
            points = _detour_candidate(length, side*offset, spacing_m)
            audit = audit_path(points, wheelbase_m)
            attempts += 1
            if attempts == 1:
                initial_steering = audit.max_required_steering_deg
            last_points, last_audit = points, audit
            steering_ok = (
                audit.feasible and
                audit.max_curvature <= planner_curvature+1.0e-9 and
                audit.max_required_steering_deg <=
                float(planner_max_steering_deg)+1.0e-7)
            collision_free = path_collision_free(
                points, obstacles, clearance)
            boundary_clear = path_within_boundaries(
                points, left_boundary_m, right_boundary_m)
            if steering_ok and collision_free and boundary_clear:
                replans = attempts-1
                state = "READY_REPLANNED" if replans else "READY"
                reason = "+".join(sorted(set(reasons))) if replans else ""
                return _plan(True, state, points, audit, replans, reason,
                             initial_steering)
            if not steering_ok:
                reasons.append("STEERING_LIMIT")
                # At a fixed transition length, a larger lateral offset only
                # increases curvature. Move the rejoin target forward first.
                break
            if not collision_free:
                reasons.append("COLLISION")
            if not boundary_clear:
                reasons.append("CURB_BOUNDARY")
        if attempts > int(maximum_replans):
            break
    reason = ("STEERING_LIMIT" if "STEERING_LIMIT" in reasons else
              "CURB_BOUNDARY" if "CURB_BOUNDARY" in reasons else "COLLISION")
    return _plan(False, "NO_FEASIBLE_DETOUR", last_points, last_audit,
                 max(0, attempts-1), reason, initial_steering)


def _parking_schedule(mode, branch, requested_steering_deg=None):
    schedules = {
        (7, "A"): ((1.55, 0.0), (0.65, 18.0), (1.20, -18.0)),
        (7, "B"): ((1.00, 0.0), (1.10, -18.0), (1.40, 18.0)),
        (10, "A"): ((2.40, -22.0), (1.00, 22.0), (0.15, 0.0)),
        (10, "B"): ((1.00, 0.0), (1.15, -22.0), (0.70, 22.0)),
    }
    output = schedules[(int(mode), str(branch))]
    if requested_steering_deg is None:
        return output
    requested = abs(float(requested_steering_deg))
    return tuple((distance, math.copysign(requested, steering))
                 if steering else (distance, 0.0)
                 for distance, steering in output)


def _regenerate_schedule(schedule, planner_max_steering_deg):
    """Increase arc length while reducing curvature; never clamp a path."""
    regenerated = []
    changed = False
    for distance, steering in schedule:
        if abs(steering) <= planner_max_steering_deg:
            regenerated.append((distance, steering))
            continue
        changed = True
        target = math.copysign(planner_max_steering_deg, steering)
        scale = abs(math.tan(math.radians(steering)) /
                    math.tan(math.radians(target)))
        regenerated.append((distance*scale, target))
    return tuple(regenerated), changed


def plan_parking(mode, branch, wheelbase_m=0.73,
                 planner_max_steering_deg=21.0, step_m=0.08,
                 requested_steering_deg=None):
    """Generate T/parallel paths with radius expansion before integration."""
    if int(mode) not in (7, 10) or str(branch) not in ("A", "B"):
        empty = PathAudit(0, math.inf, math.inf, 0.0, False)
        return _plan(False, "INVALID_PARKING_REQUEST", (), empty)
    try:
        planner_curvature, _ = steering_geometry(
            wheelbase_m, planner_max_steering_deg)
    except ValueError:
        empty = PathAudit(0, math.inf, math.inf, 0.0, False)
        return _plan(False, "INVALID_GEOMETRY", (), empty)
    if step_m <= 0.0:
        empty = PathAudit(0, math.inf, math.inf, 0.0, False)
        return _plan(False, "INVALID_GEOMETRY", (), empty)

    requested = _parking_schedule(mode, branch, requested_steering_deg)
    initial_steering = max(abs(steering) for _, steering in requested)
    schedule, regenerated = _regenerate_schedule(
        requested, float(planner_max_steering_deg))
    x = y = yaw = 0.0
    points = [(x, y, yaw)]
    for distance, steering in schedule:
        steps = max(1, int(math.ceil(distance/step_m)))
        delta = -distance/steps
        curvature = math.tan(math.radians(steering))/float(wheelbase_m)
        for _ in range(steps):
            next_yaw = yaw+delta*curvature
            midpoint_yaw = 0.5*(yaw+next_yaw)
            x += delta*math.cos(midpoint_yaw)
            y += delta*math.sin(midpoint_yaw)
            yaw = next_yaw
            points.append((x, y, yaw))
    audit = audit_path(points, wheelbase_m)
    feasible = (
        audit.feasible and
        audit.max_curvature <= planner_curvature+1.0e-9 and
        audit.max_required_steering_deg <=
        float(planner_max_steering_deg)+1.0e-7)
    if not feasible:
        return _plan(False, "PLANNER_GIVE_UP", points, audit,
                     int(regenerated), "STEERING_LIMIT", initial_steering)
    return _plan(True, "READY_REPLANNED" if regenerated else "READY",
                 points, audit, int(regenerated),
                 "STEERING_LIMIT" if regenerated else "",
                 initial_steering)
