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
    route_index: int = -1

    @property
    def path_point_count(self):
        return len(self.points)


def steering_geometry(wheelbase_m=0.73, planner_max_steering_deg=20.0):
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
          initial_steering_deg=0.0, route_index=-1):
    return LocalPlan(
        bool(valid), str(state), tuple(points),
        float(audit.max_required_steering_deg), float(audit.max_curvature),
        float(audit.min_turn_radius_m), int(replans), str(replan_reason),
        float(initial_steering_deg), int(route_index))


def _local_reference_samples(current_pose, route, current_index,
                             maximum_ahead_m):
    """Express the forward, same-segment CSV centreline in vehicle axes."""
    px, py, pyaw = (float(value) for value in current_pose)
    cosine, sine = math.cos(pyaw), math.sin(pyaw)
    start = int(current_index)
    current = route[start]
    samples = [(0.0, 0.0, 0.0, 0.0, start)]
    previous_x, previous_y = px, py
    distance = 0.0
    for index in range(start+1, len(route)):
        point = route[index]
        if (point.segment_id != current.segment_id or
                int(point.direction) != int(current.direction)):
            break
        step = math.hypot(
            float(point.x)-previous_x, float(point.y)-previous_y)
        previous_x, previous_y = float(point.x), float(point.y)
        if step <= 1.0e-6:
            continue
        distance += step
        if distance > float(maximum_ahead_m)+step:
            break
        dx, dy = float(point.x)-px, float(point.y)-py
        local_x = cosine*dx+sine*dy
        local_y = -sine*dx+cosine*dy
        local_yaw = math.atan2(
            math.sin(float(point.yaw)-pyaw),
            math.cos(float(point.yaw)-pyaw))
        samples.append((distance, local_x, local_y, local_yaw, index))
        if distance >= float(maximum_ahead_m):
            break
    # Recorded yaw contains normal steering/measurement jitter. Using every
    # raw yaw as a Hermite tangent can fold a 20 cm segment back on itself.
    # Derive continuous interior tangents from neighbouring CSV positions;
    # the exact recorded yaw is restored only at the selected rejoin pose.
    smooth = []
    for index, sample in enumerate(samples):
        before = samples[max(0, index-1)]
        after = samples[min(len(samples)-1, index+1)]
        yaw = (0.0 if index == 0 else math.atan2(
            after[2]-before[2], after[1]-before[1]))
        smooth.append((sample[0], sample[1], sample[2], yaw, sample[4]))
    return tuple(smooth)


def _interpolate_reference(samples, distance):
    value = max(0.0, min(float(distance), samples[-1][0]))
    for first, second in zip(samples, samples[1:]):
        if value > second[0]:
            continue
        span = max(1.0e-9, second[0]-first[0])
        ratio = (value-first[0])/span
        t2, t3 = ratio*ratio, ratio*ratio*ratio
        h00, h10 = 2.0*t3-3.0*t2+1.0, t3-2.0*t2+ratio
        h01, h11 = -2.0*t3+3.0*t2, t3-t2
        m0x, m0y = span*math.cos(first[3]), span*math.sin(first[3])
        m1x, m1y = span*math.cos(second[3]), span*math.sin(second[3])
        x = h00*first[1]+h10*m0x+h01*second[1]+h11*m1x
        y = h00*first[2]+h10*m0y+h01*second[2]+h11*m1y
        dh00, dh10 = 6.0*t2-6.0*ratio, 3.0*t2-4.0*ratio+1.0
        dh01, dh11 = -dh00, 3.0*t2-2.0*ratio
        dx = (dh00*first[1]+dh10*m0x+dh01*second[1]+dh11*m1x)
        dy = (dh00*first[2]+dh10*m0y+dh01*second[2]+dh11*m1y)
        return x, y, math.atan2(dy, dx)
    return samples[-1][1], samples[-1][2], samples[-1][3]


def _endpoint_reference(rejoin, spacing_m):
    """Smooth local CSV approximation with exact position/yaw endpoints."""
    length, goal_x, goal_y, goal_yaw, route_index = rejoin
    count = max(2, int(math.ceil(length/float(spacing_m)))+1)
    tangent_scale = max(math.hypot(goal_x, goal_y), 0.5)
    m0x, m0y = tangent_scale, 0.0
    m1x = tangent_scale*math.cos(goal_yaw)
    m1y = tangent_scale*math.sin(goal_yaw)
    output = []
    for index in range(count):
        ratio = index/(count-1)
        t2, t3 = ratio*ratio, ratio*ratio*ratio
        h10 = t3-2.0*t2+ratio
        h01 = -2.0*t3+3.0*t2
        h11 = t3-t2
        x = h10*m0x+h01*goal_x+h11*m1x
        y = h10*m0y+h01*goal_y+h11*m1y
        dh10 = 3.0*t2-4.0*ratio+1.0
        dh01 = -6.0*t2+6.0*ratio
        dh11 = 3.0*t2-2.0*ratio
        dx = dh10*m0x+dh01*goal_x+dh11*m1x
        dy = dh10*m0y+dh01*goal_y+dh11*m1y
        output.append((ratio*length, x, y, math.atan2(dy, dx),
                       route_index))
    return tuple(output)


def _project_local_point(samples, point):
    best = (math.inf, 0.0, 0.0)
    px, py = float(point[0]), float(point[1])
    for first, second in zip(samples, samples[1:]):
        vx, vy = second[1]-first[1], second[2]-first[2]
        denominator = vx*vx+vy*vy
        ratio = (0.0 if denominator <= 1.0e-12 else max(
            0.0, min(1.0, ((px-first[1])*vx+(py-first[2])*vy) /
                         denominator)))
        x, y = first[1]+ratio*vx, first[2]+ratio*vy
        separation = math.hypot(px-x, py-y)
        if separation < best[0]:
            tangent = math.atan2(vy, vx)
            lateral = -(px-x)*math.sin(tangent)+(py-y)*math.cos(tangent)
            along = first[0]+ratio*(second[0]-first[0])
            best = separation, along, lateral
    return best[1], best[2]


def _offset_path(samples, outbound_end, hold_end, transition_end, target_d,
                 spacing_m, final_yaw=None, path_end=None):
    path_end = float(transition_end if path_end is None else path_end)
    if not 0.5 <= outbound_end < hold_end < transition_end <= path_end:
        return ()
    count = max(1, int(math.ceil(path_end/float(spacing_m))))
    distances = [path_end*index/count for index in range(count+1)]
    raw = []
    for distance in distances:
        if distance <= outbound_end:
            offset = target_d*_blend(distance/outbound_end)
        elif distance <= hold_end:
            offset = target_d
        elif distance <= transition_end:
            ratio = (distance-hold_end)/(transition_end-hold_end)
            offset = target_d*(1.0-_blend(ratio))
        else:
            offset = 0.0
        x, y, yaw = _interpolate_reference(samples, distance)
        raw.append([x-offset*math.sin(yaw),
                    y+offset*math.cos(yaw), yaw])
    for index, point in enumerate(raw):
        before = raw[max(0, index-1)]
        after = raw[min(len(raw)-1, index+1)]
        point[2] = math.atan2(after[1]-before[1], after[0]-before[0])
    raw[0] = [0.0, 0.0, 0.0]
    end_x, end_y, end_yaw = _interpolate_reference(samples, path_end)
    raw[-1] = [end_x, end_y,
               end_yaw if final_yaw is None else float(final_yaw)]
    return tuple(tuple(point) for point in raw)


def plan_route_detour(current_pose, route, current_index, obstacle_y,
                      minimum_ahead_m=1.5, maximum_ahead_m=10.0,
                      lateral_m=0.65, spacing_m=0.10,
                      wheelbase_m=0.73, planner_max_steering_deg=20.0,
                      obstacles=(), vehicle_width_m=0.80,
                      vehicle_length_m=1.30, obstacle_margin_m=0.15,
                      maximum_replans=48,
                      left_boundary_m=None, right_boundary_m=None):
    """Generate the shortest feasible CSV-relative lateral-offset detour."""
    empty = PathAudit(0, math.inf, math.inf, 0.0, False)
    try:
        planner_curvature, _ = steering_geometry(
            wheelbase_m, planner_max_steering_deg)
        start = int(current_index)
        route = tuple(route)
        current = route[start]
    except (ValueError, TypeError, IndexError):
        return _plan(False, "ROUTE_CONTEXT_INVALID", (), empty)
    if (minimum_ahead_m <= 0.5 or maximum_ahead_m < minimum_ahead_m or
            lateral_m <= 0.0 or spacing_m <= 0.0 or
            vehicle_width_m <= 0.0 or vehicle_length_m <= 0.0 or
            int(current.direction) <= 0):
        return _plan(False, "ROUTE_CONTEXT_INVALID", (), empty)
    samples = _local_reference_samples(
        current_pose, route, start, maximum_ahead_m)
    if len(samples) < 3:
        return _plan(False, "NO_FORWARD_CSV_REJOIN", (), empty)

    obstacle_points = tuple(
        (float(point[0]), float(point[1])) for point in obstacles)
    if not obstacle_points:
        obstacle_points = ((1.5, float(obstacle_y)),)
    if max(point[0] for point in obstacle_points) <= 0.0:
        return _plan(False, "OBSTACLE_NOT_FORWARD", (), empty)

    clearance = float(vehicle_width_m)/2.0+float(obstacle_margin_m)
    rejoin_samples = tuple(
        sample for sample in samples[1:]
        if sample[0] >= max(point[0] for point in obstacle_points)+
        float(vehicle_length_m)/2.0+float(obstacle_margin_m)+
        float(minimum_ahead_m)-1.0e-9)
    if not rejoin_samples:
        return _plan(False, "NO_FORWARD_CSV_REJOIN", (), empty)

    boundary_margin = float(vehicle_width_m)/2.0+0.05
    left_center = (None if left_boundary_m is None else
                   float(left_boundary_m)-boundary_margin)
    right_center = (None if right_boundary_m is None else
                    float(right_boundary_m)+boundary_margin)
    attempts = 0
    last_points, last_audit = (), empty
    reasons = []
    initial_steering = 0.0
    for rejoin in rejoin_samples:
        goal_yaw = math.atan2(
            math.sin(float(route[rejoin[4]].yaw)-float(current_pose[2])),
            math.cos(float(route[rejoin[4]].yaw)-float(current_pose[2])))
        reference = _endpoint_reference(
            (rejoin[0], rejoin[1], rejoin[2], goal_yaw, rejoin[4]),
            spacing_m)
        projections = tuple(
            _project_local_point(reference, point)
            for point in obstacle_points)
        obstacle_s_min = min(value[0] for value in projections)
        obstacle_s_max = max(value[0] for value in projections)
        obstacle_d_min = min(value[1] for value in projections)
        obstacle_d_max = max(value[1] for value in projections)
        obstacle_d = sum(value[1] for value in projections)/len(projections)
        # Choose the opposite side in the CSV Frenet frame, not raw sensor y.
        # This remains correct on a curve where the same world point can be
        # left of the route even when it is right of the vehicle centreline.
        side = -1.0 if obstacle_d >= 0.0 else 1.0
        required = (obstacle_d_min-clearance if side < 0.0 else
                    obstacle_d_max+clearance)
        # Shift only as far as the observed obstacle envelope requires. The
        # old fixed 0.65 m minimum created unnecessarily wide detours.
        magnitude = max(0.20, abs(required)+0.02)
        offsets = tuple(side*(magnitude+0.05*step) for step in range(5))
        # Complete the lateral move as the rear footprint clears the object,
        # hold it only for the observed object length, then return immediately.
        outbound_end = (obstacle_s_max+float(vehicle_length_m)/2.0+
                        float(obstacle_margin_m))
        hold_end = outbound_end+max(
            0.20, obstacle_s_max-obstacle_s_min)
        if rejoin[0] < hold_end+float(minimum_ahead_m)-1.0e-9:
            continue
        for offset in offsets:
            if attempts >= int(maximum_replans):
                break
            points = _offset_path(
                reference, outbound_end, hold_end,
                rejoin[0]-min(0.50, 0.25*(rejoin[0]-hold_end)), offset,
                spacing_m, goal_yaw, rejoin[0])
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
            collision_free = path_collision_free(points, obstacles, clearance)
            boundary_clear = path_within_boundaries(
                points, left_center, right_center)
            if steering_ok and collision_free and boundary_clear:
                return _plan(
                    True, "ROUTE_REJOIN_READY", points, audit,
                    attempts-1, "FRENET_OFFSET_SHORTEST",
                    initial_steering, rejoin[4])
            if not steering_ok:
                reasons.append("STEERING_LIMIT")
            if not collision_free:
                reasons.append("COLLISION")
            if not boundary_clear:
                reasons.append("CURB_BOUNDARY")
        if attempts >= int(maximum_replans):
            break
    reason = "+".join(sorted(set(reasons))) or "NO_FEASIBLE_ROUTE_DETOUR"
    return _plan(False, "NO_FEASIBLE_ROUTE_DETOUR", last_points,
                 last_audit, max(0, attempts-1), reason)


def plan_detour(obstacle_y, length_m=5.5, lateral_m=0.65,
                wheelbase_m=0.73, planner_max_steering_deg=20.0,
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
                 planner_max_steering_deg=20.0, step_m=0.08,
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
