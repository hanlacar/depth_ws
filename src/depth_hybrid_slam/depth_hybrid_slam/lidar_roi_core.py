"""Steering-aware LiDAR ROI, temporal tracking, and CSV sweep geometry."""

from dataclasses import dataclass, replace
import math


STATIC = "STATIC"
DYNAMIC = "DYNAMIC"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Cluster:
    points: tuple
    centroid: tuple
    track_id: int = -1
    motion: str = UNKNOWN
    speed_mps: float = 0.0


@dataclass(frozen=True)
class RoiAssessment:
    centerline: tuple
    curvature: float
    radius_m: float
    zone1_count: int
    zone2_count: int
    zone3_dynamic_count: int
    nearest_m: float
    hard_stop: bool
    slowdown: bool


@dataclass(frozen=True)
class BroadAssessment:
    obstacles: tuple
    curbs: tuple
    collision_obstacles: tuple
    path_blocked: bool


def ackermann_centerline(steering_deg, wheelbase_m=0.73,
                         length_m=1.5, spacing_m=0.05):
    """Return (arc-length, x, y) samples; positive steer bends left."""
    steering = max(-22.0, min(22.0, float(steering_deg)))
    curvature = math.tan(math.radians(steering))/float(wheelbase_m)
    count = max(2, int(math.ceil(float(length_m)/float(spacing_m)))+1)
    output = []
    for index in range(count):
        distance = float(length_m)*index/(count-1)
        if abs(curvature) <= 1.0e-9:
            x, y = distance, 0.0
        else:
            x = math.sin(curvature*distance)/curvature
            y = (1.0-math.cos(curvature*distance))/curvature
        output.append((distance, x, y))
    radius = math.inf if abs(curvature) <= 1.0e-9 else 1.0/abs(curvature)
    return tuple(output), curvature, radius


def cluster_points(points, gap_m=0.16, minimum_points=2):
    """Cluster angularly ordered LaserScan points without inventing returns."""
    valid = tuple((float(x), float(y)) for x, y in (points or ())
                  if math.isfinite(float(x)) and math.isfinite(float(y)))
    groups = []
    current = []
    for point in valid:
        if current and math.hypot(point[0]-current[-1][0],
                                  point[1]-current[-1][1]) > float(gap_m):
            if len(current) >= int(minimum_points):
                groups.append(tuple(current))
            current = []
        current.append(point)
    if len(current) >= int(minimum_points):
        groups.append(tuple(current))
    return tuple(Cluster(
        group,
        (sum(point[0] for point in group)/len(group),
         sum(point[1] for point in group)/len(group))) for group in groups)


def _projection(point, centerline):
    best = (math.inf, math.inf)
    px, py = float(point[0]), float(point[1])
    last_index = len(centerline)-2
    for index, (first, second) in enumerate(zip(centerline, centerline[1:])):
        s0, x0, y0 = first
        s1, x1, y1 = second
        dx, dy = x1-x0, y1-y0
        length2 = dx*dx+dy*dy
        raw_along = (0.0 if length2 <= 1.0e-12 else
                     ((px-x0)*dx+(py-y0)*dy)/length2)
        along = max(0.0, min(1.0, raw_along))
        # Preserve signed longitudinal distance outside the two end caps so a
        # return just beyond 1.5 m cannot be rounded onto the 1.5 m boundary.
        if (index == 0 and raw_along < 0.0) or (
                index == last_index and raw_along > 1.0):
            along = raw_along
        x, y = x0+along*dx, y0+along*dy
        distance = s0+along*(s1-s0)
        lateral = math.hypot(px-x, py-y)
        if lateral < best[1]:
            best = (distance, lateral)
    return best


def clusters_in_centerline_corridor(clusters, centerline, *,
                                    half_width_m=0.30, length_m=1.5):
    """Return clusters intersecting the displayed three-zone corridor."""
    selected = []
    for cluster in clusters:
        if any(0.0 < distance <= float(length_m) and
               lateral <= float(half_width_m)
               for distance, lateral in (
                   _projection(point, centerline) for point in cluster.points)):
            selected.append(cluster)
    return tuple(selected)


def assess_curved_roi(clusters, steering_deg, *, front_active=True,
                      fresh=True, half_width_m=0.30, wheelbase_m=0.73,
                      length_m=1.5):
    """Apply 0.5/1.0/1.5 m rules along an Ackermann arc corridor."""
    centerline, curvature, radius = ackermann_centerline(
        steering_deg, wheelbase_m, length_m=length_m)
    if not front_active:
        return RoiAssessment(centerline, curvature, radius, 0, 0, 0,
                             math.inf, False, False)
    if not fresh:
        return RoiAssessment(centerline, curvature, radius, 0, 0, 0,
                             math.inf, True, False)
    zones = [0, 0, 0]
    nearest = math.inf
    for cluster in clusters:
        projections = [_projection(point, centerline)
                       for point in cluster.points]
        in_roi = [distance for distance, lateral in projections
                  if lateral <= float(half_width_m) and distance > 0.0]
        if not in_roi:
            continue
        distance = min(in_roi)
        nearest = min(nearest, distance)
        if distance <= 0.5:
            zones[0] += 1
        elif distance <= 1.0:
            zones[1] += 1
        elif distance <= 1.5 and cluster.motion == DYNAMIC:
            zones[2] += 1
    return RoiAssessment(
        centerline, curvature, radius, zones[0], zones[1], zones[2], nearest,
        zones[0] > 0, zones[1] > 0)


def mode_gates(mode):
    value = int(mode)
    # The front scanner owns emergency detection throughout every live mode.
    # A rear scanner, when installed, adds parking evidence in Modes 7/10 but
    # is never a prerequisite for the front emergency path.
    return 1 <= value <= 11, value in (7, 10)


def transform_point(point, pose):
    x, y, yaw = (float(value) for value in pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (x+cosine*point[0]-sine*point[1],
            y+sine*point[0]+cosine*point[1])


class DynamicClusterTracker:
    """Nearest-neighbour association in a fixed frame with N-frame state."""

    def __init__(self, velocity_threshold_mps=0.25, confirmation_count=3,
                 timeout_s=0.75, association_distance_m=0.50):
        self.velocity_threshold_mps = float(velocity_threshold_mps)
        self.confirmation_count = max(2, int(confirmation_count))
        self.timeout_s = float(timeout_s)
        self.association_distance_m = float(association_distance_m)
        self._tracks = {}
        self._next_id = 1

    def update(self, clusters, now, fixed_pose=None):
        now = float(now)
        self._tracks = {key: value for key, value in self._tracks.items()
                        if now-value[1] <= self.timeout_s}
        if fixed_pose is None:
            return tuple(replace(value, motion=UNKNOWN, speed_mps=0.0,
                                 track_id=-1) for value in clusters)
        output, used = [], set()
        for cluster in clusters:
            fixed = transform_point(cluster.centroid, fixed_pose)
            candidate = None
            distance = math.inf
            for track_id, track in self._tracks.items():
                if track_id in used:
                    continue
                separation = math.hypot(fixed[0]-track[0][0],
                                        fixed[1]-track[0][1])
                if separation < distance and separation <= self.association_distance_m:
                    candidate, distance = track_id, separation
            if candidate is None:
                candidate, self._next_id = self._next_id, self._next_id+1
                self._tracks[candidate] = (fixed, now, 0, 0, UNKNOWN)
                motion, speed = UNKNOWN, 0.0
            else:
                previous, previous_at, dynamic_count, static_count, _ = \
                    self._tracks[candidate]
                elapsed = max(1.0e-6, now-previous_at)
                speed = math.hypot(fixed[0]-previous[0],
                                   fixed[1]-previous[1])/elapsed
                moving = speed >= self.velocity_threshold_mps
                dynamic_count = dynamic_count+1 if moving else 0
                static_count = static_count+1 if not moving else 0
                motion = (DYNAMIC if dynamic_count >= self.confirmation_count else
                          STATIC if static_count >= self.confirmation_count else
                          UNKNOWN)
                self._tracks[candidate] = (
                    fixed, now, dynamic_count, static_count, motion)
            used.add(candidate)
            output.append(replace(cluster, track_id=candidate,
                                  motion=motion, speed_mps=speed))
        return tuple(output)


def _curb(cluster):
    xs = [point[0] for point in cluster.points]
    ys = [point[1] for point in cluster.points]
    return (len(cluster.points) >= 4 and abs(cluster.centroid[1]) >= 0.45 and
            max(xs)-min(xs) >= 0.35 and max(ys)-min(ys) <= 0.22)


def path_collision(cluster, route_samples, clearance_m=0.55):
    for point in cluster.points:
        for sample in route_samples:
            if math.hypot(point[0]-sample[0], point[1]-sample[1]) \
                    <= float(clearance_m):
                return True
    return False


def _cluster_in_sensor_frame(cluster, sensor_pose):
    """Express a base_link cluster in the sensor frame used for sensing."""
    px, py, yaw = (float(value) for value in sensor_pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)

    def local(point):
        dx, dy = float(point[0])-px, float(point[1])-py
        return (cosine*dx+sine*dy, -sine*dx+cosine*dy)

    points = tuple(local(point) for point in cluster.points)
    return replace(cluster, points=points, centroid=local(cluster.centroid))


def assess_mode5_broad(clusters, route_samples, *, range_m=2.0,
                       fov_deg=80.0, lateral_m=1.0, clearance_m=0.55,
                       sensor_pose=(0.0, 0.0, 0.0)):
    broad = []
    for cluster in clusters:
        sensor_cluster = _cluster_in_sensor_frame(cluster, sensor_pose)
        x, y = sensor_cluster.centroid
        angle = abs(math.degrees(math.atan2(y, x)))
        if (x > 0.0 and math.hypot(x, y) <= float(range_m) and
                angle <= float(fov_deg) and abs(y) <= float(lateral_m)):
            broad.append((cluster, sensor_cluster))
    # Return original base_link clusters: CSV swept-footprint collision tests
    # intentionally remain in the vehicle/body frame.
    curbs = tuple(base for base, local in broad if _curb(local))
    obstacles = tuple(base for base, local in broad if not _curb(local))
    collisions = tuple(value for value in obstacles
                       if path_collision(value, route_samples, clearance_m))
    return BroadAssessment(obstacles, curbs, collisions, bool(collisions))


def forward_route_samples(path, active_index, pose, window_m=2.0):
    """Transform a monotonic, forward-only Path slice from map to base_link."""
    if pose is None or not path:
        return ()
    px, py, yaw = (float(value) for value in pose)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    output, traveled = [], 0.0
    previous = None
    for point in tuple(path)[max(0, int(active_index)):]:
        mx, my = float(point[0]), float(point[1])
        if previous is not None:
            traveled += math.hypot(mx-previous[0], my-previous[1])
        if traveled > float(window_m):
            break
        dx, dy = mx-px, my-py
        output.append((cosine*dx+sine*dy, -sine*dx+cosine*dy))
        previous = (mx, my)
    return tuple(output)


def speed_bump_suppressed(mode, route_index, zones, *, camera_fresh,
                          drivable_road, object_evidence_valid,
                          object_detected, motion):
    """Fail closed: all six independent pieces of evidence are mandatory."""
    in_zone = any(int(zone[0]) == int(mode) and
                  int(zone[1]) <= int(route_index) <= int(zone[2])
                  for zone in zones)
    return (int(mode) in (8, 9) and in_zone and bool(camera_fresh) and
            bool(drivable_road) and bool(object_evidence_valid) and
            not bool(object_detected) and str(motion) == STATIC)
