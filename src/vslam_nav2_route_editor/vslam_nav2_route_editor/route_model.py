"""Route data model, geometry, policy checks, and lossless CSV persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

CSV_FIELDS = [
    "index", "segment_id", "x", "y", "yaw_deg", "direction", "speed",
    "required_steering_deg", "event",
]
LEGACY_CSV_FIELDS = CSV_FIELDS[:-1]
ROUTE_NAMES = (
    "START_A", "START_B", "COMMON_1", "T_PARK_A", "T_PARK_B",
    "COMMON_2", "PARALLEL_A", "PARALLEL_B", "COMMON_3", "END_A", "END_B",
)
VALID_DIRECTIONS = {"F", "R"}
VALID_EVENTS = {"NONE", "STOP", "ACCEL"}


@dataclass
class VehiclePolicy:
    wheelbase_m: float = 0.73
    steering_left_deg: float = 22.0
    steering_right_deg: float = -22.0
    forward_speed: float = 2.0
    turning_speed: float = 1.0
    reverse_speed: float = -1.0
    turning_threshold_deg: float = 10.0


@dataclass
class RoutePoint:
    index: int
    segment_id: int
    x: float
    y: float
    yaw_deg: float
    direction: str = "F"
    speed: float = 2.0
    required_steering_deg: float = 0.0
    event: str = "NONE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _motion_yaw(points: list[RoutePoint], index: int) -> float:
    if len(points) < 2:
        return points[index].yaw_deg
    if index == len(points) - 1:
        a, b = points[index - 1], points[index]
    else:
        a, b = points[index], points[index + 1]
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def _curvature(a: RoutePoint, b: RoutePoint, c: RoutePoint) -> float:
    ab = math.hypot(b.x - a.x, b.y - a.y)
    bc = math.hypot(c.x - b.x, c.y - b.y)
    ac = math.hypot(c.x - a.x, c.y - a.y)
    if min(ab, bc, ac) < 1e-8:
        return 0.0
    cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    return 2.0 * cross / (ab * bc * ac)


def apply_vehicle_policy(
    points: list[RoutePoint], policy: VehiclePolicy = VehiclePolicy(),
    update_forward_yaw: bool = True,
) -> list[RoutePoint]:
    """Calculate steering and speed without ever flipping reverse vehicle yaw."""
    for i, point in enumerate(points):
        point.index = i
        point.direction = point.direction.upper()
        point.event = point.event.upper()
        if point.direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction at index {i}: {point.direction}")
        if point.event not in VALID_EVENTS:
            raise ValueError(f"invalid event at index {i}: {point.event}")
        if point.event == "STOP":
            point.speed = 0.0
            point.required_steering_deg = 0.0
            continue
        if update_forward_yaw and point.direction == "F":
            point.yaw_deg = normalize_angle_deg(_motion_yaw(points, i))
        # Reverse yaw is intentionally retained as the vehicle heading.
        if len(points) < 3:
            curvature = 0.0
        elif i == 0:
            curvature = _curvature(points[0], points[1], points[2])
        elif i == len(points) - 1:
            curvature = _curvature(points[-3], points[-2], points[-1])
        else:
            curvature = _curvature(points[i - 1], points[i], points[i + 1])
        if point.direction == "R":
            curvature = -curvature
        steering = math.degrees(math.atan(policy.wheelbase_m * curvature))
        point.required_steering_deg = steering
        if point.direction == "R":
            point.speed = policy.reverse_speed
        elif abs(steering) >= policy.turning_threshold_deg:
            point.speed = policy.turning_speed
        else:
            point.speed = policy.forward_speed
    return points


def _checked_index_range(points: list[RoutePoint], start: int, end: int) -> range:
    if start < 0 or end < start or end >= len(points):
        raise ValueError(
            f"index range must satisfy 0 <= start <= end < {len(points)}: "
            f"start={start}, end={end}")
    return range(start, end + 1)


def assign_segment_ids(points: list[RoutePoint]) -> list[RoutePoint]:
    """Assign motion/STOP-event blocks without changing pose or yaw."""
    segment = 1
    previous = None
    for point in points:
        if point.event == "STOP":
            if previous not in (None, "STOP"):
                segment += 1
        elif previous == "STOP":
            segment += 1
        point.segment_id = segment
        previous = point.event
    return points


def set_direction_range(
    points: list[RoutePoint], start: int, end: int, direction: str,
    policy: VehiclePolicy = VehiclePolicy(),
) -> list[RoutePoint]:
    """Return a policy-updated copy with an inclusive F/R direction range."""
    direction = direction.upper()
    if direction not in {"F", "R"}:
        raise ValueError("set-direction accepts only F or R; use set-stop for STOP")
    selected = _checked_index_range(points, start, end)
    updated = [replace(point) for point in points]
    for index in selected:
        updated[index].direction = direction
    # Preserve every stored vehicle heading, especially when switching to R.
    assign_segment_ids(updated)
    apply_vehicle_policy(updated, policy, update_forward_yaw=False)
    missing = validate_route(updated, policy)["direction_changes_without_stop"]
    if missing:
        joined = ", ".join(map(str, missing))
        raise ValueError(
            f"validation failed: F/R direction change requires STOP before index {joined}")
    return updated


def set_stop_point(
    points: list[RoutePoint], index: int,
    policy: VehiclePolicy = VehiclePolicy(),
) -> list[RoutePoint]:
    """Compatibility alias: mark a STOP event without changing direction."""
    _checked_index_range(points, index, index)
    updated = [replace(point) for point in points]
    updated[index].event = "STOP"
    assign_segment_ids(updated)
    return apply_vehicle_policy(updated, policy, update_forward_yaw=False)


def set_event_point(
    points: list[RoutePoint], index: int, event: str,
    policy: VehiclePolicy = VehiclePolicy(),
) -> list[RoutePoint]:
    """Return a copy with NONE/STOP/ACCEL set at one waypoint."""
    _checked_index_range(points, index, index)
    event = event.upper()
    if event not in VALID_EVENTS:
        raise ValueError(f"event must be one of {sorted(VALID_EVENTS)}")
    updated = [replace(point) for point in points]
    updated[index].event = event
    assign_segment_ids(updated)
    return apply_vehicle_policy(updated, policy, update_forward_yaw=False)


def nearest_route_point(
    points: list[RoutePoint], x: float, y: float,
    maximum_distance_m: float = 0.5,
) -> tuple[int, float]:
    """Return nearest index/distance, rejecting clicks too far from the route."""
    if not points:
        raise ValueError("route is empty")
    index, distance = min(
        ((point.index, math.hypot(point.x - x, point.y - y)) for point in points),
        key=lambda item: item[1])
    if distance > maximum_distance_m:
        raise ValueError(
            f"nearest route point is {distance:.3f}m away; "
            f"maximum is {maximum_distance_m:.3f}m")
    return index, distance


def straight_points(
    x0: float, y0: float, x1: float, y1: float, spacing: float = 0.20,
    direction: str = "F", segment_id: int = 1,
    vehicle_yaw_deg: float | None = None,
) -> list[RoutePoint]:
    direction = direction.upper()
    if direction not in VALID_DIRECTIONS:
        raise ValueError("a generated line must be F or R")
    if direction == "R" and vehicle_yaw_deg is None:
        raise ValueError("reverse line requires --vehicle-yaw-deg; yaw is never auto-flipped")
    distance = math.hypot(x1 - x0, y1 - y0)
    if distance <= 0.0 or spacing <= 0.0:
        raise ValueError("line length and spacing must be positive")
    count = max(1, math.ceil(distance / spacing))
    travel_yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
    yaw = travel_yaw if vehicle_yaw_deg is None else float(vehicle_yaw_deg)
    result = []
    for i in range(count + 1):
        ratio = i / count
        result.append(RoutePoint(i, segment_id, x0 + ratio * (x1 - x0),
                                 y0 + ratio * (y1 - y0), yaw, direction))
    return apply_vehicle_policy(result, update_forward_yaw=True)


def insert_required_stops(points: list[RoutePoint]) -> list[RoutePoint]:
    """Insert STOP between adjacent F/R blocks and assign monotonic segments."""
    if not points:
        return points
    output: list[RoutePoint] = []
    previous_motion = None
    segment = 1
    for point in points:
        direction = point.direction.upper()
        if direction in {"F", "R"} and previous_motion and direction != previous_motion:
            anchor = output[-1]
            output.append(RoutePoint(0, segment, anchor.x, anchor.y,
                                     anchor.yaw_deg, previous_motion,
                                     0.0, 0.0, "STOP"))
            segment += 1
        point.segment_id = segment
        output.append(point)
        if point.event == "STOP":
            segment += 1
            previous_motion = None
        else:
            previous_motion = direction
    assign_segment_ids(output)
    return apply_vehicle_policy(output, update_forward_yaw=False)


def smooth_route(points: list[RoutePoint], iterations: int = 2) -> list[RoutePoint]:
    """Chaikin smoothing per uninterrupted motion block; STOP anchors are fixed."""
    if iterations < 1:
        return apply_vehicle_policy(points, update_forward_yaw=True)
    result: list[RoutePoint] = []
    block: list[RoutePoint] = []

    def flush() -> None:
        nonlocal block
        if not block:
            return
        current = block
        for _ in range(iterations):
            if len(current) < 3:
                break
            nxt = [current[0]]
            for a, b in zip(current[:-1], current[1:]):
                for ratio in (0.25, 0.75):
                    yaw = normalize_angle_deg(a.yaw_deg + ratio * normalize_angle_deg(b.yaw_deg - a.yaw_deg))
                    nxt.append(RoutePoint(0, a.segment_id,
                                          a.x + ratio * (b.x - a.x),
                                          a.y + ratio * (b.y - a.y), yaw,
                                          a.direction))
            nxt.append(current[-1])
            current = nxt
        result.extend(current)
        block = []

    for point in points:
        if point.event == "STOP" or (block and point.direction != block[-1].direction):
            flush()
            if point.event == "STOP":
                result.append(point)
            else:
                block.append(point)
        else:
            block.append(point)
    flush()
    return apply_vehicle_policy(result, update_forward_yaw=True)


def load_route(path: str | Path) -> list[RoutePoint]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (LEGACY_CSV_FIELDS, CSV_FIELDS):
            raise ValueError(
                f"CSV fields must be legacy {LEGACY_CSV_FIELDS} or {CSV_FIELDS}")
        rows = list(reader)
    points = []
    for i, row in enumerate(rows):
        direction = row["direction"].upper()
        event = row.get("event", "NONE").upper() or "NONE"
        # Legacy direction=STOP is migrated in memory to an event while using
        # the nearest real motion direction as the vehicle heading contract.
        if direction == "STOP":
            event = "STOP"
            prior = next((p.direction for p in reversed(points)
                          if p.direction in VALID_DIRECTIONS), None)
            following = next((candidate["direction"].upper()
                              for candidate in rows[i + 1:]
                              if candidate["direction"].upper() in VALID_DIRECTIONS), None)
            direction = prior or following or "F"
        point = RoutePoint(
            int(row["index"]), int(row["segment_id"]), float(row["x"]),
            float(row["y"]), float(row["yaw_deg"]), direction,
            float(row["speed"]), float(row["required_steering_deg"]), event)
        if point.direction not in VALID_DIRECTIONS:
            raise ValueError(f"invalid direction at row {i}: {point.direction}")
        if point.event not in VALID_EVENTS:
            raise ValueError(f"invalid event at row {i}: {point.event}")
        points.append(point)
    return points


def save_route(
    path: str | Path, points: list[RoutePoint], route_name: str,
    policy: VehiclePolicy = VehiclePolicy(), source_db: str = "",
    source_db_sha256: str = "",
) -> dict:
    if route_name not in ROUTE_NAMES:
        raise ValueError(f"route name must be one of: {', '.join(ROUTE_NAMES)}")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, point in enumerate(points):
        point.index = i
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                     dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            row = asdict(point)
            for key in ("x", "y"):
                row[key] = f"{row[key]:.6f}"
            for key in ("yaw_deg", "speed", "required_steering_deg"):
                row[key] = f"{row[key]:.6f}"
            writer.writerow(row)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    report = validate_route(points, policy)
    metadata = {
        "format_version": 2,
        "route_name": route_name,
        "frame_id": "map",
        "route_csv": str(path),
        "route_sha256": _sha256(path),
        "point_count": len(points),
        "vehicle_policy": asdict(policy),
        "source_vslam_db": source_db,
        "source_vslam_db_sha256": source_db_sha256,
        "validation": report,
        "direction_storage": "F/R in CSV; nav_msgs/Path carries pose only",
        "event_storage": "NONE/STOP/ACCEL in CSV; separate from direction",
    }
    metadata_path = path.with_suffix(".metadata.yaml")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8",
                                     dir=path.parent, delete=False) as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
        temporary = Path(stream.name)
    os.replace(temporary, metadata_path)
    return metadata


def validate_route(points: list[RoutePoint], policy: VehiclePolicy = VehiclePolicy()) -> dict:
    invalid = []
    policy_errors = []
    transitions = []
    maximum = 0.0
    for point in points:
        steer = point.required_steering_deg
        maximum = max(maximum, abs(steer))
        if steer < policy.steering_right_deg - 1e-9 or steer > policy.steering_left_deg + 1e-9:
            invalid.append(point.index)
        expected = (0.0 if point.event == "STOP" else
                    policy.reverse_speed if point.direction == "R" else
                    policy.turning_speed if abs(steer) >= policy.turning_threshold_deg else
                    policy.forward_speed)
        if abs(point.speed - expected) > 1e-6:
            policy_errors.append(point.index)
    last_motion = None
    stop_since_motion = False
    for point in points:
        if point.event == "STOP":
            stop_since_motion = True
            continue
        if last_motion and point.direction != last_motion:
            transitions.append({"index": point.index, "from": last_motion,
                                "to": point.direction, "has_stop": stop_since_motion})
            last_motion = point.direction
            stop_since_motion = False
        else:
            last_motion = point.direction
            stop_since_motion = False
    missing_stops = [x["index"] for x in transitions if not x["has_stop"]]
    return {
        "valid": not invalid and not policy_errors and not missing_stops,
        "invalid_steering_indices": invalid,
        "speed_policy_error_indices": policy_errors,
        "direction_changes_without_stop": missing_stops,
        "max_abs_required_steering_deg": maximum,
        "counts": {key: sum(p.direction == key for p in points)
                   for key in ("F", "R")},
        "event_counts": {key: sum(p.event == key for p in points)
                         for key in ("NONE", "STOP", "ACCEL")},
    }


def summary_json(points: list[RoutePoint], policy: VehiclePolicy = VehiclePolicy()) -> str:
    return json.dumps(validate_route(points, policy), ensure_ascii=False, indent=2)
