"""Load map-frame routes and the competition segmented route network."""

from collections import OrderedDict
import csv
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path

import yaml

from .models import RoutePoint


SEGMENTED_COLUMNS = (
    "segment_id", "segment_type", "point_index", "latitude", "longitude",
    "x_m", "y_m", "direction", "mode", "drive_level", "event",
    "from_node", "to_node",
)
ALIGNED_OPTIONAL_COLUMNS = (
    "source_x_m", "source_y_m", "map_yaw_rad", "alignment_zone",
)


def is_segmented_columns(fieldnames):
    fields = tuple(fieldnames or ())
    if fields[:len(SEGMENTED_COLUMNS)] != SEGMENTED_COLUMNS:
        return False
    return fields[len(SEGMENTED_COLUMNS):] in ((), ALIGNED_OPTIONAL_COLUMNS)

# This is the route-network topology, not a substring filter.  AAA_BASE is the
# unbranched source/base recording and is deliberately not part of either
# competition traversal.  In the latest reference schema V_foword is a separate
# common segment; in this workspace CSV it is included at the head of V_A/V_B.
DEFAULT_BRANCH = "A"
A_EXCLUSIVE_SEGMENTS = ("START_A", "T_A", "V_A", "END_AA")
B_EXCLUSIVE_SEGMENTS = ("START_B", "T_B", "V_B", "END_AB")


@dataclass(frozen=True)
class SegmentedRoute:
    points: tuple
    source_points: tuple
    raw_row_count: int
    active_row_count: int
    excluded_b_row_count: int
    segment_order: tuple
    excluded_b_segments: tuple
    common_segments: tuple
    source_frame: str
    frame_id: str
    coordinate_transform: dict
    maximum_step_m: float
    maximum_connection_m: float
    metadata: dict
    selected_branch: str
    excluded_branch_row_count: int
    excluded_branch_segments: tuple


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_route_binding(route_path, map_path, metadata_path=""):
    """Verify hashes, output frame, and independently approved alignment."""
    route, database = Path(route_path), Path(map_path)
    candidates = ([Path(metadata_path)] if metadata_path else []) + [
        route.parent / "metadata.yaml", route.parent / "route_metadata.yaml"]
    metadata = next((item for item in candidates if item.is_file()), None)
    if not route.is_file() or not database.is_file() or metadata is None:
        return False, "MAP_ROUTE_FILES_MISSING"
    with metadata.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not values.get("finalized", True):
        return False, "MAP_ROUTE_NOT_FINALIZED"
    route_hashes = {str(values.get(name, "")) for name in (
        "route_sha256", "route_csv_sha256", "final_route_csv_sha256")}
    if sha256(route) not in route_hashes:
        return False, "ROUTE_CHECKSUM_MISMATCH"
    if sha256(database) != str(values.get("rtabmap_db_sha256", "")):
        return False, "MAP_CHECKSUM_MISMATCH"
    alignment = values.get("alignment", {}) or {}
    route_validation = values.get("route_validation")
    if (route_validation is not None and
            not bool((route_validation or {}).get("validated", False))):
        return False, "MAP_ROUTE_NOT_VALIDATED"
    if (bool(alignment.get("required_for_runtime", False)) and
            not bool(alignment.get("validated", False))):
        return False, "CSV_MAP_ALIGNMENT_NOT_VALIDATED"
    if values.get("route_coordinate_frame", "map") != "map":
        if str(values.get("frame_id", "")) != "map":
            return False, "ROUTE_OUTPUT_FRAME_NOT_MAP"
        if not bool(alignment.get("validated", False)):
            return False, "CSV_MAP_ALIGNMENT_NOT_VALIDATED"
        try:
            if float(alignment["symmetric_rms_m"]) > float(
                    alignment["maximum_allowed_rms_m"]):
                return False, "CSV_MAP_ALIGNMENT_ERROR_TOO_HIGH"
        except (KeyError, TypeError, ValueError):
            return False, "CSV_MAP_ALIGNMENT_QUALITY_MISSING"
    return True, "VERIFIED"


def _finite(row, fields, row_number):
    values = []
    for field in fields:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid segmented CSV row {row_number} field {field}: {error}") \
                from error
        if not math.isfinite(value):
            raise ValueError(
                f"segmented CSV row {row_number} field {field} is NaN or Inf")
        values.append(value)
    return values


def _metadata(path, metadata_path):
    csv_path = Path(path)
    candidates = ([Path(metadata_path)] if metadata_path else []) + [
        csv_path.with_suffix(".metadata.yaml"),
        csv_path.parent / "route_metadata.yaml",
        csv_path.parent / "metadata.yaml",
    ]
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    if selected is None:
        return {}, None
    with selected.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}, selected


def _active_segment_order(segments, branch=DEFAULT_BRANCH):
    branch = str(branch).strip().upper()
    if branch not in ("A", "B"):
        raise ValueError("route branch must be A or B")
    order = [f"START_{branch}", "COMMON_1", "T_foword",
             f"T_{branch}", "COMMON_2"]
    if "V_foword" in segments:
        order.append("V_foword")
    order.extend((f"V_{branch}", "END_common", f"END_A{branch}"))
    missing = [name for name in order if name not in segments]
    if missing:
        raise ValueError(
            f"segmented route is missing branch-{branch} topology segments: " +
            ", ".join(missing))
    return tuple(order)


def _assign_body_yaw(points):
    for index, point in enumerate(points):
        neighbour = None
        before = False
        if (index + 1 < len(points) and
                points[index + 1].direction == point.direction):
            neighbour = points[index + 1]
        elif index > 0 and points[index - 1].direction == point.direction:
            neighbour = points[index - 1]
            before = True
        if neighbour is None:
            motion_yaw = 0.0
        elif before:
            motion_yaw = math.atan2(
                point.y - neighbour.y, point.x - neighbour.x)
        else:
            motion_yaw = math.atan2(
                neighbour.y - point.y, neighbour.x - point.x)
        body_yaw = motion_yaw if point.direction > 0 else motion_yaw + math.pi
        points[index] = replace(
            point, yaw=math.atan2(math.sin(body_yaw), math.cos(body_yaw)))


def _transform_to_map(points, metadata):
    if "route_coordinate_frame" not in metadata or "frame_id" not in metadata:
        raise ValueError(
            "segmented route metadata must declare route_coordinate_frame and frame_id")
    source_frame = str(metadata["route_coordinate_frame"])
    frame_id = str(metadata["frame_id"])
    transform = dict(metadata.get("csv_to_map", {}) or {})
    if source_frame == "map":
        if frame_id != "map":
            raise ValueError("route metadata frame_id must be map")
        return points, source_frame, frame_id, {
            "x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0, "scale": 1.0}
    required = {"x_m", "y_m", "yaw_deg", "scale"}
    missing = required - set(transform)
    if missing or frame_id != "map":
        raise ValueError(
            "non-map CSV requires a complete scale-1 csv_to_map transform")
    tx, ty, yaw_deg, scale = (
        float(transform["x_m"]), float(transform["y_m"]),
        float(transform["yaw_deg"]), float(transform["scale"]))
    if not all(math.isfinite(value) for value in (tx, ty, yaw_deg, scale)):
        raise ValueError("csv_to_map transform contains NaN or Inf")
    if abs(scale - 1.0) > 1.0e-9:
        raise ValueError("csv_to_map scale must remain exactly 1.0")
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    converted = [replace(
        point,
        x=tx + cosine * point.x - sine * point.y,
        y=ty + sine * point.x + cosine * point.y,
    ) for point in points]
    return converted, source_frame, frame_id, {
        "x_m": tx, "y_m": ty, "yaw_deg": yaw_deg, "scale": scale}


def load_segmented_route(path, metadata_path="", max_step_m=1.25,
                         max_connection_m=1.25, branch=DEFAULT_BRANCH):
    """Strictly parse the final network and assemble COMMON plus A or B."""
    branch = str(branch).strip().upper()
    if branch not in ("A", "B"):
        raise ValueError("route branch must be A or B")
    segments = OrderedDict()
    source_coordinates = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not is_segmented_columns(reader.fieldnames):
            raise ValueError(
                f"unexpected segmented CSV columns: {reader.fieldnames}; "
                f"expected {list(SEGMENTED_COLUMNS)} plus optional aligned fields")
        for row_number, row in enumerate(reader, 2):
            numeric = _finite(
                row, ("point_index", "latitude", "longitude", "x_m", "y_m",
                      "direction", "mode", "drive_level"), row_number)
            point_index = int(numeric[0])
            direction = int(numeric[5])
            mode = int(numeric[6])
            if numeric[0] != point_index or point_index < 0:
                raise ValueError(f"invalid point_index at CSV row {row_number}")
            if numeric[5] != direction or direction not in (-1, 1):
                raise ValueError(f"direction must be 1 or -1 at CSV row {row_number}")
            if numeric[6] != mode or not 1 <= mode <= 11:
                raise ValueError(f"mode must be an integer 1..11 at CSV row {row_number}")
            if numeric[7] not in (1.0, 2.0, 3.0):
                raise ValueError(f"drive_level must be 1, 2, or 3 at CSV row {row_number}")
            segment_id = str(row["segment_id"]).strip()
            event = str(row["event"]).strip().upper()
            if not segment_id or not event:
                raise ValueError(f"empty segment_id/event at CSV row {row_number}")
            point = RoutePoint(
                -1, numeric[3], numeric[4], 0.0, direction, numeric[7],
                event, segment_id, segment_id, str(row["segment_type"]).strip(),
                point_index, numeric[1], numeric[2], mode, event,
                str(row["from_node"]).strip(), str(row["to_node"]).strip())
            segments.setdefault(segment_id, []).append(point)
            if "source_x_m" in row:
                source_xy = _finite(row, ("source_x_m", "source_y_m"), row_number)
                source_coordinates[(segment_id, point_index)] = tuple(source_xy)
    raw_count = sum(len(points) for points in segments.values())
    for name, points in segments.items():
        actual = [point.point_index for point in points]
        if actual != list(range(len(points))):
            raise ValueError(
                f"segment {name} point_index must be contiguous from zero")

    order = _active_segment_order(segments, branch)
    active = []
    common = []
    for name in order:
        if name not in A_EXCLUSIVE_SEGMENTS+B_EXCLUSIVE_SEGMENTS:
            common.append(name)
        start_index = len(active)
        active.extend(replace(point, index=start_index + offset)
                      for offset, point in enumerate(segments[name]))
    excluded_segments = (B_EXCLUSIVE_SEGMENTS if branch == "A" else
                         A_EXCLUSIVE_SEGMENTS)
    if any(point.segment_id in excluded_segments for point in active):
        raise ValueError(
            f"branch-{branch} route contains an opposite-branch segment")
    active_indexes = [point.index for point in active]
    if active_indexes != list(range(len(active))):
        raise ValueError("active route indexes are not monotonic and contiguous")

    metadata, _ = _metadata(path, metadata_path)
    if not metadata:
        raise ValueError("segmented route metadata is required")
    _assign_body_yaw(active)
    source = [replace(
        point,
        x=source_coordinates.get((point.segment_id, point.point_index),
                                 (point.x, point.y))[0],
        y=source_coordinates.get((point.segment_id, point.point_index),
                                 (point.x, point.y))[1],
    ) for point in active]
    _assign_body_yaw(source)
    source_points = tuple(source)
    active, source_frame, frame_id, transform = _transform_to_map(active, metadata)
    _assign_body_yaw(active)
    if not all(math.isfinite(value) for point in active
               for value in (point.x, point.y, point.yaw)):
        raise ValueError("active route contains NaN or Inf after map transform")

    steps = [math.hypot(after.x - before.x, after.y - before.y)
             for before, after in zip(active, active[1:])
             if before.segment_id == after.segment_id]
    connections = [math.hypot(
        segments[after][0].x - segments[before][-1].x,
        segments[after][0].y - segments[before][-1].y)
        for before, after in zip(order, order[1:])]
    maximum_step = max(steps, default=0.0)
    maximum_connection = max(connections, default=0.0)
    if maximum_step > float(max_step_m):
        raise ValueError(
            f"active route point jump {maximum_step:.3f} m exceeds {max_step_m:.3f} m")
    if maximum_connection > float(max_connection_m):
        raise ValueError(
            "active route segment connection "
            f"{maximum_connection:.3f} m exceeds {max_connection_m:.3f} m")
    excluded_b = B_EXCLUSIVE_SEGMENTS if branch == "A" else ()
    excluded_b_count = sum(len(segments.get(name, ())) for name in excluded_b)
    excluded_count = sum(len(segments.get(name, ()))
                         for name in excluded_segments)
    return SegmentedRoute(
        tuple(active), source_points, raw_count, len(active), excluded_b_count, order,
        tuple(name for name in excluded_b if name in segments),
        tuple(common), source_frame, frame_id, transform,
        maximum_step, maximum_connection, metadata, branch, excluded_count,
        tuple(name for name in excluded_segments if name in segments))


def load_map_route(path, include_invalid=False):
    points = []
    previous_stamp = None
    previous_distance = -1.0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_sec", "timestamp_nanosec", "route_index",
                    "map_x_m", "map_y_m", "map_z_m", "yaw_rad",
                    "cumulative_distance_m", "valid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("route CSV missing fields: " + ", ".join(sorted(missing)))
        for row in reader:
            stamp = int(row["timestamp_sec"]) * 1_000_000_000 + int(row["timestamp_nanosec"])
            distance = float(row["cumulative_distance_m"])
            values = tuple(float(row[name]) for name in
                           ("map_x_m", "map_y_m", "map_z_m", "yaw_rad"))
            if not all(math.isfinite(value) for value in values + (distance,)):
                raise ValueError("route CSV contains NaN or Inf")
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError("route CSV timestamps are not strictly increasing")
            if distance < previous_distance:
                raise ValueError("route CSV cumulative distance decreased")
            valid = str(row["valid"]).lower() in ("1", "true", "yes")
            if valid or include_invalid:
                points.append({
                    "stamp_ns": stamp, "index": int(row["route_index"]),
                    "x": values[0], "y": values[1], "z": values[2],
                    "yaw": values[3], "valid": valid,
                })
            previous_stamp, previous_distance = stamp, distance
    return points
