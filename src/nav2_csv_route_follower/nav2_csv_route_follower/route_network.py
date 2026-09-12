"""Strict parser and branch assembler for route_network_segmented."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path

import yaml


REQUIRED_COLUMNS = (
    "segment_id", "segment_type", "point_index", "latitude", "longitude",
    "x_m", "y_m", "direction", "mode", "drive_level", "event",
    "from_node", "to_node",
)


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass
class RoutePoint:
    route_index: int
    segment_id: str
    point_index: int
    x: float
    y: float
    yaw: float
    direction: str
    mode: int
    drive_level: float
    event: str
    segment_type: str = ""
    from_node: str = ""
    to_node: str = ""


@dataclass(frozen=True)
class RouteMetadata:
    format_version: int
    origin_lat: float
    origin_lon: float
    segmented: bool
    values: dict


class RouteNetwork:
    def __init__(self, segments, metadata):
        self.segments = segments
        self.metadata = metadata

    @classmethod
    def load(cls, csv_path, yaml_path):
        csv_path, yaml_path = Path(csv_path), Path(yaml_path)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                raise ValueError(
                    f"unexpected CSV columns: {reader.fieldnames}; "
                    f"expected {list(REQUIRED_COLUMNS)}")
            raw_segments = OrderedDict()
            for row_number, row in enumerate(reader, 2):
                try:
                    direction_value = int(row["direction"])
                    direction = "F" if direction_value == 1 else (
                        "R" if direction_value == -1 else "")
                    if not direction:
                        raise ValueError("direction must be 1 or -1")
                    event = row["event"].strip().upper()
                    if event not in {"NONE", "STOP_LINE"}:
                        raise ValueError(f"unsupported event {event!r}")
                    drive = float(row["drive_level"])
                    if drive not in {1.0, 2.0, 3.0}:
                        raise ValueError(f"unsupported drive_level {drive}")
                    point = RoutePoint(
                        route_index=-1,
                        segment_id=row["segment_id"].strip(),
                        point_index=int(row["point_index"]),
                        x=float(row["x_m"]), y=float(row["y_m"]), yaw=0.0,
                        direction=direction, mode=int(row["mode"]),
                        drive_level=drive, event=event,
                        segment_type=row["segment_type"].strip(),
                        from_node=row["from_node"].strip(),
                        to_node=row["to_node"].strip())
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid CSV row {row_number}: {error}") from error
                raw_segments.setdefault(point.segment_id, []).append(point)
        for name, points in raw_segments.items():
            expected = list(range(len(points)))
            actual = [point.point_index for point in points]
            if actual != expected:
                raise ValueError(f"segment {name} point_index is not contiguous")
            cls._assign_yaw(points)
        values = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for field in ("format_version", "origin_lat", "origin_lon", "segmented"):
            if field not in values:
                raise ValueError(f"route YAML missing {field}")
        metadata = RouteMetadata(
            int(values["format_version"]), float(values["origin_lat"]),
            float(values["origin_lon"]), bool(values["segmented"]), values)
        return cls(raw_segments, metadata)

    @staticmethod
    def _assign_yaw(points):
        for index, point in enumerate(points):
            # Use a neighbour with the same direction when possible so an
            # F/R boundary does not contaminate vehicle heading.
            candidate = None
            if index + 1 < len(points) and points[index + 1].direction == point.direction:
                candidate = points[index + 1]
                dx, dy = candidate.x - point.x, candidate.y - point.y
            elif index > 0 and points[index - 1].direction == point.direction:
                candidate = points[index - 1]
                dx, dy = point.x - candidate.x, point.y - candidate.y
            else:
                dx, dy = 1.0, 0.0
            motion_yaw = math.atan2(dy, dx)
            point.yaw = normalize_angle(
                motion_yaw if point.direction == "F" else motion_yaw + math.pi)

    @staticmethod
    def normalize_branch(value) -> str:
        return "B" if str(value).strip().upper() == "B" else "A"

    def assemble(self, start="A", t_branch="A", parallel="A", end="A"):
        start = self.normalize_branch(start)
        t_branch = self.normalize_branch(t_branch)
        parallel = self.normalize_branch(parallel)
        end = self.normalize_branch(end)
        names = [
            f"START_{start}", "COMMON_1", "T_foword", f"T_{t_branch}",
            "COMMON_2", f"V_{parallel}", "END_common", f"END_A{end}",
        ]
        missing = [name for name in names if name not in self.segments]
        if missing:
            raise ValueError(f"route network missing selected segments: {missing}")
        result = []
        for name in names:
            for point in self.segments[name]:
                result.append(replace(point, route_index=len(result)))
        return result

    def analysis(self):
        result = {
            "waypoint_count": sum(len(v) for v in self.segments.values()),
            "metadata": self.metadata.values,
            "segments": [], "stop_lines": [], "direction_changes": [],
            "drive_3_ranges": [], "drive_level_3_points": [],
        }
        for name, points in self.segments.items():
            item = {
                "segment_id": name, "segment_type": points[0].segment_type,
                "count": len(points), "start_index": points[0].point_index,
                "end_index": points[-1].point_index,
                "start_xy": [points[0].x, points[0].y],
                "end_xy": [points[-1].x, points[-1].y],
                "directions": sorted(set(p.direction for p in points)),
                "modes": sorted(set(p.mode for p in points)),
                "drive_levels": sorted(set(p.drive_level for p in points)),
            }
            result["segments"].append(item)
            result["stop_lines"].extend({
                "segment_id": name, "point_index": p.point_index,
                "x": p.x, "y": p.y} for p in points if p.event == "STOP_LINE")
            result["drive_level_3_points"].extend({
                "segment_id": name, "point_index": p.point_index,
                "mode": p.mode, "x": p.x, "y": p.y,
            } for p in points if p.drive_level == 3.0)
            for before, after in zip(points, points[1:]):
                if before.direction != after.direction:
                    result["direction_changes"].append({
                        "segment_id": name,
                        "before_index": before.point_index,
                        "after_index": after.point_index,
                        "from": before.direction, "to": after.direction,
                        "x": after.x, "y": after.y})
            start = None
            for index, point in enumerate(points + [None]):
                is_three = point is not None and point.drive_level == 3.0
                if is_three and start is None:
                    start = index
                if not is_three and start is not None:
                    end = index - 1
                    result["drive_3_ranges"].append({
                        "segment_id": name,
                        "start_index": points[start].point_index,
                        "end_index": points[end].point_index,
                        "start_xy": [points[start].x, points[start].y],
                        "end_xy": [points[end].x, points[end].y]})
                    start = None
        return result


def transform_points(points, tx: float, ty: float, yaw_deg: float):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    transformed = []
    for point in points:
        transformed.append(replace(
            point,
            x=tx + c * point.x - s * point.y,
            y=ty + s * point.x + c * point.y,
            yaw=normalize_angle(point.yaw + yaw)))
    return transformed
