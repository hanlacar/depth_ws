"""Rebuild a raw route against the final optimized RTAB-Map graph."""

import csv
import json
import math
from pathlib import Path

from .geometry import wrap_angle
from .route_recorder_core import FIELDS, file_sha256


def _stamp(row):
    return int(row["timestamp_sec"])*1_000_000_000+int(row["timestamp_nanosec"])


def _anchored(row, anchors):
    anchor = anchors.get(str(int(row["nearest_rtabmap_node_id"])))
    if anchor is None or any(row.get(name, "") == "" for name in (
            "node_relative_x_m", "node_relative_y_m", "node_relative_yaw_rad")):
        return None
    relative_x, relative_y = float(row["node_relative_x_m"]), float(row["node_relative_y_m"])
    cosine, sine = math.cos(float(anchor["yaw"])), math.sin(float(anchor["yaw"]))
    return (float(anchor["x"])+cosine*relative_x-sine*relative_y,
            float(anchor["y"])+sine*relative_x+cosine*relative_y,
            wrap_angle(float(anchor["yaw"])+float(row["node_relative_yaw_rad"])))


def _interpolate(first, second, fraction):
    row = dict(first)
    for name in ("map_x_m", "map_y_m", "map_z_m"):
        row[name] = float(first[name])+fraction*(float(second[name])-float(first[name]))
    row["yaw_rad"] = wrap_angle(float(first["yaw_rad"])+fraction*wrap_angle(
        float(second["yaw_rad"])-float(first["yaw_rad"])))
    row["timestamp_ns"] = int(round(first["timestamp_ns"]+fraction*(
        second["timestamp_ns"]-first["timestamp_ns"])))
    if fraction >= 0.5:
        for name in ("localization_state", "localization_confidence",
                     "tracking_valid", "reset_count", "nearest_rtabmap_node_id",
                     "node_relative_x_m", "node_relative_y_m",
                     "node_relative_yaw_rad", "pose_source", "direction",
                     "drive_level", "mission_marker", "stop_line_id", "section_id"):
            row[name] = second.get(name, row.get(name, ""))
    return row


def realign_route(raw_path, graph_path, final_path, spacing_m=0.05,
                  maximum_jump_m=0.75, maximum_yaw_jump_deg=120.0):
    """Create, never overwrite, a smoothed final map-frame CSV."""
    raw_path, graph_path, final_path = map(Path, (raw_path, graph_path, final_path))
    if final_path.exists() or final_path.with_suffix(final_path.suffix+".partial").exists():
        raise FileExistsError(f"refusing to overwrite final route: {final_path}")
    with open(graph_path, encoding="utf-8") as stream:
        graph = json.load(stream)
    with open(raw_path, newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    corrected, rejected = [], 0
    for source in raw_rows:
        if str(source.get("valid", source.get("record_valid", ""))).lower() != "true":
            rejected += 1
            continue
        pose = _anchored(source, graph.get("poses", {}))
        if pose is None:
            rejected += 1
            continue
        row = dict(source, map_x_m=pose[0], map_y_m=pose[1], yaw_rad=pose[2],
                   timestamp_ns=_stamp(source))
        if corrected:
            distance = math.hypot(pose[0]-float(corrected[-1]["map_x_m"]),
                                  pose[1]-float(corrected[-1]["map_y_m"]))
            yaw_jump = abs(wrap_angle(pose[2]-float(corrected[-1]["yaw_rad"])))
            same_direction = str(source.get("direction", "1")) == str(
                corrected[-1].get("direction", "1"))
            if (distance > float(maximum_jump_m) or
                    (same_direction and distance < 0.15 and
                     yaw_jump > math.radians(maximum_yaw_jump_deg))):
                rejected += 1
                continue
        corrected.append(row)
    if len(corrected) < 2:
        raise RuntimeError("fewer than two valid anchored route points")
    sampled = [corrected[0]]
    for first, second in zip(corrected, corrected[1:]):
        distance = math.hypot(float(second["map_x_m"])-float(first["map_x_m"]),
                              float(second["map_y_m"])-float(first["map_y_m"]))
        count = max(1, int(math.ceil(distance/float(spacing_m))))
        for index in range(1, count+1):
            sampled.append(_interpolate(first, second, index/count))
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial = final_path.with_suffix(final_path.suffix+".partial")
    cumulative, previous, last_stamp = 0.0, None, None
    with open(partial, "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, row in enumerate(sampled):
            x, y, yaw = (float(row["map_x_m"]), float(row["map_y_m"]),
                         float(row["yaw_rad"]))
            if previous is not None:
                cumulative += math.hypot(x-previous[0], y-previous[1])
            stamp = max(int(row["timestamp_ns"]), (last_stamp+1 if last_stamp is not None else 0))
            output = {name: row.get(name, "") for name in FIELDS}
            output.update({
                "timestamp_sec": stamp//1_000_000_000,
                "timestamp_nanosec": stamp % 1_000_000_000,
                "route_index": index, "map_x_m": x, "map_y_m": y,
                "yaw_rad": yaw, "yaw_deg": math.degrees(yaw),
                "cumulative_distance_m": cumulative, "valid": True,
                "index": index, "timestamp": stamp/1.0e9,
                "x": x, "y": y, "z": output.get("map_z_m", 0.0),
                "yaw": yaw, "record_valid": True,
            })
            writer.writerow(output)
            previous, last_stamp = (x, y), stamp
    partial.replace(final_path)
    return {
        "raw_point_count": len(raw_rows), "anchored_point_count": len(corrected),
        "rejected_point_count": rejected, "final_point_count": len(sampled),
        "valid_route_point_percent": 100.0*len(corrected)/len(raw_rows) if raw_rows else 0.0,
        "total_distance_m": cumulative, "final_route_csv_sha256": file_sha256(final_path),
        "graph_session_id": graph.get("session_id", ""),
    }
