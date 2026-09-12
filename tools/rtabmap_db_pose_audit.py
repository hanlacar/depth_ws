#!/usr/bin/env python3
"""Read-only low-memory pose, link, payload and graph audit for RTAB-Map DBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


LINK_TYPES = {
    0: "Neighbor", 1: "GlobalClosure", 2: "LocalSpaceClosure",
    3: "LocalTimeClosure", 4: "UserClosure", 5: "VirtualClosure",
    6: "NeighborMerged", 7: "PosePrior", 8: "Landmark",
}


def matrix_from_blob(blob: bytes | None) -> np.ndarray | None:
    if blob is None or len(blob) != 48:
        return None
    values = struct.unpack("<12f", blob)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(values).reshape(3, 4)
    return matrix


def pose_fields(matrix: np.ndarray) -> dict:
    rotation = matrix[:3, :3]
    # ZYX Euler angles, matching RTAB-Map's common roll/pitch/yaw convention.
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return {
        "x": float(matrix[0, 3]), "y": float(matrix[1, 3]),
        "z": float(matrix[2, 3]), "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw),
    }


def rotation_error_deg(matrix: np.ndarray) -> float:
    value = float(np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(value))


def relative_error(a: np.ndarray, b: np.ndarray, measurement: np.ndarray) -> tuple[float, float]:
    predicted = np.linalg.inv(a) @ b
    error = np.linalg.inv(measurement) @ predicted
    return float(np.linalg.norm(error[:3, 3])), rotation_error_deg(error)


def load_exported_poses(path: Path | None) -> dict[int, np.ndarray]:
    poses = {}
    if path is None or not path.exists():
        return poses
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 9:
                continue
            _, x, y, z, qx, qy, qz, qw, node_id = fields
            x, y, z, qx, qy, qz, qw = map(float, (x, y, z, qx, qy, qz, qw))
            # Quaternion to homogeneous matrix.
            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            qx, qy, qz, qw = (qx/norm, qy/norm, qz/norm, qw/norm)
            matrix = np.eye(4)
            matrix[:3, :3] = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])
            matrix[:3, 3] = (x, y, z)
            poses[int(node_id)] = matrix
    return poses


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values)
    return {
        "min": float(np.min(array)), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "max": float(np.max(array)),
    }


def ranges(poses: dict[int, np.ndarray]) -> dict:
    if not poses:
        return {}
    values = np.array([matrix[:3, 3] for matrix in poses.values()])
    return {
        axis: {"min": float(values[:, index].min()),
               "max": float(values[:, index].max()),
               "span": float(np.ptp(values[:, index]))}
        for index, axis in enumerate("xyz")
    }


def connected_components(ids: set[int], edges: set[tuple[int, int]]) -> list[list[int]]:
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    remaining = set(ids)
    result = []
    while remaining:
        start = next(iter(remaining))
        queue = deque([start])
        component = []
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in graph[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component))
    return sorted(result, key=len, reverse=True)


def audit(db_path: Path, optimized_path: Path | None) -> dict:
    uri = f"file:{db_path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    node_rows = list(connection.execute(
        "SELECT id,map_id,stamp,pose,weight,label,time_enter FROM Node ORDER BY stamp,id"))
    raw = {row[0]: matrix_from_blob(row[3]) for row in node_rows}
    raw = {key: value for key, value in raw.items() if value is not None}
    optimized = load_exported_poses(optimized_path)
    map_counts = [dict(zip(
        ("map_id", "count", "id_min", "id_max", "stamp_min", "stamp_max"), row))
        for row in connection.execute(
            "SELECT map_id,count(*),min(id),max(id),min(stamp),max(stamp) "
            "FROM Node GROUP BY map_id ORDER BY map_id")]

    payload_row = connection.execute(
        "SELECT count(*),"
        "sum(image IS NOT NULL AND length(image)>0),"
        "sum(depth IS NOT NULL AND length(depth)>0),"
        "sum(calibration IS NOT NULL AND length(calibration)>0),"
        "sum(depth_confidence IS NOT NULL AND length(depth_confidence)>0),"
        "sum(scan IS NOT NULL AND length(scan)>0),"
        "sum(ground_cells IS NOT NULL AND length(ground_cells)>0),"
        "sum(obstacle_cells IS NOT NULL AND length(obstacle_cells)>0),"
        "sum(empty_cells IS NOT NULL AND length(empty_cells)>0),"
        "min(cell_size),max(cell_size) FROM Data").fetchone()
    calibration_hashes = Counter()
    for calibration, in connection.execute("SELECT calibration FROM Data"):
        calibration_hashes[hashlib.sha256(calibration or b"").hexdigest()] += 1
    admin = connection.execute(
        "SELECT version,length(preview_image),length(opt_cloud),length(opt_ids),"
        "length(opt_poses),length(opt_last_localization),length(opt_map),"
        "opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin").fetchone()

    # Sequential-node deltas, including whether the database has an odometry edge.
    link_rows = list(connection.execute(
        "SELECT from_id,to_id,type,transform,information_matrix FROM Link"))
    neighbor_pairs = {tuple(sorted((a, b))) for a, b, link_type, _, _ in link_rows
                      if link_type == 0}
    sequential = []
    for first, second in zip(node_rows, node_rows[1:]):
        a, b = first[0], second[0]
        if first[1] != second[1] or a not in raw or b not in raw:
            continue
        relative = np.linalg.inv(raw[a]) @ raw[b]
        sequential.append({
            "from": a, "to": b, "dt_s": float(second[2]-first[2]),
            "translation_m": float(np.linalg.norm(relative[:3, 3])),
            "rotation_deg": rotation_error_deg(relative),
            "yaw_change_deg": math.degrees(math.atan2(relative[1, 0], relative[0, 0])),
            "has_neighbor_link": tuple(sorted((a, b))) in neighbor_pairs,
        })

    # Deduplicate symmetric link rows while retaining naturally one-way DB links.
    dedup = {}
    raw_link_counts = Counter()
    for a, b, link_type, transform_blob, information_blob in link_rows:
        raw_link_counts[link_type] += 1
        key = (min(a, b), max(a, b), link_type)
        candidate = (a, b, transform_blob, information_blob)
        candidate_matrix = matrix_from_blob(transform_blob)
        candidate_valid = (candidate_matrix is not None and
                           abs(np.linalg.det(candidate_matrix[:3, :3])) > 1e-6)
        existing = dedup.get(key)
        existing_matrix = matrix_from_blob(existing[2]) if existing else None
        existing_valid = (existing_matrix is not None and
                          abs(np.linalg.det(existing_matrix[:3, :3])) > 1e-6)
        if (existing is None or (candidate_valid and not existing_valid)
                or (candidate_valid == existing_valid and a < b and existing[0] > existing[1])):
            dedup[key] = candidate

    link_details = []
    for (lo, hi, link_type), (a, b, transform_blob, information_blob) in dedup.items():
        measurement = matrix_from_blob(transform_blob)
        if measurement is None:
            continue
        transform_valid = bool(abs(np.linalg.det(measurement[:3, :3])) > 1e-6)
        entry = {
            "from": a, "to": b, "type": link_type,
            "type_name": LINK_TYPES.get(link_type, str(link_type)),
            "measurement_m": float(np.linalg.norm(measurement[:3, 3])),
            "measurement_deg": rotation_error_deg(measurement),
            "transform_valid": transform_valid,
        }
        if information_blob and len(information_blob) == 36 * 8:
            information = np.asarray(struct.unpack("<36d", information_blob)).reshape(6, 6)
            entry["information_diag"] = [float(x) for x in np.diag(information)]
        if transform_valid and a in raw and b in raw:
            entry["raw_error_m"], entry["raw_error_deg"] = relative_error(
                raw[a], raw[b], measurement)
            entry["raw_endpoint_distance_m"] = float(
                np.linalg.norm(raw[a][:3, 3] - raw[b][:3, 3]))
        if transform_valid and a in optimized and b in optimized:
            entry["optimized_error_m"], entry["optimized_error_deg"] = relative_error(
                optimized[a], optimized[b], measurement)
            entry["optimized_endpoint_distance_m"] = float(
                np.linalg.norm(optimized[a][:3, 3] - optimized[b][:3, 3]))
        link_details.append(entry)

    closures = [entry for entry in link_details if entry["type"] in (1, 2, 3, 4)]
    valid_closures = [entry for entry in closures if entry["transform_valid"]]
    suspect = sorted(
        [entry for entry in closures
         if entry.get("optimized_error_m", 0) > 0.5
         or entry.get("optimized_error_deg", 0) > 10.0],
        key=lambda item: (item.get("optimized_error_m", 0),
                          item.get("optimized_error_deg", 0)), reverse=True)
    top_corrections = sorted(
        valid_closures, key=lambda item: item.get("raw_error_m", -1), reverse=True)[:20]

    components = connected_components(set(raw), neighbor_pairs)
    optimized_ids = set(optimized)
    parameters = connection.execute(
        "SELECT parameters FROM Info ORDER BY rowid DESC LIMIT 1").fetchone()
    parsed_parameters = {}
    if parameters and parameters[0]:
        for field in parameters[0].split(";"):
            if ":" in field:
                key, value = field.split(":", 1)
                parsed_parameters[key] = value
    selected_parameters = {key: parsed_parameters.get(key) for key in (
        "Rtabmap/DetectionRate", "RGBD/LinearUpdate", "RGBD/AngularUpdate",
        "RGBD/CreateOccupancyGrid", "RGBD/ForceOdom3DoF", "Reg/Force3DoF",
        "Optimizer/GravitySigma", "Optimizer/Robust", "Optimizer/Strategy",
        "RGBD/OptimizeMaxError", "Mem/NotLinkedNodesKept", "Vis/MinInliers",
        "Grid/3D", "Grid/CellSize", "Grid/RangeMax")}
    feature = connection.execute(
        "SELECT count(*),count(distinct node_id),avg(CASE WHEN depth_z IS NOT NULL THEN 1.0 ELSE 0 END),"
        "avg(response),min(response),max(response) FROM Feature").fetchone()
    connection.close()

    pose_order = [row[0] for row in node_rows if row[0] in raw]
    opt_order = [node_id for node_id in pose_order if node_id in optimized]
    report = {
        "path": str(db_path.resolve()), "size_bytes": db_path.stat().st_size,
        "integrity": integrity, "node_count": len(node_rows), "map_counts": map_counts,
        "stamp_start": node_rows[0][2] if node_rows else None,
        "stamp_end": node_rows[-1][2] if node_rows else None,
        "duration_s": node_rows[-1][2]-node_rows[0][2] if node_rows else 0,
        "raw_pose_count": len(raw), "raw_ranges": ranges(raw),
        "raw_start": {"id": pose_order[0], **pose_fields(raw[pose_order[0]])} if pose_order else None,
        "raw_end": {"id": pose_order[-1], **pose_fields(raw[pose_order[-1]])} if pose_order else None,
        "optimized_pose_count": len(optimized), "optimized_ranges": ranges(optimized),
        "optimized_start": {"id": opt_order[0], **pose_fields(optimized[opt_order[0]])} if opt_order else None,
        "optimized_end": {"id": opt_order[-1], **pose_fields(optimized[opt_order[-1]])} if opt_order else None,
        "optimized_missing_count": len(set(raw)-optimized_ids),
        "optimized_missing_ranges": [
            {"count": len(component), "min": min(component), "max": max(component)}
            for component in connected_components(set(raw)-optimized_ids, set())[:10]
        ],
        "payload": dict(zip((
            "rows", "rgb", "depth", "calibration", "depth_confidence", "scan",
            "ground", "obstacle", "empty", "cell_size_min", "cell_size_max"), payload_row)),
        "calibration_hash_counts": calibration_hashes,
        "admin": dict(zip((
            "version", "preview_bytes", "opt_cloud_bytes", "opt_ids_bytes",
            "opt_poses_bytes", "opt_last_localization_bytes", "opt_map_bytes",
            "opt_map_x_min", "opt_map_y_min", "opt_map_resolution"), admin)),
        "sequential_translation": numeric_summary([x["translation_m"] for x in sequential]),
        "sequential_rotation_deg": numeric_summary([x["rotation_deg"] for x in sequential]),
        "sequential_dt_s": numeric_summary([x["dt_s"] for x in sequential]),
        "top_translation_steps": sorted(sequential, key=lambda x: x["translation_m"], reverse=True)[:20],
        "top_rotation_steps": sorted(sequential, key=lambda x: x["rotation_deg"], reverse=True)[:20],
        "missing_neighbor_steps": [x for x in sequential if not x["has_neighbor_link"]],
        "neighbor_components": [
            {"count": len(component), "min": min(component), "max": max(component)}
            for component in components[:20]
        ],
        "raw_link_counts": {LINK_TYPES.get(k, str(k)): v for k, v in raw_link_counts.items()},
        "unique_link_counts": {LINK_TYPES.get(k, str(k)): v for k, v in Counter(
            key[2] for key in dedup).items()},
        "closure_count": len(closures),
        "valid_closure_count": len(valid_closures),
        "null_closure_count": len(closures)-len(valid_closures),
        "null_closures": [entry for entry in closures if not entry["transform_valid"]],
        "closure_suspect_count": len(suspect),
        "closure_suspects": suspect[:30], "largest_raw_closure_corrections": top_corrections,
        "parameters": selected_parameters,
        "feature": dict(zip(("count", "nodes", "depth_fraction", "response_avg",
                             "response_min", "response_max"), feature)),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--optimized-poses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.database, args.optimized_poses)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: dict(x))
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
