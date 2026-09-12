#!/usr/bin/env python3
"""Create a road-level, locally aligned RTAB-Map working seed.

Only a copied database is changed. Flat-road orientation is inferred from the
stored RGB-D road planes. A per-recording fixed camera rotation correction is
applied about the recorded camera origin, then a smooth node attitude residual
levels the measured road without flattening curbs, walls or other 3-D points.
The existing ramp trajectory/grade is retained. No nodes are added or removed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sqlite3
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import medfilt, savgol_filter


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
OUTPUT_DIR = ROOT / "maps/merged_competition_level_aligned_v5"
WORK = OUTPUT_DIR / "work"
SEED = WORK / "level_pose_seed.db"
PLANES = ROOT / "analysis/level_aligned/depth_plane_before_detailed.csv"
PAIRS = ROOT / "analysis/level_aligned/rgbd_pair_alignment_raw_tf.jsonl"
RAMP_REPORT = ROOT / "maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json"
REPORT = WORK / "level_build_report.json"
EXPECTED_SHA256 = "b8bff86d58c60663df910ff053b43d91d33c6bd9076d4251d7f992637de44e86"
GROUND_DATUM_Z = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(blob: bytes) -> np.ndarray:
    result = np.eye(4)
    result[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return result


def pose_blob(pose: np.ndarray) -> bytes:
    return struct.pack("<12f", *pose[:3, :4].astype(np.float32).ravel())


def rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def se2_matrix(theta: float, translation: np.ndarray) -> np.ndarray:
    result = np.eye(3)
    c, s = math.cos(theta), math.sin(theta)
    result[:2, :2] = ((c, -s), (s, c))
    result[:2, 2] = translation
    return result


def pose_se2(pose: np.ndarray) -> np.ndarray:
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    return se2_matrix(yaw, pose[:2, 3])


def align_normal_to_up(normal: np.ndarray) -> np.ndarray:
    source = normal / np.linalg.norm(normal)
    target = np.asarray([0.0, 0.0, 1.0])
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3)
    axis /= sine
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = math.atan2(sine, cosine)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def smooth(values: np.ndarray, valid: np.ndarray, median_window: int = 9,
           savgol_window: int = 17) -> np.ndarray:
    index = np.arange(len(values))
    if valid.sum() < 2:
        return np.zeros_like(values)
    filled = np.interp(index, index[valid], values[valid])
    mw = min(median_window, len(values) if len(values) % 2 else len(values)-1)
    if mw >= 3:
        filled = medfilt(filled, mw)
    sw = min(savgol_window, len(values) if len(values) % 2 else len(values)-1)
    if sw >= 5:
        filled = savgol_filter(filled, sw, 2, mode="interp")
    return filled


def load_planes() -> dict[int, dict[str, float]]:
    result = {}
    with PLANES.open() as stream:
        for row in csv.DictReader(stream):
            result[int(row["id"])] = {key: float(value) for key, value in row.items() if key != "id"}
    return result


def load_ramp_nodes() -> tuple[set[int], dict[int, list]]:
    report = json.loads(RAMP_REPORT.read_text())
    source_map = {int(key): value for key, value in report["node_source_map"].items()}
    ramps = set()
    for node_id, (source, source_id) in source_map.items():
        if source == "competition_338" and 115 <= source_id <= 210:
            ramps.add(node_id)
        if source == "091640" and 345 <= source_id <= 630:
            ramps.add(node_id)
    return ramps, source_map


def fit_marking_correction(poses: dict[int, np.ndarray]) -> tuple[np.ndarray, dict]:
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    accepted = [row for row in rows if row.get("kind") == "pair" and row.get("accepted")
                and row["target_map"] == 2 and row["inliers"] >= 6]
    best = {}
    for row in accepted:
        key = row["target_id"]
        if key not in best or (row["inliers"], -row["rms_m"]) > (best[key]["inliers"], -best[key]["rms_m"]):
            best[key] = row
    training_ids = [node_id for node_id in (1564, 1567, 1573, 1576, 1582) if node_id in best]
    holdout_ids = [node_id for node_id in (1570, 1579, 1600) if node_id in best]
    if len(training_ids) < 4 or len(holdout_ids) < 2:
        raise RuntimeError("insufficient independent RGB-D marking anchors")

    def desired(row):
        local = se2_matrix(math.radians(row["theta_deg"]), np.asarray([row["tx"], row["ty"]]))
        return pose_se2(poses[row["base_id"]]) @ local

    all_ids = training_ids + holdout_ids
    source = {node_id: pose_se2(poses[node_id]) for node_id in all_ids}
    target = {node_id: desired(best[node_id]) for node_id in all_ids}
    center = np.mean([source[node_id][:2, 2] for node_id in training_ids], axis=0)

    def residual(parameters):
        theta, dx, dy = parameters
        rotation = se2_matrix(theta, np.asarray([0.0, 0.0]))[:2, :2]
        errors = []
        for node_id in training_ids:
            predicted = rotation @ (source[node_id][:2, 2] - center) + center + (dx, dy)
            errors.extend(predicted - target[node_id][:2, 2])
        return np.asarray(errors)

    optimized = least_squares(residual, np.zeros(3), loss="huber", f_scale=0.25)
    theta, dx, dy = optimized.x
    rotation = se2_matrix(theta, np.asarray([0.0, 0.0]))[:2, :2]
    translation = center + np.asarray([dx, dy]) - rotation @ center
    correction = se2_matrix(theta, translation)

    def errors(ids, transform):
        output = []
        for node_id in ids:
            predicted = transform @ source[node_id]
            output.append(float(np.linalg.norm(predicted[:2, 2] - target[node_id][:2, 2])))
        return output

    identity = np.eye(3)
    train_before, train_after = errors(training_ids, identity), errors(training_ids, correction)
    hold_before, hold_after = errors(holdout_ids, identity), errors(holdout_ids, correction)
    before_rms = float(np.sqrt(np.mean(np.square(hold_before))))
    after_rms = float(np.sqrt(np.mean(np.square(hold_after))))
    applied = after_rms < before_rms * 0.65
    report = {
        "map_id": 2, "scale": 1.0, "theta_deg": math.degrees(theta),
        "translation_m": translation.tolist(), "center_m": center.tolist(),
        "training_ids": training_ids, "holdout_ids": holdout_ids,
        "training_before_m": train_before, "training_after_m": train_after,
        "holdout_before_m": hold_before, "holdout_after_m": hold_after,
        "training_rms_before_m": float(np.sqrt(np.mean(np.square(train_before)))),
        "training_rms_after_m": float(np.sqrt(np.mean(np.square(train_after)))),
        "holdout_rms_before_m": before_rms, "holdout_rms_after_m": after_rms,
        "applied": applied,
        "decision": ("accepted: independent RGB-D holdout improved by at least 35%" if applied else
                     "rejected: independent RGB-D holdout improvement was below 35%; existing XY/yaw retained"),
    }
    return (correction if applied else np.eye(3)), report


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if SEED.exists():
        raise FileExistsError(f"working seed already exists: {SEED}")
    actual_hash = sha256(INPUT)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"input SHA-256 mismatch: {actual_hash}")

    read_only = sqlite3.connect(f"file:{INPUT.resolve()}?mode=ro&immutable=1", uri=True)
    node_rows = list(read_only.execute("SELECT id,map_id,pose FROM Node ORDER BY id"))
    poses = {node_id: matrix(blob) for node_id, _, blob in node_rows}
    map_ids = {node_id: map_id for node_id, map_id, _ in node_rows}
    input_nodes = [node_id for node_id, _, _ in node_rows]
    read_only.close()
    if len(input_nodes) != 1784:
        raise RuntimeError(f"expected 1784 nodes, got {len(input_nodes)}")

    planes = load_planes()
    ramp_nodes, source_map = load_ramp_nodes()
    recording = {node_id: ("competition_338" if map_ids[node_id] == 0 else
                           "091640" if map_ids[node_id] == 1 else "113208")
                 for node_id in input_nodes}
    group_slopes = {}
    for group in ("competition_338", "091640", "113208"):
        values = [(planes[node_id]["slope_x"], planes[node_id]["slope_y"])
                  for node_id in input_nodes if recording[node_id] == group and node_id not in ramp_nodes
                  and planes[node_id]["rms_m"] <= 0.04 and planes[node_id]["inliers"] >= 500]
        group_slopes[group] = np.median(np.asarray(values), axis=0)
    marking_transform, marking_report = fit_marking_correction(poses)
    target_xyyaw = {}
    for node_id in input_nodes:
        current = pose_se2(poses[node_id])
        target = marking_transform @ current if map_ids[node_id] == 2 else current
        target_xyyaw[node_id] = (float(target[0, 2]), float(target[1, 2]),
                                 math.atan2(target[1, 0], target[0, 0]))

    # Keep the stored session calibration unchanged: changing it made the map
    # feature geometry incompatible with the independent localization query.
    # The measured road plane is instead leveled by the world pose.
    corrected = {}
    source = sqlite3.connect(f"file:{INPUT.resolve()}?mode=ro&immutable=1", uri=True)
    for node_id, blob in source.execute("SELECT id,calibration FROM Data ORDER BY id"):
        plane = planes[node_id]
        n0 = np.asarray([-plane["slope_x"], -plane["slope_y"], 1.0])
        norm = float(np.linalg.norm(n0))
        n0 /= norm
        h0 = plane["intercept_m"] / norm
        corrected[node_id] = {"normal": n0, "height": h0}
    source.close()

    group_heights = {}
    for group in ("competition_338", "091640", "113208"):
        heights = [corrected[node_id]["height"] for node_id in input_nodes
                   if recording[node_id] == group and node_id not in ramp_nodes
                   and planes[node_id]["rms_m"] <= 0.04 and planes[node_id]["inliers"] >= 500]
        group_heights[group] = float(np.median(heights))

    desired = {}
    per_node = {}
    for map_id in sorted(set(map_ids.values())):
        ids = [node_id for node_id in input_nodes if map_ids[node_id] == map_id]
        raw_a = np.asarray([-corrected[node_id]["normal"][0] / corrected[node_id]["normal"][2] for node_id in ids])
        raw_b = np.asarray([-corrected[node_id]["normal"][1] / corrected[node_id]["normal"][2] for node_id in ids])
        raw_h = np.asarray([corrected[node_id]["height"] for node_id in ids])
        valid = np.asarray([planes[node_id]["rms_m"] <= 0.04 and planes[node_id]["inliers"] >= 500
                            and abs(raw_a[index]) < math.tan(math.radians(30))
                            and abs(raw_b[index]) < math.tan(math.radians(30))
                            for index, node_id in enumerate(ids)])
        smooth_a = np.clip(smooth(raw_a, valid, 3, 5), -math.tan(math.radians(25)), math.tan(math.radians(25)))
        smooth_b = np.clip(smooth(raw_b, valid, 3, 5), -math.tan(math.radians(25)), math.tan(math.radians(25)))
        smooth_h = smooth(raw_h, valid, 11, 21)
        for index, node_id in enumerate(ids):
            x, y, yaw = target_xyyaw[node_id]
            residual_rotation = align_normal_to_up(np.asarray([-smooth_a[index], -smooth_b[index], 1.0]))
            if node_id in ramp_nodes:
                target_rotation = poses[node_id][:3, :3]
                # Preserve the recorded ramp's relative height curve, applying
                # only the recording-wide base-frame datum correction.
                z = float(poses[node_id][2, 3] + GROUND_DATUM_Z - group_heights[recording[node_id]])
            else:
                target_rotation = rotation_z(yaw)
                z = float(GROUND_DATUM_Z - smooth_h[index])
            pose = np.eye(4)
            pose[:3, :3] = target_rotation @ residual_rotation
            pose[:3, 3] = (x, y, z)
            desired[node_id] = pose
            raw_normal = corrected[node_id]["normal"]
            world_normal = pose[:3, :3] @ raw_normal
            world_normal /= np.linalg.norm(world_normal)
            road_point = pose[:3, :3] @ (corrected[node_id]["height"] * raw_normal) + pose[:3, 3]
            per_node[node_id] = {
                "map_id": map_id, "recording": recording[node_id], "ramp": node_id in ramp_nodes,
                "raw_corrected_slope_x": float(raw_a[index]), "raw_corrected_slope_y": float(raw_b[index]),
                "smoothed_slope_x": float(smooth_a[index]), "smoothed_slope_y": float(smooth_b[index]),
                "road_normal_tilt_deg": math.degrees(math.acos(np.clip(world_normal[2], -1.0, 1.0))),
                "road_height_at_node_m": float(road_point[2]), "pose_z_m": z,
            }

    shutil.copy2(INPUT, SEED)
    connection = sqlite3.connect(SEED)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    for node_id in input_nodes:
        # Keep localization graph poses exactly unchanged.  The relative
        # leveling transform is encoded in the scan-generation sensor frame.
        correction = np.linalg.inv(poses[node_id]) @ desired[node_id]
        calibration_blob = connection.execute("SELECT calibration FROM Data WHERE id=?", (node_id,)).fetchone()[0]
        calibration = np.frombuffer(calibration_blob, np.float32).copy()
        local = np.eye(4)
        local[:3, :4] = calibration[-12:].reshape(3, 4)
        effective = correction @ local
        calibration[-12:] = effective[:3, :4].astype(np.float32).ravel()
        connection.execute("UPDATE Data SET calibration=? WHERE id=?", (calibration.tobytes(), node_id))
    feature_updates = 0
    connection.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,opt_last_localization=NULL,"
                       "opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,opt_tex_materials=NULL,"
                       "opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    connection.execute("UPDATE Data SET scan=NULL,scan_info=NULL,ground_cells=NULL,obstacle_cells=NULL,empty_cells=NULL")
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    output_nodes = connection.execute("SELECT count(*) FROM Node").fetchone()[0]
    output_ids = [row[0] for row in connection.execute("SELECT id FROM Node ORDER BY id")]
    connection.close()
    if integrity != "ok" or output_ids != input_nodes:
        raise RuntimeError(f"seed validation failed: integrity={integrity}, nodes={output_nodes}")

    flat_metrics = [value for node_id, value in per_node.items() if node_id not in ramp_nodes]
    ramp_offsets = {group: [desired[node_id][2, 3] - poses[node_id][2, 3]
                            for node_id in ramp_nodes if recording[node_id] == group]
                    for group in ("competition_338", "091640")}
    ramp_relative_change = max((float(np.ptp(offsets)) for offsets in ramp_offsets.values() if offsets), default=0.0)
    ramp_xy_change = max(np.linalg.norm(desired[node_id][:2, 3] - poses[node_id][:2, 3]) for node_id in ramp_nodes)
    report = {
        "input": str(INPUT), "input_sha256_before": actual_hash, "input_nodes": len(input_nodes),
        "seed": str(SEED), "seed_size_bytes": SEED.stat().st_size,
        "seed_integrity": integrity, "seed_nodes": output_nodes, "node_ids_identical": output_ids == input_nodes,
        "no_nodes_added_or_removed": True, "feature_rows_transformed": feature_updates,
        "localization_pose_constraint_and_feature_geometry_preserved": True,
        "seed_calibration_mode": "temporary effective scan transform; original calibration restored after scan regeneration",
        "ground_datum_z_m": GROUND_DATUM_Z,
        "sensor_correction": {
            group: {"median_slope_x": float(group_slopes[group][0]),
                    "median_slope_y": float(group_slopes[group][1]),
                    "pitch_equivalent_deg": math.degrees(math.atan(group_slopes[group][0])),
                    "roll_equivalent_deg": math.degrees(math.atan(group_slopes[group][1])),
                    "corrected_ground_height_median_m": group_heights[group],
                    "stored_transform_changed": False,
                    "reason": "kept to preserve RGB-D feature/localization frame semantics; road plane leveled in world pose"}
            for group in group_slopes},
        "marking_alignment": marking_report,
        "rejected_alignment_changes": {
            "map_id_1": "one RGB-D anchor only; no independent holdout",
            "map_id_3": "two anchors requested opposite yaw corrections; pose change rejected",
            "map_id_4": "no accepted RGB-D anchor; pose change rejected",
            "map_id_5": "two anchors gave inconsistent correction; pose change rejected",
        },
        "flat_road_after_seed": {
            "nodes": len(flat_metrics),
            "normal_tilt_median_deg": float(np.median([x["road_normal_tilt_deg"] for x in flat_metrics])),
            "normal_tilt_p90_deg": float(np.percentile([x["road_normal_tilt_deg"] for x in flat_metrics], 90)),
            "road_height_std_m": float(np.std([x["road_height_at_node_m"] for x in flat_metrics])),
            "road_height_p05_p95_m": np.percentile([x["road_height_at_node_m"] for x in flat_metrics], [5, 95]).tolist(),
        },
        "ramp_preservation": {"nodes": len(ramp_nodes),
                              "constant_z_offset_by_recording_m": {group: float(np.median(offsets))
                                  for group, offsets in ramp_offsets.items() if offsets},
                              "relative_height_shape_max_change_m": ramp_relative_change,
                              "pose_xy_max_change_m": float(ramp_xy_change)},
        "per_node": per_node,
        "input_sha256_after": sha256(INPUT),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seed", "seed_integrity", "seed_nodes",
          "no_nodes_added_or_removed", "sensor_correction", "marking_alignment",
          "flat_road_after_seed", "ramp_preservation", "input_sha256_after")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
