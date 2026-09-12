#!/usr/bin/env python3
"""Build a coordinate-consistent level-aligned RTAB-Map v6 seed.

The v5 displayed scans in corrected local frames while localization poses and
features stayed in the original base frame.  This script moves the same rigid
road-level correction into Node.pose, gauge-transforms every Link, restores the
official camera frame meaning, and rejects local road-plane height outliers.
No node, RGB, depth, calibration, word or feature row is added or removed.
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
from scipy.signal import medfilt, savgol_filter


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
REFERENCE = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
OUTPUT_DIR = ROOT / "maps/merged_competition_level_aligned_v6"
WORK = OUTPUT_DIR / "work"
SEED = WORK / "coordinate_consistent_seed.db"
PLANES = ROOT / "analysis/level_aligned/depth_plane_before_detailed.csv"
RAMP_REPORT = ROOT / "maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json"
REPORT = WORK / "v6_build_report.json"
EXPECTED_INPUT_SHA = "a3a375f34f0cce14b4925544632f9ca0fdfb8c24f845cde9f975a194fd54d990"
EXPECTED_REFERENCE_SHA = "b8bff86d58c60663df910ff053b43d91d33c6bd9076d4251d7f992637de44e86"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(blob: bytes) -> np.ndarray:
    output = np.eye(4)
    output[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return output


def blob(transform: np.ndarray) -> bytes:
    return struct.pack("<12f", *transform[:3, :4].astype(np.float32).ravel())


def rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_normal_to_up(normal: np.ndarray) -> np.ndarray:
    source = normal / np.linalg.norm(normal)
    target = np.asarray([0.0, 0.0, 1.0])
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis)); cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3)
    axis /= sine
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(math.atan2(sine, cosine)) * skew + \
        (1.0 - cosine) * (skew @ skew)


def smooth(values: np.ndarray, valid: np.ndarray, median_window: int, sg_window: int) -> np.ndarray:
    index = np.arange(len(values))
    if valid.sum() < 2:
        return np.zeros_like(values)
    output = np.interp(index, index[valid], values[valid])
    mw = min(median_window, len(values) if len(values) % 2 else len(values) - 1)
    if mw >= 3:
        output = medfilt(output, mw)
    sw = min(sg_window, len(values) if len(values) % 2 else len(values) - 1)
    if sw >= 5:
        output = savgol_filter(output, sw, 2, mode="interp")
    return output


def local_median(values: np.ndarray, window: int = 11) -> np.ndarray:
    size = min(window, len(values) if len(values) % 2 else len(values) - 1)
    return medfilt(values, size) if size >= 3 else values.copy()


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT_DIR}")
    if sha256(INPUT) != EXPECTED_INPUT_SHA:
        raise RuntimeError("v5 input SHA-256 mismatch")
    if sha256(REFERENCE) != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("gap-filled reference SHA-256 mismatch")
    OUTPUT_DIR.mkdir(parents=True); WORK.mkdir()

    planes = {int(row["id"]): {key: float(value) for key, value in row.items() if key != "id"}
              for row in csv.DictReader(PLANES.open())}
    ramp_data = json.loads(RAMP_REPORT.read_text())
    source_map = {int(key): value for key, value in ramp_data["node_source_map"].items()}
    ramp_nodes = set()
    for node_id, (source, source_id) in source_map.items():
        if source == "competition_338" and 115 <= source_id <= 210:
            ramp_nodes.add(node_id)
        if source == "091640" and 345 <= source_id <= 630:
            ramp_nodes.add(node_id)

    source = sqlite3.connect(f"file:{INPUT.resolve()}?mode=ro&immutable=1", uri=True)
    node_rows = list(source.execute("SELECT id,map_id,pose FROM Node ORDER BY id"))
    poses = {node_id: matrix(pose) for node_id, _, pose in node_rows}
    map_ids = {node_id: map_id for node_id, map_id, _ in node_rows}
    node_ids = [node_id for node_id, _, _ in node_rows]
    source.close()
    if len(node_ids) != 1784:
        raise RuntimeError(f"expected 1784 nodes, got {len(node_ids)}")

    recording = {node_id: ("competition_338" if map_ids[node_id] == 0 else
                           "091640" if map_ids[node_id] == 1 else "113208") for node_id in node_ids}
    corrected = {}
    for node_id in node_ids:
        plane = planes[node_id]
        normal = np.asarray([-plane["slope_x"], -plane["slope_y"], 1.0])
        norm = float(np.linalg.norm(normal)); normal /= norm
        corrected[node_id] = {"normal": normal, "height": plane["intercept_m"] / norm}

    group_height = {}
    for group in ("competition_338", "091640", "113208"):
        values = [corrected[node_id]["height"] for node_id in node_ids
                  if recording[node_id] == group and node_id not in ramp_nodes
                  and planes[node_id]["rms_m"] <= 0.04 and planes[node_id]["inliers"] >= 500]
        group_height[group] = float(np.median(values))

    desired = {}; per_node = {}; rejected = []
    for map_id in sorted(set(map_ids.values())):
        ids = [node_id for node_id in node_ids if map_ids[node_id] == map_id]
        raw_a = np.asarray([-corrected[node_id]["normal"][0] / corrected[node_id]["normal"][2] for node_id in ids])
        raw_b = np.asarray([-corrected[node_id]["normal"][1] / corrected[node_id]["normal"][2] for node_id in ids])
        raw_h = np.asarray([corrected[node_id]["height"] for node_id in ids])
        basic = np.asarray([planes[node_id]["rms_m"] <= 0.04 and planes[node_id]["inliers"] >= 500
                            and abs(raw_a[index]) < math.tan(math.radians(30))
                            and abs(raw_b[index]) < math.tan(math.radians(30))
                            for index, node_id in enumerate(ids)])
        med_a, med_b, med_h = local_median(raw_a), local_median(raw_b), local_median(raw_h)
        robust = basic & (np.abs(raw_h - med_h) <= 0.08) & \
            (np.hypot(raw_a - med_a, raw_b - med_b) <= math.tan(math.radians(8)))
        for index, node_id in enumerate(ids):
            if basic[index] and not robust[index]:
                rejected.append({"node": node_id, "map_id": map_id,
                                 "height_residual_m": float(raw_h[index] - med_h[index]),
                                 "slope_residual": float(np.hypot(raw_a[index] - med_a[index], raw_b[index] - med_b[index]))})
        smooth_a = np.clip(smooth(raw_a, robust, 5, 9), -math.tan(math.radians(25)), math.tan(math.radians(25)))
        smooth_b = np.clip(smooth(raw_b, robust, 5, 9), -math.tan(math.radians(25)), math.tan(math.radians(25)))
        smooth_h = smooth(raw_h, robust, 15, 25)
        for index, node_id in enumerate(ids):
            old = poses[node_id]
            yaw = math.atan2(old[1, 0], old[0, 0])
            if node_id in ramp_nodes:
                target_rotation = old[:3, :3]
                z = float(old[2, 3] - group_height[recording[node_id]])
            else:
                normal = np.asarray([-smooth_a[index], -smooth_b[index], 1.0])
                target_rotation = rotation_z(yaw) @ align_normal_to_up(normal)
                z = float(-smooth_h[index])
            target = np.eye(4); target[:3, :3] = target_rotation
            target[:3, 3] = (old[0, 3], old[1, 3], z)
            desired[node_id] = target
            world_normal = target[:3, :3] @ corrected[node_id]["normal"]
            world_normal /= np.linalg.norm(world_normal)
            road_point = target[:3, :3] @ (corrected[node_id]["height"] * corrected[node_id]["normal"]) + target[:3, 3]
            per_node[node_id] = {"map_id": map_id, "ramp": node_id in ramp_nodes,
                                 "robust_plane": bool(robust[index]),
                                 "road_height_m": float(road_point[2]),
                                 "road_tilt_deg": math.degrees(math.acos(np.clip(world_normal[2], -1, 1))),
                                 "pose_z_m": z}

    shutil.copy2(INPUT, SEED)
    target = sqlite3.connect(SEED)
    target.execute("PRAGMA journal_mode=OFF"); target.execute("PRAGMA synchronous=OFF")
    corrections = {node_id: np.linalg.inv(poses[node_id]) @ desired[node_id] for node_id in node_ids}
    for node_id in node_ids:
        target.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?",
                       (blob(desired[node_id]), blob(desired[node_id]), node_id))
    links_changed = 0
    for rowid, from_id, to_id, old_blob in list(target.execute("SELECT rowid,from_id,to_id,transform FROM Link ORDER BY rowid")):
        if from_id in corrections and to_id in corrections:
            old_link = matrix(old_blob)
            new_link = np.linalg.inv(corrections[from_id]) @ old_link @ corrections[to_id]
            target.execute("UPDATE Link SET transform=? WHERE rowid=?", (blob(new_link), rowid))
            links_changed += 1
    target.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
                   "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,"
                   "opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    target.execute("UPDATE Data SET scan=NULL,scan_info=NULL,ground_cells=NULL,obstacle_cells=NULL,empty_cells=NULL")
    target.commit()
    integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    counts = dict(target.execute("SELECT map_id,count(*) FROM Node GROUP BY map_id"))
    target.close()
    if integrity != "ok" or sum(counts.values()) != 1784:
        raise RuntimeError(f"seed validation failed: {integrity}, {counts}")

    flat = [value for value in per_node.values() if not value["ramp"] and value["robust_plane"]]
    report = {
        "input": str(INPUT), "input_sha256_before": EXPECTED_INPUT_SHA,
        "reference": str(REFERENCE), "reference_sha256_before": EXPECTED_REFERENCE_SHA,
        "seed": str(SEED), "integrity": integrity, "node_count_by_map_id": counts,
        "nodes_added": 0, "nodes_removed": 0, "links_gauge_transformed": links_changed,
        "coordinate_model": "Node pose contains road-level correction; original calibration and local feature geometry retained; scans regenerated in official camera frame",
        "height_outliers_rejected": rejected,
        "flat_metrics_seed": {"nodes": len(flat),
            "height_std_m": float(np.std([x["road_height_m"] for x in flat])),
            "height_abs_max_m": float(np.max(np.abs([x["road_height_m"] for x in flat]))),
            "tilt_median_deg": float(np.median([x["road_tilt_deg"] for x in flat])),
            "tilt_p90_deg": float(np.percentile([x["road_tilt_deg"] for x in flat], 90))},
        "ramp_mask": {"nodes": len(ramp_nodes), "competition_source_transition": [115, 210],
                      "competition_core": [124, 200], "091640_source_transition": [345, 630],
                      "091640_core": [365, 600],
                      "review": "contact sheet confirms masks terminate on flat return; no arrival nodes 1841-1859 are masked"},
        "xy_yaw": {"applied": False, "scale": 1.0,
                    "reason": "session marking ICP proposed only 0.016-0.048 m corrections and independent holdouts did not consistently improve; no evidence-supported warp"},
        "problem_regions": {
            "intersection": {"node_ranges": [[1564, 1628], [1629, 1780]],
                             "bbox_xy": [[25.34, 16.87, 39.38, 21.43], [14.78, 32.67, 27.25, 47.10]]},
            "straight": {"node_range": [1781, 1840], "bbox_xy": [2.69, 21.33, 20.22, 37.92]},
            "arrival": {"node_range": [1841, 1859], "bbox_xy": [-0.96, 20.70, 2.94, 21.66]}},
        "per_node": per_node,
        "input_sha256_after": sha256(INPUT), "reference_sha256_after": sha256(REFERENCE)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seed", "integrity", "node_count_by_map_id",
          "links_gauge_transformed", "height_outliers_rejected", "flat_metrics_seed", "ramp_mask", "xy_yaw")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
