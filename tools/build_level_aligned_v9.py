#!/usr/bin/env python3
"""Build v9 by correcting only the split yellow-line observation submap.

The correction is a scale-1 gauge transform of a short map_id=3 interval.
Node poses and every incident Link measurement are transformed together.  RGB-D,
features, calibration, scans and scan local transforms are retained byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import struct
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
OUTPUT = ROOT / "maps/merged_competition_level_aligned_v9/rtabmap.db"
REPORT = ROOT / "maps/merged_competition_level_aligned_v9/work/v9_build_report.json"

# The line fit is based on actual yellow RGB-D scan points.  The source fit is
# map_id=3 nodes 1641..1656; the reference fit combines the long competition
# approach and the independent 113208 return observation.
PIVOT = np.asarray([22.43784325, 29.5])
TARGET_AT_PIVOT_Y = np.asarray([21.94919120, 29.5])
FULL_THETA = math.radians(-1.8059808647418325)
FULL_DZ = -0.003274


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(blob: bytes) -> np.ndarray:
    value = np.eye(4)
    value[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return value


def blob(value: np.ndarray) -> bytes:
    return struct.pack("<12f", *value[:3, :4].astype(np.float32).ravel())


def partial_world_transform(weight: float) -> np.ndarray:
    theta = FULL_THETA * weight
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]])
    shift = (TARGET_AT_PIVOT_Y - PIVOT) * weight
    translation = PIVOT + shift - rotation @ PIVOT
    value = np.eye(4)
    value[:2, :2] = rotation
    value[:2, 3] = translation
    value[2, 3] = FULL_DZ * weight
    return value


def digest_rows(connection: sqlite3.Connection, query: str) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(query):
        for value in row:
            digest.update(value if isinstance(value, bytes) else repr(value).encode())
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True)

    source_hash_before = sha256(SOURCE)
    shutil.copy2(SOURCE, OUTPUT)
    db = sqlite3.connect(OUTPUT)
    db.execute("PRAGMA foreign_keys=OFF")

    ids = [row[0] for row in db.execute(
        "SELECT id FROM Node WHERE map_id=3 AND id BETWEEN 1629 AND 1664 ORDER BY id")]
    if ids != [1629, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639,
               1640, 1641, 1642, 1643, 1647, 1648, 1651, 1652, 1653, 1654,
               1655, 1656, 1657, 1658, 1659, 1660, 1662, 1664]:
        raise RuntimeError(f"unexpected target node set: {ids}")

    full_start = ids.index(1641)
    full_stop = ids.index(1656)
    weights = {}
    for index, node_id in enumerate(ids):
        if index < full_start:
            weights[node_id] = index / full_start
        elif index <= full_stop:
            weights[node_id] = 1.0
        else:
            weights[node_id] = (len(ids) - 1 - index) / (len(ids) - 1 - full_stop)

    before_digests = {
        "rgb_depth_calibration_scan": digest_rows(
            db, "SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"),
        "feature_word": digest_rows(db, "SELECT * FROM Feature ORDER BY rowid"),
    }
    old_poses = {node_id: matrix(pose) for node_id, pose in db.execute(
        "SELECT id,pose FROM Node ORDER BY id")}
    corrections = {}
    changes = []
    for node_id in ids:
        world = partial_world_transform(weights[node_id])
        old = old_poses[node_id]
        new = world @ old
        corrections[node_id] = np.linalg.inv(old) @ new
        gt_blob = db.execute("SELECT ground_truth_pose FROM Node WHERE id=?", (node_id,)).fetchone()[0]
        if gt_blob:
            new_gt = world @ matrix(gt_blob)
            db.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?",
                       (blob(new), blob(new_gt), node_id))
        else:
            db.execute("UPDATE Node SET pose=? WHERE id=?", (blob(new), node_id))
        changes.append({
            "id": node_id, "weight": weights[node_id],
            "before_xyz": old[:3, 3].tolist(), "after_xyz": new[:3, 3].tolist(),
            "xy_shift_m": float(np.linalg.norm(new[:2, 3] - old[:2, 3])),
            "yaw_shift_deg": math.degrees(FULL_THETA * weights[node_id]),
        })

    changed_links = 0
    for rowid, from_id, to_id, transform_blob in list(db.execute(
        "SELECT rowid,from_id,to_id,transform FROM Link ORDER BY rowid")):
        if from_id not in corrections and to_id not in corrections:
            continue
        c_from = corrections.get(from_id, np.eye(4))
        c_to = corrections.get(to_id, np.eye(4))
        transformed = np.linalg.inv(c_from) @ matrix(transform_blob) @ c_to
        db.execute("UPDATE Link SET transform=? WHERE rowid=?", (blob(transformed), rowid))
        changed_links += 1

    # Stale optimized display cache is removed; payload data are untouched and a
    # fresh official export will be saved after validation.
    db.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
               "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
               "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
               "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    db.commit()
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    after_digests = {
        "rgb_depth_calibration_scan": digest_rows(
            db, "SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"),
        "feature_word": digest_rows(db, "SELECT * FROM Feature ORDER BY rowid"),
    }
    db.close()

    report = {
        "source": str(SOURCE), "source_sha256_before": source_hash_before,
        "output": str(OUTPUT), "integrity": integrity,
        "scale": 1.0, "line_split_bbox_map_xy": [21.7, 27.8, 22.65, 31.2],
        "full_correction": {
            "theta_deg": math.degrees(FULL_THETA),
            "rotation_pivot_xy_m": PIVOT.tolist(),
            "pivot_translation_xy_m": (TARGET_AT_PIVOT_Y - PIVOT).tolist(),
            "equivalent_world_translation_xy_m": partial_world_transform(1.0)[:2, 3].tolist(),
            "z_translation_m": FULL_DZ,
        },
        "full_weight_nodes": [node_id for node_id in ids if weights[node_id] == 1.0],
        "taper_nodes": [node_id for node_id in ids if 0.0 < weights[node_id] < 1.0],
        "zero_weight_boundary_nodes": [node_id for node_id in ids if weights[node_id] == 0.0],
        "modified_pose_nodes": [node_id for node_id in ids if weights[node_id] > 0.0],
        "changed_link_rows": changed_links, "changes": changes,
        "payload_digests_identical": before_digests == after_digests,
        "payload_digests_before": before_digests, "payload_digests_after": after_digests,
        "source_sha256_after": sha256(SOURCE), "output_sha256": sha256(OUTPUT),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
