#!/usr/bin/env python3
"""Build a v7 seed with one consistent virtual sensor frame.

Node poses and Links remain byte-identical to v5.  The robust road-level
correction computed for v6 is encoded in calibration, Feature 3-D coordinates,
and subsequently regenerated scans.  RGB/depth pixels and descriptors remain
unchanged.  This avoids v5's scan-only frame mismatch without moving the graph.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
V6_SEED = ROOT / "maps/merged_competition_level_aligned_v6/work/coordinate_consistent_seed.db"
OUTPUT_DIR = ROOT / "maps/merged_competition_level_aligned_v7"
WORK = OUTPUT_DIR / "work"
SEED = WORK / "virtual_sensor_seed.db"
REPORT = WORK / "v7_build_report.json"
EXPECTED_SHA = "a3a375f34f0cce14b4925544632f9ca0fdfb8c24f845cde9f975a194fd54d990"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mat(blob: bytes) -> np.ndarray:
    output = np.eye(4); output[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4); return output


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT_DIR}")
    if sha(INPUT) != EXPECTED_SHA:
        raise RuntimeError("v5 SHA-256 mismatch")
    OUTPUT_DIR.mkdir(parents=True); WORK.mkdir()
    old = sqlite3.connect(f"file:{INPUT.resolve()}?mode=ro&immutable=1", uri=True)
    leveled = sqlite3.connect(f"file:{V6_SEED.resolve()}?mode=ro&immutable=1", uri=True)
    old_pose = {i: mat(p) for i, p in old.execute("SELECT id,pose FROM Node ORDER BY id")}
    new_pose = {i: mat(p) for i, p in leveled.execute("SELECT id,pose FROM Node ORDER BY id")}
    if old_pose.keys() != new_pose.keys():
        raise RuntimeError("node IDs differ")
    corrections = {i: np.linalg.inv(old_pose[i]) @ new_pose[i] for i in old_pose}
    leveled.close(); old.close()

    shutil.copy2(INPUT, SEED)
    connection = sqlite3.connect(SEED)
    connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF")
    calibration_updates = 0
    for node_id, calibration_blob in connection.execute("SELECT id,calibration FROM Data ORDER BY id"):
        calibration = np.frombuffer(calibration_blob, np.float32).copy()
        local = np.eye(4); local[:3, :4] = calibration[-12:].reshape(3, 4)
        effective = corrections[node_id] @ local
        calibration[-12:] = effective[:3, :4].astype(np.float32).ravel()
        connection.execute("UPDATE Data SET calibration=? WHERE id=?", (calibration.tobytes(), node_id))
        calibration_updates += 1
    feature_updates = 0
    cursor = connection.execute("SELECT rowid,node_id,depth_x,depth_y,depth_z FROM Feature ORDER BY rowid")
    batch = []
    while True:
        rows = cursor.fetchmany(20000)
        if not rows:
            break
        for rowid, node_id, x, y, z in rows:
            if x is None or y is None or z is None:
                continue
            point = corrections[node_id] @ np.asarray([x, y, z, 1.0])
            batch.append((float(point[0]), float(point[1]), float(point[2]), rowid))
        connection.executemany("UPDATE Feature SET depth_x=?,depth_y=?,depth_z=? WHERE rowid=?", batch)
        feature_updates += len(batch); batch.clear()
    connection.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
                       "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,"
                       "opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    connection.execute("UPDATE Data SET scan=NULL,scan_info=NULL,ground_cells=NULL,obstacle_cells=NULL,empty_cells=NULL")
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    nodes = connection.execute("SELECT count(*) FROM Node").fetchone()[0]
    features = connection.execute("SELECT count(*) FROM Feature").fetchone()[0]
    connection.close()
    report = {"input": str(INPUT), "input_sha256_before": EXPECTED_SHA, "seed": str(SEED),
              "integrity": integrity, "nodes": nodes, "calibration_updates": calibration_updates,
              "feature_3d_updates": feature_updates, "features": features,
              "node_pose_changed": False, "links_changed": False, "rgb_depth_descriptor_changed": False,
              "frame_model": "virtual level sensor frame shared by calibration, Feature.depth_xyz and regenerated scan",
              "v6_pose_graph_attempt": "rejected: 129/273 localization",
              "input_sha256_after": sha(INPUT)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
