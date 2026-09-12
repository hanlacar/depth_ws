#!/usr/bin/env python3
"""Restore graph identity while retaining v7's consistent virtual sensor data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
SEED = ROOT / "maps/merged_competition_level_aligned_v7/work/virtual_sensor_seed.db"
TARGET = ROOT / "maps/merged_competition_level_aligned_v7/rtabmap.db"
REPORT = TARGET.parent / "work/v7_finalize_report.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    original = sqlite3.connect(f"file:{INPUT.resolve()}?mode=ro&immutable=1", uri=True)
    seed = sqlite3.connect(f"file:{SEED.resolve()}?mode=ro&immutable=1", uri=True)
    target = sqlite3.connect(TARGET)

    original_nodes = list(original.execute("SELECT id,pose,ground_truth_pose,map_id,weight,stamp,velocity,label,gps,env_sensors,time_enter FROM Node ORDER BY id"))
    target_ids = [row[0] for row in target.execute("SELECT id FROM Node ORDER BY id")]
    if target_ids != [row[0] for row in original_nodes]:
        raise RuntimeError("node IDs changed during reprocessing")
    target.executemany(
        "UPDATE Node SET pose=?,ground_truth_pose=?,map_id=?,weight=?,stamp=?,velocity=?,label=?,gps=?,env_sensors=?,time_enter=? WHERE id=?",
        [(pose, gt, map_id, weight, stamp, velocity, label, gps, env, entered, node_id)
         for node_id, pose, gt, map_id, weight, stamp, velocity, label, gps, env, entered in original_nodes],
    )

    # The graph is deliberately unchanged from v5. Reprocessing may replace a
    # sparse neighbour edge, so restore every official Link row byte-for-byte.
    original_links = list(original.execute("SELECT from_id,to_id,type,information_matrix,transform,user_data FROM Link ORDER BY rowid"))
    target.execute("DELETE FROM Link")
    target.executemany("INSERT INTO Link(from_id,to_id,type,information_matrix,transform,user_data) VALUES(?,?,?,?,?,?)", original_links)

    # Reprocessing preserves RGB/depth pixels but can rewrite the vocabulary.
    # Retain v7 seed calibration and transformed 3-D features as a single
    # virtual sensor frame, while preserving descriptors and visual words.
    seed_calibration = list(seed.execute("SELECT id,calibration FROM Data ORDER BY id"))
    target.executemany("UPDATE Data SET calibration=? WHERE id=?", [(cal, node_id) for node_id, cal in seed_calibration])
    target.execute("DELETE FROM Feature")
    target.execute("DELETE FROM Word")
    target.executemany("INSERT INTO Word(id,descriptor_size,descriptor,time_enter) VALUES(?,?,?,?)",
                       seed.execute("SELECT id,descriptor_size,descriptor,time_enter FROM Word ORDER BY id"))
    target.executemany("INSERT INTO Feature(node_id,word_id,pos_x,pos_y,size,dir,response,octave,depth_x,depth_y,depth_z,descriptor_size,descriptor) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       seed.execute("SELECT node_id,word_id,pos_x,pos_y,size,dir,response,octave,depth_x,depth_y,depth_z,descriptor_size,descriptor FROM Feature ORDER BY rowid"))

    target.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    target.commit()

    max_local_difference = 0.0
    for calibration, scan_info in target.execute("SELECT calibration,scan_info FROM Data ORDER BY id"):
        c = np.frombuffer(calibration, np.float32)[-12:]
        s = np.frombuffer(scan_info, np.float32)[7:19]
        max_local_difference = max(max_local_difference, float(np.max(np.abs(c - s))))
    integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    counts = target.execute("SELECT count(*),sum(image IS NOT NULL),sum(depth IS NOT NULL),sum(calibration IS NOT NULL),sum(scan IS NOT NULL) FROM Data").fetchone()
    features = target.execute("SELECT count(distinct node_id),count(*) FROM Feature").fetchone()
    links = target.execute("SELECT count(*) FROM Link").fetchone()[0]
    target.close(); seed.close(); original.close()

    report = {
        "target": str(TARGET), "integrity": integrity, "data_counts": counts,
        "feature_counts": features, "links": links,
        "scan_info_vs_calibration_max_matrix_difference": max_local_difference,
        "restored_from_v5": ["Node all fields", "complete Link rows"],
        "restored_from_v7_seed": ["Data.calibration", "Word", "Feature including transformed depth_xyz"],
        "retained_from_reprocess": ["Data.scan", "Data.scan_info", "occupancy cells", "Statistics"],
        "input_sha256_after": sha(INPUT), "seed_sha256_after": sha(SEED),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
