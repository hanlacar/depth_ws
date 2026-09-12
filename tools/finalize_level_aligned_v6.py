#!/usr/bin/env python3
"""Finalize v6 without reverting its coordinate-consistent pose graph."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
SEED = ROOT / "maps/merged_competition_level_aligned_v6/work/coordinate_consistent_seed.db"
TARGET = ROOT / "maps/merged_competition_level_aligned_v6/rtabmap.db"
REPORT = TARGET.parent / "work/v6_finalize_report.json"


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
    seed_nodes = list(seed.execute("SELECT id,pose,ground_truth_pose FROM Node ORDER BY id"))
    target_ids = [row[0] for row in target.execute("SELECT id FROM Node ORDER BY id")]
    if target_ids != [row[0] for row in seed_nodes]:
        raise RuntimeError("node IDs changed during reprocessing")
    for node_id, pose, ground_truth in seed_nodes:
        target.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?", (pose, ground_truth, node_id))
    # Preserve original node identity/working-memory metadata.  Reprocess had
    # marked node 8 as intermediate (-9), which truncated optimized export at
    # node 6 even though the SQL Link graph was connected.
    for row in original.execute("SELECT id,map_id,weight,stamp,velocity,label,gps,env_sensors,time_enter "
                                "FROM Node ORDER BY id"):
        node_id, map_id, weight, stamp, velocity, label, gps, env, entered = row
        target.execute("UPDATE Node SET map_id=?,weight=?,stamp=?,velocity=?,label=?,gps=?,env_sensors=?,time_enter=? "
                       "WHERE id=?", (map_id, weight, stamp, velocity, label, gps, env, entered, node_id))
    # Reprocess replaced the sparse 6-8-19 neighbor chain with 6-19.  Restore
    # the complete, gauge-transformed RTAB-Map Link rows from the validated
    # seed, including information matrices and user data.
    seed_links = list(seed.execute("SELECT from_id,to_id,type,information_matrix,transform,user_data "
                                   "FROM Link ORDER BY rowid"))
    target.execute("DELETE FROM Link")
    target.executemany("INSERT INTO Link(from_id,to_id,type,information_matrix,transform,user_data) "
                       "VALUES(?,?,?,?,?,?)", seed_links)
    for node_id, calibration in original.execute("SELECT id,calibration FROM Data ORDER BY id"):
        target.execute("UPDATE Data SET calibration=? WHERE id=?", (calibration, node_id))
    target.execute("DELETE FROM Feature"); target.execute("DELETE FROM Word")
    target.executemany("INSERT INTO Word(id,descriptor_size,descriptor,time_enter) VALUES(?,?,?,?)",
                       original.execute("SELECT id,descriptor_size,descriptor,time_enter FROM Word ORDER BY id"))
    target.executemany("INSERT INTO Feature(node_id,word_id,pos_x,pos_y,size,dir,response,octave,"
                       "depth_x,depth_y,depth_z,descriptor_size,descriptor) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       original.execute("SELECT node_id,word_id,pos_x,pos_y,size,dir,response,octave,"
                                        "depth_x,depth_y,depth_z,descriptor_size,descriptor FROM Feature ORDER BY rowid"))
    target.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
                   "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,"
                   "opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    target.commit()

    max_local_difference = 0.0
    for calibration, scan_info in target.execute("SELECT calibration,scan_info FROM Data ORDER BY id"):
        c = np.frombuffer(calibration, np.float32)[-12:]
        s = np.frombuffer(scan_info, np.float32)[7:19]
        max_local_difference = max(max_local_difference, float(np.max(np.abs(c - s))))
    integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    counts = target.execute("SELECT count(*),sum(image IS NOT NULL),sum(depth IS NOT NULL),"
                            "sum(calibration IS NOT NULL),sum(scan IS NOT NULL) FROM Data").fetchone()
    features = target.execute("SELECT count(distinct node_id),count(*) FROM Feature").fetchone()
    target.close(); seed.close(); original.close()
    report = {"target": str(TARGET), "integrity": integrity, "data_counts": counts,
              "feature_counts": features, "scan_info_vs_calibration_max_matrix_difference": max_local_difference,
              "restored_from_v5": ["Data.calibration", "Word", "Feature"],
              "restored_from_v6_seed": ["Node.pose", "Node.ground_truth_pose", "complete Link rows"],
              "retained_from_reprocess": ["Data.scan", "Data.scan_info", "Data.ground_cells",
                                           "Data.obstacle_cells", "Data.empty_cells", "Statistics"],
              "input_sha256_after": sha(INPUT), "seed_sha256_after": sha(SEED)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
