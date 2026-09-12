#!/usr/bin/env python3
"""Restore localization geometry after corrected RGB-D scan regeneration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
TARGET = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
REPORT = TARGET.parent / "work/localization_geometry_restore.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = sqlite3.connect(f"file:{SOURCE.resolve()}?mode=ro&immutable=1", uri=True)
    target = sqlite3.connect(TARGET)
    source_nodes = list(source.execute("SELECT id,pose,ground_truth_pose FROM Node ORDER BY id"))
    target_ids = [row[0] for row in target.execute("SELECT id FROM Node ORDER BY id")]
    if target_ids != [row[0] for row in source_nodes]:
        raise RuntimeError("node IDs changed during scan regeneration")
    for node_id, pose, ground_truth in source_nodes:
        target.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?",
                       (pose, ground_truth, node_id))
    for node_id, calibration in source.execute("SELECT id,calibration FROM Data ORDER BY id"):
        target.execute("UPDATE Data SET calibration=? WHERE id=?", (calibration, node_id))
    links = list(source.execute("SELECT from_id,to_id,type,transform FROM Link ORDER BY rowid"))
    target_links = list(target.execute("SELECT rowid,from_id,to_id,type FROM Link ORDER BY rowid"))
    if [(a,b,t) for _,a,b,t in target_links] != [(a,b,t) for a,b,t,_ in links]:
        raise RuntimeError("link rows changed during scan regeneration")
    for (rowid, _, _, _), (_, _, _, transform) in zip(target_links, links):
        target.execute("UPDATE Link SET transform=? WHERE rowid=?", (transform, rowid))
    # rtabmap-reprocess re-quantizes the visual vocabulary even when odometry
    # features are reused. Restore the exact input vocabulary and feature rows
    # so the localization map remains byte-for-byte compatible with its query.
    target.execute("DELETE FROM Feature")
    target.execute("DELETE FROM Word")
    word_rows = source.execute("SELECT id,descriptor_size,descriptor,time_enter FROM Word ORDER BY id")
    target.executemany("INSERT INTO Word(id,descriptor_size,descriptor,time_enter) VALUES(?,?,?,?)", word_rows)
    feature_rows = source.execute("SELECT node_id,word_id,pos_x,pos_y,size,dir,response,octave,"
                                  "depth_x,depth_y,depth_z,descriptor_size,descriptor FROM Feature ORDER BY rowid")
    target.executemany("INSERT INTO Feature(node_id,word_id,pos_x,pos_y,size,dir,response,octave,"
                       "depth_x,depth_y,depth_z,descriptor_size,descriptor) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       feature_rows)
    target.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,opt_last_localization=NULL,"
                   "opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,opt_tex_materials=NULL,"
                   "opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    target.commit()
    integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    counts = target.execute("SELECT count(*),sum(image is not null),sum(depth is not null),"
                            "sum(calibration is not null),sum(scan is not null) FROM Data").fetchone()
    feature_counts = target.execute("SELECT count(distinct node_id),count(*) FROM Feature").fetchone()
    target.close()
    source.close()
    report = {
        "source": str(SOURCE), "target": str(TARGET),
        "restored": ["Node.pose", "Node.ground_truth_pose", "Link.transform", "Data.calibration",
                     "Word", "Feature"],
        "retained_regenerated": ["Data.scan", "Data.scan_info", "Data.ground_cells",
                                  "Data.obstacle_cells", "Data.empty_cells"],
        "integrity": integrity, "data_counts": counts, "feature_counts": feature_counts,
        "source_sha256_after": sha(SOURCE),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
