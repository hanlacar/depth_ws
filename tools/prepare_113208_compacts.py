#!/usr/bin/env python3
"""Apply the accepted scale-1 SE(2) to selected 113208 working DBs."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path

import numpy as np

ROOT = Path("/home/qor/depth_ws")
POSES = Path("/tmp/rtabmap_audit_poses/map_113208_poses.txt")
REPORT = ROOT / "analysis/full_course_premerge/premerge_report.json"
DATABASES = [
    ROOT / "maps/merged_competition_full_course/work/113208_corner_compact.db",
    ROOT / "maps/merged_competition_full_course/work/113208_intersection_compact.db",
]


def pose_blob(p):
    return struct.pack("<12f", *p[:3, :4].astype(np.float32).ravel())


def compose(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    p[:3, 3] = (x, y, 0)
    return p


def load():
    result = {}
    for line in POSES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        qx, qy, qz, qw = map(float, f[4:8])
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        result[int(f[8])] = (float(f[1]), float(f[2]), yaw)
    return result


def main():
    raw = load()
    fit = json.loads(REPORT.read_text())["fits"]["intersection"]
    theta = math.radians(fit["theta_deg"])
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]])
    translation = np.asarray(fit["translation_m"])
    summary = {"transform": fit, "databases": {}}
    for path in DATABASES:
        con = sqlite3.connect(path)
        ids = [r[0] for r in con.execute("SELECT id FROM Node ORDER BY id")]
        desired = {}
        for node_id in ids:
            x, y, yaw = raw[node_id]
            xy = np.asarray([x, y])@rotation.T+translation
            desired[node_id] = compose(float(xy[0]), float(xy[1]), yaw+theta)
            b = pose_blob(desired[node_id])
            con.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?", (b, b, node_id))
        for rowid, from_id, to_id in con.execute("SELECT rowid,from_id,to_id FROM Link"):
            if from_id in desired and to_id in desired:
                relative = np.linalg.inv(desired[from_id])@desired[to_id]
                con.execute("UPDATE Link SET transform=? WHERE rowid=?", (pose_blob(relative), rowid))
        con.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
                    "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
                    "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
                    "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
        con.commit()
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
        summary["databases"][str(path)] = {"nodes": len(ids), "id_min": min(ids),
                                             "id_max": max(ids), "integrity": integrity}
    out = ROOT / "maps/merged_competition_full_course/work/prepared_compacts.json"
    out.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
