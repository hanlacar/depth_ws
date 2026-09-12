#!/usr/bin/env python3
"""Apply the accepted unit-scale 113208 SE(2) to gap-fill compact DBs.

Only disposable compact databases below the gap-fill work directory are
modified.  Original recordings and the existing integrated map are read-only.
The candidate observations are on the flat return/intersection area, so their
Z, roll and pitch are placed on the existing flat datum while XY/yaw retain the
previously validated 113208 alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from pathlib import Path

import numpy as np

ROOT = Path("/home/qor/depth_ws")
POSES = Path("/tmp/rtabmap_audit_poses/map_113208_poses.txt")
DBS = [
    ROOT / "maps/merged_competition_gap_filled/work/113208_hold_1661_1881_compact.db",
    ROOT / "maps/merged_competition_gap_filled/work/113208_gap_2303_2399_compact.db",
    ROOT / "maps/merged_competition_gap_filled/work/113208_gap_2400_2803_compact.db",
    ROOT / "maps/merged_competition_gap_filled/work/113208_gap_2400_2803_selected.db",
]
OUT = ROOT / "maps/merged_competition_gap_filled/work/prepared_gap_compacts.json"
INTERSECTION = (-81.09913688302755,
                np.asarray([0.8876411571083551, -3.4107036975498914]))
RETURN = (-81.152550349049,
          np.asarray([0.4394632851288556, -1.9924492786768795]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_poses() -> dict[int, tuple[float, float, float]]:
    result = {}
    for line in POSES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        qx, qy, qz, qw = map(float, f[4:8])
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        result[int(f[8])] = (float(f[1]), float(f[2]), yaw)
    return result


def pose(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    p[:3, 3] = (x, y, 0)
    return p


def blob(p: np.ndarray) -> bytes:
    return struct.pack("<12f", *p[:3, :4].astype(np.float32).ravel())


def main() -> None:
    raw = load_poses()
    summary = {
        "scale": 1.0,
        "flat_datum_z_m": 0.0,
        "databases": {},
    }
    for path in DBS:
        transform_name = "return" if "2400_2803" in path.name else "intersection"
        theta_deg, translation = RETURN if transform_name == "return" else INTERSECTION
        theta = math.radians(theta_deg)
        r = np.asarray([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
        before = sha256(path)
        con = sqlite3.connect(path)
        ids = [r0[0] for r0 in con.execute("SELECT id FROM Node ORDER BY id")]
        desired = {}
        for node_id in ids:
            x, y, yaw = raw[node_id]
            xy = np.asarray([x, y]) @ r.T + translation
            desired[node_id] = pose(float(xy[0]), float(xy[1]), yaw + theta)
            b = blob(desired[node_id])
            con.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?", (b, b, node_id))
        for rowid, from_id, to_id in con.execute("SELECT rowid,from_id,to_id FROM Link"):
            if from_id in desired and to_id in desired:
                rel = np.linalg.inv(desired[from_id]) @ desired[to_id]
                con.execute("UPDATE Link SET transform=? WHERE rowid=?", (blob(rel), rowid))
        con.execute(
            "UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
            "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
            "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
            "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL"
        )
        con.commit()
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
        summary["databases"][str(path)] = {
            "sha256_before": before,
            "sha256_after": sha256(path),
            "nodes": len(ids),
            "source_ids": [min(ids), max(ids)],
            "transform": transform_name,
            "theta_deg": theta_deg,
            "translation_m": translation.tolist(),
            "integrity": integrity,
        }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
