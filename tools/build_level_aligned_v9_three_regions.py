#!/usr/bin/env python3
"""Build v9 from v8, correcting only the three photographed road defects.

All edits are scale-1 SE(2) gauge changes.  Node poses and incident Link
measurements are updated together.  RGB-D, calibration, features, and local
scan payloads remain byte-identical; the global display cache is regenerated.
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
REPORT = ROOT / "maps/merged_competition_level_aligned_v9/work/v9_three_region_build.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mat(blob: bytes) -> np.ndarray:
    t = np.eye(4)
    t[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return t


def pack(t: np.ndarray) -> bytes:
    return struct.pack("<12f", *t[:3, :4].astype(np.float32).ravel())


def smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def world_about_pose(pose: np.ndarray, dx: float, dy: float, dz: float,
                     yaw_deg: float) -> np.ndarray:
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    r = np.eye(4)
    r[:2, :2] = ((c, -s), (s, c))
    pivot = pose[:3, 3].copy()
    a = np.eye(4); a[:3, 3] = pivot + np.asarray([dx, dy, dz])
    b = np.eye(4); b[:3, 3] = -pivot
    return a @ r @ b


def interpolate_controls(node: int, controls):
    if node <= controls[0][0]:
        return controls[0][1:]
    if node >= controls[-1][0]:
        return controls[-1][1:]
    for left, right in zip(controls, controls[1:]):
        if left[0] <= node <= right[0]:
            w = smooth((node-left[0])/(right[0]-left[0]))
            return tuple((1-w)*a+w*b for a, b in zip(left[1:], right[1:]))
    raise AssertionError(node)


def taper(node: int, start: int, full_start: int, full_stop: int, stop: int) -> float:
    if node <= start or node >= stop:
        return 0.0
    if node < full_start:
        return smooth((node-start)/(full_start-start))
    if node <= full_stop:
        return 1.0
    return smooth((stop-node)/(stop-full_stop))


def digest_rows(db, query):
    h = hashlib.sha256()
    for row in db.execute(query):
        for value in row:
            h.update(value if isinstance(value, bytes) else repr(value).encode())
    return h.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True)
    source_before = sha256(SOURCE)
    shutil.copy2(SOURCE, OUTPUT)
    db = sqlite3.connect(OUTPUT)
    db.execute("PRAGMA foreign_keys=OFF")
    poses = {i: mat(blob) for i, blob in db.execute("SELECT id,pose FROM Node")}
    payload_before = {
        "data": digest_rows(db, "SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"),
        "features": digest_rows(db, "SELECT * FROM Feature ORDER BY rowid"),
    }

    corrections = {}
    labels = {}
    details = []

    # 1) Previously verified yellow-line split: same scale-1 model as the
    # preserved yellow-only v9.  Zero-weight endpoints are not written.
    yellow_ids = [i for i in poses if 1629 <= i <= 1664 and
                  db.execute("SELECT map_id FROM Node WHERE id=?", (i,)).fetchone()[0] == 3]
    yellow_ids.sort()
    first_full, last_full = yellow_ids.index(1641), yellow_ids.index(1656)
    pivot = np.asarray([22.43784325, 29.5])
    shift = np.asarray([21.94919120, 29.5]) - pivot
    for index, node in enumerate(yellow_ids):
        if index < first_full:
            w = index / first_full
        elif index <= last_full:
            w = 1.0
        else:
            w = (len(yellow_ids)-1-index)/(len(yellow_ids)-1-last_full)
        if w <= 0:
            continue
        angle = math.radians(-1.8059808647418325*w)
        c, s = math.cos(angle), math.sin(angle)
        r = np.asarray(((c, -s), (s, c)))
        translation = pivot + shift*w - r@pivot
        world = np.eye(4); world[:2, :2] = r; world[:2, 3] = translation
        world[2, 3] = -0.003274*w
        corrections[node] = world
        labels[node] = "yellow_split"

    # 2) Torn intersection connection.  Controls are actual RGB-D world
    # residuals.  1707->646 has 119 inliers / 0.0186 m RMS.  The late control
    # combines two independent reverse-view checks; correction reaches zero at
    # the existing map3->map4 temporal boundary, so map4 is untouched.
    triangle_controls = [
        (1690, 0.0, 0.0, 0.0, 0.0),
        (1707, -0.1546, 0.1192, 0.0, -3.94),
        (1738, 0.2470, 0.0850, 0.0, 1.34),
        (1751, 0.0, 0.0, 0.0, 0.0),
    ]
    for node in range(1691, 1751):
        if node not in poses:
            continue
        dx, dy, dz, yaw = interpolate_controls(node, triangle_controls)
        corrections[node] = world_about_pose(poses[node], dx, dy, dz, yaw)
        labels[node] = "triangle_intersection"

    # 3) Long straight road.  Actual RGB-D yellow/edge markings contain two
    # corresponding longitudinal lines in both repeat passes.  Their widths
    # are 3.5550 m and 3.3990 m (kept unchanged: no scale).  The two observed
    # midlines were 2.0521 m apart, although they are the same physical road.
    # Align both rigidly to the mean midline/direction instead of using vehicle
    # trajectory separation, which would retain the visually duplicated road.
    middle_line = np.asarray([-1.40263530, -0.073602105])
    late_line = np.asarray([-4.85874492, -0.048530080])
    center_y = 56.0
    common_b = (middle_line[1] + late_line[1]) / 2.0
    common_x = ((middle_line[0] + middle_line[1]*center_y) +
                (late_line[0] + late_line[1]*center_y)) / 2.0
    common_a = common_x-common_b*center_y
    road_models = [
        ("straight_middle", 807, 813, 840, 846, middle_line, common_a, common_b),
        ("straight_late", 1150, 1156, 1185, 1191, late_line, common_a, common_b),
    ]
    for label, start, full_start, full_stop, stop, old_line, new_a, new_b in road_models:
        old_heading = math.degrees(math.atan2(1.0, old_line[1]))
        new_heading = math.degrees(math.atan2(1.0, new_b))
        delta_yaw = new_heading-old_heading
        for node in range(start+1, stop):
            if node not in poses:
                continue
            w = taper(node, start, full_start, full_stop, stop)
            y = poses[node][1, 3]
            dx = ((new_a+new_b*y) - (old_line[0]+old_line[1]*y))*w
            world = world_about_pose(poses[node], dx, 0.0, 0.0, delta_yaw*w)
            corrections[node] = world
            labels[node] = label

    old_poses = poses.copy()
    for node, world in corrections.items():
        old = old_poses[node]; new = world@old
        gt = db.execute("SELECT ground_truth_pose FROM Node WHERE id=?", (node,)).fetchone()[0]
        if gt:
            db.execute("UPDATE Node SET pose=?,ground_truth_pose=? WHERE id=?",
                       (pack(new), pack(world@mat(gt)), node))
        else:
            db.execute("UPDATE Node SET pose=? WHERE id=?", (pack(new), node))
        details.append({"id": node, "region": labels[node],
                        "before_xyz": old[:3, 3].tolist(), "after_xyz": new[:3, 3].tolist(),
                        "xy_change_m": float(np.linalg.norm(new[:2, 3]-old[:2, 3])),
                        "yaw_change_deg": math.degrees(math.atan2(world[1, 0], world[0, 0]))})

    # Gauge-consistent link update.  Local feature/scan coordinates stay valid.
    changed_links = 0
    for rowid, a, b, transform in list(db.execute(
            "SELECT rowid,from_id,to_id,transform FROM Link ORDER BY rowid")):
        if a not in corrections and b not in corrections:
            continue
        ca = np.linalg.inv(old_poses[a]) @ (corrections[a]@old_poses[a]) if a in corrections else np.eye(4)
        cb = np.linalg.inv(old_poses[b]) @ (corrections[b]@old_poses[b]) if b in corrections else np.eye(4)
        updated = np.linalg.inv(ca) @ mat(transform) @ cb
        db.execute("UPDATE Link SET transform=? WHERE rowid=?", (pack(updated), rowid))
        changed_links += 1

    db.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
               "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
               "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
               "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    db.commit()
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    payload_after = {
        "data": digest_rows(db, "SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"),
        "features": digest_rows(db, "SELECT * FROM Feature ORDER BY rowid"),
    }
    nodes = db.execute("SELECT count(*) FROM Node").fetchone()[0]
    db.close()
    report = {
        "source": str(SOURCE), "source_sha256_before": source_before,
        "source_sha256_after": sha256(SOURCE), "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT), "integrity": integrity, "nodes": nodes,
        "scale": 1.0, "changed_pose_nodes": sorted(corrections),
        "changed_nodes_by_region": {name: sorted(i for i in corrections if labels[i] == name)
                                    for name in sorted(set(labels.values()))},
        "changed_links": changed_links, "payload_identical": payload_before == payload_after,
        "line_models": {"middle_before": middle_line.tolist(), "late_before": late_line.tolist(),
                        "common_after": [float(common_a), float(common_b)],
                        "duplicate_midline_offset_before_m": float(abs(
                            (middle_line[0]+middle_line[1]*center_y)-
                            (late_line[0]+late_line[1]*center_y))),
                        "observed_widths_preserved_m": [3.5550, 3.3990]},
        "triangle_controls": triangle_controls, "details": details,
    }
    REPORT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("output", "output_sha256", "integrity", "nodes", "scale", "changed_links", "payload_identical", "line_models")}, indent=2))


if __name__ == "__main__":
    main()
