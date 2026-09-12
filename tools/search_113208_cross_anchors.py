#!/usr/bin/env python3
"""Brute-force nearby RGB-D cross-session anchors on the working merge."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from validate_113208_rgbd_anchors import (
    DB, icp_metrics, matrix, orb_metrics, scan, scan_local_transform,
)

OUT = Path("/home/qor/depth_ws/maps/merged_competition_full_course/work/cross_anchor_search.json")


def angle(a):
    return math.atan2(a[1, 0], a[0, 0])


def difference(a, b):
    return abs((a-b+math.pi)%(2*math.pi)-math.pi)


def main():
    con = sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1", uri=True)
    rows = [(i, m, matrix(p)) for i, m, p in con.execute(
        "SELECT id,map_id,pose FROM Node ORDER BY id")]
    base = [(i, p) for i, m, p in rows if m in (0, 1)]
    target = [(i, p) for i, m, p in rows if m in (2, 3)]
    cache = {}

    def image(node_id):
        if node_id not in cache:
            blob = con.execute("SELECT image FROM Data WHERE id=?", (node_id,)).fetchone()[0]
            cache[node_id] = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
        return cache[node_id]

    visual = []
    for target_id, tp in target[::2]:
        candidates = []
        for base_id, bp in base:
            distance = np.linalg.norm(tp[:2, 3]-bp[:2, 3])
            da = difference(angle(tp), angle(bp))
            if distance <= 3.0 and da <= math.radians(55):
                candidates.append((distance, base_id, bp))
        for distance, base_id, bp in sorted(candidates)[:18]:
            rgb = orb_metrics(image(base_id), image(target_id))
            if rgb["inliers"] >= 10 and rgb["fraction"] >= .28:
                visual.append({"base_id": base_id, "target_id": target_id,
                               "pose_distance_m": float(distance),
                               "yaw_difference_deg": math.degrees(difference(angle(tp), angle(bp))),
                               "rgb": rgb})
    visual.sort(key=lambda q: (-q["rgb"]["inliers"], q["pose_distance_m"]))

    # ICP only on the strongest spatially distributed visual candidates.
    selected = []
    used_target = []
    for item in visual:
        if any(abs(item["target_id"]-i) < 10 for i in used_target):
            continue
        selected.append(item)
        used_target.append(item["target_id"])
        if len(selected) >= 24:
            break
    by_id = {i: p for i, _, p in rows}
    for item in selected:
        records = []
        for node_id in (item["base_id"], item["target_id"]):
            scan_blob, info = con.execute(
                "SELECT scan,scan_info FROM Data WHERE id=?", (node_id,)).fetchone()
            records.append((scan(scan_blob), scan_local_transform(info)))
        item["depth_icp"] = icp_metrics(records[0][0], records[1][0],
                                         by_id[item["base_id"]], by_id[item["target_id"]],
                                         records[0][1], records[1][1])
        q = item["depth_icp"]
        item["accepted"] = bool(
            item["rgb"]["inliers"] >= 12 and item["rgb"]["fraction"] >= .30 and
            q["rmse_m"] is not None and q["rmse_m"] <= .12 and
            q["inlier_ratio_0_25m"] >= .20 and
            np.linalg.norm(q["correction_translation_m"]) <= 1.5 and
            abs(q["correction_theta_deg"]) <= 15)
    con.close()
    report = {"visual_candidate_count": len(visual), "tested_rgbd": selected}
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
