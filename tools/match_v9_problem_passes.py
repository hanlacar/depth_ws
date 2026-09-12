#!/usr/bin/env python3
"""Verify repeated observations in the three v9 repair regions."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/qor/depth_ws")
DB = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
OUT = ROOT / "analysis/level_aligned_v9_three/work/visual_feature_matches.json"


def pose(blob):
    value = np.eye(4)
    value[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return value


def features(db, node):
    rows = list(db.execute(
        "SELECT pos_x,pos_y,descriptor FROM Feature WHERE node_id=?", (node,)))
    rows = [row for row in rows if row[2] and len(row[2]) == 32]
    if not rows:
        return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
    return (np.asarray([(row[0], row[1]) for row in rows], np.float32),
            np.asarray([np.frombuffer(row[2], np.uint8) for row in rows]))


def verify(a, b):
    pa, da = a; pb, db = b
    if len(da) < 12 or len(db) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x for x, y in pairs if x.distance < .77*y.distance]
    if len(good) < 10:
        return None
    aa = np.asarray([pa[m.queryIdx] for m in good])
    bb = np.asarray([pb[m.trainIdx] for m in good])
    _, mask = cv2.findFundamentalMat(aa, bb, cv2.FM_RANSAC, 1.5, .995)
    if mask is None:
        return None
    count = int(mask.sum())
    if count < 8:
        return None
    return {"good": len(good), "inliers": count,
            "fraction": count/len(good),
            "median_hamming": float(np.median([m.distance for m in good]))}


def run(name, target_ids, reference_ids, max_pose_distance):
    db = sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1", uri=True)
    poses = {i: pose(b) for i, b in db.execute("SELECT id,pose FROM Node")}
    target_ids = [i for i in target_ids if i in poses]
    reference_ids = [i for i in reference_ids if i in poses]
    cache = {}
    def load(i):
        if i not in cache:
            cache[i] = features(db, i)
        return cache[i]
    found = []
    for target in target_ids:
        pt = poses[target][:2, 3]
        candidates = sorted(reference_ids,
                            key=lambda ref: np.linalg.norm(poses[ref][:2, 3]-pt))[:14]
        for ref in candidates:
            distance = float(np.linalg.norm(poses[ref][:2, 3]-pt))
            if distance > max_pose_distance:
                continue
            result = verify(load(target), load(ref))
            if result:
                found.append({"target": target, "reference": ref,
                              "pose_distance_m": distance, **result})
    db.close()
    found.sort(key=lambda x: (x["inliers"], x["fraction"]), reverse=True)
    # Keep the strongest correspondence per target plus all very strong pairs.
    best = {}
    for row in found:
        best.setdefault(row["target"], row)
    selected = list(best.values()) + [row for row in found if row["inliers"] >= 35]
    unique = {(r["target"], r["reference"]): r for r in selected}
    return sorted(unique.values(), key=lambda x: x["target"])


def main():
    result = {
        "long_road": run("long_road", range(728, 873, 2), range(1137, 1235, 2), 12),
        "triangle_map3": run("triangle_map3", range(1690, 1751),
                             list(range(408, 441))+list(range(640, 699))+
                             list(range(728, 803))+list(range(932, 966)), 15),
        "triangle_map4": run("triangle_map4", range(1751, 1768),
                             list(range(408, 441))+list(range(640, 699))+
                             list(range(728, 803))+list(range(932, 966)), 15),
    }
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    for key, rows in result.items():
        print("\n", key, len(rows))
        for row in sorted(rows, key=lambda x: x["inliers"], reverse=True)[:30]:
            print(row)


if __name__ == "__main__":
    main()
