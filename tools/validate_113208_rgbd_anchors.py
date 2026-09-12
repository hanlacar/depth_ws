#!/usr/bin/env python3
"""Validate selected cross-session anchors with stored RGB and depth scans."""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/qor/depth_ws")
DB = ROOT / "maps/merged_competition_full_course/work/append_candidate_no_cross.db"
OUT = ROOT / "maps/merged_competition_full_course/work/rgbd_anchor_validation.json"

PAIRS = [
    (334, 1564, "113208:1457 corner entry"),
    (643, 1707, "113208:2156 intersection exit"),
    (949, 1737, "113208:2247 intersection turn"),
    (948, 1747, "113208:2293 intersection return"),
]


def matrix(blob):
    p = np.eye(4)
    p[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return p


def scan(blob):
    obj = zlib.decompressobj()
    raw = obj.decompress(blob)+obj.flush()
    rows, cols, cv_type = struct.unpack("<3i", obj.unused_data)
    channels = ((cv_type >> 3)+1)
    points = np.frombuffer(raw, np.float32).reshape(rows*cols, channels)[:, :3]
    return points[np.all(np.isfinite(points), axis=1)]


def scan_local_transform(blob):
    values = np.frombuffer(blob, np.float32)
    if values.size != 19:
        raise ValueError(f"unexpected scan_info length {values.size}")
    pose = np.eye(4)
    pose[:3, :4] = values[7:19].reshape(3, 4)
    return pose


def orb_metrics(left, right):
    detector = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
    ka, da = detector.detectAndCompute(left, None)
    kb, db = detector.detectAndCompute(right, None)
    if da is None or db is None:
        return {"keypoints": [len(ka), len(kb)], "good": 0, "inliers": 0,
                "fraction": 0.0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < .75*n.distance]
    inliers = 0
    if len(good) >= 8:
        a = np.float32([ka[m.queryIdx].pt for m in good])
        b = np.float32([kb[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        inliers = int(mask.sum()) if mask is not None else 0
    return {"keypoints": [len(ka), len(kb)], "good": len(good),
            "inliers": inliers, "fraction": inliers/max(len(good), 1)}


def estimate_se2(source, target):
    sc, tc = source.mean(0), target.mean(0)
    s, t = source-sc, target-tc
    theta = math.atan2(np.sum(s[:, 0]*t[:, 1]-s[:, 1]*t[:, 0]), np.sum(s*t))
    r = np.asarray([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    tr = tc-sc@r.T
    return r, tr


def icp_metrics(base_scan, target_scan, base_pose, target_pose,
                base_local, target_local):
    b = base_scan[(base_scan[:, 2] > -.5)&(base_scan[:, 2] < 3.0)][::2]
    q = target_scan[(target_scan[:, 2] > -.5)&(target_scan[:, 2] < 3.0)][::2]
    b = b@base_local[:3, :3].T+base_local[:3, 3]
    q = q@target_local[:3, :3].T+target_local[:3, 3]
    bw = b@base_pose[:3, :3].T+base_pose[:3, 3]
    qw = q@target_pose[:3, :3].T+target_pose[:3, 3]
    base_xy = bw[:, :2]
    moving = qw[:, :2]
    total_r = np.eye(2); total_t = np.zeros(2)
    tree = cKDTree(base_xy)
    iterations = 0
    for iterations in range(30):
        distance, index = tree.query(moving)
        valid = distance < .50
        if np.count_nonzero(valid) < 80:
            break
        threshold = np.quantile(distance[valid], .75)
        valid &= distance <= threshold
        r, t = estimate_se2(moving[valid], base_xy[index[valid]])
        moving = moving@r.T+t
        total_t = total_t@r.T+t
        total_r = r@total_r
        if abs(math.atan2(r[1, 0], r[0, 0])) < 1e-5 and np.linalg.norm(t) < 1e-4:
            break
    distance, _ = tree.query(moving)
    inlier = distance < .25
    rmse = float(np.sqrt(np.mean(distance[inlier]**2))) if np.any(inlier) else None
    return {"iterations": iterations+1, "points": [len(base_xy), len(moving)],
            "inlier_ratio_0_25m": float(np.mean(inlier)), "rmse_m": rmse,
            "correction_theta_deg": math.degrees(math.atan2(total_r[1, 0], total_r[0, 0])),
            "correction_translation_m": total_t.tolist()}


def main():
    result = []
    con = sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1", uri=True)
    for base_id, target_id, label in PAIRS:
        records = []
        for node_id in (base_id, target_id):
            pose_blob = con.execute("SELECT pose FROM Node WHERE id=?", (node_id,)).fetchone()[0]
            image_blob, scan_blob, info_blob = con.execute(
                "SELECT image,scan,scan_info FROM Data WHERE id=?", (node_id,)).fetchone()
            image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_GRAYSCALE)
            records.append((matrix(pose_blob), image, scan(scan_blob),
                            scan_local_transform(info_blob)))
        rgb = orb_metrics(records[0][1], records[1][1])
        icp = icp_metrics(records[0][2], records[1][2], records[0][0], records[1][0],
                          records[0][3], records[1][3])
        accepted = (rgb["inliers"] >= 15 and rgb["fraction"] >= .30 and
                    icp["rmse_m"] is not None and icp["rmse_m"] <= .18 and
                    icp["inlier_ratio_0_25m"] >= .08 and
                    np.linalg.norm(icp["correction_translation_m"]) <= 1.5 and
                    abs(icp["correction_theta_deg"]) <= 20)
        result.append({"base_id": base_id, "target_id": target_id, "label": label,
                       "rgb": rgb, "depth_icp": icp, "accepted": bool(accepted)})
    con.close()
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
