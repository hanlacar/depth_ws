#!/usr/bin/env python3
"""Cross-check gap-fill RGB and depth scans against retained map observations."""

from __future__ import annotations

import json, math, sqlite3, struct, zlib
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/qor/depth_ws")
BASE = ROOT / "maps/merged_competition_full_course/rtabmap.db"
RETURN = ROOT / "maps/merged_competition_gap_filled/work/113208_gap_2400_2803_compact.db"
OUT = ROOT / "analysis/gap_fill/rgbd_geometric_validation.json"
# The last two pairs deliberately use spatially adjacent end-of-base nodes,
# not retrieval-only matches from visually repetitive road scenes.
PAIRS = [
    (1251, 2400, "return/front"), (1253, 2421, "return/front-2"),
    (1259, 2470, "return/corner"), (1267, 2544, "return/middle"),
    (1276, 2604, "return/back"), (1278, 2616, "return/holdout"),
    (1285, 2680, "late-return/front"), (1290, 2782, "late-return/back"),
]


def mat(blob):
    p = np.eye(4); p[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4); return p


def read_scan(blob):
    o = zlib.decompressobj(); raw = o.decompress(blob) + o.flush()
    rows, cols, typ = struct.unpack("<3i", o.unused_data)
    ch = (typ >> 3) + 1
    p = np.frombuffer(raw, np.float32).reshape(rows * cols, ch)[:, :3]
    return p[np.all(np.isfinite(p), axis=1)]


def local(blob):
    v = np.frombuffer(blob, np.float32); p = np.eye(4)
    p[:3, :4] = v[7:19].reshape(3, 4); return p


def image_and_scan(con, node_id):
    pose = mat(con.execute("SELECT pose FROM Node WHERE id=?", (node_id,)).fetchone()[0])
    im, scan, info = con.execute("SELECT image,scan,scan_info FROM Data WHERE id=?", (node_id,)).fetchone()
    return pose, cv2.imdecode(np.frombuffer(im, np.uint8), cv2.IMREAD_GRAYSCALE), read_scan(scan), local(info)


def orb(a, b):
    d = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
    ka, da = d.detectAndCompute(a, None); kb, db = d.detectAndCompute(b, None)
    if da is None or db is None: return {"good": 0, "inliers": 0, "fraction": 0.0}
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < .75 * y.distance]
    n = 0
    if len(good) >= 8:
        aa = np.float32([ka[x.queryIdx].pt for x in good]); bb = np.float32([kb[x.trainIdx].pt for x in good])
        _, mask = cv2.findHomography(aa, bb, cv2.RANSAC, 3.0)
        n = int(mask.sum()) if mask is not None else 0
    return {"good": len(good), "inliers": n, "fraction": n / max(1, len(good))}


def se2(a, b):
    ac, bc = a.mean(0), b.mean(0); aa, bb = a-ac, b-bc
    th = math.atan2(np.sum(aa[:, 0]*bb[:, 1]-aa[:, 1]*bb[:, 0]), np.sum(aa*bb))
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return r, bc-ac@r.T


def icp(a, b, pa, pb, la, lb):
    a = a[(a[:, 2] > -.5) & (a[:, 2] < 3.0)][::2]
    b = b[(b[:, 2] > -.5) & (b[:, 2] < 3.0)][::2]
    aw = (a@la[:3, :3].T+la[:3, 3])@pa[:3, :3].T+pa[:3, 3]
    bw = (b@lb[:3, :3].T+lb[:3, 3])@pb[:3, :3].T+pb[:3, 3]
    fixed, moving = aw[:, :2], bw[:, :2]; tree = cKDTree(fixed)
    tr, tt = np.eye(2), np.zeros(2)
    for _ in range(30):
        dist, idx = tree.query(moving); use = dist < .5
        if use.sum() < 80: break
        use &= dist <= np.quantile(dist[use], .75)
        r, t = se2(moving[use], fixed[idx[use]])
        moving = moving@r.T+t; tt = tt@r.T+t; tr = r@tr
        if abs(math.atan2(r[1,0], r[0,0])) < 1e-5 and np.linalg.norm(t) < 1e-4: break
    dist, _ = tree.query(moving); use = dist < .25
    return {
        "points": [len(fixed), len(moving)], "overlap_0_25m": float(use.mean()),
        "rmse_m": float(np.sqrt(np.mean(dist[use]**2))) if use.any() else None,
        "correction_m": tt.tolist(),
        "correction_norm_m": float(np.linalg.norm(tt)),
        "correction_deg": math.degrees(math.atan2(tr[1,0], tr[0,0])),
    }


def main():
    b = sqlite3.connect(f"file:{BASE}?mode=ro&immutable=1", uri=True)
    q = sqlite3.connect(f"file:{RETURN}?mode=ro&immutable=1", uri=True)
    out = []
    for bi, qi, label in PAIRS:
        br, qr = image_and_scan(b, bi), image_and_scan(q, qi)
        out.append({"label": label, "base_id": bi, "source_id": qi,
                    "rgb": orb(br[1], qr[1]),
                    "depth_icp": icp(br[2], qr[2], br[0], qr[0], br[3], qr[3])})
    b.close(); q.close(); OUT.write_text(json.dumps(out, indent=2)+"\n"); print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
