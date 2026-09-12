#!/usr/bin/env python3
"""Find real 113208 ground/boundary observations missing from the current map.

The input databases are opened immutable and sequentially.  Candidate cells are
accepted only when they are RTAB-Map local ground observations falling in cells
that are unknown in the current global occupancy.  This script does not mutate
any database.
"""

from __future__ import annotations

import collections
import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path("/home/qor/depth_ws")
BASE = ROOT / "maps/merged_competition_full_course/rtabmap.db"
SOURCE = ROOT / "maps/map_20260906_113208/rtabmap.db"
SOURCE_POSES = Path("/tmp/rtabmap_audit_poses/map_113208_poses.txt")
BASE_POSES = ROOT / "analysis/full_course_final/full_course_global_poses.txt"
PRE = ROOT / "analysis/full_course_premerge/premerge_report.json"
OUT = ROOT / "analysis/gap_fill"
REPORT = OUT / "candidate_analysis.json"
RANGES = [(1661, 1881), (2303, 2399), (2400, 2803)]


def ro(path: Path):
    c = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    c.execute("PRAGMA query_only=ON")
    return c


def load_pose_xyyaw(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"): continue
        f = line.split(); qx, qy, qz, qw = map(float, f[4:8])
        out[int(f[8])] = np.asarray([float(f[1]), float(f[2]),
            math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))])
    return out


def decode_grid(path: Path):
    with ro(path) as c:
        blob, x0, y0, res = c.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin").fetchone()
    obj = zlib.decompressobj(); raw = obj.decompress(blob)+obj.flush()
    w, h, _ = struct.unpack("<3i", obj.unused_data)
    return np.frombuffer(raw, np.int8).reshape(h, w), float(x0), float(y0), float(res)


def decode_cells(blob):
    if not blob: return np.empty((0, 3), np.float32)
    obj = zlib.decompressobj(); raw = obj.decompress(blob)+obj.flush()
    rows, cols, kind = struct.unpack("<3i", obj.unused_data)
    if kind != 29: raise ValueError((rows, cols, kind))
    return np.frombuffer(raw, np.float32).reshape(rows*cols, 4)[:, :3]


def in_ranges(node):
    return any(a <= node <= b for a, b in RANGES)


def transform(node, pose, fits):
    name = "intersection" if node <= 2399 else "return"
    fit = fits[name]; th = math.radians(fit["theta_deg"])
    r = np.asarray([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    xy = pose[:2]@r.T+np.asarray(fit["translation_m"])
    return xy, pose[2]+th, name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid, x0, y0, res = decode_grid(BASE)
    base_pose = load_pose_xyyaw(BASE_POSES)
    source_pose = load_pose_xyyaw(SOURCE_POSES)
    fits = json.loads(PRE.read_text())["fits"]
    h, w = grid.shape
    ground_votes = np.zeros((h, w), np.uint16)
    obstacle_votes = np.zeros((h, w), np.uint16)
    node_cells = {}; stats = []

    # Source is opened only after the base cache has been closed and decoded.
    with ro(SOURCE) as c:
        for node, ground_blob, obstacle_blob in c.execute(
            "SELECT id,ground_cells,obstacle_cells FROM Data ORDER BY id"):
            if not in_ranges(node) or node not in source_pose: continue
            xy, heading, fit_name = transform(node, source_pose[node], fits)
            cr, sr = math.cos(heading), math.sin(heading)
            rotation = np.asarray([[cr, -sr], [sr, cr]])
            counts = {}
            stored = []
            for kind, blob, votes in (("ground", ground_blob, ground_votes),
                                      ("obstacle", obstacle_blob, obstacle_votes)):
                local = decode_cells(blob)
                world = local[:, :2]@rotation.T+xy if len(local) else np.empty((0, 2))
                ix = np.floor((world[:, 0]-x0)/res).astype(np.int32)
                iy = np.floor((world[:, 1]-y0)/res).astype(np.int32)
                valid = (ix>=0)&(ix<w)&(iy>=0)&(iy<h)
                flat = np.unique(iy[valid]*w+ix[valid])
                values = grid.ravel()[flat]
                counts[kind+"_cells"] = int(len(flat))
                counts[kind+"_unknown"] = int(np.count_nonzero(values < 0))
                counts[kind+"_free"] = int(np.count_nonzero(values == 0))
                counts[kind+"_occupied"] = int(np.count_nonzero(values == 100))
                new_flat = flat[values < 0]
                votes.ravel()[new_flat] = np.minimum(votes.ravel()[new_flat]+1, 65535)
                if kind == "ground": stored = new_flat.tolist()
            node_cells[node] = stored
            stats.append({"node": node, "fit": fit_name, "x": float(xy[0]),
                          "y": float(xy[1]), **counts})

    # Require two independent node observations and join nearby cells at 5 cm.
    observed = ground_votes >= 2
    joined = ndimage.binary_closing(ndimage.binary_dilation(observed, iterations=2), iterations=2)
    labels, count = ndimage.label(joined)
    raw_regions = []
    for label in range(1, count+1):
        area = labels == label
        original = area & observed
        cells = int(np.count_nonzero(original))
        if cells < 35: continue
        yy, xx = np.where(original)
        bbox = [x0+xx.min()*res, y0+yy.min()*res,
                x0+(xx.max()+1)*res, y0+(yy.max()+1)*res]
        nodes = []
        region_flat = set((yy*w+xx).tolist())
        for node, flat in node_cells.items():
            overlap = len(region_flat.intersection(flat))
            if overlap: nodes.append((node, overlap))
        nodes.sort(key=lambda x: (-x[1], x[0]))
        raw_regions.append({"cells": cells, "area_m2": cells*res*res,
            "bbox_xy": bbox, "nodes": nodes[:20], "node_min": min(x[0] for x in nodes),
            "node_max": max(x[0] for x in nodes), "peak_votes": int(ground_votes[original].max())})
    raw_regions.sort(key=lambda q: -q["cells"])
    regions = []
    for index, item in enumerate(raw_regions[:12], 1):
        item["region_id"] = f"G{index}"; regions.append(item)

    # Window summaries help select compact contiguous source segments.
    windows = []
    for a, b in RANGES:
        for lo in range(a, b+1, 40):
            hi = min(lo+39, b); q = [x for x in stats if lo <= x["node"] <= hi]
            if not q: continue
            windows.append({"range": [lo, hi], "nodes": len(q),
                "ground_unknown": sum(x["ground_unknown"] for x in q),
                "ground_total": sum(x["ground_cells"] for x in q),
                "obstacle_unknown": sum(x["obstacle_unknown"] for x in q),
                "conflict_ground_on_occupied": sum(x["ground_occupied"] for x in q)})

    # Candidate overview using real unknown-ground observations.
    extent = [x0, x0+w*res, y0, y0+h*res]
    display = np.zeros_like(grid, np.uint8); display[grid==0]=1; display[grid==100]=2
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    ax.imshow(display, origin="lower", extent=extent,
              cmap=ListedColormap([(1,1,1,0),(.9,.91,.92,.7),(.03,.03,.03,.95)]),
              interpolation="nearest")
    yy, xx = np.where(ground_votes >= 2)
    ax.scatter(x0+(xx+.5)*res, y0+(yy+.5)*res, s=.8, color="#13a449", alpha=.7,
               label="113208 ground in currently unknown cells (≥2 nodes)")
    for item in regions:
        x1,y1,x2,y2=item["bbox_xy"]
        ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec="#006b2f",lw=2))
        ax.text((x1+x2)/2,(y1+y2)/2,item["region_id"],weight="bold",fontsize=9,
                bbox=dict(fc="white",ec="#006b2f",alpha=.9))
    ax.set_title("Gap candidates from actual 113208 ground observations\nscale=1.000, current map coordinates",weight="bold")
    ax.set_xlabel("X [m] →");ax.set_ylabel("Y [m] →");ax.set_aspect("equal");ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/"02_candidate_regions_numbered.png",dpi=180);plt.close(fig)

    report = {"input_ranges": RANGES, "alignment": {
        k: {x: fits[k][x] for x in ("theta_deg","translation_m","rms_m")} for k in ("intersection","return")},
        "grid": {"shape": [h,w], "origin": [x0,y0], "resolution":res},
        "nodes_analyzed": len(stats), "ground_unknown_cells_ge2": int(np.count_nonzero(observed)),
        "obstacle_unknown_cells_ge2": int(np.count_nonzero(obstacle_votes>=2)),
        "regions": regions, "windows": windows,
        "top_nodes_by_unknown_ground": sorted(stats,key=lambda x:-x["ground_unknown"])[:60],
        "notes": ["Unknown-ground cells are actual RTAB-Map local ground observations, not interpolation.",
                  "Regions still require RGB/depth and alignment review before insertion."]}
    REPORT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("nodes_analyzed","ground_unknown_cells_ge2","obstacle_unknown_cells_ge2","regions","windows")},indent=2))


if __name__ == "__main__": main()
