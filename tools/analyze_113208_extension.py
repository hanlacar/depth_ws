#!/usr/bin/env python3
"""Read-only pre-merge diagnostics for the 113208 RTAB-Map session.

All plots are made from stored optimized poses, RGB and node-local RTAB-Map
ground/obstacle grids.  Source databases are opened immutable and one at a
time.  No map data is modified by this script.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path("/home/qor/depth_ws")
BASE_DB = ROOT / "maps/merged_competition_1lane_2lane_ramp/rtabmap.db"
AUX_DB = ROOT / "maps/map_20260906_113208/rtabmap.db"
BASE_POSES = Path("/tmp/ramp_final_export/ramp_final3_global_poses.txt")
AUX_POSES = Path("/tmp/rtabmap_audit_poses/map_113208_poses.txt")
OUT = ROOT / "analysis/full_course_premerge"

ANCHORS = {
    "ramp": [(289, 94), (331, 110), (374, 121), (416, 132), (459, 148)],
    "corner_entry": [(1012, 285), (1032, 287), (1083, 298), (1243, 314),
                     (1283, 323), (1358, 328), (1415, 330), (1457, 334)],
    "intersection": [(1457, 334), (2154, 643), (2246, 949), (2292, 948)],
    "return": [(2400, 1251), (2420, 1253), (2471, 1259),
               (2546, 1267), (2606, 1276), (2616, 1278)],
}

STATUS_RANGES = {
    "add": [(1457, 1660), (1882, 2302)],
    "hold": [(1661, 1881), (2303, 2399)],
    "exclude": [(1, 1456), (2400, 2851)],
}


def ro(path: Path):
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only=ON")
    return con


def load_poses(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        x, y, z = map(float, f[1:4])
        qx, qy, qz, qw = map(float, f[4:8])
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        out[int(f[8])] = np.asarray([x, y, z, yaw])
    return out


def fit_se2(aux, base, pairs):
    src = np.asarray([aux[a][:2] for a, _ in pairs])
    dst = np.asarray([base[b][:2] for _, b in pairs])
    sc, dc = src.mean(0), dst.mean(0)
    s, d = src-sc, dst-dc
    theta = math.atan2(np.sum(s[:, 0]*d[:, 1]-s[:, 1]*d[:, 0]), np.sum(s*d))
    rotation = np.asarray([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]])
    translation = dc-sc@rotation.T
    residual = np.linalg.norm(src@rotation.T+translation-dst, axis=1)
    scale = float(np.sum(d*(s@rotation.T))/np.sum(s*s))
    return rotation, translation, residual, scale


def decode_matrix(blob):
    if not blob:
        return np.empty((0, 3), np.float32)
    obj = zlib.decompressobj()
    raw = obj.decompress(blob)+obj.flush()
    if len(obj.unused_data) != 12:
        raise ValueError("compressed matrix trailer missing")
    rows, cols, cv_type = struct.unpack("<3i", obj.unused_data)
    if cv_type != 29 or len(raw) != rows*cols*16:
        raise ValueError((rows, cols, cv_type, len(raw)))
    return np.frombuffer(raw, np.float32).reshape(-1, 4)[:, :3]


def decode_global_map(path):
    with ro(path) as con:
        blob, x0, y0, res = con.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin").fetchone()
    obj = zlib.decompressobj()
    raw = obj.decompress(blob)+obj.flush()
    if len(obj.unused_data) == 12:
        width, height, _ = struct.unpack("<3i", obj.unused_data)
    else:
        raise ValueError("global map dimensions unavailable")
    grid = np.frombuffer(raw, np.int8).reshape(height, width)
    return grid, (x0, x0+width*res, y0, y0+height*res), float(res)


def transform_for(node_id, fits):
    if node_id <= 856:
        return fits["ramp"][:2]
    if node_id <= 1456:
        return fits["corner_entry"][:2]
    if node_id <= 2399:
        return fits["intersection"][:2]
    return fits["return"][:2]


def aligned_pose(node_id, pose, fits):
    r, t = transform_for(node_id, fits)
    xy = pose[:2]@r.T+t
    return np.r_[xy, pose[2], pose[3]+math.atan2(r[1, 0], r[0, 0])]


def status(node_id):
    for name, ranges in STATUS_RANGES.items():
        if any(a <= node_id <= b for a, b in ranges):
            return name
    return "hold"


def occupancy_points(grid, extent):
    yy, xx = np.where(grid == 100)
    dx = (extent[1]-extent[0])/grid.shape[1]
    dy = (extent[3]-extent[2])/grid.shape[0]
    return np.c_[extent[0]+(xx+.5)*dx, extent[2]+(yy+.5)*dy]


def load_aux_cells(aux, fits, stride=3):
    groups = {"add": [], "hold": [], "exclude": []}
    per_node = {}
    with ro(AUX_DB) as con:
        rows = con.execute("SELECT id,ground_cells,obstacle_cells FROM Data ORDER BY id")
        for index, (node_id, ground_blob, obstacle_blob) in enumerate(rows):
            if index % stride:
                continue
            pose = aligned_pose(node_id, aux[node_id], fits)
            c, s = math.cos(pose[3]), math.sin(pose[3])
            rotation = np.asarray([[c, -s], [s, c]])
            obs = decode_matrix(obstacle_blob)
            ground = decode_matrix(ground_blob)
            world_obs = obs[:, :2]@rotation.T+pose[:2] if len(obs) else np.empty((0, 2))
            world_ground = ground[:, :2]@rotation.T+pose[:2] if len(ground) else np.empty((0, 2))
            st = status(node_id)
            if len(world_obs):
                groups[st].append(world_obs)
            center = int(np.count_nonzero((obs[:, 0] > .3)&(obs[:, 0] < 5.0)&
                                          (np.abs(obs[:, 1]) < 1.1)&(obs[:, 2] > .15))) if len(obs) else 0
            per_node[node_id] = {"obstacles": int(len(obs)), "center_obstacles": center,
                                 "ground": int(len(ground)), "status": st}
    return {k: np.vstack(v) if v else np.empty((0, 2)) for k, v in groups.items()}, per_node


def finish(ax, title):
    ax.set_title(title, weight="bold", fontsize=14)
    ax.set_xlabel("X [m] →")
    ax.set_ylabel("Y [m] →")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=.25)


def draw_grid(ax, grid, extent, alpha=1.0):
    show = np.zeros_like(grid, np.uint8)
    show[grid == 0] = 1
    show[grid == 100] = 2
    cmap = ListedColormap([(1, 1, 1, 0), (.88, .89, .91, .42*alpha),
                           (.06, .06, .06, .95*alpha)])
    ax.imshow(show, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")


def plot_maps(base, aux, fits, grid, extent, cell_groups):
    base_xy = np.asarray([p[:2] for p in base.values()])
    aux_ids = np.asarray(sorted(aux))
    aligned = np.asarray([aligned_pose(i, aux[i], fits)[:2] for i in aux_ids])

    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    draw_grid(ax, grid, extent)
    ax.plot(*base_xy.T, color="black", lw=1.1)
    finish(ax, "1. Existing integrated occupancy — BEFORE")
    fig.savefig(OUT/"01_existing_integrated_occupancy.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    for st, color, label in [("exclude", "#d62728", "EXCLUDE/duplicate"),
                             ("hold", "#f1b514", "HOLD"),
                             ("add", "#159447", "ADD candidate")]:
        q = cell_groups[st]
        if len(q): ax.scatter(q[:, 0], q[:, 1], s=.25, c=color, alpha=.32, label=label)
    ax.plot(*aligned.T, color="#2456a6", lw=.8)
    finish(ax, "2. 113208 node-local occupancy — piecewise ALIGNED, scale=1.000")
    ax.legend(markerscale=10)
    fig.savefig(OUT/"02_113208_aligned_occupancy.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    draw_grid(ax, grid, extent, .65)
    q = np.vstack([cell_groups[k] for k in ("exclude", "hold", "add")])
    ax.scatter(q[:, 0], q[:, 1], s=.22, c="#1769d2", alpha=.25,
               label="113208 observed obstacle cells")
    ax.plot(*base_xy.T, color="black", lw=1.0, label="existing trajectory")
    finish(ax, "3. Existing + 113208 occupancy overlay — BEFORE merge")
    ax.legend(markerscale=10)
    fig.savefig(OUT/"03_occupancy_overlay.png", dpi=180); plt.close(fig)


def boundary_analysis(base, aux, fits, grid, extent, cell_groups):
    existing = occupancy_points(grid, extent)
    tree = cKDTree(existing)
    add = cell_groups["add"]
    distance = tree.query(add, workers=-1)[0]
    unique = add[distance > .30]
    duplicate = add[distance <= .18]

    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    draw_grid(ax, grid, extent, .55)
    if len(unique): ax.scatter(unique[:, 0], unique[:, 1], s=.45, c="#00a83b", alpha=.7,
                               label=f"new boundary observations >0.30 m ({len(unique):,})")
    if len(duplicate): ax.scatter(duplicate[:, 0], duplicate[:, 1], s=.20, c="#d62728", alpha=.20,
                                  label=f"duplicate observations ≤0.18 m ({len(duplicate):,})")
    finish(ax, "4. Missing road-boundary evidence supplied by ADD candidates")
    ax.legend(markerscale=8)
    fig.savefig(OUT/"04_missing_road_boundaries.png", dpi=180); plt.close(fig)

    # Direction-aware left/right boundary availability along the retained path.
    union = np.vstack([existing, unique])
    trees = {"before": cKDTree(existing), "after": cKDTree(union)}
    ids = [i for i in sorted(aux) if status(i) == "add"][::12]
    coverage = {}
    for name, tr in trees.items():
        left = right = both = 0
        for i in ids:
            p = aligned_pose(i, aux[i], fits)
            near = tr.query_ball_point(p[:2], 6.0)
            q = tr.data[near]-p[:2]
            c, s = math.cos(p[3]), math.sin(p[3])
            local = q@np.asarray([[c, -s], [s, c]])
            valid = (local[:, 0] > -1.5)&(local[:, 0] < 5.0)
            l = np.any(valid&(local[:, 1] > 1.0)&(local[:, 1] < 5.5))
            r = np.any(valid&(local[:, 1] < -1.0)&(local[:, 1] > -5.5))
            left += l; right += r; both += l and r
        coverage[name] = {"samples": int(len(ids)), "left": int(left),
                          "right": int(right), "both": int(both)}
    return {"new_obstacle_samples": int(len(add)),
            "unique_over_0_30m": int(len(unique)),
            "duplicate_under_0_18m": int(len(duplicate)),
            "coverage": coverage}


def plot_node_status(aux, fits, per_node):
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    colors = {"add": "#159447", "exclude": "#d62728", "hold": "#f1b514"}
    labels = {"add": "ADD", "exclude": "EXCLUDE", "hold": "HOLD"}
    for st in ("exclude", "hold", "add"):
        ids = [i for i in sorted(aux) if status(i) == st]
        xy = np.asarray([aligned_pose(i, aux[i], fits)[:2] for i in ids])
        ax.plot(*xy.T, color=colors[st], lw=3 if st == "add" else 1.8,
                ls="-" if st == "add" else "--", label=labels[st])
    for i in [146, 856, 939, 1457, 1660, 1882, 2302, 2400, 2803, 2851]:
        if i not in aux: continue
        p = aligned_pose(i, aux[i], fits)[:2]
        ax.annotate(str(i), p, xytext=(4, 4), textcoords="offset points", fontsize=8,
                    bbox=dict(fc="white", ec=colors[status(i)], alpha=.9))
    finish(ax, "5. 113208 pre-merge node decision")
    ax.legend()
    fig.savefig(OUT/"05_nodes_to_add.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ids = np.asarray(sorted(per_node))
    center = np.asarray([per_node[i]["center_obstacles"] for i in ids])
    ax.plot(ids, center, color="#555", lw=.7)
    for st in ("exclude", "hold", "add"):
        mask = np.asarray([status(i) == st for i in ids])
        ax.scatter(ids[mask], center[mask], s=7, color=colors[st], alpha=.7, label=labels[st])
    ax.axvspan(1661, 1881, color="#f1b514", alpha=.18, label="people-visible HOLD")
    ax.axvspan(2804, 2851, color="#d62728", alpha=.16, label="garage/vehicles EXCLUDE")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("113208 node id →"); ax.set_ylabel("local forward corridor obstacle cells")
    ax.set_title("6. Duplicate / dynamic-object exclusion evidence", weight="bold")
    ax.grid(True, alpha=.3); ax.legend(ncol=3)
    fig.savefig(OUT/"06_excluded_duplicate_dynamic_nodes.png", dpi=180); plt.close(fig)


def load_rgb(path, ids):
    result = {}
    with ro(path) as con:
        for i in ids:
            row = con.execute("SELECT image FROM Data WHERE id=?", (i,)).fetchone()
            if not row: continue
            im = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
            result[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return result


def contact_sheet(per_node):
    rows = [
        ("start / ramp", 94, 289, "EXCLUDE duplicate; preserve existing ramp"),
        ("upper stop line", 148, 459, "EXCLUDE duplicate; preserve existing Z/pitch"),
        ("right-angle corner", 334, 1457, "ADD boundary candidate"),
        ("intersection entry", 354, 1528, "ADD clean candidate"),
        ("intersection people", 394, 1712, "HOLD dynamic-object review"),
        ("intersection turn", 503, 1987, "ADD curb/island boundary"),
        ("intersection exit", 643, 2154, "ADD crosswalk/edge candidate"),
        ("final return", 1253, 2420, "EXCLUDE duplicate; anchor only"),
        ("arrival / garage", 1290, 2851, "EXCLUDE vehicle/garage"),
    ]
    bi = load_rgb(BASE_DB, [r[1] for r in rows])
    ai = load_rgb(AUX_DB, [r[2] for r in rows])
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 24), constrained_layout=True)
    for rr, (section, b, a, decision) in enumerate(rows):
        color = "#159447" if decision.startswith("ADD") else ("#f1b514" if decision.startswith("HOLD") else "#d62728")
        for cc, (name, nid, image) in enumerate((("integrated/base", b, bi[b]), ("113208", a, ai[a]))):
            axes[rr, cc].imshow(image); axes[rr, cc].axis("off")
            for spine in axes[rr, cc].spines.values():
                spine.set_visible(True); spine.set_color(color); spine.set_linewidth(4)
            extra = ""
            if name == "113208" and nid in per_node:
                extra = f" | obstacle={per_node[nid]['obstacles']} center={per_node[nid]['center_obstacles']}"
            axes[rr, cc].set_title(f"{section}: {name} node {nid}{extra}\n{decision}",
                                   fontsize=8.5, color=color, weight="bold")
    fig.suptitle("7. Segment-by-segment actual RGB comparison", fontsize=17, weight="bold")
    fig.savefig(OUT/"07_segment_rgb_comparison.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, aux = load_poses(BASE_POSES), load_poses(AUX_POSES)
    fits = {name: fit_se2(aux, base, pairs) for name, pairs in ANCHORS.items()}
    grid, extent, resolution = decode_global_map(BASE_DB)
    groups, per_node = load_aux_cells(aux, fits, stride=3)
    plot_maps(base, aux, fits, grid, extent, groups)
    boundary = boundary_analysis(base, aux, fits, grid, extent, groups)
    plot_node_status(aux, fits, per_node)
    contact_sheet(per_node)
    report = {"scale_applied": 1.0, "status_ranges_premerge": STATUS_RANGES,
              "fits": {k: {"theta_deg": math.degrees(math.atan2(v[0][1, 0], v[0][0, 0])),
                            "translation_m": v[1].tolist(),
                            "rms_m": float(np.sqrt(np.mean(v[2]**2))),
                            "residuals_m": v[2].tolist(),
                            "free_scale_diagnostic": v[3], "pairs": ANCHORS[k]}
                       for k, v in fits.items()},
              "boundary": boundary, "grid_resolution": resolution,
              "notes": ["113208 local grids sampled every third stored node for plots",
                        "green cells are observations, not manually drawn boundaries"]}
    (OUT/"premerge_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
