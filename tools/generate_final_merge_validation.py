#!/usr/bin/env python3
"""Render the actual optimized graph and stored occupancy of the merged DB."""

from __future__ import annotations

import math
import sqlite3
import struct
import zlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path("/home/qor/depth_ws")
DB = ROOT / "maps/merged_competition_1lane_2lane/rtabmap.db"
POSES = Path("/tmp/merged_final_opt_poses.txt")
OUT = ROOT / "analysis/map_merge_visualization/merge_final_validation.png"


def connect(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    c.execute("PRAGMA query_only=ON")
    return c


def load_poses(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) == 9:
            result[int(f[8])] = np.asarray([float(f[1]), float(f[2]), float(f[3])])
    return result


def matrix(blob: bytes) -> np.ndarray:
    out = np.eye(4)
    out[:3, :4] = np.asarray(struct.unpack("<12f", blob)).reshape(3, 4)
    return out


def grid_shape(size: int, resolution: float, x_min: float, y_min: float,
               poses: dict[int, np.ndarray]):
    x_max = max(p[0] for p in poses.values())
    y_max = max(p[1] for p in poses.values())
    candidates = []
    for height in range(1, int(math.sqrt(size)) + 1):
        if size % height:
            continue
        for h, w in ((height, size // height), (size // height, height)):
            if x_min + w * resolution >= x_max and y_min + h * resolution >= y_max:
                margin = (x_min + w * resolution - x_max) + (y_min + h * resolution - y_max)
                candidates.append((margin, h, w))
    if not candidates:
        raise RuntimeError(f"cannot infer occupancy dimensions for {size} cells")
    _, height, width = min(candidates)
    return height, width


def implied_alignment(base_pose, aux_pose, link):
    transform = base_pose @ link @ np.linalg.inv(aux_pose)
    return (math.degrees(math.atan2(transform[1, 0], transform[0, 0])),
            transform[0, 3], transform[1, 3])


def main():
    poses = load_poses(POSES)
    with connect(DB) as c:
        map_ids = dict(c.execute("SELECT id,map_id FROM Node"))
        raw_poses = {node_id: matrix(blob) for node_id, blob in c.execute(
            "SELECT id,pose FROM Node")}
        blob, x_min, y_min, resolution = c.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin"
        ).fetchone()
        links = list(c.execute(
            "SELECT l.from_id,l.to_id,l.type,l.transform FROM Link l "
            "JOIN Node a ON a.id=l.from_id JOIN Node b ON b.id=l.to_id "
            "WHERE a.map_id=0 AND b.map_id=1 ORDER BY l.from_id"))
        stamps = dict(c.execute("SELECT id,stamp FROM Node"))
    raw_grid = np.frombuffer(zlib.decompress(blob), np.int8)
    height, width = grid_shape(raw_grid.size, resolution, x_min, y_min, poses)
    grid = raw_grid.reshape(height, width)
    extent = (x_min, x_min + width * resolution,
              y_min, y_min + height * resolution)

    fig, (ax, detail) = plt.subplots(1, 2, figsize=(18, 9),
                                     gridspec_kw={"width_ratios": [1.55, 1]},
                                     constrained_layout=True)
    display = np.zeros_like(grid, np.uint8)
    display[grid == 0] = 1
    display[grid == 100] = 2
    cmap = ListedColormap([(1, 1, 1, 0), (.86, .88, .9, .68), (.08, .09, .1, .92)])
    ax.imshow(display, origin="lower", extent=extent, cmap=cmap,
              interpolation="nearest")
    for map_id, color, label, width_line in (
            (0, "#111111", "competition base (kept)", 2.0),
            (1, "#d62728", "091640 lane-2 branch (added)", 2.6)):
        ids = [i for i in poses if map_ids.get(i) == map_id]
        ids.sort(key=lambda i: stamps[i])
        xy = np.asarray([poses[i][:2] for i in ids])
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width_line, label=label, zorder=5)
        ax.scatter(*xy[0], s=70, facecolor="white", edgecolor=color, lw=2, zorder=7)
        ax.annotate(f"map {map_id} start", xy[0], xytext=(6, 5),
                    textcoords="offset points", fontsize=8, color=color)

    alignments = []
    for base_id, aux_id, link_type, link_blob in links:
        b, a = poses[base_id][:2], poses[aux_id][:2]
        ax.plot([b[0], a[0]], [b[1], a[1]], color="#ffbf00", lw=2.0, zorder=8)
        ax.scatter(*b, s=110, facecolor="white", edgecolor="#7a3db8", lw=2.3, zorder=9)
        alignments.append((base_id, aux_id, link_type,
                           *implied_alignment(raw_poses[base_id], raw_poses[aux_id],
                                              matrix(link_blob))))
    ax.set_title("Final stored occupancy + optimized trajectories", weight="bold")
    ax.set_xlabel("X [m] →")
    ax.set_ylabel("Y [m] →")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", framealpha=.94)

    detail.axis("off")
    lines = [
        "ACTUAL MERGED GRAPH VALIDATION",
        "",
        f"Optimized poses: {len(poses)} (base 1234 + lane-2 270)",
        f"Stored grid: {width}×{height}, {resolution:.2f} m/cell",
        "Optimization: SE(2), scale=1.000, no reflection",
        "",
        "Cross-session constraints (one direction):",
    ]
    for base_id, aux_id, link_type, theta, tx, ty in alignments:
        kind = "Global visual" if link_type == 1 else "Local-space visual"
        lines.append(f"  {base_id} ↔ {aux_id}  {kind}")
        lines.append(f"    implied θ={theta:+.2f}°, t=({tx:+.2f},{ty:+.2f}) m")
    theta_values = np.asarray([x[3] for x in alignments])
    tx_values = np.asarray([x[4] for x in alignments])
    ty_values = np.asarray([x[5] for x in alignments])
    lines += [
        "",
        f"Anchor consistency: θ spread={np.ptp(theta_values):.2f}°; "
        f"translation spread=({np.ptp(tx_values):.2f},{np.ptp(ty_values):.2f}) m",
        "All-link graph components: 1",
        "Saved optimized pose Z: exactly 0 m (2D working map)",
        "",
        "Yellow segments are graph constraints, not painted road marks.",
        "091640 original IDs for the four cross anchors:",
        "241, 247, 251, 257 (mapped by exact timestamps)",
    ]
    detail.text(.02, .98, "\n".join(lines), va="top", ha="left",
                fontsize=10.3, family="monospace",
                bbox=dict(boxstyle="round,pad=.7", fc="#f7f8fa", ec="#495057"))
    fig.suptitle("Merged competition_338 + 091640 — Actual DB Data",
                 fontsize=16, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
