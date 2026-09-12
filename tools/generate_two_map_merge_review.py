#!/usr/bin/env python3
"""Visual review for competition_338 + 091640, read-only and low-memory."""

from __future__ import annotations

import math
import sqlite3
import zlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from generate_map_merge_visualizations import (
    DB_PATHS, COLORS, Trajectory, align_xy, connect, decode_pose,
    draw_occupancy, finish_xy, fit_se2, load_base_occupancy, load_trajectory,
)

OUT = Path("/home/qor/depth_ws/analysis/map_merge_visualization")


def occupancy_091640():
    with connect(DB_PATHS["091640"]) as connection:
        blob, x0, y0, resolution = connection.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin"
        ).fetchone()
    raw = np.frombuffer(zlib.decompress(blob), np.int8)
    height, width = 1869, 1800
    assert raw.size == height * width
    grid = raw.reshape(height, width)
    return grid, (x0, x0 + width * resolution,
                  y0, y0 + height * resolution)


def occupied_points(grid, extent, stride=1):
    yy, xx = np.where(grid == 100)
    if stride > 1:
        yy, xx = yy[::stride], xx[::stride]
    dx = (extent[1] - extent[0]) / grid.shape[1]
    dy = (extent[3] - extent[2]) / grid.shape[0]
    return np.c_[extent[0] + (xx + .5) * dx,
                 extent[2] + (yy + .5) * dy]


def plot_raw_aligned(base, aux, global_transform, branch_transform):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, title, aux_xy in [
        (axes[0], "RAW native frames", aux.xyz[:, :2]),
        (axes[1], "ALIGNED global SE(2), scale=1.000", align_xy(aux, global_transform)),
    ]:
        ax.plot(*base.xyz[:, :2].T, color="#111", lw=2.4, label="competition_338")
        ax.plot(*aux_xy.T, color=COLORS["091640"], lw=1.7, label="091640")
        ax.scatter(*base.xyz[0, :2], s=80, facecolor="white", edgecolor="#111", lw=2)
        ax.scatter(*aux_xy[0], s=80, facecolor="white", edgecolor=COLORS["091640"], lw=2)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("X [m] →"); ax.set_ylabel("Y [m] →")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.45)
        ax.legend()
    fig.suptitle("competition_338 vs 091640 Trajectory Registration", fontsize=16,
                 weight="bold")
    fig.savefig(OUT / "merge_trajectory_raw_vs_aligned.png", dpi=180)
    plt.close(fig)

    # Branch-specific local alignment exposes lateral lane separation without
    # allowing distant-course drift to pull the starting corridor.
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    bmask = base.mask(1, 370)
    amask = aux.mask(186, 812)
    excluded_mask = aux.mask(813, 1018)
    branch_xy = align_xy(aux, branch_transform)
    ax.plot(*base.xyz[bmask, :2].T, color="#159447", lw=3.2,
            label="competition selected 1–370")
    ax.plot(*branch_xy[amask].T, color="#d62728", lw=3.2,
            label="091640 added 186–812")
    ax.plot(*branch_xy[excluded_mask].T, color="#777777", lw=1.8, ls="--",
            label="091640 excluded duplicate 813–1018")
    for name, trajectory, xy, ids, color in [
        ("base", base, base.xyz[:, :2], [1, 100, 200, 280, 307, 334, 370], "#159447"),
        ("091640", aux, branch_xy, [186, 306, 546, 698, 772, 812], "#d62728")]:
        for nid in ids:
            ii = np.flatnonzero(trajectory.ids == nid)
            if not len(ii): continue
            p = xy[int(ii[0])]
            ax.scatter(*p, s=38, color=color, zorder=8)
            ax.annotate(f"{name}:{nid}", p, xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=color,
                        bbox=dict(fc="white", ec=color, alpha=.85, pad=.15))
    ax.scatter(*base.by_id(307)[:2], s=170, facecolor="white", edgecolor="#6a3d9a",
               lw=2.5, zorder=10)
    ax.annotate("branch rejoins / strongest anchor\n091640:772 ↔ base:307",
                base.by_id(307)[:2], xytext=(15, -35), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#6a3d9a"), fontsize=9,
                weight="bold", color="#6a3d9a",
                bbox=dict(fc="white", ec="#6a3d9a", alpha=.92))
    finish_xy(ax, "Starting-lane Branch — Locally ALIGNED (actual selection)",
              "Rigid SE(2), scale=1.000; red=added 186–812; gray=excluded duplicate")
    ax.legend(loc="best")
    fig.savefig(OUT / "merge_start_lane_branch.png", dpi=190)
    plt.close(fig)


def plot_occupancy_overlay(base_grid, base_extent, aux_grid, aux_extent,
                           transform, base, aux):
    base_occ = occupied_points(base_grid, base_extent, 2)
    aux_occ = occupied_points(aux_grid, aux_extent, 2)
    rotation, translation = transform[:2]
    aux_aligned = aux_occ @ rotation.T + translation
    aux_traj = align_xy(aux, transform)
    fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
    ax.scatter(base_occ[:, 0], base_occ[:, 1], s=.35, color="#111", alpha=.28,
               label="competition occupied cells")
    ax.scatter(aux_aligned[:, 0], aux_aligned[:, 1], s=.35, color="#e31a1c", alpha=.25,
               label="091640 occupied cells (aligned)")
    ax.plot(*base.xyz[:, :2].T, color="#111", lw=1.2, alpha=.7)
    ax.plot(*aux_traj.T, color="#e31a1c", lw=1.1, alpha=.7)
    # Areas near high-residual anchors are factual disagreement review zones.
    for aux_id, base_id in [(772, 307), (1692, 639), (2441, 944),
                            (2894, 1134), (3258, 1254)]:
        q = base.by_id(base_id)[:2]
        p = aux_traj[int(np.flatnonzero(aux.ids == aux_id)[0])]
        residual = np.linalg.norm(p-q)
        color = "#ffbf00" if residual < 4 else "#d62728"
        ax.plot([p[0],q[0]],[p[1],q[1]], color=color, ls=":", lw=1.4)
        ax.scatter(*q, s=80, facecolor="white", edgecolor=color, lw=2)
        ax.annotate(f"{aux_id}↔{base_id}\nres={residual:.1f}m", q,
                    xytext=(5,5), textcoords="offset points", fontsize=7,
                    color=color, weight="bold",
                    bbox=dict(fc="white", ec=color, alpha=.88, pad=.15))
    finish_xy(ax, "Stored Occupancy Maps — ALIGNED Overlay",
              "Black=competition, red=091640; actual occupied cells; rigid SE(2), scale=1.000")
    ax.legend(markerscale=8, framealpha=.94)
    fig.savefig(OUT / "merge_occupancy_overlay.png", dpi=190)
    plt.close(fig)


def load_rgb(path, ids):
    out = {}
    with connect(path) as connection:
        for nid in ids:
            blob = connection.execute("SELECT image FROM Data WHERE id=?", (nid,)).fetchone()[0]
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            out[nid] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return out


def plot_selection_sheet():
    base_ids = [1, 61, 91, 131, 285, 307, 334, 351, 361, 370]
    aux_ids = [186, 306, 546, 698, 728, 772, 812, 906, 1018]
    base_images = load_rgb(DB_PATHS["competition_338"], base_ids)
    aux_images = load_rgb(DB_PATHS["091640"], aux_ids)
    # Actual local obstacle-cell counts provide a depth/occupancy check. Counts
    # alone do not classify a person, so visible-person frames remain AMBIGUOUS.
    def counts(path, ids):
        result = {}
        with connect(path) as connection:
            for nid in ids:
                blob = connection.execute("SELECT obstacle_cells FROM Data WHERE id=?", (nid,)).fetchone()[0]
                result[nid] = len(zlib.decompress(blob)) // 16 if blob else 0
        return result
    bc, ac = counts(DB_PATHS["competition_338"], base_ids), counts(DB_PATHS["091640"], aux_ids)
    rows = [
        ("GREEN keep: lane branches", ("base", 1), ("aux", 186), "green"),
        ("YELLOW: people visible; occupancy check", ("base", 91), ("base", 131), "gold"),
        ("GREEN: clean 091640 departure", ("aux", 306), ("aux", 546), "green"),
        ("GREEN/YELLOW overlap anchors", ("aux", 698), ("base", 285), "gold"),
        ("GREEN strongest anchor", ("aux", 772), ("base", 307), "green"),
        ("RED duplicate common course", ("aux", 906), ("base", 334), "red"),
        ("YELLOW: people visible; local grid", ("base", 351), ("base", 361), "gold"),
    ]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 18), constrained_layout=True)
    for rr, (title, left, right, status) in enumerate(rows):
        for cc, (source, nid) in enumerate((left,right)):
            images = base_images if source == "base" else aux_images
            cnt = bc[nid] if source == "base" else ac[nid]
            name = "competition" if source == "base" else "091640"
            axes[rr,cc].imshow(images[nid]); axes[rr,cc].axis("off")
            for spine in axes[rr,cc].spines.values():
                spine.set_visible(True); spine.set_color(status); spine.set_linewidth(5)
            axes[rr,cc].set_title(
                f"{title}\n{name}:{nid} | local obstacle cells={cnt}",
                fontsize=8.3, color=status, weight="bold", pad=4)
    fig.suptitle("Node Selection Review: GREEN keep / RED exclude / YELLOW hold",
                 fontsize=16, weight="bold")
    fig.savefig(OUT / "merge_node_selection_review.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_trajectory(DB_PATHS["competition_338"])
    aux = load_trajectory(DB_PATHS["091640"])
    global_pairs = [(772,307),(1692,639),(2441,944),(2894,1134),(3258,1254)]
    # Two anchors at the painted start and two at the rejoin.  These span the
    # whole departure branch and avoid the ill-conditioned short-baseline fit.
    local_pairs = [(243,5),(251,21),(772,307),(812,313)]
    global_transform = fit_se2(aux, base, global_pairs)
    branch_transform = fit_se2(aux, base, local_pairs)
    for name, result in (("GLOBAL",global_transform),("BRANCH",branch_transform)):
        r,t,e,s=result
        print(name, "theta_deg", math.degrees(math.atan2(r[1,0],r[0,0])),
              "translation",t,"rms",math.sqrt(np.mean(e*e)),"free_scale",s,
              "residuals",e)
    bg,be=load_base_occupancy(DB_PATHS["competition_338"])
    ag,ae=occupancy_091640()
    plot_raw_aligned(base,aux,global_transform,branch_transform)
    plot_occupancy_overlay(bg,be,ag,ae,global_transform,base,aux)
    plot_selection_sheet()
    for path in sorted(OUT.glob("merge_*.png")):
        print(path.name,path.stat().st_size)


if __name__ == "__main__":
    main()
