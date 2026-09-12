#!/usr/bin/env python3
"""Generate read-only, data-driven RTAB-Map merge diagnostics.

The script opens one SQLite database at a time with mode=ro/immutable=1.  It
uses raw Node poses for trajectories, the base database's stored optimized
occupancy map for context, and stored RGB/Feature data for anchor sheets.
"""

from __future__ import annotations

import math
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


ROOT = Path("/home/qor/depth_ws")
MAPS = ROOT / "maps"
OUT = ROOT / "analysis" / "map_merge_visualization"

DB_PATHS = {
    "competition_338": MAPS / "competition_338.db",
    "090057": MAPS / "map_20260906_090057" / "rtabmap.db",
    "090420": MAPS / "map_20260906_090420" / "rtabmap.db",
    "091640": MAPS / "map_20260906_091640" / "rtabmap.db",
    "094430": MAPS / "map_20260906_094430" / "rtabmap.db",
    "113208": MAPS / "map_20260906_113208" / "rtabmap.db",
    "root_rtabmap": MAPS / "rtabmap.db",
}

COLORS = {
    "competition_338": "#111111",
    "090057": "#9467bd",
    "090420": "#159447",
    "091640": "#d62728",
    "094430": "#ff8c00",
    "113208": "#1769d2",
    "root_rtabmap": "#7f7f7f",
}


@dataclass
class Trajectory:
    ids: np.ndarray
    stamps: np.ndarray
    xyz: np.ndarray

    def by_id(self, node_id: int) -> np.ndarray:
        index = np.flatnonzero(self.ids == node_id)
        if len(index) != 1:
            raise KeyError(f"node {node_id} not found")
        return self.xyz[index[0]]

    def mask(self, first: int, last: int) -> np.ndarray:
        return (self.ids >= first) & (self.ids <= last)


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def decode_pose(blob: bytes) -> np.ndarray:
    values = np.frombuffer(blob, dtype=np.float32)
    if values.size == 12:
        return np.asarray((values[3], values[7], values[11]), dtype=float)
    if values.size == 16:
        matrix = values.reshape(4, 4)
        return matrix[:3, 3].astype(float)
    raise ValueError(f"unexpected pose blob size: {len(blob)}")


def load_trajectory(path: Path) -> Trajectory:
    # Only compact pose metadata is retained after this database is closed.
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT id,stamp,pose FROM Node WHERE pose IS NOT NULL "
            "ORDER BY stamp,id")
        ids, stamps, xyz = [], [], []
        for node_id, stamp, pose in rows:
            ids.append(node_id)
            stamps.append(stamp)
            xyz.append(decode_pose(pose))
    return Trajectory(np.asarray(ids), np.asarray(stamps), np.asarray(xyz))


def load_base_occupancy(path: Path):
    with connect(path) as connection:
        blob, x_min, y_min, resolution = connection.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution "
            "FROM Admin").fetchone()
    raw = np.frombuffer(zlib.decompress(blob), dtype=np.int8)
    # For this database the stored 89.5 m x 93.4 m grid is 1790 x 1868.
    height, width = 1868, 1790
    if raw.size != height * width:
        raise ValueError(f"unexpected base occupancy size {raw.size}")
    grid = raw.reshape(height, width)
    extent = (x_min, x_min + width * resolution,
              y_min, y_min + height * resolution)
    return grid, extent


def draw_occupancy(ax, grid, extent, alpha=0.52):
    # unknown=transparent white, free=light gray, occupied=dark gray
    shown = np.zeros_like(grid, dtype=np.uint8)
    shown[grid == 0] = 1
    shown[grid == 100] = 2
    cmap = ListedColormap([(1, 1, 1, 0), (0.87, 0.89, 0.91, alpha * 0.45),
                           (0.23, 0.25, 0.27, alpha)])
    ax.imshow(shown, origin="lower", extent=extent, cmap=cmap,
              interpolation="nearest", zorder=0)


def fit_se2(target: Trajectory, base: Trajectory, pairs):
    source = np.asarray([target.by_id(t)[:2] for t, _ in pairs])
    destination = np.asarray([base.by_id(b)[:2] for _, b in pairs])
    sc = source.mean(axis=0)
    dc = destination.mean(axis=0)
    src = source - sc
    dst = destination - dc
    cross = np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0])
    dot = np.sum(src[:, 0] * dst[:, 0] + src[:, 1] * dst[:, 1])
    theta = math.atan2(cross, dot)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.asarray(((c, -s), (s, c)))
    translation = dc - sc @ rotation.T
    aligned = source @ rotation.T + translation
    residual = np.linalg.norm(aligned - destination, axis=1)
    free_scale = np.sum(dst * (src @ rotation.T)) / np.sum(src * src)
    return rotation, translation, residual, free_scale


def align_xy(trajectory: Trajectory, transform):
    rotation, translation = transform[:2]
    return trajectory.xyz[:, :2] @ rotation.T + translation


def annotate_nodes(ax, trajectory, xy, node_ids, color, prefix=""):
    for node_id in node_ids:
        index = np.flatnonzero(trajectory.ids == node_id)
        if not len(index):
            continue
        i = int(index[0])
        ax.annotate(f"{prefix}{node_id}", xy[i], xytext=(5, 5),
                    textcoords="offset points", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec=color,
                              alpha=.82, lw=.6), zorder=20)


def add_direction_arrows(ax, xy, color, count=5, lw=1.2, zorder=8):
    if len(xy) < 3:
        return
    indexes = np.linspace(1, len(xy) - 2, count + 2, dtype=int)[1:-1]
    for index in indexes:
        p0, p1 = xy[index - 1], xy[index + 1]
        if np.linalg.norm(p1 - p0) < 1e-5:
            continue
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=9), zorder=zorder)


def mark_start_end(ax, xy, color, label=None):
    if not len(xy):
        return
    ax.scatter(*xy[0], marker="o", s=52, facecolor="white", edgecolor=color,
               linewidth=1.8, zorder=12)
    ax.scatter(*xy[-1], marker="X", s=60, color=color, edgecolor="white",
               linewidth=.7, zorder=12)
    if label:
        ax.annotate(f"{label} start", xy[0], xytext=(5, -12),
                    textcoords="offset points", fontsize=7, color=color)
        ax.annotate(f"{label} end", xy[-1], xytext=(5, 5),
                    textcoords="offset points", fontsize=7, color=color)


def finish_xy(ax, title, subtitle=None):
    ax.set_xlabel("X [m] →")
    ax.set_ylabel("Y [m] →")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", lw=.45, alpha=.65)
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    if subtitle:
        ax.text(.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color="#444444")


def plot_all_raw(trajectories):
    fig, ax = plt.subplots(figsize=(13.5, 11), constrained_layout=True)
    widths = {"competition_338": 3.0, "090420": 1.8, "091640": 1.8,
              "113208": 1.8, "090057": 1.2, "094430": 1.2,
              "root_rtabmap": 1.2}
    for name, trajectory in trajectories.items():
        xy = trajectory.xyz[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[name], lw=widths[name],
                alpha=.94, label=name, zorder=7 if name == "competition_338" else 5)
        mark_start_end(ax, xy, COLORS[name], name if name in
                       {"competition_338", "090420", "091640", "113208"} else None)
        add_direction_arrows(ax, xy, COLORS[name], count=6 if len(xy) > 100 else 2)
    labels = {
        "competition_338": [1, 285, 393, 728, 944, 1253, 1290],
        "090420": [1, 748, 908, 1103],
        "091640": [1, 772, 1692, 2441, 3258, 3399],
        "113208": [1, 1012, 1457, 2400, 2851],
    }
    for name, node_ids in labels.items():
        annotate_nodes(ax, trajectories[name], trajectories[name].xyz[:, :2],
                       node_ids, COLORS[name], prefix=f"{name}:")
    finish_xy(ax, "All RTAB-Map XY Trajectories — RAW",
              "Native DB map frames overlaid; no rotation, translation, reflection, or scale applied")
    ax.legend(loc="upper left", framealpha=.94, ncol=2)
    ax.text(.99, .01, "○ start   ✕ end   arrows = travel direction",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(fc="white", ec="#999", alpha=.9))

    # Small valid databases are otherwise hidden at the origin.
    inset = ax.inset_axes([.68, .66, .30, .30])
    for name in ("090057", "094430", "root_rtabmap"):
        xy = trajectories[name].xyz[:, :2]
        inset.plot(xy[:, 0], xy[:, 1], color=COLORS[name], lw=1.6, label=name)
        mark_start_end(inset, xy, COLORS[name])
    inset.set_aspect("equal", adjustable="box")
    inset.grid(True, lw=.35, alpha=.6)
    inset.set_title("RAW origin detail", fontsize=9)
    inset.tick_params(labelsize=7)
    inset.legend(fontsize=7, loc="best")
    fig.savefig(OUT / "all_maps_xy_overlay.png", dpi=190, facecolor="white")
    plt.close(fig)


def draw_anchor_residuals(ax, trajectories, aligned, name, pairs, color,
                          maximum=5):
    target = trajectories[name]
    base = trajectories["competition_338"]
    for target_id, base_id in pairs[:maximum]:
        ti = int(np.flatnonzero(target.ids == target_id)[0])
        p = aligned[name][ti]
        q = base.by_id(base_id)[:2]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=.8,
                ls=":", alpha=.75, zorder=13)
        ax.scatter(*q, s=95, facecolor="white", edgecolor=color,
                   linewidth=2.1, zorder=15)
        ax.annotate(f"{name}:{target_id}\n↔ base:{base_id}", q,
                    xytext=(6, 6), textcoords="offset points", fontsize=7,
                    color=color, weight="bold",
                    bbox=dict(boxstyle="round,pad=.18", fc="white", ec=color,
                              alpha=.90, lw=.8), zorder=20)


def plot_selected(trajectories, aligned, transforms, grid, extent, pairs):
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    draw_occupancy(ax, grid, extent)
    base_xy = trajectories["competition_338"].xyz[:, :2]
    ax.plot(*base_xy.T, color="#222", lw=1.8, alpha=.75,
            label="competition_338 BASE (RAW)", zorder=6)

    selected = {
        "113208": [(146, 856), (939, 2302), (2400, 2803)],
        "091640": [(1018, 2457), (2458, 3077)],
    }
    for name in ("113208", "091640"):
        trajectory, xy = trajectories[name], aligned[name]
        selected_mask = np.zeros(len(trajectory.ids), dtype=bool)
        for first, last in selected[name]:
            mask = trajectory.mask(first, last)
            selected_mask |= mask
            ax.plot(xy[mask, 0], xy[mask, 1], color=COLORS[name], lw=4.4,
                    solid_capstyle="round", zorder=10,
                    label=f"{name} selected {first}–{last}")
            add_direction_arrows(ax, xy[mask], COLORS[name], count=3,
                                 lw=1.6, zorder=14)
        # Every non-selected pose remains visible as explicitly excluded context.
        for indexes in np.ma.clump_unmasked(np.ma.masked_where(selected_mask,
                                                               np.arange(len(selected_mask)))):
            section = xy[indexes]
            if len(section) > 1:
                ax.plot(section[:, 0], section[:, 1], color="#858585", lw=1.5,
                        ls="--", alpha=.72, zorder=4)
    draw_anchor_residuals(ax, trajectories, aligned, "113208", pairs["113208"],
                          COLORS["113208"], maximum=5)
    draw_anchor_residuals(ax, trajectories, aligned, "091640", pairs["091640"],
                          COLORS["091640"], maximum=5)
    finish_xy(ax, "Selected Candidate Segments on BASE Occupancy — ALIGNED",
              "Rigid SE(2), scale=1.000 fixed; dotted anchor residuals expose non-rigid disagreement")
    ax.legend(loc="upper left", fontsize=8, framealpha=.94, ncol=2)
    fig.savefig(OUT / "selected_segments_overlay.png", dpi=190, facecolor="white")
    plt.close(fig)


def cumulative_distance(xyz):
    if len(xyz) < 2:
        return np.zeros(len(xyz))
    delta = np.diff(xyz, axis=0)
    return np.r_[0.0, np.cumsum(np.linalg.norm(delta[:, :2], axis=1))]


def plot_height(trajectories):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 11),
                                  gridspec_kw={"height_ratios": [2.2, 1]},
                                  constrained_layout=True)
    for name in ("competition_338", "090420", "091640", "113208"):
        trajectory = trajectories[name]
        distance = cumulative_distance(trajectory.xyz)
        z = trajectory.xyz[:, 2]
        lw = 2.8 if name == "competition_338" else 1.6
        ax.plot(distance, z, color=COLORS[name], lw=lw, alpha=.94,
                label=f"{name} (Z span={np.ptp(z):.2f} m)")
    base = trajectories["competition_338"]
    bmin, bmax = np.argmin(base.xyz[:, 2]), np.argmax(base.xyz[:, 2])
    bd = cumulative_distance(base.xyz)
    ax.annotate(f"BASE abnormal span = {np.ptp(base.xyz[:, 2]):.2f} m",
                xy=(bd[bmax], base.xyz[bmax, 2]),
                xytext=(bd[bmax] + 72, 13.7),
                arrowprops=dict(arrowstyle="->", color="#111", lw=1.4),
                fontsize=10, weight="bold",
                bbox=dict(fc="#fff5bf", ec="#555", alpha=.95))
    ax.scatter([bd[bmin], bd[bmax]], [base.xyz[bmin, 2], base.xyz[bmax, 2]],
               color="#111", s=35, zorder=9)
    ax.set_xlabel("Cumulative XY travel distance [m] →")
    ax.set_ylabel("Raw pose Z [m] ↑")
    ax.grid(True, lw=.45, alpha=.65)
    ax.legend(loc="best", ncol=2, framealpha=.94)
    ax.set_title("Height Profiles from Raw DB Poses", fontsize=15,
                 weight="bold", pad=27)
    ax.text(.5, 1.01, "RAW native poses; each session starts in its own map frame",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")

    compare = [
        ("113208:939\nbase:268", "113208", 939, 268),
        ("113208:1012\nbase:285", "113208", 1012, 285),
        ("113208:1457\nbase:334", "113208", 1457, 334),
        ("091640:772\nbase:307", "091640", 772, 307),
        ("091640:1692\nbase:639", "091640", 1692, 639),
        ("113208:2400\nbase:1251", "113208", 2400, 1251),
        ("113208:2606\nbase:1276", "113208", 2606, 1276),
    ]
    for x, (label, name, target_id, base_id) in enumerate(compare):
        target_z = trajectories[name].by_id(target_id)[2]
        base_z = base.by_id(base_id)[2]
        ax2.plot([x, x], [target_z, base_z], color=COLORS[name], lw=2.3,
                 alpha=.8)
        ax2.scatter(x, base_z, s=55, color="#111", zorder=5)
        ax2.scatter(x, target_z, s=55, color=COLORS[name], zorder=5)
        ax2.text(x, max(base_z, target_z) + .55,
                 f"Δ={base_z-target_z:+.1f}m", ha="center", va="bottom",
                 fontsize=8, weight="bold")
    ax2.set_xticks(range(len(compare)), [x[0] for x in compare], fontsize=8)
    ax2.set_ylabel("Raw pose Z [m]")
    ax2.set_title("Same-place visual anchors: raw Z disagreement", fontsize=11,
                  weight="bold")
    ax2.grid(True, axis="y", lw=.45, alpha=.65)
    ax2.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#111", label="BASE Z"),
        Line2D([], [], marker="o", ls="", color=COLORS["113208"],
               label="Auxiliary Z")], loc="upper right")
    fig.savefig(OUT / "height_comparison.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_base_problems(base, grid, extent):
    disconnected = {2, 3, 4, 7, *range(9, 19), *range(35, 46),
                    66, 67, 68, *range(72, 78), *range(347, 354),
                    *range(707, 718), 1210, 1211, 1212, 1225}
    xy = base.xyz[:, :2]
    normal = np.asarray([node not in disconnected for node in base.ids])
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    draw_occupancy(ax, grid, extent)
    ax.plot(xy[:, 0], xy[:, 1], color="#b9b9b9", lw=1.1, zorder=4)
    ax.scatter(xy[normal, 0], xy[normal, 1], s=5, color="#294f78",
               alpha=.72, label="normal connected graph nodes", zorder=6)
    ax.scatter(xy[~normal, 0], xy[~normal, 1], s=36, marker="x",
               color="#e31a1c", linewidth=1.5,
               label=f"disconnected/not-optimized nodes ({np.sum(~normal)})", zorder=12)
    for first, last, color, label in [
        (392, 393, "#ff8c00", "392→393: 2.11m / 8.87° / 4.27s"),
        (727, 728, "#ff00a8", "727→728: 2.58m / 46.47° / 8.60s")]:
        p, q = base.by_id(first)[:2], base.by_id(last)[:2]
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", lw=3, color=color,
                                    mutation_scale=16), zorder=16)
        ax.annotate(label, (p + q) / 2, xytext=(8, 8), textcoords="offset points",
                    fontsize=9, weight="bold", color=color,
                    bbox=dict(fc="white", ec=color, alpha=.93), zorder=20)
    for first, last, offset in [(768, 780, (8, -22)), (768, 782, (8, 8))]:
        p, q = base.by_id(first)[:2], base.by_id(last)[:2]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#7b2cbf", lw=2.7,
                ls="--", zorder=15)
        ax.scatter([p[0], q[0]], [p[1], q[1]], s=60, facecolor="white",
                   edgecolor="#7b2cbf", linewidth=2, zorder=16)
        ax.annotate(f"suspect closure {first}↔{last}", (p + q) / 2,
                    xytext=offset, textcoords="offset points", fontsize=8,
                    color="#7b2cbf", weight="bold",
                    bbox=dict(fc="white", ec="#7b2cbf", alpha=.92), zorder=20)
    finish_xy(ax, "competition_338 Problem Nodes and Links — RAW",
              "Stored optimized occupancy backdrop; trajectory and marked node positions use raw Node.pose")
    ax.legend(loc="upper left", framealpha=.95)
    ax.text(.985, .018,
            "56 disconnected IDs:\n2–4, 7, 9–18, 35–45, 66–68, 72–77,\n"
            "347–353, 707–717, 1210–1212, 1225",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#b30000",
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#e31a1c",
                      alpha=.94), zorder=25)
    fig.savefig(OUT / "base_problem_nodes.png", dpi=190, facecolor="white")
    plt.close(fig)


def plot_merge_plan(trajectories, aligned, grid, extent, pairs):
    fig, ax = plt.subplots(figsize=(15, 11), constrained_layout=True)
    draw_occupancy(ax, grid, extent, alpha=.42)
    base_xy = trajectories["competition_338"].xyz[:, :2]
    ax.plot(*base_xy.T, color="#111", lw=3.0, label="BASE retained",
            zorder=8)

    # Excluded context is rendered before additions.
    for name in ("113208", "091640"):
        ax.plot(*aligned[name].T, color="#aaa", lw=1.0, ls="--", alpha=.7,
                zorder=3)
    for first, last in [(146, 856), (939, 2302), (2400, 2803)]:
        mask = trajectories["113208"].mask(first, last)
        section = aligned["113208"][mask]
        ax.plot(*section.T, color=COLORS["113208"], lw=4.3,
                label=f"113208 add {first}–{last}", zorder=11)
        add_direction_arrows(ax, section, COLORS["113208"], count=3,
                             lw=1.6, zorder=15)
    for first, last in [(1018, 2457), (2458, 3077)]:
        mask = trajectories["091640"].mask(first, last)
        section = aligned["091640"][mask]
        ax.plot(*section.T, color=COLORS["091640"], lw=4.3,
                label=f"091640 2-lane add {first}–{last}", zorder=10)
        add_direction_arrows(ax, section, COLORS["091640"], count=4,
                             lw=1.6, zorder=15)
    ax.plot(*aligned["090420"].T, color=COLORS["090420"], lw=2.2,
            ls="--", alpha=.9, label="090420 alignment validation only",
            zorder=7)

    sequence = [
        ("1", "113208", 2400, 1251, "return anchor first"),
        ("2", "113208", 1012, 285, "start/intersection"),
        ("3", "091640", 772, 307, "2-lane submaps"),
        ("4", "090420", 908, 334, "validation only"),
    ]
    base = trajectories["competition_338"]
    for number, name, target_id, base_id, text in sequence:
        target = trajectories[name]
        ti = int(np.flatnonzero(target.ids == target_id)[0])
        p = aligned[name][ti]
        q = base.by_id(base_id)[:2]
        color = COLORS[name]
        ax.annotate("", xy=q, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                    mutation_scale=15), zorder=18)
        ax.scatter(*q, s=170, facecolor="white", edgecolor=color,
                   linewidth=2.5, zorder=19)
        ax.text(q[0], q[1], number, ha="center", va="center", color=color,
                fontsize=10, weight="bold", zorder=20)
        ax.annotate(f"{number}. {text}\n{name}:{target_id} → base:{base_id}", q,
                    xytext=(10, 10), textcoords="offset points", fontsize=8,
                    weight="bold", color=color,
                    bbox=dict(fc="white", ec=color, alpha=.93), zorder=20)
    finish_xy(ax, "Proposed Integrated Map Geometry — ALIGNED DATA",
              "Actual raw trajectories transformed by rigid SE(2), scale=1.000; gray = excluded context")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8,
              framealpha=.95, ncol=2)
    fig.savefig(OUT / "merge_plan.png", dpi=190, facecolor="white")
    plt.close(fig)


def load_images(path: Path, node_ids):
    output = {}
    with connect(path) as connection:
        for node_id in node_ids:
            row = connection.execute(
                "SELECT image FROM Data WHERE id=?", (node_id,)).fetchone()
            if not row or not row[0]:
                raise KeyError(f"image missing: {path}:{node_id}")
            image = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
            output[node_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return output


def load_features(path: Path, node_ids):
    output = {}
    with connect(path) as connection:
        for node_id in node_ids:
            points, descriptors = [], []
            for x, y, descriptor in connection.execute(
                    "SELECT pos_x,pos_y,descriptor FROM Feature WHERE node_id=?",
                    (node_id,)):
                if descriptor and len(descriptor) == 32:
                    points.append((x, y))
                    descriptors.append(np.frombuffer(descriptor, np.uint8))
            output[node_id] = (np.asarray(points, np.float32),
                               np.asarray(descriptors, np.uint8))
    return output


def geometric_matches(base_feature, target_feature):
    base_points, base_desc = base_feature
    target_points, target_desc = target_feature
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(base_desc, target_desc, k=2)
    good = [first for first, second in matches if first.distance < .80 * second.distance]
    source = np.asarray([base_points[m.queryIdx] for m in good])
    destination = np.asarray([target_points[m.trainIdx] for m in good])
    _, mask = cv2.findFundamentalMat(source, destination, cv2.FM_RANSAC, 2.0, .995)
    if mask is None:
        return []
    return [match for match, keep in zip(good, mask.ravel()) if keep]


def plot_contact_sheets(images, features):
    requested = [
        ("090420", 908, 334),
        ("091640", 772, 307),
        ("113208", 1012, 285),
        ("113208", 2420, 1253),
        ("113208", 2430, 1253),
    ]
    match_rows = []
    for name, target_id, base_id in requested:
        matches = geometric_matches(features["competition_338"][base_id],
                                    features[name][target_id])
        match_rows.append((name, target_id, base_id, matches))

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)
    for row, (name, target_id, base_id, matches) in enumerate(match_rows):
        axes[row, 0].imshow(images[name][target_id])
        axes[row, 1].imshow(images["competition_338"][base_id])
        axes[row, 0].text(.5, -.035,
                          f"{name} | node {target_id} | {len(matches)} geometric inliers",
                          transform=axes[row, 0].transAxes, ha="center", va="top",
                          fontsize=10, weight="bold", color=COLORS[name])
        axes[row, 1].text(.5, -.035,
                          f"competition_338 | node {base_id} | {len(matches)} geometric inliers",
                          transform=axes[row, 1].transAxes, ha="center", va="top",
                          fontsize=10, weight="bold", color="#111111")
        for column in range(2):
            axes[row, column].axis("off")
    fig.suptitle("Anchor RGB Pairs from Stored RTAB-Map Images", fontsize=17,
                 weight="bold")
    fig.savefig(OUT / "anchor_matches_contact_sheet.png", dpi=170,
                facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(16, 15), constrained_layout=True)
    rng = np.random.default_rng(338)
    for ax, (name, target_id, base_id, matches) in zip(axes, match_rows):
        left = images[name][target_id]
        right = images["competition_338"][base_id]
        canvas = np.concatenate((left, right), axis=1)
        ax.imshow(canvas)
        base_points, _ = features["competition_338"][base_id]
        target_points, _ = features[name][target_id]
        selected = matches
        if len(selected) > 45:
            indexes = np.linspace(0, len(selected)-1, 45, dtype=int)
            selected = [selected[i] for i in indexes]
        colors = plt.cm.turbo(rng.random(len(selected)))
        for match, color in zip(selected, colors):
            source = target_points[match.trainIdx]
            destination = base_points[match.queryIdx] + np.asarray((left.shape[1], 0))
            ax.plot([source[0], destination[0]], [source[1], destination[1]],
                    color=color, lw=.8, alpha=.85)
            ax.scatter([source[0], destination[0]], [source[1], destination[1]],
                       s=5, color=color)
        ax.set_title(f"{name}:{target_id}  ↔  competition_338:{base_id}   |   "
                     f"{len(matches)} geometric inliers", fontsize=10,
                     weight="bold")
        ax.axis("off")
    fig.suptitle("Stored BRIEF Feature Matches after Fundamental-Matrix RANSAC",
                 fontsize=16, weight="bold")
    fig.savefig(OUT / "anchor_feature_matches.png", dpi=180, facecolor="white")
    plt.close(fig)
    return [(name, target_id, base_id, len(matches))
            for name, target_id, base_id, matches in match_rows]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # SQLite DBs are opened and closed strictly one at a time here.
    trajectories = {name: load_trajectory(path) for name, path in DB_PATHS.items()}
    grid, extent = load_base_occupancy(DB_PATHS["competition_338"])
    pairs = {
        "090420": [(908, 334), (748, 280), (768, 284), (903, 332), (933, 343)],
        "091640": [(772, 307), (1692, 639), (2441, 944),
                   (2894, 1134), (3258, 1254)],
        "113208": [(1012, 285), (1243, 314), (1457, 334), (2154, 643),
                   (2400, 1251), (2420, 1253), (2606, 1276)],
    }
    transforms, aligned = {}, {}
    for name in ("090420", "091640", "113208"):
        result = fit_se2(trajectories[name], trajectories["competition_338"],
                         pairs[name])
        transforms[name] = result
        aligned[name] = align_xy(trajectories[name], result)
        rotation, translation, residual, free_scale = result
        theta = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        print(f"ALIGN {name}: fixed_scale=1.000 theta_deg={theta:.6f} "
              f"tx={translation[0]:.6f} ty={translation[1]:.6f} "
              f"anchor_rms={math.sqrt(np.mean(residual**2)):.6f} "
              f"free_similarity_scale={free_scale:.6f}")

    plot_all_raw(trajectories)
    plot_selected(trajectories, aligned, transforms, grid, extent, pairs)
    plot_height(trajectories)
    plot_base_problems(trajectories["competition_338"], grid, extent)
    plot_merge_plan(trajectories, aligned, grid, extent, pairs)

    requested = {
        "competition_338": [285, 307, 334, 1253],
        "090420": [908], "091640": [772], "113208": [1012, 2420, 2430],
    }
    images, features = {}, {}
    for name, node_ids in requested.items():
        images[name] = load_images(DB_PATHS[name], node_ids)
        features[name] = load_features(DB_PATHS[name], node_ids)
    print("FEATURE_MATCHES", plot_contact_sheets(images, features))
    for path in sorted(OUT.glob("*.png")):
        print(f"OUTPUT {path.name} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
