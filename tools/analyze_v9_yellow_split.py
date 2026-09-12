#!/usr/bin/env python3
"""Locate duplicated yellow-road markings in the v8 stored cloud."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/qor/depth_ws")
PLY = ROOT / "analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"
OUT = ROOT / "analysis/level_aligned_v9/work"
DB = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
MARKINGS = OUT / "markings"


def load_ply(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    count = int(next(line.split()[2] for line in header.decode().splitlines()
                     if line.startswith("element vertex")))
    dtype = np.dtype({
        "names": ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "curvature"],
        "formats": ["<f4"] * 6 + ["u1"] * 3 + ["<f4"],
        "offsets": [0, 4, 8, 12, 16, 20, 24, 25, 26, 27],
        "itemsize": 31,
    })
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def yellow_mask(cloud):
    rgb = np.column_stack((cloud["r"], cloud["g"], cloud["b"])).astype(np.uint8)
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3)[:, :, ::-1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Broad yellow/orange marking candidate.  Geometry, not this color mask alone,
    # is used for the final correspondence measurement.
    return ((hsv[:, 0] >= 8) & (hsv[:, 0] <= 38) &
            (hsv[:, 1] >= 62) & (hsv[:, 2] >= 85) &
            (cloud["z"] >= -0.18) & (cloud["z"] <= 0.28))


def load_markings(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    count = int(next(line.split()[2] for line in header.decode().splitlines()
                     if line.startswith("element vertex")))
    dtype = np.dtype({"names": ["x", "y", "z", "type", "node"],
                      "formats": ["<f4", "<f4", "<f4", "u1", "<i4"],
                      "offsets": [0, 4, 8, 12, 13], "itemsize": 17})
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def marking_crop_figure():
    bbox = (18.0, 16.0, 26.2, 37.0)
    colors = {0: "#111111", 1: "#d81b60", 2: "#1976d2", 3: "#7b1fa2",
              4: "#00897b", 5: "#fb8c00"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 17), constrained_layout=True)
    node_summary = {}
    for map_id in range(6):
        cloud = load_markings(MARKINGS / f"map{map_id}.ply")
        m = ((cloud["type"] == 2) & (cloud["x"] >= bbox[0]) & (cloud["x"] <= bbox[2]) &
             (cloud["y"] >= bbox[1]) & (cloud["y"] <= bbox[3]))
        p = cloud[m]
        if not len(p):
            continue
        axes[0].scatter(p["x"], p["y"], s=.6, color=colors[map_id], alpha=.55,
                        linewidths=0, rasterized=True, label=f"map_id={map_id}")
        ids, counts = np.unique(p["node"], return_counts=True)
        node_summary[str(map_id)] = {str(int(i)): int(n) for i, n in zip(ids, counts)}
        # Color each contributing node to expose consecutive observation bands.
        axes[1].scatter(p["x"], p["y"], s=.55, c=p["node"], cmap="turbo", alpha=.6,
                        linewidths=0, rasterized=True, label=f"map_id={map_id}")
    for ax, title in zip(axes, ("yellow RGB-D observations by map/session",
                                "same observations colored by integrated node ID")):
        ax.set_facecolor("#eeeeee")
        ax.set_aspect("equal")
        ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]),
               xlabel="map X [m] →", ylabel="map Y [m] →", title=title)
        ax.grid(alpha=.25)
    axes[0].legend(loc="lower right")
    fig.savefig(OUT / "v8_split_crop_by_source.png", dpi=230)
    plt.close(fig)

    # Pose centers in and around the crop, recorded independently of projected points.
    pose_rows = []
    db = sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1", uri=True)
    for node_id, map_id, blob in db.execute("SELECT id,map_id,pose FROM Node ORDER BY id"):
        pose = np.eye(4); pose[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
        x, y = pose[0, 3], pose[1, 3]
        if bbox[0] - 4 <= x <= bbox[2] + 4 and bbox[1] - 4 <= y <= bbox[3] + 4:
            pose_rows.append({"node": node_id, "map_id": map_id, "x": float(x), "y": float(y),
                              "z": float(pose[2, 3]),
                              "yaw_deg": float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))})
    db.close()
    (OUT / "v8_split_crop_sources.json").write_text(json.dumps({
        "bbox": bbox, "yellow_points_by_node": node_summary, "nearby_pose_rows": pose_rows,
    }, indent=2) + "\n")


def actual_cloud_crop_figure(cloud):
    bbox = (20.5, 26.0, 23.7, 32.5)
    m = ((cloud["x"] >= bbox[0]) & (cloud["x"] <= bbox[2]) &
         (cloud["y"] >= bbox[1]) & (cloud["y"] <= bbox[3]) &
         (cloud["z"] >= -0.15) & (cloud["z"] <= 0.25))
    p = cloud[m]
    rgb = np.column_stack((p["r"], p["g"], p["b"])).astype(float) / 255.0
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8).reshape(-1, 1, 3)[:, :, ::-1],
                       cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ym = ((hsv[:, 0] >= 8) & (hsv[:, 0] <= 38) & (hsv[:, 1] >= 62) & (hsv[:, 2] >= 85))
    fig, axes = plt.subplots(1, 2, figsize=(11, 16), constrained_layout=True)
    axes[0].set_facecolor("#111111")
    axes[0].scatter(p["x"], p["y"], c=rgb, s=1.1, linewidths=0, rasterized=True)
    axes[1].set_facecolor("#111111")
    axes[1].scatter(p["x"][~ym], p["y"][~ym], c="#555555", s=.35, alpha=.18,
                    linewidths=0, rasterized=True)
    axes[1].scatter(p["x"][ym], p["y"][ym], c="#ffd400", s=2.2, alpha=.9,
                    linewidths=0, rasterized=True)
    for ax, title in zip(axes, ("actual stored RGB cloud", "yellow candidate geometry")):
        ax.set_aspect("equal"); ax.grid(alpha=.16); ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]),
            xlabel="map X [m] →", ylabel="map Y [m] →", title=title)
    fig.savefig(OUT / "v8_actual_split_closeup.png", dpi=260)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cloud = load_ply(PLY)
    mask = yellow_mask(cloud)
    yellow = np.column_stack((cloud["x"][mask], cloud["y"][mask], cloud["z"][mask]))
    bg_idx = np.arange(0, len(cloud), max(1, len(cloud) // 300_000))

    fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
    ax.set_facecolor("#111111")
    ax.scatter(cloud["x"][bg_idx], cloud["y"][bg_idx], s=.12, c="#666666", alpha=.22,
               linewidths=0, rasterized=True)
    ax.scatter(yellow[:, 0], yellow[:, 1], s=1.6, c="#ffd400", alpha=.82,
               linewidths=0, rasterized=True)
    for x in np.arange(-10, 71, 5):
        ax.axvline(x, color="white", lw=.25, alpha=.18)
    for y in np.arange(-5, 86, 5):
        ax.axhline(y, color="white", lw=.25, alpha=.18)
    ax.set_aspect("equal")
    ax.set(xlabel="map X [m] →", ylabel="map Y [m] →",
           title="v8 actual stored cloud — yellow/orange road-marking candidates")
    fig.savefig(OUT / "v8_yellow_candidates_full.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(OUT / "v8_yellow_candidates.npz", xyz=yellow)
    (OUT / "yellow_candidate_summary.json").write_text(json.dumps({
        "ply": str(PLY), "cloud_points": int(len(cloud)),
        "yellow_candidate_points": int(len(yellow)),
        "bounds_xyz": [[float(yellow[:, i].min()), float(yellow[:, i].max())] for i in range(3)],
        "mask": "HSV H=8..38 S>=62 V>=85, map Z=-0.18..0.28",
    }, indent=2) + "\n")
    actual_cloud_crop_figure(cloud)
    marking_crop_figure()


if __name__ == "__main__":
    main()
