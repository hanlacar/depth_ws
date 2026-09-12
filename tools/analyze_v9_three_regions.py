#!/usr/bin/env python3
"""Attribute the three photographed v8 defects to map/session node groups."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/qor/depth_ws")
DB = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
PLY = ROOT / "analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"
MARK = ROOT / "analysis/level_aligned_v9/work/markings"
OUT = ROOT / "analysis/level_aligned_v9_three/work"

REGIONS = {
    "yellow_split": (18.0, 25.0, 25.0, 33.0),
    "triangle_intersection": (0.0, 38.0, 30.0, 57.0),
    "bent_long_road": (-12.0, -4.0, 7.0, 80.0),
}

GROUPS = {
    "yellow_split": [
        ("competition approach", 0, 372, 403, "#111111"),
        ("competition return", 0, 1250, 1268, "#555555"),
        ("113208 turn", 3, 1629, 1684, "#d81b60"),
        ("113208 return", 5, 1781, 1811, "#ff8f00"),
    ],
    "triangle_intersection": [
        ("competition pass A", 0, 408, 440, "#111111"),
        ("competition pass B", 0, 640, 698, "#555555"),
        ("competition pass C", 0, 728, 802, "#8d6e63"),
        ("competition pass D", 0, 932, 965, "#9e9e9e"),
        ("113208 intersection", 3, 1690, 1750, "#d81b60"),
        ("113208 continuation", 4, 1751, 1767, "#1e88e5"),
    ],
    "bent_long_road": [
        ("competition early", 0, 1, 93, "#111111"),
        ("competition middle", 0, 729, 872, "#d81b60"),
        ("competition late", 0, 1137, 1234, "#1e88e5"),
        ("competition closing", 0, 1281, 1290, "#8e24aa"),
        ("091640 start", 1, 1291, 1321, "#00897b"),
        ("113208 arrival", 5, 1828, 1859, "#fb8c00"),
    ],
}


def load_ply(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    count = int(next(line.split()[2] for line in header.decode().splitlines()
                     if line.startswith("element vertex")))
    dtype = np.dtype({"names": ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "c"],
                      "formats": ["<f4"] * 6 + ["u1"] * 3 + ["<f4"],
                      "offsets": [0, 4, 8, 12, 16, 20, 24, 25, 26, 27], "itemsize": 31})
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def load_marking(map_id: int):
    path = MARK / f"map{map_id}.ply"
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


def pose_rows():
    result = []
    connection = sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1", uri=True)
    for node_id, map_id, blob in connection.execute("SELECT id,map_id,pose FROM Node ORDER BY id"):
        pose = np.frombuffer(blob, np.float32).reshape(3, 4)
        result.append((node_id, map_id, float(pose[0, 3]), float(pose[1, 3]),
                       float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))))
    connection.close()
    return result


def region_figure(name, bbox, cloud, poses):
    xmin, ymin, xmax, ymax = bbox
    mask = ((cloud["x"] >= xmin) & (cloud["x"] <= xmax) &
            (cloud["y"] >= ymin) & (cloud["y"] <= ymax))
    idx = np.flatnonzero(mask)
    if len(idx) > 400_000:
        idx = idx[::math.ceil(len(idx) / 400_000)]
    rgb = np.column_stack((cloud["r"][idx], cloud["g"][idx], cloud["b"][idx])).astype(float) / 255

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)
    axes[0].set_facecolor("#111")
    axes[0].scatter(cloud["x"][idx], cloud["y"][idx], c=rgb, s=.28, linewidths=0,
                    rasterized=True)
    axes[0].set_title("actual v8 stored RGB cloud")

    caches = {}
    for label, map_id, start, stop, color in GROUPS[name]:
        if map_id not in caches:
            caches[map_id] = load_marking(map_id)
        mark = caches[map_id]
        use = ((mark["node"] >= start) & (mark["node"] <= stop) &
               (mark["x"] >= xmin) & (mark["x"] <= xmax) &
               (mark["y"] >= ymin) & (mark["y"] <= ymax))
        q = mark[use]
        axes[1].scatter(q["x"], q["y"], color=color, s=.34, alpha=.45,
                        linewidths=0, rasterized=True, label=f"{label}: {start}–{stop}")
        trajectory = np.asarray([(x, y) for node_id, m, x, y, yaw in poses
                                 if m == map_id and start <= node_id <= stop])
        if len(trajectory):
            axes[1].plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=2.2)
    axes[1].set_facecolor("#eeeeee"); axes[1].set_title("RGB-D road markings + pose paths by source group")
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        ax.set_aspect("equal"); ax.grid(alpha=.2)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="map X [m] →", ylabel="map Y [m] →")
    fig.suptitle(name.replace("_", " ").upper(), fontsize=16)
    fig.savefig(OUT / f"{name}_source_attribution.png", dpi=220)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cloud, poses = load_ply(PLY), pose_rows()
    summary = {"input_db": str(DB), "input_ply": str(PLY), "regions": {}}
    for name, bbox in REGIONS.items():
        region_figure(name, bbox, cloud, poses)
        nearby = [(i, m, x, y, yaw) for i, m, x, y, yaw in poses
                  if bbox[0] - 5 <= x <= bbox[2] + 5 and bbox[1] - 5 <= y <= bbox[3] + 5]
        summary["regions"][name] = {"bbox_xy": bbox, "nearby_pose_rows": nearby}
    (OUT / "region_source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
