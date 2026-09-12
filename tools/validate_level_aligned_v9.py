#!/usr/bin/env python3
"""Validate the minimal v8→v9 yellow-line connection correction."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/qor/depth_ws")
V8 = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
V9 = ROOT / "maps/merged_competition_level_aligned_v9/rtabmap.db"
P8 = ROOT / "analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"
P9 = ROOT / "analysis/level_aligned_v9/final/merged_competition_level_aligned_v9.ply"
M8 = ROOT / "analysis/level_aligned_v9/work/markings/map3.ply"
M9 = ROOT / "analysis/level_aligned_v9/work/markings/map3_v9.ply"
M5 = ROOT / "analysis/level_aligned_v9/work/markings/map5.ply"
OUT = ROOT / "analysis/level_aligned_v9/final"
MODIFIED = {1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640,
            1641, 1642, 1643, 1647, 1648, 1651, 1652, 1653, 1654, 1655,
            1656, 1657, 1658, 1659, 1660, 1662}
FULL = {1641, 1642, 1643, 1647, 1648, 1651, 1652, 1653, 1654, 1655, 1656}


def ro(path: Path):
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ply(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    count = int(next(row.split()[2] for row in header.decode().splitlines()
                     if row.startswith("element vertex")))
    dtype = np.dtype({"names": ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "c"],
                      "formats": ["<f4"] * 6 + ["u1"] * 3 + ["<f4"],
                      "offsets": [0, 4, 8, 12, 16, 20, 24, 25, 26, 27], "itemsize": 31})
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def marking(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    count = int(next(row.split()[2] for row in header.decode().splitlines()
                     if row.startswith("element vertex")))
    dtype = np.dtype({"names": ["x", "y", "z", "type", "node"],
                      "formats": ["<f4", "<f4", "<f4", "u1", "<i4"],
                      "offsets": [0, 4, 8, 12, 13], "itemsize": 17})
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def yellow_from_ply(cloud, bbox):
    xmin, ymin, xmax, ymax = bbox
    m = ((cloud["x"] >= xmin) & (cloud["x"] <= xmax) &
         (cloud["y"] >= ymin) & (cloud["y"] <= ymax) &
         (cloud["z"] >= -.15) & (cloud["z"] <= .25))
    p = cloud[m]
    rgb8 = np.column_stack((p["r"], p["g"], p["b"])).astype(np.uint8)
    hsv = cv2.cvtColor(rgb8.reshape(-1, 1, 3)[:, :, ::-1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    yellow = ((hsv[:, 0] >= 8) & (hsv[:, 0] <= 38) &
              (hsv[:, 1] >= 62) & (hsv[:, 2] >= 85))
    return p, rgb8.astype(float) / 255.0, yellow


def compare_figure():
    bbox = (20.5, 26.0, 23.7, 32.5)
    clouds = [ply(P8), ply(P9)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 15), constrained_layout=True)
    fig2, detail = plt.subplots(1, 2, figsize=(10, 15), constrained_layout=True)
    for ax, dax, cloud, title in zip(axes, detail, clouds, ("v8 BEFORE", "v9 AFTER")):
        p, rgb, yellow = yellow_from_ply(cloud, bbox)
        ax.set_facecolor("#111")
        ax.scatter(p["x"], p["y"], c=rgb, s=1.1, linewidths=0, rasterized=True)
        dax.set_facecolor("#111")
        dax.scatter(p["x"][~yellow], p["y"][~yellow], c="#666", s=.25, alpha=.12,
                    linewidths=0, rasterized=True)
        dax.scatter(p["x"][yellow], p["y"][yellow], c="#ffd400", s=2.2, alpha=.9,
                    linewidths=0, rasterized=True)
        for current in (ax, dax):
            current.set_aspect("equal"); current.grid(alpha=.14)
            current.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]), xlabel="map X [m] →",
                        ylabel="map Y [m] →", title=title)
    fig.suptitle("Split yellow centerline — identical map crop and scale", fontsize=15)
    fig.savefig(OUT / "yellow_split_before_after.png", dpi=260)
    plt.close(fig)
    fig2.suptitle("Yellow candidate geometry — identical crop and scale", fontsize=15)
    fig2.savefig(OUT / "yellow_split_candidates_before_after.png", dpi=260)
    plt.close(fig2)

    wide = (18.0, 27.0, 27.0, 34.0)
    fig3, waxes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, cloud, title in zip(waxes, clouds, ("v8 BEFORE", "v9 AFTER")):
        p, rgb, _ = yellow_from_ply(cloud, wide)
        ax.set_facecolor("#111"); ax.scatter(p["x"], p["y"], c=rgb, s=.7,
                                              linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.grid(alpha=.14)
        ax.set(xlim=(wide[0], wide[2]), ylim=(wide[1], wide[3]), xlabel="map X [m] →",
               ylabel="map Y [m] →", title=title)
    fig3.suptitle("Connection road, both boundaries — identical crop and scale", fontsize=15)
    fig3.savefig(OUT / "connection_road_boundaries_before_after.png", dpi=240)
    plt.close(fig3)


def fit_line(path: Path, nodes: set[int]):
    q = marking(path)
    m = ((q["type"] == 2) & np.isin(q["node"], list(nodes)) &
         (q["x"] > 21.4) & (q["x"] < 22.9) &
         (q["y"] > 27.75) & (q["y"] < 31.15))
    x, y = q["x"][m].astype(float), q["y"][m].astype(float)
    rng = np.random.default_rng(4); centered = y - 29.5; best = None
    for _ in range(6000):
        a, b = rng.choice(len(x), 2, replace=False)
        if abs(centered[b] - centered[a]) < .4:
            continue
        slope = (x[b] - x[a]) / (centered[b] - centered[a])
        intercept = x[a] - slope * centered[a]
        if abs(slope) > .5:
            continue
        keep = abs(x - (intercept + slope * centered)) < .055
        if best is None or keep.sum() > best.sum():
            best = keep
    keep = best
    for _ in range(5):
        design = np.column_stack((np.ones(keep.sum()), centered[keep]))
        coefficient = np.linalg.lstsq(design, x[keep], rcond=None)[0]
        residual = abs(x - (coefficient[0] + coefficient[1] * centered))
        keep = residual < .065
    return coefficient, int(len(x)), int(keep.sum())


def line_metrics():
    before, n8, i8 = fit_line(M8, FULL)
    after, n9, i9 = fit_line(M9, FULL)
    reference, nr, ir = fit_line(M5, set(range(1784, 1794)))
    y = np.linspace(28.0, 31.0, 121) - 29.5
    def metric(coefficient):
        delta = coefficient[0] + coefficient[1] * y - (reference[0] + reference[1] * y)
        return {"rms_m": float(np.sqrt(np.mean(delta * delta))),
                "max_m": float(np.max(abs(delta))), "signed_mean_m": float(np.mean(delta))}
    return {"reference": reference.tolist(),
            "before_fit": before.tolist(), "after_fit": after.tolist(),
            "before": metric(before), "after": metric(after),
            "fit_points": {"before": [n8, i8], "after": [n9, i9], "reference": [nr, ir]}}


def mat(blob: bytes):
    value = np.eye(4); value[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4); return value


def scan_points(connection, ids):
    output = []
    for node_id in ids:
        row = connection.execute(
            "SELECT n.pose,d.scan,d.scan_info FROM Node n JOIN Data d USING(id) WHERE id=?", (node_id,)).fetchone()
        if not row or not row[1]:
            continue
        obj = zlib.decompressobj(); raw = obj.decompress(row[1]) + obj.flush()
        rows, cols, kind = struct.unpack("<3i", obj.unused_data); channels = (kind >> 3) + 1
        points = np.frombuffer(raw, np.float32).reshape(rows * cols, channels)[:, :3]
        local = np.eye(4); local[:3, :4] = np.frombuffer(row[2], np.float32)[7:19].reshape(3, 4)
        world = mat(row[0]) @ local
        points = points @ world[:3, :3].T + world[:3, 3]
        use = ((points[:, 0] > 18) & (points[:, 0] < 27) &
               (points[:, 1] > 27) & (points[:, 1] < 34) &
               (points[:, 2] > -.12) & (points[:, 2] < .12))
        output.append(points[use])
    return np.vstack(output)


def plane(points):
    keep = np.ones(len(points), dtype=bool)
    for _ in range(6):
        design = np.column_stack((points[keep, 0] - 22, points[keep, 1] - 30.5,
                                  np.ones(keep.sum())))
        coefficient = np.linalg.lstsq(design, points[keep, 2], rcond=None)[0]
        residual = points[:, 2] - (coefficient[0] * (points[:, 0] - 22) +
                                   coefficient[1] * (points[:, 1] - 30.5) + coefficient[2])
        keep = abs(residual) < .025
    return coefficient


def height_metrics():
    result = {}
    for label, path in (("before", V8), ("after", V9)):
        with ro(path) as db:
            reference = plane(scan_points(db, list(range(374, 393))))
            moving = plane(scan_points(db, sorted(FULL)))
        grid_x, grid_y = np.meshgrid(np.linspace(20.5, 23.5, 25), np.linspace(28, 32, 25))
        rz = reference[0] * (grid_x - 22) + reference[1] * (grid_y - 30.5) + reference[2]
        mz = moving[0] * (grid_x - 22) + moving[1] * (grid_y - 30.5) + moving[2]
        result[label] = {"reference_plane": reference.tolist(), "moving_plane": moving.tolist(),
                         "rms_difference_m": float(np.sqrt(np.mean((mz-rz)**2))),
                         "max_difference_m": float(np.max(abs(mz-rz)))}
    return result


def row_digest(path, query, parameters=()):
    value = hashlib.sha256()
    with ro(path) as db:
        for row in db.execute(query, parameters):
            for field in row:
                value.update(field if isinstance(field, bytes) else repr(field).encode())
    return value.hexdigest()


def graph_metrics():
    with ro(V9) as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        ids = [r[0] for r in db.execute("SELECT id FROM Node")]
        adjacency = {i: set() for i in ids}
        for a, b in db.execute("SELECT from_id,to_id FROM Link"):
            adjacency[a].add(b); adjacency[b].add(a)
        remain = set(ids); components = []
        while remain:
            stack = [remain.pop()]; count = 0
            while stack:
                u = stack.pop(); count += 1
                for v in adjacency[u]:
                    if v in remain: remain.remove(v); stack.append(v)
            components.append(count)
    placeholders = ",".join("?" for _ in MODIFIED)
    outside_query = f"SELECT * FROM Node WHERE id NOT IN ({placeholders}) ORDER BY id"
    scan_query = "SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"
    feature_query = "SELECT * FROM Feature ORDER BY rowid"
    return {"integrity": integrity, "node_count": len(ids),
            "components": sorted(components, reverse=True),
            "outside_node_rows_identical": row_digest(V8, outside_query, tuple(sorted(MODIFIED))) ==
                                             row_digest(V9, outside_query, tuple(sorted(MODIFIED))),
            "all_rgb_depth_calibration_scan_identical": row_digest(V8, scan_query) == row_digest(V9, scan_query),
            "all_features_identical": row_digest(V8, feature_query) == row_digest(V9, feature_query)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    compare_figure()
    report = {"input": {"path": str(V8), "sha256": sha(V8)},
              "output": {"path": str(V9), "size_bytes": V9.stat().st_size, "sha256": sha(V9)},
              "ply": {"path": str(P9), "size_bytes": P9.stat().st_size, "sha256": sha(P9)},
              "modified_nodes": sorted(MODIFIED), "full_correction_nodes": sorted(FULL),
              "map_bbox_xy": [21.7, 27.8, 22.65, 31.2],
              "yellow_lateral_error": line_metrics(), "road_height_error": height_metrics(),
              "graph_and_preservation": graph_metrics()}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
