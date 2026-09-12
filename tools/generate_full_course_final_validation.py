#!/usr/bin/env python3
"""Final read-only validation metrics and visualizations for the full-course DB."""

from __future__ import annotations

import collections
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
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path("/home/qor/depth_ws")
FINAL = ROOT / "maps/merged_competition_full_course/rtabmap.db"
BASE = ROOT / "maps/merged_competition_1lane_2lane_ramp/rtabmap.db"
AUX = ROOT / "maps/map_20260906_113208/rtabmap.db"
OLD_POSES = Path("/tmp/ramp_final_export/ramp_final3_global_poses.txt")
NEW_POSES = ROOT / "analysis/full_course_final/full_course_global_poses.txt"
PRE = ROOT / "analysis/full_course_premerge/premerge_report.json"
LINK_REPORT = ROOT / "maps/merged_competition_full_course/work/manual_cross_links_v3.json"
OUT = ROOT / "analysis/full_course_final"
REPORT = ROOT / "maps/merged_competition_full_course/work/final_validation_report.json"


def ro(path: Path):
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_matrix(x, y, z, qx, qy, qz, qw):
    value = np.eye(4)
    value[:3, :3] = np.asarray([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])
    value[:3, 3] = (x, y, z)
    return value


def load_poses(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        result[int(f[8])] = pose_matrix(*map(float, f[1:8]))
    return result


def matrix(blob):
    value = np.eye(4)
    value[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return value


def yaw(value):
    return math.atan2(value[1, 0], value[0, 0])


def angle_difference(a, b):
    return (a-b+math.pi) % (2*math.pi)-math.pi


def decode_global(path: Path):
    with ro(path) as connection:
        blob, x0, y0, resolution = connection.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin"
        ).fetchone()
    obj = zlib.decompressobj()
    raw = obj.decompress(blob)+obj.flush()
    width, height, _ = struct.unpack("<3i", obj.unused_data)
    grid = np.frombuffer(raw, np.int8).reshape(height, width)
    extent = (x0, x0+width*resolution, y0, y0+height*resolution)
    return grid, extent, float(resolution)


def occupied_points(grid, extent):
    yy, xx = np.where(grid == 100)
    rx = (extent[1]-extent[0])/grid.shape[1]
    ry = (extent[3]-extent[2])/grid.shape[0]
    return np.c_[extent[0]+(xx+.5)*rx, extent[2]+(yy+.5)*ry]


def draw_grid(ax, grid, extent):
    shown = np.zeros_like(grid, np.uint8)
    shown[grid == 0] = 1
    shown[grid == 100] = 2
    cmap = ListedColormap([(1, 1, 1, 0), (.90, .91, .92, .68), (.03, .03, .03, .97)])
    ax.imshow(shown, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")


def graph_components(connection):
    remaining = {row[0] for row in connection.execute("SELECT id FROM Node")}
    graph = collections.defaultdict(set)
    for a, b in connection.execute("SELECT from_id,to_id FROM Link"):
        graph[a].add(b); graph[b].add(a)
    sizes = []
    while remaining:
        stack = [remaining.pop()]; count = 0
        while stack:
            node = stack.pop(); count += 1
            for neighbor in graph[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor)
        sizes.append(count)
    return sorted(sizes, reverse=True)


def decompressed_scan(blob):
    obj = zlib.decompressobj(); raw = obj.decompress(blob)+obj.flush()
    rows, cols, kind = struct.unpack("<3i", obj.unused_data)
    channels = (kind >> 3)+1
    return np.frombuffer(raw, np.float32).reshape(rows*cols, channels)[:, :3]


def scan_local(blob):
    values = np.frombuffer(blob, np.float32)
    value = np.eye(4); value[:3, :4] = values[7:19].reshape(3, 4)
    return value


def source_ids_by_stamp():
    with ro(AUX) as source, ro(FINAL) as final:
        source_rows = list(source.execute("SELECT id,stamp FROM Node ORDER BY stamp"))
        source_stamps = np.asarray([x[1] for x in source_rows])
        source_ids = np.asarray([x[0] for x in source_rows])
        result = {}
        for final_id, map_id, stamp in final.execute(
            "SELECT id,map_id,stamp FROM Node WHERE map_id IN (2,3) ORDER BY id"
        ):
            index = int(np.argmin(np.abs(source_stamps-stamp)))
            result[final_id] = int(source_ids[index])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, new = load_poses(OLD_POSES), load_poses(NEW_POSES)
    common = sorted(set(old)&set(new))
    xy = np.asarray([np.linalg.norm(old[i][:2, 3]-new[i][:2, 3]) for i in common])
    dz = np.asarray([abs(old[i][2, 3]-new[i][2, 3]) for i in common])
    dyaw = np.asarray([abs(angle_difference(yaw(old[i]), yaw(new[i]))) for i in common])

    with ro(FINAL) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        components = graph_components(connection)
        maps = [dict(zip(("map_id", "nodes", "id_min", "id_max"), row)) for row in
                connection.execute("SELECT map_id,count(*),min(id),max(id) FROM Node GROUP BY map_id")]
        payload = connection.execute(
            "SELECT count(*),sum(length(image)>0),sum(length(depth)>0),"
            "sum(length(calibration)>0),sum(length(scan)>0),"
            "sum(length(ground_cells)>0),sum(length(obstacle_cells)>0) FROM Data"
        ).fetchone()
        admin = connection.execute(
            "SELECT length(opt_cloud),length(opt_ids),length(opt_poses),length(opt_map),"
            "opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin"
        ).fetchone()
        cross_residuals = []
        seen = set()
        for a, b, blob in connection.execute(
            "SELECT from_id,to_id,transform FROM Link WHERE type=4"
        ):
            key = tuple(sorted((a, b)))
            if key in seen: continue
            seen.add(key)
            measurement = matrix(blob)
            if a > b:
                measurement = np.linalg.inv(measurement); a, b = b, a
            predicted = np.linalg.inv(new[a])@new[b]
            error = np.linalg.inv(measurement)@predicted
            cross_residuals.append({"from": a, "to": b,
                "translation_m": float(np.linalg.norm(error[:3, 3])),
                "rotation_deg": math.degrees(math.acos(float(np.clip(
                    (np.trace(error[:3, :3])-1)/2, -1, 1))))})

    base_grid, base_extent, _ = decode_global(BASE)
    final_grid, final_extent, resolution = decode_global(FINAL)
    base_occ, final_occ = occupied_points(base_grid, base_extent), occupied_points(final_grid, final_extent)
    distances = cKDTree(base_occ).query(final_occ)[0]

    source_map = source_ids_by_stamp()
    added_source = sorted(source_map.values())
    added_final = sorted(source_map)

    pre = json.loads(PRE.read_text())
    applied = pre["fits"]["intersection"]

    # Final global occupancy and graph.
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    draw_grid(ax, final_grid, final_extent)
    base_ids = [i for i in sorted(new) if i <= 1563]
    add_ids = [i for i in sorted(new) if i >= 1564]
    base_xy = np.asarray([new[i][:2, 3] for i in base_ids])
    add_xy = np.asarray([new[i][:2, 3] for i in add_ids])
    ax.plot(*base_xy.T, color="black", lw=1.2, label="retained integrated map")
    for map_id, color in ((2, "#159447"), (3, "#159447")):
        ids = [i for i in add_ids if (1564 <= i <= 1628) == (map_id == 2)]
        q = np.asarray([new[i][:2, 3] for i in ids])
        ax.plot(*q.T, color=color, lw=3, label="113208 selected" if map_id == 2 else None)
    for item in json.loads(LINK_REPORT.read_text())["anchors"]:
        a, b = item["base_id"], item["appended_id"]
        color = "#b31bcb" if item["inserted_as_graph_edge"] else "#f2b705"
        ax.plot([new[a][0, 3], new[b][0, 3]], [new[a][1, 3], new[b][1, 3]],
                color=color, lw=2, ls="-" if item["inserted_as_graph_edge"] else ":")
        ax.scatter(new[a][0, 3], new[a][1, 3], s=55, color=color, edgecolor="white", zorder=8)
    ax.set_title("Final full-course occupancy + one-component pose graph\nactual local-grid fusion, scale=1.000", weight="bold")
    ax.set_xlabel("X [m] →"); ax.set_ylabel("Y [m] →"); ax.set_aspect("equal")
    ax.grid(alpha=.22); ax.legend()
    fig.savefig(OUT/"final_occupancy_graph.png", dpi=180); plt.close(fig)

    # Before / after map comparison.
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    for ax, grid, extent, title in ((axes[0], base_grid, base_extent, "BEFORE: ramp-integrated base"),
                                    (axes[1], final_grid, final_extent, "AFTER: selected 113208 reinforcement")):
        draw_grid(ax, grid, extent); ax.set_title(title, weight="bold")
        ax.set_xlabel("X [m] →"); ax.set_ylabel("Y [m] →"); ax.set_aspect("equal"); ax.grid(alpha=.2)
    fig.savefig(OUT/"final_occupancy_before_after.png", dpi=180); plt.close(fig)

    # Ramp preservation and flat added nodes.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    ids = [i for i in sorted(old) if i <= 1290]
    axes[0].plot(ids, [old[i][2, 3] for i in ids], color="#777", lw=2, label="base before")
    axes[0].plot(ids, [new[i][2, 3] for i in ids], color="#0067c5", lw=1, ls="--", label="final")
    axes[0].axvspan(115, 210, color="#f28e2b", alpha=.25, label="competition ramp")
    axes[0].set_ylabel("Z [m]"); axes[0].set_title("Existing ramp Z preserved", weight="bold")
    axes[0].grid(alpha=.3); axes[0].legend()
    axes[1].plot(added_final, [new[i][2, 3] for i in added_final], color="#159447", lw=2)
    axes[1].set_xlabel("final node id →"); axes[1].set_ylabel("Z [m]")
    axes[1].set_title("Added 113208 flat intersection/corner submaps", weight="bold"); axes[1].grid(alpha=.3)
    fig.savefig(OUT/"final_ramp_preservation.png", dpi=180); plt.close(fig)

    # RGB-D anchor contact sheet using actual stored RGB.
    anchor_rows = json.loads(LINK_REPORT.read_text())["anchors"]
    with ro(FINAL) as connection:
        fig, axes = plt.subplots(len(anchor_rows), 2, figsize=(13, 13), constrained_layout=True)
        for row, item in enumerate(anchor_rows):
            for col, node_id in enumerate((item["base_id"], item["appended_id"])):
                blob = connection.execute("SELECT image FROM Data WHERE id=?", (node_id,)).fetchone()[0]
                image = cv2.cvtColor(cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(image); axes[row, col].axis("off")
                source_id = item["source_113208_id"] if col else node_id
                side = "integrated" if col == 0 else "113208"
                edge = "GRAPH EDGE" if item["inserted_as_graph_edge"] else "VALIDATION ONLY"
                axes[row, col].set_title(
                    f"{side} node {source_id}\nRGB inliers={item['rgb_inliers']}, "
                    f"Depth ICP={item['depth_icp_rmse_m']:.3f} m | {edge}", fontsize=9,
                    color="#8a128c" if item["inserted_as_graph_edge"] else "#a37700")
        fig.suptitle("Actual RGB-D cross-session anchors", fontsize=16, weight="bold")
        fig.savefig(OUT/"final_anchor_rgbd_contact_sheet.png", dpi=180); plt.close(fig)

    # Low-memory 3D rendering reconstructed from stored scans and optimized poses.
    clouds = {"base": [], "add": []}
    with ro(FINAL) as connection:
        for index, (node_id, map_id, scan_blob, info_blob) in enumerate(connection.execute(
            "SELECT n.id,n.map_id,d.scan,d.scan_info FROM Node n JOIN Data d ON n.id=d.id ORDER BY n.id"
        )):
            if index % 6 or not scan_blob or node_id not in new:
                continue
            points = decompressed_scan(scan_blob)
            points = points[np.all(np.isfinite(points), axis=1)][::5]
            local = scan_local(info_blob)
            points = points@local[:3, :3].T+local[:3, 3]
            points = points@new[node_id][:3, :3].T+new[node_id][:3, 3]
            points = points[(points[:, 2] > -.8)&(points[:, 2] < 4.0)]
            clouds["base" if map_id <= 1 else "add"].append(points)
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for name, color, alpha in (("base", "#555555", .16), ("add", "#18a34a", .48)):
        q = np.vstack(clouds[name]) if clouds[name] else np.empty((0, 3))
        if len(q): ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=.18, c=color, alpha=alpha, label=name)
    ramp_ids = [i for i in range(115, 211) if i in new]
    ramp = np.asarray([new[i][:3, 3] for i in ramp_ids])
    ax.plot(ramp[:, 0], ramp[:, 1], ramp[:, 2], color="#ff7f0e", lw=4, label="preserved ramp")
    ax.set_xlabel("X [m] →"); ax.set_ylabel("Y [m] →"); ax.set_zlabel("Z [m] ↑")
    ax.set_title("Final 3D view — actual stored scans + optimized poses", weight="bold")
    ax.view_init(elev=28, azim=-130); ax.legend()
    fig.savefig(OUT/"final_3d_cloud_view.png", dpi=180); plt.close(fig)

    flat_ids = [i for i in common if not (115 <= i <= 210 or 1354 <= i <= 1482)]
    flat_z = np.asarray([new[i][2, 3] for i in flat_ids])
    add_z = np.asarray([new[i][2, 3] for i in added_final])
    hashes = {}
    for path in (ROOT/"maps/competition_338.db",
                 ROOT/"maps/map_20260906_091640/rtabmap.db", BASE, AUX, FINAL):
        hashes[str(path)] = {"size_bytes": path.stat().st_size, "sha256": hash_file(path)}
    report = {
        "integrity_check": integrity, "all_link_components": components,
        "map_sessions": maps,
        "payload": dict(zip(("rows", "rgb", "depth", "calibration", "scan", "ground", "obstacle"), payload)),
        "admin": dict(zip(("opt_cloud_bytes", "opt_ids_bytes", "opt_poses_bytes", "opt_map_bytes",
                            "map_x_min", "map_y_min", "map_resolution"), admin)),
        "base_deformation": {"xy_rms_m": float(np.sqrt(np.mean(xy*xy))),
            "xy_median_m": float(np.median(xy)), "xy_max_m": float(np.max(xy)),
            "z_max_m": float(np.max(dz)), "yaw_rms_deg": math.degrees(float(np.sqrt(np.mean(dyaw*dyaw)))),
            "yaw_max_deg": math.degrees(float(np.max(dyaw)))},
        "height": {"flat_z_std_m": float(np.std(flat_z)), "flat_z_abs_max_m": float(np.max(abs(flat_z))),
            "added_z_std_m": float(np.std(add_z)), "added_z_abs_max_m": float(np.max(abs(add_z))),
            "ramp_z_min_m": float(min(new[i][2, 3] for i in range(115, 211) if i in new)),
            "ramp_z_max_m": float(max(new[i][2, 3] for i in range(115, 211) if i in new)),
            "ramp_old_new_max_abs_difference_m": float(np.max(dz))},
        "cross_session_graph_residuals": cross_residuals,
        "added_final_node_count": len(added_final), "added_final_id_range": [min(added_final), max(added_final)],
        "added_113208_source_count": len(added_source),
        "added_113208_source_ranges": [[min(x for x in added_source if x < 1800), max(x for x in added_source if x < 1800)],
                                         [min(x for x in added_source if x >= 1800), max(x for x in added_source if x >= 1800)]],
        "applied_se2": {"scale": 1.0, "theta_deg": applied["theta_deg"],
            "translation_m": applied["translation_m"], "intersection_anchor_rms_m": applied["rms_m"]},
        "occupancy": {"resolution_m": resolution, "shape_hw": list(final_grid.shape),
            "occupied_before": int(len(base_occ)), "occupied_after": int(len(final_occ)),
            "after_cells_over_0_30m_from_before": int(np.count_nonzero(distances > .30)),
            "after_cells_under_0_18m_from_before": int(np.count_nonzero(distances <= .18)),
            "premerge_directional_boundary_coverage": pre["boundary"]["coverage"]},
        "localization": {"baseline": {"success": 259, "queries": 273, "percent": 259/273*100},
            "final": {"success": 262, "queries": 273, "percent": 262/273*100},
            "delta_percentage_points": (262-259)/273*100,
            "average_ms": 110.674950, "peak_rss_kib": 2223460, "swaps": 0},
        "hashes": hashes,
        "decision": "PARTIAL PASS",
        "reason": "Selected submaps are connected and localization is not degraded, but global 113208 alignment RMS is about 1.07 m and boundary gain is modest; held ambiguous ranges remain excluded."
    }
    REPORT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    for path in sorted(OUT.glob("final_*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
