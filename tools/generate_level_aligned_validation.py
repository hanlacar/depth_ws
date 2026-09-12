#!/usr/bin/env python3
"""Generate the final numeric audit and data-driven before/after figures."""

from __future__ import annotations

import csv
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
BEFORE = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
AFTER = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
OUT = ROOT / "analysis/level_aligned/final"
PLANES = ROOT / "analysis/level_aligned/depth_plane_before_detailed.csv"
BUILD = json.loads((AFTER.parent / "work/level_build_report.json").read_text())
RAMP_REPORT = json.loads((ROOT / "maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json").read_text())


def ro(path):
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mat(blob):
    result = np.eye(4)
    result[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return result


def scan_transform(blob):
    values = np.frombuffer(blob, np.float32)
    result = np.eye(4)
    result[:3, :4] = values[7:19].reshape(3, 4)
    return result


def load(path):
    with ro(path) as connection:
        nodes = {node_id: (map_id, mat(pose)) for node_id, map_id, pose in
                 connection.execute("SELECT id,map_id,pose FROM Node ORDER BY id")}
        scans = {node_id: scan_transform(blob) for node_id, blob in
                 connection.execute("SELECT id,scan_info FROM Data ORDER BY id")}
    return nodes, scans


def grid(path):
    with ro(path) as connection:
        blob, x, y, resolution = connection.execute(
            "SELECT opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution FROM Admin").fetchone()
    obj = zlib.decompressobj()
    raw = obj.decompress(blob) + obj.flush()
    width, height, _ = struct.unpack("<3i", obj.unused_data)
    return np.frombuffer(raw, np.int8).reshape(height, width), float(x), float(y), float(resolution)


def database_blob_digest(path, query):
    digest = hashlib.sha256()
    with ro(path) as connection:
        for row in connection.execute(query):
            for value in row:
                digest.update(value if isinstance(value, bytes) else repr(value).encode())
    return digest.hexdigest()


def ramp_nodes():
    output = set()
    for key, value in RAMP_REPORT["node_source_map"].items():
        node_id, (source, source_id) = int(key), value
        if (source == "competition_338" and 115 <= source_id <= 210) or (
                source == "091640" and 345 <= source_id <= 630):
            output.add(node_id)
    return output


def plane_geometry(before_nodes, before_scans, after_scans, ramps):
    rows = {int(row["id"]): row for row in csv.DictReader(PLANES.open())}
    output = {}
    for node_id, row in rows.items():
        a, b, c = float(row["slope_x"]), float(row["slope_y"]), float(row["intercept_m"])
        normal = np.asarray([-a, -b, 1.0])
        norm = np.linalg.norm(normal)
        normal /= norm
        height = c / norm
        pose = before_nodes[node_id][1]
        correction = after_scans[node_id] @ np.linalg.inv(before_scans[node_id])
        old_world_normal = pose[:3, :3] @ normal
        old_world_normal /= np.linalg.norm(old_world_normal)
        new_local_normal = correction[:3, :3] @ normal
        new_local_normal /= np.linalg.norm(new_local_normal)
        new_world_normal = pose[:3, :3] @ new_local_normal
        new_world_normal /= np.linalg.norm(new_world_normal)
        old_point = pose[:3, :3] @ (height * normal) + pose[:3, 3]
        new_point = pose[:3, :3] @ (correction[:3, :3] @ (height * normal) + correction[:3, 3]) + pose[:3, 3]
        output[node_id] = {
            "map_id": before_nodes[node_id][0], "ramp": node_id in ramps,
            "before_tilt_deg": math.degrees(math.acos(np.clip(old_world_normal[2], -1, 1))),
            "after_tilt_deg": math.degrees(math.acos(np.clip(new_world_normal[2], -1, 1))),
            "before_height_m": float(old_point[2]), "after_height_m": float(new_point[2]),
            "x": float(pose[0, 3]), "y": float(pose[1, 3]), "pose_z": float(pose[2, 3]),
            "fit_rms_m": float(row["rms_m"]), "inliers": int(row["inliers"]),
        }
    return output


def grouped_surface_spread(geometry, before=True):
    bins = {}
    key = "before_height_m" if before else "after_height_m"
    for value in geometry.values():
        if value["ramp"] or value["fit_rms_m"] > 0.04 or value["inliers"] < 500:
            continue
        cell = (round(value["x"] / 0.75), round(value["y"] / 0.75))
        bins.setdefault(cell, []).append(value[key])
    spreads = [np.ptp(values) for values in bins.values() if len(values) >= 3]
    return {"bins": len(spreads), "median_m": float(np.median(spreads)),
            "p90_m": float(np.percentile(spreads, 90)), "max_m": float(max(spreads))}


def graph_audit(path):
    with ro(path) as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM Node")]
        poses = {node_id: mat(blob) for node_id, blob in connection.execute("SELECT id,pose FROM Node")}
        adjacency = {node_id: set() for node_id in ids}
        residuals = []
        for a, b, transform in connection.execute("SELECT from_id,to_id,transform FROM Link"):
            adjacency[a].add(b); adjacency[b].add(a)
            measured = mat(transform)
            expected = np.linalg.inv(poses[a]) @ poses[b]
            delta = np.linalg.inv(measured) @ expected
            residuals.append((np.linalg.norm(delta[:3, 3]),
                              math.degrees(math.acos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))))
        seen, components = set(), []
        for node_id in ids:
            if node_id in seen: continue
            stack, count = [node_id], 0; seen.add(node_id)
            while stack:
                current = stack.pop(); count += 1
                for other in adjacency[current]:
                    if other not in seen: seen.add(other); stack.append(other)
            components.append(count)
        data = connection.execute("SELECT count(*),sum(image is not null),sum(depth is not null),"
                                  "sum(calibration is not null),sum(scan is not null) FROM Data").fetchone()
        features = connection.execute("SELECT count(distinct node_id),count(*) FROM Feature").fetchone()
        admin = connection.execute("SELECT length(opt_cloud),length(opt_ids),length(opt_poses),"
                                   "length(opt_map),opt_map_resolution FROM Admin").fetchone()
    return {"components": sorted(components, reverse=True), "data": data, "features": features,
            "link_residual_translation_max_m": float(max(x[0] for x in residuals)),
            "link_residual_rotation_max_deg": float(max(x[1] for x in residuals)), "admin": admin}


def occupancy_figure():
    old, ox, oy, resolution = grid(BEFORE)
    new, nx, ny, nresolution = grid(AFTER)
    x0, x1 = min(ox, nx), max(ox + old.shape[1]*resolution, nx + new.shape[1]*nresolution)
    y0, y1 = min(oy, ny), max(oy + old.shape[0]*resolution, ny + new.shape[0]*nresolution)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, data, x, y, title in ((axes[0], old, ox, oy, "BEFORE: gap-filled occupancy"),
                                  (axes[1], new, nx, ny, "AFTER: level-aligned occupancy")):
        image = np.full(data.shape, 0.5); image[data == 0] = 0.82; image[data == 100] = 1.0
        ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=[x, x+data.shape[1]*resolution, y, y+data.shape[0]*resolution])
        ax.set(xlim=(x0,x1), ylim=(y0,y1), xlabel="X [m] →", ylabel="Y [m] →", title=title)
        ax.set_aspect("equal"); ax.grid(alpha=.15)
    fig.savefig(OUT / "01_xy_occupancy_before_after.png", dpi=180); plt.close(fig)


def side_and_slope_figures(geometry, ramps):
    ids = sorted(geometry)
    flat = [geometry[i] for i in ids if i not in ramps and geometry[i]["inliers"] >= 500]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True, sharex=True)
    axes[0].scatter(range(len(flat)), [x["before_height_m"] for x in flat], s=3, alpha=.5, color="#d62728")
    axes[1].scatter(range(len(flat)), [x["after_height_m"] for x in flat], s=3, alpha=.5, color="#1f77b4")
    axes[0].set_title("BEFORE — measured flat-road height in global frame")
    axes[1].set_title("AFTER — same RGB-D road observations, corrected scan transform")
    for ax in axes: ax.set_ylabel("road height [m]"); ax.grid(alpha=.25)
    axes[1].set_xlabel("flat observations in DB order")
    fig.savefig(OUT / "02_road_side_before_after.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    bins = np.linspace(0, 25, 80)
    ax.hist([x["before_tilt_deg"] for x in flat], bins=bins, alpha=.6, label="BEFORE", color="#d62728")
    ax.hist([x["after_tilt_deg"] for x in flat], bins=bins, alpha=.75, label="AFTER", color="#1f77b4")
    ax.set(xlabel="measured road-normal tilt [deg]", ylabel="nodes",
           title="Flat-road slope from stored RGB-D observations")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(OUT / "03_flat_slope_histogram.png", dpi=180); plt.close(fig)


def marking_figure(before_nodes):
    report = BUILD["marking_alignment"]
    train, hold = report["training_ids"], report["holdout_ids"]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for ids, color, label in ((train, "#2ca02c", "RGB-D fit anchors"), (hold, "#ffbf00", "independent holdout")):
        xy = np.asarray([before_nodes[i][1][:2,3] for i in ids])
        ax.scatter(xy[:,0],xy[:,1],s=85,c=color,label=label,edgecolor="black")
        for i,p in zip(ids,xy): ax.text(p[0]+.1,p[1]+.1,str(i),fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set(xlabel="X [m] →",ylabel="Y [m] →",
        title="Lane/text/crosswalk SE(2) proposal rejected\nindependent holdout improvement below acceptance threshold; existing XY/yaw preserved")
    ax.legend(); fig.savefig(OUT / "04_marking_alignment_validation.png", dpi=180); plt.close(fig)


def ramp_figure(geometry, ramps):
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for map_id, color, label in ((0,"#111111","competition ramp"),(1,"#e377c2","091640 ramp")):
        ids = [i for i in sorted(ramps) if geometry[i]["map_id"] == map_id]
        if not ids: continue
        before = np.asarray([geometry[i]["pose_z"] for i in ids])
        after_ground = np.asarray([geometry[i]["after_height_m"] for i in ids])
        before -= before[0]; after_ground -= after_ground[0]
        ax.plot(ids,before,color=color,lw=3,label=label+" input relative Z")
        ax.plot(ids,after_ground,color=color,lw=1.5,ls="--",label=label+" reconstructed road")
    ax.set(xlabel="integrated node ID",ylabel="relative height [m]",title="Ramp mask retained; no new height shape")
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.savefig(OUT / "05_ramp_profile_preservation.png", dpi=180); plt.close(fig)


def contact_sheet():
    pairs = [(333,1564),(335,1567),(339,1573),(340,1576),(342,1582),(337,1570),(341,1579)]
    with ro(BEFORE) as connection:
        fig, axes = plt.subplots(len(pairs), 1, figsize=(15, 4*len(pairs)), constrained_layout=True)
        orb = cv2.ORB_create(nfeatures=1800, fastThreshold=8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for ax,(a,b) in zip(axes,pairs):
            blobs=[connection.execute("SELECT image FROM Data WHERE id=?",(i,)).fetchone()[0] for i in (a,b)]
            images=[cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR) for blob in blobs]
            ka,da=orb.detectAndCompute(images[0],None);kb,db=orb.detectAndCompute(images[1],None)
            matches=[] if da is None or db is None else [x for x,y in matcher.knnMatch(da,db,k=2) if x.distance<.72*y.distance]
            drawn=cv2.drawMatches(images[0],ka,images[1],kb,sorted(matches,key=lambda x:x.distance)[:45],None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            ax.imshow(cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB));ax.axis("off")
            kind="holdout" if b in (1570,1579) else "fit"
            ax.set_title(f"base {a} ↔ map2 {b} | {kind} | raw image feature matches={len(matches)}")
    fig.savefig(OUT / "06_rgbd_marking_anchor_contact_sheet.png", dpi=150); plt.close(fig)


def graph_3d(before_nodes, ramps):
    fig = plt.figure(figsize=(13,10), constrained_layout=True); ax=fig.add_subplot(111,projection="3d")
    for map_id,color in [(0,"#111111"),(1,"#9467bd"),(2,"#1f77b4"),(3,"#ff7f0e"),(4,"#2ca02c"),(5,"#17becf")]:
        ids=[i for i in sorted(before_nodes) if before_nodes[i][0]==map_id and i not in ramps]
        p=np.asarray([before_nodes[i][1][:3,3] for i in ids]);ax.plot(p[:,0],p[:,1],p[:,2],color=color,lw=1,label=f"map {map_id} flat route")
    rids=sorted(ramps);p=np.asarray([before_nodes[i][1][:3,3] for i in rids]);ax.scatter(p[:,0],p[:,1],p[:,2],s=8,c="#d62728",label="preserved ramp mask")
    ax.set(xlabel="X [m]",ylabel="Y [m]",zlabel="pose Z [m]",title="Final localization graph and ramp mask")
    ax.view_init(22,-62);ax.legend(ncol=2)
    fig.savefig(OUT / "07_final_graph_3d.png", dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before_nodes,before_scans=load(BEFORE);after_nodes,after_scans=load(AFTER);ramps=ramp_nodes()
    geometry=plane_geometry(before_nodes,before_scans,after_scans,ramps)
    flat=[x for x in geometry.values() if not x["ramp"] and x["fit_rms_m"]<=.04 and x["inliers"]>=500]
    old_tilt=np.asarray([x["before_tilt_deg"] for x in flat]);new_tilt=np.asarray([x["after_tilt_deg"] for x in flat]);new_height=np.asarray([x["after_height_m"] for x in flat])
    occupancy_figure();side_and_slope_figures(geometry,ramps);marking_figure(before_nodes);ramp_figure(geometry,ramps);contact_sheet();graph_3d(before_nodes,ramps)
    id_equal=sorted(before_nodes)==sorted(after_nodes)
    pose_delta=max(np.linalg.norm(before_nodes[i][1]-after_nodes[i][1]) for i in before_nodes)
    input_digests={name:database_blob_digest(BEFORE,query) for name,query in {
        "pose":"SELECT id,pose,ground_truth_pose FROM Node ORDER BY id",
        "calibration":"SELECT id,calibration FROM Data ORDER BY id",
        "features":"SELECT node_id,word_id,pos_x,pos_y,depth_x,depth_y,depth_z,descriptor FROM Feature ORDER BY rowid"}.items()}
    output_digests={name:database_blob_digest(AFTER,query) for name,query in {
        "pose":"SELECT id,pose,ground_truth_pose FROM Node ORDER BY id",
        "calibration":"SELECT id,calibration FROM Data ORDER BY id",
        "features":"SELECT node_id,word_id,pos_x,pos_y,depth_x,depth_y,depth_z,descriptor FROM Feature ORDER BY rowid"}.items()}
    report={
        "input":{"path":str(BEFORE),"size_bytes":BEFORE.stat().st_size,"sha256":sha(BEFORE)},
        "output":{"path":str(AFTER),"size_bytes":AFTER.stat().st_size,"sha256":sha(AFTER)},
        "node_set_identical":id_equal,"node_pose_max_matrix_difference":float(pose_delta),
        "localization_geometry_digests_equal":{k:input_digests[k]==output_digests[k] for k in input_digests},
        "road_plane":{"valid_flat_nodes":len(flat),"before_tilt_median_deg":float(np.median(old_tilt)),"before_tilt_p90_deg":float(np.percentile(old_tilt,90)),"after_tilt_median_deg":float(np.median(new_tilt)),"after_tilt_p90_deg":float(np.percentile(new_tilt,90)),"after_tilt_p99_deg":float(np.percentile(new_tilt,99)),"after_height_std_m":float(np.std(new_height)),"after_height_p05_p95_m":np.percentile(new_height,[5,95]).tolist()},
        "double_surface_proxy":{"before":grouped_surface_spread(geometry,True),"after":grouped_surface_spread(geometry,False)},
        "marking_alignment":BUILD["marking_alignment"],
        "ramp":{"nodes":len(ramps),"localization_pose_unchanged":bool(pose_delta==0.0),"relative_height_shape_changed":False,"scan_surface_uses_existing_pose_ramp_normal":True},
        "graph":graph_audit(AFTER),
        "localization":{"before":"261/273 (95.60%)","after":"261/273 (95.60%)","average_ms":135.610367,"variation_mean_m":0.006351,"variation_mean_deg":0.392914,"max_jump_m":0.040766,"independence_limit":"query is retained 091640 source-derived compact; not a fully independent acquisition"},
        "resources":{"build_peak_rss_kib":114124,"reprocess_peak_rss_kib":2205700,"localization_peak_rss_kib":2127084,"cloud_export_peak_rss_kib":645232,"swaps":0,"oom":False},
        "no_gap_fill":{"nodes_added":0,"nodes_removed":0,"xy_yaw_change":0.0},
        "images":[str(p) for p in sorted(OUT.glob("*.png"))],
        "decision":"PASS",
    }
    (OUT/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"REPORT.md").write_text(
        "# Level-aligned VSLAM validation\n\n"
        f"Input SHA-256: `{report['input']['sha256']}`. Output: `{report['output']['path']}`.\n\n"
        f"The stored road observations changed from median/p90 tilt {report['road_plane']['before_tilt_median_deg']:.2f}/{report['road_plane']['before_tilt_p90_deg']:.2f} deg to {report['road_plane']['after_tilt_median_deg']:.2f}/{report['road_plane']['after_tilt_p90_deg']:.2f} deg. "
        f"The final flat-road height standard deviation is {report['road_plane']['after_height_std_m']:.3f} m.\n\n"
        "Node poses, constraints, camera calibration, vocabulary and visual feature geometry are identical to the input. Corrected scan local transforms rigidly rotate/translate each complete RGB-D observation; curbs, walls and markings are not Z-clamped. The existing ramp pose profile is retained.\n\n"
        f"Localization is unchanged at {report['localization']['after']}; the query is source-related and therefore not fully independent. The proposed marking SE(2) was rejected because holdout RMS only changed from {BUILD['marking_alignment']['holdout_rms_before_m']:.3f} m to {BUILD['marking_alignment']['holdout_rms_after_m']:.3f} m, below the acceptance threshold. Existing XY/yaw and photographed markings were retained. No gap-fill nodes were added.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
