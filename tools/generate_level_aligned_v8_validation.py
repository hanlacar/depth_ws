#!/usr/bin/env python3
"""Generate numeric and visual v5→v8 audit from stored scans and exported PLYs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/qor/depth_ws")
REFERENCE = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
BEFORE = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
AFTER = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
PLY_BEFORE = ROOT / "analysis/level_aligned_v8/before/merged_competition_level_aligned_v5_cloud.ply"
PLY_AFTER = ROOT / "analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"
OUT = ROOT / "analysis/level_aligned_v8/final"
PLANES = ROOT / "analysis/level_aligned/depth_plane_before_detailed.csv"
RAMP_REPORT = ROOT / "maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json"

REGIONS = {
    "intersection": {"ranges": [(1564, 1628), (1629, 1780)], "bbox": (14.0, 31.8, 28.0, 47.8)},
    "straight": {"ranges": [(1781, 1840)], "bbox": (2.0, 20.5, 21.0, 38.5)},
    "arrival": {"ranges": [(1841, 1859)], "bbox": (-2.0, 19.5, 4.2, 23.0)},
}


def ro(path: Path):
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matrix(blob: bytes) -> np.ndarray:
    out = np.eye(4); out[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4); return out


def scan_matrix(blob: bytes) -> np.ndarray:
    values = np.frombuffer(blob, np.float32)
    out = np.eye(4); out[:3, :4] = values[7:19].reshape(3, 4); return out


def load_db(path: Path):
    with ro(path) as db:
        nodes = {i: (m, matrix(p)) for i, m, p in db.execute("SELECT id,map_id,pose FROM Node ORDER BY id")}
        scans = {i: scan_matrix(s) for i, s in db.execute("SELECT id,scan_info FROM Data ORDER BY id")}
    return nodes, scans


def ramp_nodes() -> set[int]:
    source_map = json.loads(RAMP_REPORT.read_text())["node_source_map"]
    return {int(i) for i, (source, source_id) in source_map.items()
            if (source == "competition_338" and 115 <= source_id <= 210)
            or (source == "091640" and 345 <= source_id <= 630)}


def geometry(reference_nodes, reference_scans, display_scans):
    rows = {int(r["id"]): r for r in csv.DictReader(PLANES.open())}
    output = {}
    for node_id, row in rows.items():
        normal = np.asarray([-float(row["slope_x"]), -float(row["slope_y"]), 1.0])
        normal /= np.linalg.norm(normal)
        height = float(row["intercept_m"]) / np.linalg.norm(
            np.asarray([-float(row["slope_x"]), -float(row["slope_y"]), 1.0]))
        pose = reference_nodes[node_id][1]
        correction = display_scans[node_id] @ np.linalg.inv(reference_scans[node_id])
        new_normal = pose[:3, :3] @ correction[:3, :3] @ normal
        new_normal /= np.linalg.norm(new_normal)
        point = pose[:3, :3] @ (correction[:3, :3] @ (height * normal) + correction[:3, 3]) + pose[:3, 3]
        output[node_id] = {
            "map_id": reference_nodes[node_id][0], "x": float(pose[0, 3]), "y": float(pose[1, 3]),
            "height": float(point[2]),
            "tilt": math.degrees(math.acos(np.clip(new_normal[2], -1, 1))),
            "rms": float(row["rms_m"]), "inliers": int(row["inliers"]),
        }
    return output


def region_ids(spec, geom, ramps):
    ids = [i for i, v in geom.items() if any(a <= i <= b for a, b in spec["ranges"])
           and i not in ramps and v["rms"] <= 0.04 and v["inliers"] >= 500]
    return sorted(ids)


def region_metrics(spec, before, after, ramps):
    ids = region_ids(spec, before, ramps)
    def summary(g):
        h = np.asarray([g[i]["height"] for i in ids]); tilt = np.asarray([g[i]["tilt"] for i in ids])
        jumps = [abs(g[b]["height"] - g[a]["height"]) for a, b in zip(ids[:-1], ids[1:])
                 if g[a]["map_id"] == g[b]["map_id"]]
        return {"valid_nodes": len(ids), "height_std_m": float(np.std(h)),
                "height_range_m": [float(np.min(h)), float(np.max(h))],
                "max_adjacent_height_step_m": float(max(jumps, default=0)),
                "tilt_median_deg": float(np.median(tilt)), "tilt_max_deg": float(np.max(tilt))}
    return {"before": summary(before), "after": summary(after)}


def ply_memmap(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += stream.readline()
        offset = stream.tell()
    text = header.decode("ascii")
    count = int(next(line.split()[2] for line in text.splitlines() if line.startswith("element vertex")))
    dtype = np.dtype({"names": ["x","y","z","nx","ny","nz","r","g","b","curvature"],
                      "formats": ["<f4"]*6 + ["u1"]*3 + ["<f4"],
                      "offsets": [0,4,8,12,16,20,24,25,26,27], "itemsize": 31})
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,)), count


def crop(cloud, bbox):
    xmin, ymin, xmax, ymax = bbox
    mask = (cloud["x"] >= xmin) & (cloud["x"] <= xmax) & (cloud["y"] >= ymin) & (cloud["y"] <= ymax)
    idx = np.flatnonzero(mask)
    if len(idx) > 180000:
        idx = idx[::math.ceil(len(idx)/180000)]
    xyz = np.column_stack((cloud["x"][idx], cloud["y"][idx], cloud["z"][idx]))
    rgb = np.column_stack((cloud["r"][idx], cloud["g"][idx], cloud["b"][idx])).astype(float) / 255.0
    return xyz, rgb


def region_figure(name, spec, before_cloud, after_cloud):
    before_xyz, before_rgb = crop(before_cloud, spec["bbox"])
    after_xyz, after_rgb = crop(after_cloud, spec["bbox"])
    xmin, ymin, xmax, ymax = spec["bbox"]
    all_xy = np.vstack((before_xyz[:, :2], after_xyz[:, :2])); center = np.mean(all_xy, axis=0)
    _, _, vh = np.linalg.svd(all_xy - center, full_matrices=False); axis = vh[0]
    sb = (before_xyz[:, :2] - center) @ axis; sa = (after_xyz[:, :2] - center) @ axis
    smin, smax = min(sb.min(), sa.min()), max(sb.max(), sa.max())
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for ax, xyz, rgb, title in ((axes[0,0], before_xyz, before_rgb, "v5 BEFORE — top"),
                                (axes[0,1], after_xyz, after_rgb, "v8 AFTER — top")):
        ax.set_facecolor("#151515"); ax.scatter(xyz[:,0], xyz[:,1], c=rgb, s=.28, linewidths=0)
        ax.set(xlim=(xmin,xmax), ylim=(ymin,ymax), xlabel="X [m] →", ylabel="Y [m] →", title=title)
        ax.set_aspect("equal")
    for ax, xyz, rgb, s, title in ((axes[1,0], before_xyz, before_rgb, sb, "v5 BEFORE — side"),
                                   (axes[1,1], after_xyz, after_rgb, sa, "v8 AFTER — side")):
        floor = (xyz[:,2] > -0.45) & (xyz[:,2] < 0.65)
        ax.set_facecolor("#151515"); ax.scatter(s[floor], xyz[floor,2], c=rgb[floor], s=.28, linewidths=0)
        ax.set(xlim=(smin,smax), ylim=(-.45,.65), xlabel="principal road axis [m]", ylabel="Z [m] ↑", title=title)
    fig.suptitle(f"{name.upper()} — same crop, scale and stored scan geometry", fontsize=15)
    fig.savefig(OUT / f"{name}_top_side_before_after.png", dpi=190); plt.close(fig)


def arrival_profile(before, after):
    ids = sorted(i for i in before if 1825 <= i <= 1859 and before[i]["rms"] <= .04 and before[i]["inliers"] >= 500)
    fig, ax = plt.subplots(figsize=(13,6), constrained_layout=True)
    ax.plot(ids, [before[i]["height"] for i in ids], "o-", ms=3, lw=1.5, label="v5")
    ax.plot(ids, [after[i]["height"] for i in ids], "o--", ms=3, lw=1.5, label="v8")
    ax.axvspan(1841,1859,color="#ffd54f",alpha=.25,label="arrival nodes")
    ax.set(xlabel="integrated node ID",ylabel="measured road-plane height [m]",title="Arrival approach / marking / exit height profile")
    ax.grid(alpha=.3); ax.legend(); fig.savefig(OUT/"arrival_height_profile.png",dpi=190); plt.close(fig)


def ramp_figure(before, after, ramps):
    fig, axes = plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
    for ax, map_id in zip(axes,(0,1)):
        ids = sorted(i for i in ramps if before[i]["map_id"] == map_id)
        b=np.asarray([before[i]["height"] for i in ids]); a=np.asarray([after[i]["height"] for i in ids])
        b-=b[0];a-=a[0]
        ax.plot(ids,b,lw=3,label="v5");ax.plot(ids,a,"--",lw=2,label="v8")
        ax.set(title=f"ramp map_id={map_id}",xlabel="integrated node ID",ylabel="relative road height [m]")
        ax.grid(alpha=.3);ax.legend()
    fig.suptitle("Ramp observations preserved byte-for-byte")
    fig.savefig(OUT/"ramp_preservation.png",dpi=190);plt.close(fig)


def full_cloud_figure(cloud):
    idx=np.arange(0,len(cloud),max(1,len(cloud)//350000))
    xyz=np.column_stack((cloud["x"][idx],cloud["y"][idx],cloud["z"][idx]));rgb=np.column_stack((cloud["r"][idx],cloud["g"][idx],cloud["b"][idx]))/255
    fig,axes=plt.subplots(1,2,figsize=(17,8),constrained_layout=True)
    axes[0].set_facecolor("#111");axes[0].scatter(xyz[:,0],xyz[:,1],c=rgb,s=.2,linewidths=0);axes[0].set_aspect("equal");axes[0].set(xlabel="X [m] →",ylabel="Y [m] →",title="v8 full stored cloud — top")
    axes[1].set_facecolor("#111");axes[1].scatter(xyz[:,0],xyz[:,2],c=rgb,s=.2,linewidths=0);axes[1].set(xlabel="X [m] →",ylabel="Z [m] ↑",title="v8 full stored cloud — side")
    fig.savefig(OUT/"full_cloud_top_side.png",dpi=190);plt.close(fig)


def digest(path, query):
    h=hashlib.sha256()
    with ro(path) as db:
        for row in db.execute(query):
            for v in row: h.update(v if isinstance(v,bytes) else repr(v).encode())
    return h.hexdigest()


def graph_audit(path):
    with ro(path) as db:
        ids=[r[0] for r in db.execute("SELECT id FROM Node")];adj={i:set() for i in ids}
        for a,b in db.execute("SELECT from_id,to_id FROM Link"):adj[a].add(b);adj[b].add(a)
        seen=set();components=[]
        for root in ids:
            if root in seen:continue
            stack=[root];seen.add(root);n=0
            while stack:
                u=stack.pop();n+=1
                for v in adj[u]:
                    if v not in seen:seen.add(v);stack.append(v)
            components.append(n)
        integrity=db.execute("PRAGMA integrity_check").fetchone()[0]
        data=db.execute("SELECT count(*),sum(image IS NOT NULL),sum(depth IS NOT NULL),sum(calibration IS NOT NULL),sum(scan IS NOT NULL) FROM Data").fetchone()
        admin=db.execute("SELECT length(opt_cloud),length(opt_ids),length(opt_poses) FROM Admin").fetchone()
    return {"integrity":integrity,"components":sorted(components,reverse=True),"data_counts":data,"stored_cloud_blobs":admin}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ref_nodes,ref_scans=load_db(REFERENCE);_,before_scans=load_db(BEFORE);after_nodes,after_scans=load_db(AFTER)
    ramps=ramp_nodes();before_geom=geometry(ref_nodes,ref_scans,before_scans);after_geom=geometry(ref_nodes,ref_scans,after_scans)
    before_cloud,before_count=ply_memmap(PLY_BEFORE);after_cloud,after_count=ply_memmap(PLY_AFTER)
    for name,spec in REGIONS.items():region_figure(name,spec,before_cloud,after_cloud)
    arrival_profile(before_geom,after_geom);ramp_figure(before_geom,after_geom,ramps);full_cloud_figure(after_cloud)
    metrics={name:region_metrics(spec,before_geom,after_geom,ramps) for name,spec in REGIONS.items()}
    digests={}
    queries={
        "node":"SELECT * FROM Node ORDER BY id", "link":"SELECT * FROM Link ORDER BY rowid",
        "calibration":"SELECT id,calibration FROM Data ORDER BY id",
        "features":"SELECT * FROM Feature ORDER BY rowid", "words":"SELECT * FROM Word ORDER BY id",
        "rgb_depth_scan_and_transform":"SELECT id,image,depth,scan,scan_info FROM Data ORDER BY id"}
    for name,query in queries.items():digests[name]=digest(BEFORE,query)==digest(AFTER,query)
    with ro(AFTER) as db:
        pose_count=db.execute("SELECT count(*) FROM Node").fetchone()[0]
    report={
        "input_v5":{"path":str(BEFORE),"size_bytes":BEFORE.stat().st_size,"sha256":sha(BEFORE)},
        "reference_gap_filled":{"path":str(REFERENCE),"sha256":sha(REFERENCE)},
        "output":{"path":str(AFTER),"size_bytes":AFTER.stat().st_size,"sha256":sha(AFTER)},
        "problem_regions":metrics,
        "intersection_node_1659":{"before_height_m":before_geom[1659]["height"],"after_height_m":after_geom[1659]["height"],"before_tilt_deg":before_geom[1659]["tilt"],"after_tilt_deg":after_geom[1659]["tilt"]},
        "marking_holdout":{
            "scale":1.0,"xy_yaw_applied":False,
            "intersection_map3_nearest_marking_median_p90_m":[0.03426,0.26242],
            "straight_map4_nearest_marking_median_p90_m":[0.03471,0.20623],
            "arrival_map5_nearest_marking_median_p90_m":[0.08743,1.57130],
            "reason":"all proposed session SE(2) fits were rejected because independent holdout p90 worsened or improvement was not sufficient"},
        "localization":{"v5":"261/273 (95.60%)","v8":"260/273 (95.24%)","average_ms":110.528191,"variation_mean_m":0.006228,"variation_mean_deg":0.381802,"max_jump_m":0.040782,"max_jump_deg":7.232603,"independence_limit":"retained 091640 source-derived query; not independent driving validation"},
        "preservation_digests_equal":digests,
        "graph":graph_audit(AFTER),
        "no_gap_fill":{"nodes":pose_count,"nodes_added":0,"nodes_removed":0,"xy_yaw_change_m_deg":[0.0,0.0]},
        "ramp":{"masked_nodes":len(ramps),"scan_info_max_difference":float(max(np.max(np.abs(before_scans[i]-after_scans[i])) for i in ramps)),"relative_shape_changed":False},
        "cloud":{"ply":str(PLY_AFTER),"ply_vertices":after_count,"db_export_vertices":2283287,"counts_equal":after_count==2283287,"v5_vertices":before_count},
        "resources":{"build_peak_rss_kib":55472,"cloud_export_peak_rss_kib":645172,"localization_peak_rss_kib":int(next(line.split(':')[1].strip().split()[0] for line in (AFTER.parent/'work/localization/time_final.txt').read_text().splitlines() if 'Maximum resident set size' in line)),"swaps":0,"oom":False},
        "remaining_defects":["intersection/straight/arrival XY marking overlap not force-warped: holdout did not validate transforms","arrival visual split has no corresponding large measured Z step; color/overlap rendering remains unresolved","scan display frame and visual feature frame remain different as in v5 because coordinate-wide consistency attempts failed localization","all candidate problem-region corrections were reverted after validation; v8 is not approved as a corrected driving map"],
        "decision":"FAIL",
        "images":[str(p) for p in sorted(OUT.glob("*.png"))],
    }
    (OUT/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"REPORT.md").write_text(
        "# v5 → v8 conservative correction audit\n\n"
        "v6 (pose/link gauge correction) localized 129/273 and v7 (virtual camera/feature frame) localized 133/273, so neither was approved. A v8 scan candidate also increased the measured intersection step and was reverted. Final v8 therefore preserves all v5 mapping/localization data byte-for-byte; only a freshly optimized full-cloud cache was stored. No node, XY/yaw, link, RGB, depth, descriptor, scan transform, or ramp observation changed.\n\n"
        f"Intersection node 1659 remains {before_geom[1659]['height']:.3f} m / {before_geom[1659]['tilt']:.2f} deg because RGB inspection showed that the plane detector selected the curb/flowerbed rather than road. Straight and arrival transforms were not changed because holdout geometry did not support a correction.\n\n"
        f"Localization: 261/273→260/273. Integrity={report['graph']['integrity']}, graph components={report['graph']['components']}. The query is source-related and is not independent driving validation.\n\n"
        "Decision: FAIL. The requested lane/crosswalk alignment and arrival visual split remain unresolved; this DB is a preserved diagnostic copy and is not approved as a corrected driving map.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
