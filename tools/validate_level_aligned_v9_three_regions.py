#!/usr/bin/env python3
"""Numerical and visual validation for the three-region v8 -> v9 repair."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/qor/depth_ws")
V8 = ROOT/"maps/merged_competition_level_aligned_v8/rtabmap.db"
V9 = ROOT/"maps/merged_competition_level_aligned_v9/rtabmap.db"
P8 = ROOT/"analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"
P9 = ROOT/"analysis/level_aligned_v9/final/merged_competition_level_aligned_v9.ply"
M8_0 = ROOT/"analysis/level_aligned_v9_yellow_only_backup/work/markings/map0.ply"
M8_3 = ROOT/"analysis/level_aligned_v9_yellow_only_backup/work/markings/map3.ply"
M8_5 = ROOT/"analysis/level_aligned_v9_yellow_only_backup/work/markings/map5.ply"
M9_0 = ROOT/"analysis/level_aligned_v9/work/markings/map0.ply"
M9_3 = ROOT/"analysis/level_aligned_v9/work/markings/map3.ply"
M9_5 = ROOT/"analysis/level_aligned_v9/work/markings/map5.ply"
OUT = ROOT/"analysis/level_aligned_v9/final"

REGIONS = {
    "yellow_split": (18.0, 25.0, 25.0, 33.5),
    "triangle_intersection": (4.5, 42.8, 28.0, 54.1),
    "long_straight": (-12.0, 38.0, 0.0, 70.0),
}


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def load_ply(path):
    with path.open("rb") as f:
        header=b""
        while not header.endswith(b"end_header\n"): header+=f.readline()
        off=f.tell()
    n=int(next(x.split()[2] for x in header.decode().splitlines() if x.startswith("element vertex")))
    d=np.dtype({"names":["x","y","z","nx","ny","nz","r","g","b","c"],
                "formats":["<f4"]*6+["u1"]*3+["<f4"],
                "offsets":[0,4,8,12,16,20,24,25,26,27],"itemsize":31})
    return np.memmap(path,dtype=d,mode="r",offset=off,shape=(n,))


def load_mark(path):
    with path.open("rb") as f:
        header=b""
        while not header.endswith(b"end_header\n"):header+=f.readline()
        off=f.tell()
    n=int(next(x.split()[2] for x in header.decode().splitlines() if x.startswith("element vertex")))
    d=np.dtype({"names":["x","y","z","type","node"],
                "formats":["<f4","<f4","<f4","u1","<i4"],
                "offsets":[0,4,8,12,13],"itemsize":17})
    return np.memmap(path,dtype=d,mode="r",offset=off,shape=(n,))


def visual_figures():
    clouds=(load_ply(P8),load_ply(P9))
    for name,bbox in REGIONS.items():
        fig,axes=plt.subplots(1,2,figsize=(14,9),constrained_layout=True)
        for ax,cloud,title in zip(axes,clouds,("v8 BEFORE","v9 AFTER")):
            m=((cloud["x"]>=bbox[0])&(cloud["x"]<=bbox[2])&
               (cloud["y"]>=bbox[1])&(cloud["y"]<=bbox[3])&
               (cloud["z"]>=-.25)&(cloud["z"]<=.5))
            ids=np.flatnonzero(m)
            if len(ids)>450000:ids=ids[::math.ceil(len(ids)/450000)]
            rgb=np.c_[cloud["r"][ids],cloud["g"][ids],cloud["b"][ids]].astype(float)/255
            ax.set_facecolor("#111");ax.scatter(cloud["x"][ids],cloud["y"][ids],c=rgb,s=.45,
                                                linewidths=0,rasterized=True)
            ax.set_aspect("equal");ax.grid(alpha=.14)
            ax.set(xlim=(bbox[0],bbox[2]),ylim=(bbox[1],bbox[3]),title=title,
                   xlabel="map X [m] →",ylabel="map Y [m] →")
        fig.suptitle(name.replace("_"," ").upper()+" — identical top view and scale")
        fig.savefig(OUT/f"{name}_top_before_after.png",dpi=240);plt.close(fig)

    # Side view of the corrected road and intersection connection.
    for name,bbox in {"long_straight":REGIONS["long_straight"],
                      "triangle_intersection":REGIONS["triangle_intersection"]}.items():
        fig,axes=plt.subplots(2,1,figsize=(15,7),constrained_layout=True)
        for ax,cloud,title in zip(axes,clouds,("v8 BEFORE","v9 AFTER")):
            m=((cloud["x"]>=bbox[0])&(cloud["x"]<=bbox[2])&
               (cloud["y"]>=bbox[1])&(cloud["y"]<=bbox[3])&
               (cloud["z"]>=-.3)&(cloud["z"]<=.5))
            ids=np.flatnonzero(m)
            if len(ids)>350000:ids=ids[::math.ceil(len(ids)/350000)]
            ax.scatter(cloud["y"][ids],cloud["z"][ids],c="#555",s=.22,alpha=.3,
                       linewidths=0,rasterized=True)
            ax.set(xlim=(bbox[1],bbox[3]),ylim=(-.25,.35),title=title,
                   xlabel="map Y [m] →",ylabel="Z [m] →");ax.grid(alpha=.2)
        fig.suptitle(name.replace("_"," ").upper()+" — identical side view and scale")
        fig.savefig(OUT/f"{name}_side_before_after.png",dpi=220);plt.close(fig)


def fit_line(q, nodes, x_range, y_range=(47,66), threshold=.09):
    m=((q["type"]==2)&np.isin(q["node"],list(nodes))&
       (q["x"]>x_range[0])&(q["x"]<x_range[1])&
       (q["y"]>y_range[0])&(q["y"]<y_range[1])&(abs(q["z"])<.1))
    x=q["x"][m].astype(float);y=q["y"][m].astype(float)
    rng=np.random.default_rng(4);best=None
    for _ in range(12000):
        i,j=rng.integers(0,len(x),2)
        if abs(y[j]-y[i])<min(4.0, .25*(y_range[1]-y_range[0])):continue
        b=(x[j]-x[i])/(y[j]-y[i]);a=x[i]-b*y[i]
        if abs(b)>.3:continue
        keep=abs(x-(a+b*y))<threshold
        if best is None or keep.sum()>best.sum():best=keep
    for _ in range(5):
        coef=np.linalg.lstsq(np.c_[np.ones(best.sum()),y[best]],x[best],rcond=None)[0]
        best=abs(x-(coef[0]+coef[1]*y))<threshold
    residual=x[best]-(coef[0]+coef[1]*y[best])
    return {"coef_x_a_plus_b_y":coef.tolist(),"points":int(best.sum()),
            "fit_rms_m":float(np.sqrt(np.mean(residual**2))),
            "fit_max_m":float(np.max(abs(residual)))}


def road_metrics(path,before):
    q=load_mark(path)
    specs=([(813,840,(-9,-6)),(813,840,(-6,-2)),
            (1156,1185,(-11,-7.5)),(1156,1185,(-7.5,-3))] if before else
           [(813,840,(-11,-7)),(813,840,(-7,-2)),
            (1156,1185,(-11,-7)),(1156,1185,(-7,-2))])
    fits=[fit_line(q,range(a,b+1),xr) for a,b,xr in specs]
    co=[np.asarray(x["coef_x_a_plus_b_y"]) for x in fits]
    centers=[(co[0]+co[1])/2,(co[2]+co[3])/2]
    yy=np.linspace(47,66,192)
    delta=(centers[0][0]+centers[0][1]*yy)-(centers[1][0]+centers[1][1]*yy)
    widths=[]
    for left,right in ((co[0],co[1]),(co[2],co[3])):
        w=(right[0]+right[1]*yy)-(left[0]+left[1]*yy)
        widths.append({"mean_m":float(np.mean(w)),"range_m":[float(w.min()),float(w.max())]})
    return {"line_fits":fits,"midline_lateral_difference_rms_m":float(np.sqrt(np.mean(delta**2))),
            "midline_lateral_difference_max_m":float(np.max(abs(delta))),
            "midline_slope_difference":float(abs(centers[0][1]-centers[1][1])),
            "observed_road_widths":widths}


def yellow_metric(path3,path5):
    q=load_mark(path3);r=load_mark(path5)
    moving=fit_line(q,range(1641,1657),(21.0,23.2),(27.75,31.15),.065)
    ref=fit_line(r,range(1784,1794),(21.0,23.2),(27.75,31.15),.065)
    a=np.asarray(moving["coef_x_a_plus_b_y"]);b=np.asarray(ref["coef_x_a_plus_b_y"])
    yy=np.linspace(28,31,121);d=(a[0]+a[1]*yy)-(b[0]+b[1]*yy)
    return {"moving":moving,"reference":ref,"rms_m":float(np.sqrt(np.mean(d*d))),
            "max_m":float(np.max(abs(d)))}


def db_metrics():
    def ro(p):return sqlite3.connect(f"file:{p.resolve()}?mode=ro&immutable=1",uri=True)
    build=json.loads((V9.parent/"work/v9_three_region_build.json").read_text())
    modified=set(build["changed_pose_nodes"])
    with ro(V8) as a,ro(V9) as b:
        integrity=b.execute("PRAGMA integrity_check").fetchone()[0]
        ids=[x[0] for x in b.execute("SELECT id FROM Node")]
        graph={i:set() for i in ids}
        for x,y in b.execute("SELECT from_id,to_id FROM Link"):
            graph[x].add(y);graph[y].add(x)
        remain=set(ids);components=[]
        while remain:
            todo=[remain.pop()];n=0
            while todo:
                x=todo.pop();n+=1
                for y in graph[x]:
                    if y in remain:remain.remove(y);todo.append(y)
            components.append(n)
        outside_equal=True;outside_max=0.0
        for node,pa in a.execute("SELECT id,pose FROM Node ORDER BY id"):
            pb=b.execute("SELECT pose FROM Node WHERE id=?",(node,)).fetchone()[0]
            if node not in modified:
                outside_equal &= pa==pb
                ta=np.frombuffer(pa,np.float32).reshape(3,4);tb=np.frombuffer(pb,np.float32).reshape(3,4)
                outside_max=max(outside_max,float(np.max(abs(ta-tb))))
        def digest(db,query):
            h=hashlib.sha256()
            for row in db.execute(query):
                for v in row:h.update(v if isinstance(v,bytes) else repr(v).encode())
            return h.hexdigest()
        data="SELECT id,image,depth,calibration,scan,scan_info FROM Data ORDER BY id"
        feat="SELECT * FROM Feature ORDER BY rowid"
        preserved=(digest(a,data)==digest(b,data) and digest(a,feat)==digest(b,feat))
    return {"integrity":integrity,"node_count":len(ids),"components":sorted(components,reverse=True),
            "modified_nodes":sorted(modified),"outside_pose_byte_identical":outside_equal,
            "outside_pose_max_matrix_change":outside_max,"rgbd_scan_feature_payload_identical":preserved}


def main():
    OUT.mkdir(parents=True,exist_ok=True);visual_figures()
    report={"input":{"path":str(V8),"sha256":sha(V8)},
            "output":{"path":str(V9),"size_bytes":V9.stat().st_size,"sha256":sha(V9)},
            "ply":{"path":str(P9),"size_bytes":P9.stat().st_size,"sha256":sha(P9)},
            "regions":{"yellow_split_bbox_xy":REGIONS["yellow_split"],
                       "triangle_intersection_bbox_xy":REGIONS["triangle_intersection"],
                       "long_straight_bbox_xy":REGIONS["long_straight"]},
            "yellow_split":{"before":yellow_metric(M8_3,M8_5),"after":yellow_metric(M9_3,M9_5)},
            "long_straight":{"before":road_metrics(M8_0,True),"after":road_metrics(M9_0,False)},
            "database":db_metrics()}
    (OUT/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
