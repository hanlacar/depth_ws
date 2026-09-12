#!/usr/bin/env python3
"""Render the three v10 intersection trials at one fixed map view."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/qor/depth_ws")
FILES = [
    ("v8 reference", ROOT/"analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"),
    ("move 1691–1750", ROOT/"analysis/level_aligned_v10/candidates/intersection/move.ply"),
    ("interpolate 1691–1750", ROOT/"analysis/level_aligned_v10/candidates/intersection/interpolate.ply"),
    ("exclude duplicate scans 1718–1750", ROOT/"analysis/level_aligned_v10/candidates/intersection/exclude_cloud.ply"),
]
OUT = ROOT/"analysis/level_aligned_v10/candidates/intersection/intersection_trials_top.png"
BBOX = (4.5, 42.8, 28.0, 54.1)

def load(path):
    with path.open("rb") as f:
        header=b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        offset=f.tell()
    n=int(next(s.split()[2] for s in header.decode().splitlines() if s.startswith("element vertex")))
    dtype=np.dtype({"names":["x","y","z","nx","ny","nz","r","g","b","c"],
                    "formats":["<f4"]*6+["u1"]*3+["<f4"],
                    "offsets":[0,4,8,12,16,20,24,25,26,27],"itemsize":31})
    return np.memmap(path,dtype=dtype,mode="r",offset=offset,shape=(n,))

fig, axes = plt.subplots(2,2,figsize=(14,14),constrained_layout=True)
for ax,(title,path) in zip(axes.flat,FILES):
    q=load(path)
    m=((q["x"]>=BBOX[0])&(q["x"]<=BBOX[2])&(q["y"]>=BBOX[1])&(q["y"]<=BBOX[3])&
       (q["z"]>=-.25)&(q["z"]<=.5))
    ids=np.flatnonzero(m)
    if len(ids)>450000:
        ids=ids[::math.ceil(len(ids)/450000)]
    rgb=np.c_[q["r"][ids],q["g"][ids],q["b"][ids]].astype(float)/255
    ax.set_facecolor("#101010")
    ax.scatter(q["x"][ids],q["y"][ids],c=rgb,s=.45,linewidths=0,rasterized=True)
    ax.set(xlim=(BBOX[0],BBOX[2]),ylim=(BBOX[1],BBOX[3]),title=title,
           xlabel="map X [m] →",ylabel="map Y [m] →")
    ax.set_aspect("equal"); ax.grid(alpha=.13)
fig.suptitle("Intersection candidate trials — identical top view / map coordinates",weight="bold")
OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=220); plt.close(fig)
print(OUT)
