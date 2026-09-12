#!/usr/bin/env python3
"""Fixed-scale v8/v10 visual checks for the three requested candidates."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("/home/qor/depth_ws")
FILES=[("v8",ROOT/"analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply"),
       ("v10",ROOT/"analysis/level_aligned_v10/final/merged_competition_level_aligned_v10.ply")]
OUT=ROOT/"analysis/level_aligned_v10/final"
REGIONS={"yellow_connection":(18,25,25,33.5),
         "long_straight":(-12,38,0,70),
         "intersection":(4.5,42.8,28,54.1)}

def load(path):
    with path.open("rb") as f:
        header=b""
        while not header.endswith(b"end_header\n"): header+=f.readline()
        off=f.tell()
    n=int(next(x.split()[2] for x in header.decode().splitlines() if x.startswith("element vertex")))
    d=np.dtype({"names":["x","y","z","nx","ny","nz","r","g","b","c"],
                "formats":["<f4"]*6+["u1"]*3+["<f4"],
                "offsets":[0,4,8,12,16,20,24,25,26,27],"itemsize":31})
    return np.memmap(path,dtype=d,mode="r",offset=off,shape=(n,))

clouds=[load(p) for _,p in FILES]
fig,axes=plt.subplots(3,2,figsize=(15,22),constrained_layout=True)
for row,(name,b) in enumerate(REGIONS.items()):
    for ax,(version,_),q in zip(axes[row],FILES,clouds):
        m=((q["x"]>=b[0])&(q["x"]<=b[2])&(q["y"]>=b[1])&(q["y"]<=b[3])&
           (q["z"]>=-.25)&(q["z"]<=.5))
        ids=np.flatnonzero(m)
        if len(ids)>400000: ids=ids[::math.ceil(len(ids)/400000)]
        c=np.c_[q["r"][ids],q["g"][ids],q["b"][ids]].astype(float)/255
        ax.set_facecolor("#101010");ax.scatter(q["x"][ids],q["y"][ids],c=c,s=.45,linewidths=0,rasterized=True)
        ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),title=f"{name}: {version}",xlabel="map X [m] →",ylabel="map Y [m] →")
        ax.set_aspect("equal");ax.grid(alpha=.13)
fig.suptitle("v10 candidate outcome — identical top view and scale",weight="bold")
fig.savefig(OUT/"v10_target_regions_before_after.png",dpi=220);plt.close(fig)

# Side-view check around the retained intersection surface.
b=REGIONS["intersection"]
fig,axes=plt.subplots(2,1,figsize=(15,7),constrained_layout=True)
for ax,(version,_),q in zip(axes,FILES,clouds):
    m=((q["x"]>=15)&(q["x"]<=28)&(q["y"]>=43)&(q["y"]<=50)&
       (q["z"]>=-.15)&(q["z"]<=.2))
    ids=np.flatnonzero(m)
    if len(ids)>300000: ids=ids[::math.ceil(len(ids)/300000)]
    ax.scatter(q["x"][ids],q["z"][ids],s=.3,c="#555",alpha=.35,linewidths=0,rasterized=True)
    ax.set(xlim=(15,28),ylim=(-.1,.12),title=f"intersection surface: {version}",xlabel="map X [m] →",ylabel="Z [m] →");ax.grid(alpha=.2)
fig.savefig(OUT/"v10_intersection_side_before_after.png",dpi=220);plt.close(fig)
print(OUT/"v10_target_regions_before_after.png")
