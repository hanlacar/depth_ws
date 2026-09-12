#!/usr/bin/env python3
"""Render actual stored scans by source pass in the photographed intersection."""
import sqlite3,struct,zlib
from pathlib import Path
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("/home/qor/depth_ws")
DB=ROOT/"maps/merged_competition_level_aligned_v8/rtabmap.db"
OUT=ROOT/"analysis/level_aligned_v9_three/work/triangle_actual_scans_by_pass.png"
GROUPS=[("competition A",408,440,"#111111"),("competition B",640,698,"#555555"),
        ("competition C",728,802,"#8d6e63"),("competition D",932,965,"#999999"),
        ("113208 map3",1690,1750,"#d81b60"),("113208 map4",1751,1767,"#1e88e5")]
BBOX=(4.5,42.8,28.0,54.1)

def mat(b,off=0):
 t=np.eye(4);t[:3,:4]=np.frombuffer(b,np.float32)[off:off+12].reshape(3,4);return t

def load(db,lo,hi):
 xyz=[];rgb=[]
 for pose,scan,info in db.execute("SELECT n.pose,d.scan,d.scan_info FROM Node n JOIN Data d USING(id) WHERE id BETWEEN ? AND ? ORDER BY id",(lo,hi)):
  if not scan:continue
  z=zlib.decompressobj();raw=z.decompress(scan)+z.flush();rows,cols,kind=struct.unpack("<3i",z.unused_data);ch=(kind>>3)+1
  q=np.frombuffer(raw,np.float32).reshape(rows*cols,ch);p=q[:,:3];w=mat(pose)@mat(info,7);p=p@w[:3,:3].T+w[:3,3]
  use=(p[:,0]>=BBOX[0])&(p[:,0]<=BBOX[2])&(p[:,1]>=BBOX[1])&(p[:,1]<=BBOX[3])&(p[:,2]>-.25)&(p[:,2]<.5)
  if not np.any(use):continue
  color=q[use,3].copy().view(np.uint8).reshape(-1,4)[:,:3][:,::-1] if ch>=4 else np.full((use.sum(),3),150,np.uint8)
  xyz.append(p[use]);rgb.append(color)
 return (np.vstack(xyz),np.vstack(rgb)) if xyz else (np.empty((0,3)),np.empty((0,3)))

def main():
 db=sqlite3.connect(f"file:{DB.resolve()}?mode=ro&immutable=1",uri=True)
 fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
 for ax,(name,lo,hi,color) in zip(axes.flat,GROUPS):
  p,c=load(db,lo,hi);ids=np.arange(len(p));
  if len(ids)>250000:ids=ids[::int(np.ceil(len(ids)/250000))]
  ax.set_facecolor("#111");ax.scatter(p[ids,0],p[ids,1],c=c[ids].astype(float)/255,s=.35,linewidths=0,rasterized=True)
  ax.set(xlim=(BBOX[0],BBOX[2]),ylim=(BBOX[1],BBOX[3]),title=f"{name}: {lo}–{hi}",xlabel="X →",ylabel="Y →")
  ax.set_aspect("equal");ax.grid(alpha=.15)
 db.close();fig.suptitle("Actual stored scan contribution by pass — v8")
 fig.savefig(OUT,dpi=230);plt.close(fig)

if __name__=="__main__":main()
