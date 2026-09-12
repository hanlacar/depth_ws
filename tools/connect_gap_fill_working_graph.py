#!/usr/bin/env python3
"""Connect accepted gap-fill submaps without changing either input DB."""

from __future__ import annotations

import hashlib, json, shutil, sqlite3
from pathlib import Path
import numpy as np

ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_gap_filled/work/gap_append_no_cross.db"
OUTPUT = ROOT / "maps/merged_competition_gap_filled/work/gap_connected_seed.db"
ORIGINAL = ROOT / "maps/map_20260906_113208/rtabmap.db"
BASE_POSES = ROOT / "analysis/full_course_final/full_course_global_poses.txt"
REPORT = ROOT / "maps/merged_competition_gap_filled/work/gap_graph_connections.json"


def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def matrix(blob):
    p=np.eye(4); p[:3,:4]=np.frombuffer(blob,np.float32).reshape(3,4); return p


def blob(p): return np.asarray(p[:3,:4],np.float32).tobytes()


def load_poses(p):
    out={}
    for line in p.read_text().splitlines():
        if not line or line.startswith("#"): continue
        f=line.split(); x,y,z,qx,qy,qz,qw=map(float,f[1:8]); m=np.eye(4)
        m[:3,:3]=np.array([
          [1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
          [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
          [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
        m[:3,3]=(x,y,z); out[int(f[8])]=m
    return out


def main():
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source_hash=sha256(SOURCE); shutil.copy2(SOURCE,OUTPUT)
    opt=load_poses(BASE_POSES)
    con=sqlite3.connect(OUTPUT); orig=sqlite3.connect(f"file:{ORIGINAL}?mode=ro&immutable=1",uri=True)
    # Reprocessing preserved timestamps.  This is the auditable source-to-final ID map.
    src_stamp={i:s for i,s in orig.execute("SELECT id,stamp FROM Node WHERE id BETWEEN 2303 AND 2803")}
    final_stamp={s:i for i,s in con.execute("SELECT id,stamp FROM Node WHERE id>1750")}
    mapping={i:final_stamp[s] for i,s in src_stamp.items() if s in final_stamp}
    orig.close()
    edges=[
      # temporal continuation of already retained source node 2302 (final 1750)
      (1750,mapping[2303],2303,"intersection temporal continuation"),
      # independent return submap fixed by the strongest multi-view anchor
      (1253,mapping[2420],2420,"return RGB-D anchor: 44 prior / strong recheck"),
    ]
    info=np.diag([4.0,4.0,2.0,8.0,8.0,25.0]).astype(np.float64)
    records=[]
    for base_id,target_id,source_id,label in edges:
        bp=opt[base_id]
        tp=matrix(con.execute("SELECT pose FROM Node WHERE id=?",(target_id,)).fetchone()[0])
        rel=np.linalg.inv(bp)@tp
        for a,b,m in ((base_id,target_id,rel),(target_id,base_id,np.linalg.inv(rel))):
            con.execute("INSERT INTO Link(from_id,to_id,type,information_matrix,transform,user_data) VALUES(?,?,?,?,?,NULL)",
                        (a,b,4,info.tobytes(),blob(m)))
        records.append({"label":label,"base_id":base_id,"final_id":target_id,
                        "source_113208_id":source_id,"relative_translation_m":rel[:3,3].tolist()})
    con.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,"
                "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
                "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
                "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    con.commit(); integrity=con.execute("PRAGMA integrity_check").fetchone()[0]
    counts={str(k):v for k,v in con.execute("SELECT map_id,count(*) FROM Node GROUP BY map_id")}
    con.close()
    if sha256(SOURCE)!=source_hash: raise RuntimeError("append candidate changed")
    report={"input":str(SOURCE),"input_sha256":source_hash,"output":str(OUTPUT),
            "integrity_check":integrity,"node_count_by_map_id":counts,
            "information_diagonal":np.diag(info).tolist(),"edges":records,
            "source_to_working_id":{str(k):v for k,v in sorted(mapping.items())}}
    REPORT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
