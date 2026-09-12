#!/usr/bin/env python3
"""Generate final data-driven figures and numeric validation for ramp revision."""

from __future__ import annotations

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
BASE = ROOT/"maps/competition_338.db"
AUX = ROOT/"maps/map_20260906_091640/rtabmap.db"
FINAL = ROOT/"maps/merged_competition_1lane_2lane_ramp/rtabmap.db"
OUT = ROOT/"analysis/map_ramp_revision"
REPORT_PATH = ROOT/"maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json"
FINAL_POSES = Path("/tmp/ramp_final_export/ramp_final3_global_poses.txt")
FLAT_POSES = Path("/tmp/ramp_export/merged_flat_poses.txt")
BASE_POSES = Path("/tmp/rtabmap_audit_poses/competition_338_poses.txt")
AUX_POSES = Path("/tmp/rtabmap_audit_poses/map_091640_poses.txt")


def load_pose_file(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"): continue
        f=line.split(); x,y,z=map(float,f[1:4]); qx,qy,qz,qw=map(float,f[4:8])
        r=np.asarray([
            [1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
        m=np.eye(4);m[:3,:3]=r;m[:3,3]=(x,y,z);result[int(f[8])]=m
    return result


def trajectory(poses, ids=None):
    keys=sorted(poses) if ids is None else [i for i in ids if i in poses]
    return keys,np.asarray([poses[i][:3,3] for i in keys])


def source_mapping(report):
    return {int(k):(v[0],int(v[1])) for k,v in report["node_source_map"].items()}


def source_order(final_poses, mapping, source):
    ids=[i for i in final_poses if mapping[i][0]==source]
    return sorted(ids,key=lambda i:mapping[i][1])


def load_rgb(db, node_id):
    c=sqlite3.connect(f"file:{db.resolve()}?mode=ro&immutable=1",uri=True)
    blob=c.execute("SELECT image FROM Data WHERE id=?",(node_id,)).fetchone()[0];c.close()
    image=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def contact_sheet():
    scenes=[
      ("base",73,"START sign (right lane marking)"),("aux",240,"START sign"),
      ("base",94,"crosswalk begins"),("aux",280,"crosswalk begins"),
      ("base",112,"crosswalk ends"),("aux",335,"crosswalk ends"),
      ("base",121,"stop line 1"),("aux",360,"stop line 1"),
      ("base",132,"uphill"),("aux",420,"uphill"),
      ("base",148,"stop line 2 candidate"),("aux",440,"stop line 2"),
      ("base",160,"crest"),("aux",481,"crest"),
      ("base",176,"downhill"),("aux",520,"downhill"),
      ("base",210,"flat return"),("aux",630,"flat return")]
    fig,axes=plt.subplots(3,6,figsize=(18,9),constrained_layout=True)
    for ax,(name,nid,label) in zip(axes.flat,scenes):
        ax.imshow(load_rgb(BASE if name=="base" else AUX,nid));ax.axis("off")
        color="#111111" if name=="base" else "#d62728"
        ax.set_title(f"{'competition_338' if name=='base' else '091640'}:{nid}\n{label}",fontsize=8,color=color,weight="bold")
        for s in ax.spines.values():s.set_visible(True);s.set_color(color);s.set_linewidth(3)
    fig.suptitle("Ramp landmark identification — actual stored RGB",fontsize=16,weight="bold")
    fig.savefig(OUT/"ramp_nodes_contact_sheet.png",dpi=180);plt.close(fig)


def apply_transform(points,theta_deg,t):
    a=math.radians(theta_deg);r=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    return points@r.T+np.asarray(t)


def xy_alignment(report,base,aux):
    _,b=trajectory(base); aids,a=trajectory(aux)
    global_fit=report["global_se2_rejected"]
    global_xy=apply_transform(a[:,:2],global_fit["theta_deg"],global_fit["translation_m"])
    knots=[]
    centers={"departure_rejoin":500,"s_curve":1250,"right_angle_corner":1575,"crosswalk":2420,"final_return":2910}
    for name,center in centers.items():
        f=report["piecewise_validation_fits"][name];knots.append((center,f["theta_deg"],*f["translation_m"]))
    knots=np.asarray(knots,float)
    theta=np.interp(aids,knots[:,0],knots[:,1]);tx=np.interp(aids,knots[:,0],knots[:,2]);ty=np.interp(aids,knots[:,0],knots[:,3])
    piece=np.empty((len(a),2))
    for i,p in enumerate(a[:,:2]):piece[i]=apply_transform(p[None,:],theta[i],[tx[i],ty[i]])[0]
    fig,axes=plt.subplots(1,2,figsize=(18,8),constrained_layout=True)
    axes[0].plot(*b[:,:2].T,color="black",lw=2.2,label="competition_338")
    axes[0].plot(*global_xy.T,color="#d62728",lw=1.2,label="091640 global SE(2) (rejected)")
    axes[0].set_title(f"BEFORE: one global SE(2)\nanchor RMS={global_fit['rms_m']:.3f} m")
    axes[1].plot(*b[:,:2].T,color="black",lw=2.2,label="competition_338 fixed")
    axes[1].plot(*piece.T,color="#2672d8",lw=1.3,label="091640 continuous piecewise diagnostic")
    f=report["applied_to_retained_091640"]
    mask=(np.asarray(aids)>=186)&(np.asarray(aids)<=812)
    branch=apply_transform(a[mask,:2],f["theta_deg"],f["translation_m"])
    axes[1].plot(*branch.T,color="#d62728",lw=3,label="retained 186–812 (applied)")
    axes[1].set_title(f"AFTER: local submap fits, scale=1.000\nretained-branch anchor RMS={f['rms_m']:.3f} m")
    for ax in axes:
        ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.3);ax.set_xlabel("X [m] →");ax.set_ylabel("Y [m] →");ax.legend(fontsize=8)
    fig.suptitle("091640 registration: optimized trajectories, RAW frame distinguished from ALIGNED",weight="bold",fontsize=15)
    fig.savefig(OUT/"xy_alignment_before_after.png",dpi=190);plt.close(fig)


def cumulative(xyz):
    return np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))]


def z_profiles(report,final,flat,mapping,base_raw,aux_raw):
    b_ids,b=trajectory(base_raw);a_ids,a=trajectory(aux_raw)
    final_base_ids=source_order(final,mapping,"competition_338");fb=np.asarray([final[i][:3,3] for i in final_base_ids])
    final_aux_ids=source_order(final,mapping,"091640");fa=np.asarray([final[i][:3,3] for i in final_aux_ids])
    fig,axes=plt.subplots(2,1,figsize=(15,10),constrained_layout=True)
    axes[0].plot(cumulative(b),b[:,2],color="#777",lw=1,label="competition original: ~22 m drift")
    axes[0].plot(cumulative(a),a[:,2],color="#d62728",lw=1,label="091640 original")
    axes[0].set_title("BEFORE — original optimized Z")
    axes[1].plot(cumulative(fb),fb[:,2],color="black",lw=1.8,label="final competition branch")
    axes[1].plot(cumulative(fa),fa[:,2],color="#d62728",lw=1.8,label="final 091640 branch")
    axes[1].axhline(0,color="#2ca02c",ls="--",lw=1,label="flattened level")
    axes[1].set_title(f"AFTER — relative ramp retained, peak={report['height_profile']['peak_m']:.3f} m")
    for ax in axes:
        ax.grid(alpha=.3);ax.legend();ax.set_xlabel("travel distance [m]");ax.set_ylabel("Z [m]")
    fig.savefig(OUT/"z_profile_before_after.png",dpi=190);plt.close(fig)


def segmented_map(final,mapping):
    fig,ax=plt.subplots(figsize=(12,11),constrained_layout=True)
    for source,color in (("competition_338","black"),("091640","#d62728")):
        ids=source_order(final,mapping,source);xyz=np.asarray([final[i][:3,3] for i in ids]);src=np.asarray([mapping[i][1] for i in ids])
        ramp=(src>=115)&(src<=210) if source=="competition_338" else (src>=345)&(src<=630)
        ax.plot(*xyz[~ramp,:2].T,color="#999",lw=1.3,label=f"{source} flattened")
        ax.plot(*xyz[ramp,:2].T,color=color,lw=4,label=f"{source} ramp/transition")
    ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.3);ax.legend();ax.set_xlabel("X [m] →");ax.set_ylabel("Y [m] →")
    ax.set_title("Flattened (gray) and ramp-preserved (colored) segments\nfinal optimized graph, scale=1.000",weight="bold")
    fig.savefig(OUT/"flattened_and_ramp_segments.png",dpi=190);plt.close(fig)


def load_occ():
    c=sqlite3.connect(f"file:{FINAL.resolve()}?mode=ro&immutable=1",uri=True)
    b,x,y,r=c.execute("select opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution from Admin").fetchone();c.close()
    d=zlib.decompressobj();raw=d.decompress(b)+d.flush();w,h,t=struct.unpack("<3i",d.unused_data)
    return np.frombuffer(raw,np.int8).reshape(h,w),(x,x+w*r,y,y+h*r),r


def occupancy_graph(final,mapping):
    grid,extent,res=load_occ();shown=np.zeros_like(grid,np.uint8);shown[grid==0]=1;shown[grid==100]=2
    cmap=ListedColormap([(1,1,1,0),(.82,.84,.87,.72),(.05,.06,.07,.95)])
    fig,ax=plt.subplots(figsize=(13,12),constrained_layout=True);ax.imshow(shown,origin="lower",extent=extent,cmap=cmap,interpolation="nearest")
    for source,color in (("competition_338","#111"),("091640","#d62728")):
        ids=source_order(final,mapping,source);xyz=np.asarray([final[i][:3,3] for i in ids]);ax.plot(*xyz[:,:2].T,color=color,lw=1.6,label=source)
    ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.2);ax.legend();ax.set_xlabel("X [m] →");ax.set_ylabel("Y [m] →")
    ax.set_title(f"Final regenerated global occupancy + one-component graph\n{grid.shape[1]}×{grid.shape[0]}, {res:.2f} m/cell",weight="bold")
    fig.savefig(OUT/"final_occupancy_and_graph.png",dpi=190);plt.close(fig)


def local_points(blob):
    if not blob:return np.empty((0,3))
    d=zlib.decompressobj();raw=d.decompress(blob)+d.flush();return np.frombuffer(raw,np.float32).reshape(-1,4)[:,:3]


def ramp_3d(final,mapping):
    selected=[]
    for node_id,(source,sid) in mapping.items():
        if (source=="competition_338" and 115<=sid<=210) or (source=="091640" and 345<=sid<=630):selected.append(node_id)
    c=sqlite3.connect(f"file:{FINAL.resolve()}?mode=ro&immutable=1",uri=True);ground=[];obstacle=[]
    for k,node_id in enumerate(sorted(selected)):
        if k%3:continue
        g,o=c.execute("select ground_cells,obstacle_cells from Data where id=?",(node_id,)).fetchone();m=final[node_id]
        for blob,target in ((g,ground),(o,obstacle)):
            p=local_points(blob)[::12]
            if len(p):target.append(p@m[:3,:3].T+m[:3,3])
    c.close();g=np.vstack(ground);o=np.vstack(obstacle) if obstacle else np.empty((0,3))
    fig=plt.figure(figsize=(14,10),constrained_layout=True);ax=fig.add_subplot(111,projection="3d")
    ax.scatter(g[:,0],g[:,1],g[:,2],s=.25,c=g[:,2],cmap="viridis",alpha=.32,label="ground cells")
    if len(o):ax.scatter(o[:,0],o[:,1],o[:,2],s=.6,color="#333",alpha=.35,label="obstacle cells")
    for source,color in (("competition_338","black"),("091640","#d62728")):
        ids=[i for i in selected if mapping[i][0]==source];ids.sort(key=lambda i:mapping[i][1]);p=np.asarray([final[i][:3,3] for i in ids]);ax.plot(*p.T,color=color,lw=3,label=source)
    ax.set_xlabel("X [m]");ax.set_ylabel("Y [m]");ax.set_zlabel("Z [m]");ax.view_init(24,-62);ax.legend();ax.set_title("Final 3D ramp — regenerated RTAB-Map local grids and optimized poses",weight="bold")
    fig.savefig(OUT/"final_3d_ramp_view.png",dpi=190);plt.close(fig)


def all_components(db):
    c=sqlite3.connect(f"file:{db.resolve()}?mode=ro&immutable=1",uri=True);ids={r[0] for r in c.execute('select id from Node')};adj={i:set() for i in ids}
    for a,b in c.execute('select from_id,to_id from Link'):
        if a in adj and b in adj:adj[a].add(b);adj[b].add(a)
    c.close();components=[]
    while ids:
        stack=[ids.pop()];n=0
        while stack:
            q=stack.pop();n+=1
            for v in adj[q]:
                if v in ids:ids.remove(v);stack.append(v)
        components.append(n)
    return sorted(components,reverse=True)


def validation(final,flat,mapping,report):
    base_ids=[i for i in final if mapping[i][0]=="competition_338" and i in flat]
    delta=np.asarray([final[i][:2,3]-flat[i][:2,3] for i in base_ids]);flat_nodes=[];ramp_nodes=[]
    for i,m in final.items():
        source,sid=mapping[i];ramp=(115<=sid<=210) if source=="competition_338" else (345<=sid<=630)
        (ramp_nodes if ramp else flat_nodes).append(i)
    flat_z=np.asarray([final[i][2,3] for i in flat_nodes]);ramp_ids=source_order(final,mapping,"competition_338")
    ramp_ids=[i for i in ramp_ids if 115<=mapping[i][1]<=210];rp=np.asarray([final[i][:3,3] for i in ramp_ids]);dz=np.abs(np.diff(rp[:,2]));ds=np.linalg.norm(np.diff(rp[:,:2],axis=0),axis=1)
    # Check whether differently elevated ramp/non-ramp poses occupy the same XY.
    fxyz=np.asarray([final[i][:3,3] for i in flat_nodes]);rxyz=np.asarray([final[i][:3,3] for i in ramp_nodes]);tree=cKDTree(fxyz[:,:2]);dist,index=tree.query(rxyz[:,:2],k=1)
    conflicts=int(np.count_nonzero((dist<.25)&(np.abs(rxyz[:,2]-fxyz[index,2])>1.0)))
    result={
      "all_link_components":all_components(FINAL),
      "base_xy_deformation_rms_m":float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),
      "base_xy_deformation_max_m":float(np.max(np.linalg.norm(delta,axis=1))),
      "flat_z_abs_max_m":float(np.max(np.abs(flat_z))),
      "flat_z_std_m":float(np.std(flat_z)),
      "ramp_z_min_max_m":[float(rxyz[:,2].min()),float(rxyz[:,2].max())],
      "ramp_max_adjacent_dz_m":float(dz.max()),
      "ramp_max_grade_deg":float(np.degrees(np.max(np.arctan2(dz,np.maximum(ds,1e-9))))),
      "elevated_xy_conflict_pose_count":conflicts,
      "global_se2_before_rms_m":report["global_se2_rejected"]["rms_m"],
      "segment_rms_m":{k:v["rms_m"] for k,v in report["piecewise_validation_fits"].items()},
      "overall_accepted_anchor_rms_m":report["accepted_anchor_rms_m"]}
    path=ROOT/"maps/merged_competition_1lane_2lane_ramp/work/final_validation.json";path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True);report=json.loads(REPORT_PATH.read_text());mapping=source_mapping(report)
    final=load_pose_file(FINAL_POSES);flat=load_pose_file(FLAT_POSES);base=load_pose_file(BASE_POSES);aux=load_pose_file(AUX_POSES)
    contact_sheet();xy_alignment(report,base,aux);z_profiles(report,final,flat,mapping,base,aux);segmented_map(final,mapping);occupancy_graph(final,mapping);ramp_3d(final,mapping);validation(final,flat,mapping,report)
    for p in sorted(OUT.glob('*.png')):print(p.name,p.stat().st_size)


if __name__=='__main__':main()
