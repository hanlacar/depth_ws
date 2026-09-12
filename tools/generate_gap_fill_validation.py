#!/usr/bin/env python3
"""Create auditable before/after visuals and metrics for the gap-filled map."""

from __future__ import annotations

import hashlib, json, math, sqlite3, struct, zlib
from pathlib import Path
import cv2
import matplotlib
import numpy as np
from scipy import ndimage
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path("/home/qor/depth_ws")
BASE=ROOT/"maps/merged_competition_full_course/rtabmap.db"
FINAL=ROOT/"maps/merged_competition_gap_filled/rtabmap.db"
SRC=ROOT/"maps/map_20260906_113208/rtabmap.db"
OUT=ROOT/"analysis/gap_fill/final"
PRE=json.loads((ROOT/"analysis/full_course_premerge/premerge_report.json").read_text())
MAP=json.loads((ROOT/"maps/merged_competition_gap_filled/work/gap_graph_connections.json").read_text())
RANGES=[(1661,1881),(2303,2399),(2400,2803)]


def ro(path): return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1",uri=True)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def grid(path):
 with ro(path) as c: b,x,y,r=c.execute("select opt_map,opt_map_x_min,opt_map_y_min,opt_map_resolution from Admin").fetchone()
 o=zlib.decompressobj(); raw=o.decompress(b)+o.flush(); w,h,_=struct.unpack("<3i",o.unused_data)
 return np.frombuffer(raw,np.int8).reshape(h,w),float(x),float(y),float(r)
def pts(blob):
 if not blob:return np.empty((0,3),np.float32)
 o=zlib.decompressobj();raw=o.decompress(blob)+o.flush();a,b,t=struct.unpack("<3i",o.unused_data)
 return np.frombuffer(raw,np.float32).reshape(a*b,4)[:,:3]
def poses(path):
 d={}
 for l in open(path):
  if l.startswith('#'):continue
  f=l.split();qx,qy,qz,qw=map(float,f[4:8]);d[int(f[-1])]=np.array([float(f[1]),float(f[2]),float(f[3]),math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))])
 return d
def source_candidate_masks(shape,x0,y0,res):
 sp=poses('/tmp/rtabmap_audit_poses/map_113208_poses.txt'); votes=np.zeros(shape,np.uint16); node_masks={}
 fits=PRE['fits']; h,w=shape
 with ro(SRC) as c:
  for node,blob in c.execute('select id,ground_cells from Data order by id'):
   if not any(a<=node<=b for a,b in RANGES) or node not in sp:continue
   fit=fits['intersection' if node<=2399 else 'return'];th=math.radians(fit['theta_deg']);r=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
   xy=sp[node][:2]@r.T+fit['translation_m'];heading=sp[node][3]+th;rr=np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])
   p=pts(blob);world=p[:,:2]@rr.T+xy;ix=np.floor((world[:,0]-x0)/res).astype(int);iy=np.floor((world[:,1]-y0)/res).astype(int);v=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
   flat=np.unique(iy[v]*w+ix[v]); votes.ravel()[flat]=np.minimum(votes.ravel()[flat]+1,65535);node_masks[node]=flat
 return votes,node_masks
def display(g):
 d=np.zeros_like(g,np.uint8);d[g<0]=0;d[g==0]=1;d[g==100]=2;return d
CMAP=ListedColormap([(0.02,0.02,0.02,1),(0.42,0.43,0.44,1),(0.96,0.96,0.96,1)])
def base_axes(ax,g,extent,title):
 ax.imshow(display(g),origin='lower',extent=extent,cmap=CMAP,interpolation='nearest');ax.set_title(title,weight='bold');ax.set_aspect('equal');ax.set_xlabel('X [m] →');ax.set_ylabel('Y [m] →');ax.grid(alpha=.15)
def save_full(g,extent,path,title,traj=None):
 fig,ax=plt.subplots(figsize=(11,10),constrained_layout=True);base_axes(ax,g,extent,title)
 if traj is not None:
  q=np.array([traj[i][:2] for i in sorted(traj)]);ax.plot(q[:,0],q[:,1],color='#ff8c00',lw=.6,alpha=.65,label='optimized trajectory');ax.legend()
 fig.savefig(path,dpi=180);plt.close(fig)
def load_rgb_depth(node):
 with ro(SRC) as c: im,dep=c.execute('select image,depth from Data where id=?',(node,)).fetchone()
 rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(im,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
 # RTAB-Map stores these depth frames in DEPTHRVL, exported losslessly by the
 # small C++ decoder before this reporting script is run.
 depth=cv2.imread(str(ROOT/f'analysis/gap_fill/depth_export/depth_{node}.png'),cv2.IMREAD_UNCHANGED)
 return rgb,depth
def graph_stats():
 with ro(FINAL) as c:
  ids=[x[0] for x in c.execute('select id from Node')]; adj={i:set() for i in ids}
  for a,b in c.execute('select from_id,to_id from Link'):
   if a in adj and b in adj:adj[a].add(b);adj[b].add(a)
  seen=set();comps=[]
  for i in ids:
   if i in seen:continue
   st=[i];seen.add(i);n=0
   while st:
    q=st.pop();n+=1
    for z in adj[q]:
     if z not in seen:seen.add(z);st.append(z)
   comps.append(n)
  row=c.execute('select count(*),sum(image is not null),sum(depth is not null),sum(calibration is not null),sum(scan is not null),count(distinct calibration) from Data').fetchone()
  feat=c.execute('select count(distinct node_id),count(*) from Feature').fetchone()
  return {'components':sorted(comps,reverse=True),'data_counts':row,'feature_nodes_and_rows':feat}


def main():
 OUT.mkdir(parents=True,exist_ok=True);a,x,y,res=grid(BASE);b,x2,y2,res2=grid(FINAL)
 assert a.shape==b.shape and (x,y,res)==(x2,y2,res2)
 h,w=a.shape;extent=[x,x+w*res,y,y+h*res];oldp=poses(ROOT/'analysis/full_course_final/full_course_global_poses.txt');newp=poses(OUT/'gap_filled_global_poses.txt')
 save_full(a,extent,OUT/'01_before_full_map.png','Before: merged_competition_full_course (stored global occupancy)',oldp)
 save_full(b,extent,OUT/'04_after_full_map.png','After: gap-filled map (same extent, resolution and scale)',newp)
 votes,node_masks=source_candidate_masks(a.shape,x,y,res);candidate=(votes>=2)&(a<0)
 joined=ndimage.binary_closing(ndimage.binary_dilation(candidate,iterations=2),iterations=2);labels,n=ndimage.label(joined)
 regs=[]
 for lab in range(1,n+1):
  mask=(labels==lab)&candidate;cells=int(mask.sum())
  if cells>=35:regs.append((cells,mask))
 regs.sort(key=lambda z:-z[0]);regs=regs[:6]
 selected=set(map(int,MAP['source_to_working_id'])); chosen=[2722,2523,2383,2321,1683,1874,2420]
 selected_ground=np.zeros_like(candidate)
 for node in selected:
  if node in node_masks:selected_ground.ravel()[node_masks[node]]=True
 # RGB and metric Depth evidence.
 fig,axs=plt.subplots(len(chosen),2,figsize=(11,3.0*len(chosen)),constrained_layout=True)
 for row,node in enumerate(chosen):
  rgb,d=load_rgb_depth(node);valid=d[d>0];lo,hi=(np.percentile(valid,[2,98]) if valid.size else (0,1));show=np.clip((d-lo)/max(hi-lo,1),0,1)
  axs[row,0].imshow(rgb);axs[row,0].set_title(f'113208 node {node} RGB — '+('SELECTED' if node in selected else 'HELD'),color=('#0a8f3c' if node in selected else '#c28b00'),weight='bold')
  axs[row,1].imshow(show,cmap='turbo');axs[row,1].set_title(f'node {node} stored Depth (valid={100*np.mean(d>0):.1f}%)')
  for ax in axs[row]:ax.axis('off')
 fig.savefig(OUT/'03_region_rgb_depth.png',dpi=160);plt.close(fig)
 # Exact region crops before/after.
 fig,axs=plt.subplots(3,4,figsize=(15,11),constrained_layout=True);region_metrics=[]
 for idx,(cells,mask) in enumerate(regs):
  yy,xx=np.where(mask);pad=20;i1=max(0,yy.min()-pad);i2=min(h,yy.max()+pad+1);j1=max(0,xx.min()-pad);j2=min(w,xx.max()+pad+1)
  e=[x+j1*res,x+j2*res,y+i1*res,y+i2*res];known=mask&(b>=0);rem=mask&(b<0);newfree=mask&(b==0);newocc=mask&(b==100)
  row=idx//2;col=(idx%2)*2;base_axes(axs[row,col],a[i1:i2,j1:j2],e,f'G{idx+1} before')
  base_axes(axs[row,col+1],b[i1:i2,j1:j2],e,f'G{idx+1} after — {100*known.sum()/cells:.1f}% observed')
  axs[row,col].title.set_fontsize(10);axs[row,col+1].title.set_fontsize(10)
  region_metrics.append({'region_id':f'G{idx+1}','candidate_area_m2':cells*res*res,'filled_area_m2':int(known.sum())*res*res,'filled_percent':100*known.sum()/cells,'new_free_area_m2':int(newfree.sum())*res*res,'new_boundary_area_m2':int(newocc.sum())*res*res,'remaining_candidate_area_m2':int(rem.sum())*res*res,'bbox_xy':[x+xx.min()*res,y+yy.min()*res,x+(xx.max()+1)*res,y+(yy.max()+1)*res]})
 fig.savefig(OUT/'05_regions_before_after.png',dpi=180);plt.close(fig)
 # New cells in a shared coordinate frame.  Restrict the highlight to cells
 # directly observed by retained source nodes so re-rasterization of unchanged
 # base edges is not mislabeled as new data.
 new=(a<0)&(b>=0);newfree=(a<0)&(b==0);newocc=(a<0)&(b==100)
 direct=selected_ground&new;direct_free=direct&(b==0);direct_occ=direct&(b==100)
 fig,ax=plt.subplots(figsize=(11,10),constrained_layout=True);base_axes(ax,a,extent,'Directly observed additions from retained 113208 nodes')
 yy,xx=np.where(direct_free);ax.scatter(x+(xx+.5)*res,y+(yy+.5)*res,s=.55,c='#00dc62',alpha=.85,label='new observed road/free surface')
 yy,xx=np.where(direct_occ);ax.scatter(x+(xx+.5)*res,y+(yy+.5)*res,s=1.3,c='#00ff9d',alpha=.95,label='new observed curb/wall/obstacle boundary');ax.legend(loc='upper right')
 fig.savefig(OUT/'06_new_observations_highlighted.png',dpi=180);plt.close(fig)
 # Candidate cells left unresolved; yellow means source exists but was held or did not survive filtering.
 remain=candidate&(b<0);fig,ax=plt.subplots(figsize=(11,10),constrained_layout=True);base_axes(ax,b,extent,'Remaining candidate road/boundary observations')
 yy,xx=np.where(remain);ax.scatter(x+(xx+.5)*res,y+(yy+.5)*res,s=.9,c='#ffd21c',label='source observation exists, unresolved/held')
 for idx,(cells,mask) in enumerate(regs):
  yy0,xx0=np.where(mask);ax.text(x+xx0.mean()*res,y+yy0.mean()*res,f'G{idx+1}',color='black',weight='bold',bbox=dict(fc='#ffd21c',ec='black'))
 ax.legend();fig.savefig(OUT/'07_remaining_boundaries.png',dpi=180);plt.close(fig)
 # Re-render numbered candidate overview at the final output location.
 fig,ax=plt.subplots(figsize=(11,10),constrained_layout=True);base_axes(ax,a,extent,'Candidate regions from actual 113208 ground observations')
 yy,xx=np.where(candidate);ax.scatter(x+(xx+.5)*res,y+(yy+.5)*res,s=.65,c='#1ed760',alpha=.7)
 for idx,(cells,mask) in enumerate(regs):
  yy0,xx0=np.where(mask);ax.text(x+xx0.mean()*res,y+yy0.mean()*res,f'G{idx+1}',weight='bold',bbox=dict(fc='white',ec='#15883e'))
 fig.savefig(OUT/'02_candidate_regions_numbered.png',dpi=180);plt.close(fig)
 # Deformation and ramp preservation.
 common=sorted(set(oldp)&set(newp));dxy=np.array([np.linalg.norm(oldp[i][:2]-newp[i][:2]) for i in common]);dyaw=np.array([abs(math.atan2(math.sin(newp[i][3]-oldp[i][3]),math.cos(newp[i][3]-oldp[i][3]))) for i in common]);dz=np.array([abs(newp[i][2]-oldp[i][2]) for i in common]);ramp=[i for i in range(115,211) if i in oldp and i in newp]
 # largest unresolved candidate component span, a conservative proxy for longest gap.
 rl,rn=ndimage.label(remain);spans=[]
 for k in range(1,rn+1):
  yy0,xx0=np.where(rl==k)
  if len(xx0):spans.append(max(xx0.max()-xx0.min()+1,yy0.max()-yy0.min()+1)*res)
 report={'inputs':{str(BASE):{'size_bytes':BASE.stat().st_size,'sha256':sha(BASE)},str(SRC):{'size_bytes':SRC.stat().st_size,'sha256':sha(SRC)}},'output':{'path':str(FINAL),'size_bytes':FINAL.stat().st_size,'sha256':sha(FINAL)},'occupancy':{'resolution_m':res,'before_free_cells':int((a==0).sum()),'before_occupied_cells':int((a==100).sum()),'after_free_cells':int((b==0).sum()),'after_occupied_cells':int((b==100).sum()),'new_known_area_m2':int(new.sum())*res*res,'new_road_free_area_m2':int(newfree.sum())*res*res,'new_boundary_or_obstacle_area_m2':int(newocc.sum())*res*res,'direct_selected_observation_area_m2':int(direct.sum())*res*res,'direct_selected_road_area_m2':int(direct_free.sum())*res*res,'direct_selected_boundary_area_m2':int(direct_occ.sum())*res*res,'lost_known_raster_area_m2':int(((a>=0)&(b<0)).sum())*res*res,'net_known_area_gain_m2':(int((b>=0).sum())-int((a>=0).sum()))*res*res,'candidate_area_m2':int(candidate.sum())*res*res,'candidate_filled_percent':100*int((candidate&(b>=0)).sum())/max(1,int(candidate.sum())),'remaining_candidate_area_m2':int(remain.sum())*res*res,'longest_remaining_component_span_m':max(spans,default=0)},'regions':region_metrics,'graph_and_data':graph_stats(),'base_preservation':{'nodes_compared':len(common),'xy_rms_m':float(np.sqrt(np.mean(dxy*dxy))),'xy_max_m':float(dxy.max()),'yaw_rms_deg':float(np.degrees(np.sqrt(np.mean(dyaw*dyaw)))),'yaw_max_deg':float(np.degrees(dyaw.max())),'z_max_m':float(dz.max()),'ramp_115_210_z_max_change_m':float(max(abs(oldp[i][2]-newp[i][2]) for i in ramp))},'selected_source_nodes':sorted(selected),'selected_source_to_final':MAP['source_to_working_id'],'held_ranges':[[1661,1881]],'images':[str(OUT/f'{i:02d}_{n}.png') for i,n in [(1,'before_full_map'),(2,'candidate_regions_numbered'),(3,'region_rgb_depth'),(4,'after_full_map'),(5,'regions_before_after'),(6,'new_observations_highlighted'),(7,'remaining_boundaries')]]}
 (OUT/'gap_fill_validation_report.json').write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=='__main__':main()
