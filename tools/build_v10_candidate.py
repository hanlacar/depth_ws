#!/usr/bin/env python3
"""Apply one isolated v10 candidate edit to a pre-copied v8 database."""
from __future__ import annotations
import argparse,hashlib,json,math,sqlite3,struct
from pathlib import Path
import numpy as np

def mat(b):
 t=np.eye(4);t[:3,:4]=np.frombuffer(b,np.float32).reshape(3,4);return t
def pack(t):return struct.pack('<12f',*t[:3,:4].astype(np.float32).ravel())
def smooth(x):x=max(0.,min(1.,x));return x*x*(3-2*x)
def about(p,dx=0,dy=0,dz=0,yaw=0):
 a=math.radians(yaw);c,s=math.cos(a),math.sin(a);r=np.eye(4);r[:2,:2]=((c,-s),(s,c));q=p[:3,3]
 A=np.eye(4);A[:3,3]=q+[dx,dy,dz];B=np.eye(4);B[:3,3]=-q;return A@r@B
def interp(i,c):
 for a,b in zip(c,c[1:]):
  if a[0]<=i<=b[0]:
   w=smooth((i-a[0])/(b[0]-a[0]));return tuple((1-w)*x+w*y for x,y in zip(a[1:],b[1:]))
 return c[0][1:] if i<c[0][0] else c[-1][1:]
def sh(path):
 h=hashlib.sha256();f=path.open('rb')
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 f.close();return h.hexdigest()
def clear_cache(db):
 db.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")

def main():
 ap=argparse.ArgumentParser();ap.add_argument('database',type=Path);ap.add_argument('mode',choices=['yellow','straight','intersection_move','intersection_interpolate','intersection_exclude']);ap.add_argument('report',type=Path);a=ap.parse_args()
 before=sh(a.database);db=sqlite3.connect(a.database);poses={i:mat(b) for i,b in db.execute('select id,pose from Node')};world={};excluded=[]
 if a.mode=='yellow':
  ids=sorted(i for i in poses if 1629<=i<=1664 and db.execute('select map_id from Node where id=?',(i,)).fetchone()[0]==3);f=ids.index(1641);l=ids.index(1656);p=np.array([22.43784325,29.5]);d=np.array([21.94919120,29.5])-p
  for k,i in enumerate(ids):
   w=k/f if k<f else (1 if k<=l else (len(ids)-1-k)/(len(ids)-1-l))
   if w<=0:continue
   z=math.radians(-1.8059808647418325*w);c,s=math.cos(z),math.sin(z);r=np.array(((c,-s),(s,c)));W=np.eye(4);W[:2,:2]=r;W[:2,3]=p+d*w-r@p;W[2,3]=-.003274*w;world[i]=W
 elif a.mode=='straight':
  # Dominant continuous yellow-line/curb fits from the two independent
  # observations. Other painted fragments are not averaged into the axis.
  mid=np.array([-2.6936670879,-.0822889172]);late=np.array([-6.5047873500,-.0494845300]);yc=56.;
  hm=math.atan2(1.,mid[1]);hl=math.atan2(1.,late[1]);b=1./math.tan((hm+hl)/2.)
  x=((mid[0]+mid[1]*yc)+(late[0]+late[1]*yc))/2;aa=x-b*yc
  for lo,fs,fe,hi,old in [(807,813,840,846,mid),(1150,1156,1185,1191,late)]:
   oldh=math.degrees(math.atan2(1,old[1]));newh=math.degrees(math.atan2(1,b))
   for i in range(lo+1,hi):
    if i not in poses:continue
    w=smooth((i-lo)/(fs-lo)) if i<fs else (1 if i<=fe else smooth((hi-i)/(hi-fe)))
    y=poses[i][1,3];world[i]=about(poses[i],((aa+b*y)-(old[0]+old[1]*y))*w,0,0,(newh-oldh)*w)
 elif a.mode in ('intersection_move','intersection_interpolate'):
  ctr=[(1690,0,0,0,0),(1707,-.1546,.1192,0,-3.94),(1738,.2846,.1013,0,.3694),(1751,0,0,0,0)]
  for i in range(1691,1751):
   if i not in poses:continue
   if a.mode=='intersection_move':
    w=min(smooth((i-1690)/17),smooth((1751-i)/13));v=(-.1546*w,.1192*w,0,-3.94*w)
   else:v=interp(i,ctr)
   world[i]=about(poses[i],*v)
 elif a.mode=='intersection_exclude':
  excluded=[i for i in range(1718,1751) if i in poses]
  # Only the redundant local LaserScan cache is excluded. RGB, Depth,
  # calibration, features and graph nodes remain available for localization.
  db.executemany('UPDATE Data SET scan=NULL WHERE id=?',[(i,) for i in excluded])
 old=poses.copy()
 for i,W in world.items():
  new=W@old[i];gt=db.execute('select ground_truth_pose from Node where id=?',(i,)).fetchone()[0]
  db.execute('update Node set pose=?,ground_truth_pose=? where id=?',(pack(new),pack(W@mat(gt)) if gt else gt,i))
 for rowid,x,y,b in list(db.execute('select rowid,from_id,to_id,transform from Link')):
  if x not in world and y not in world:continue
  cx=np.linalg.inv(old[x])@(world[x]@old[x]) if x in world else np.eye(4);cy=np.linalg.inv(old[y])@(world[y]@old[y]) if y in world else np.eye(4)
  db.execute('update Link set transform=? where rowid=?',(pack(np.linalg.inv(cx)@mat(b)@cy),rowid))
 clear_cache(db);db.commit();integrity=db.execute('pragma integrity_check').fetchone()[0];nodes=db.execute('select count(*) from Node').fetchone()[0];scans=db.execute('select count(scan) from Data').fetchone()[0];db.close()
 out={'mode':a.mode,'database':str(a.database.resolve()),'sha256_before':before,'sha256_after':sh(a.database),'integrity':integrity,'nodes':nodes,'modified_pose_nodes':sorted(world),'excluded_scan_nodes':excluded,'remaining_scans':scans,'scale':1.0}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
