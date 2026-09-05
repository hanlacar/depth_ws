#!/usr/bin/env python3
"""Compare legacy, original-librealsense and adapted Direct-BEV paths."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from camera_navigation.direct_bev_core import DirectBevConfig, DirectBevPlanner
from camera_navigation.librealsense_bev_path import extract_sliding_window_lanes
from camera_navigation.metric_path_quality import maximum_curvature
from camera_navigation.direct_bev_projection import (
    CameraModel, build_ground_remap, project_mask_to_bev,
)
from camera_navigation.ground_plane_calibration import rotation_matrix_rpy


class LegacyDirectBevPlanner(DirectBevPlanner):
    """Pre-migration row-nearest lane sampler, retained only for A/B tests."""
    def _sample_lane_tracks(self, lane, component):
        cfg = self.config; lane = lane & component
        left, right = [], []; previous_left = previous_right = None
        for x_m in np.linspace(cfg.x_min_m, cfg.x_max_m, cfg.sliding_windows):
            row = int(round((cfg.x_max_m-x_m)/cfg.resolution_m))
            half = max(1, int(round(.5*(cfg.x_max_m-cfg.x_min_m) /
                                      cfg.sliding_windows/cfg.resolution_m)))
            cols = np.flatnonzero(np.any(lane[max(0,row-half):row+half+1], axis=0))
            groups = np.split(cols, np.flatnonzero(np.diff(cols)>1)+1) if len(cols) else []
            values = [cfg.y_min_m+float(np.mean(g))*cfg.resolution_m for g in groups
                      if len(g) >= cfg.window_min_pixels]
            for positive, output, attr in ((True,left,"left"),(False,right,"right")):
                candidates = [v for v in values if (v>0)==positive and v != 0]
                previous = previous_left if positive else previous_right
                if candidates:
                    value=min(candidates,key=lambda y:abs(y-(previous or 0.0)))
                    if previous is None or abs(value-previous)<=cfg.window_half_width_m:
                        output.append((x_m,value))
                        if positive: previous_left=value
                        else: previous_right=value
        return np.asarray(left,float).reshape(-1,2),np.asarray(right,float).reshape(-1,2)

    def _crosscheck_lane_candidate(self, raw, mode, distance, component,
                                   warnings):
        """The pre-migration planner had no lane/road cross-check."""
        return raw, mode


class AdaptedDirectBevPlanner(DirectBevPlanner):
    """Rejected release candidate retained for reproducible A/B evidence."""
    def _sample_lane_tracks(self, lane, component):
        cfg=self.config
        result=extract_sliding_window_lanes(
            lane,component,x_max_m=cfg.x_max_m,y_min_m=cfg.y_min_m,
            resolution_m=cfg.resolution_m,windows=cfg.sliding_windows,
            margin_m=cfg.window_half_width_m,
            recenter_pixels=cfg.window_min_pixels,
            minimum_points=cfg.minimum_path_points,degree=cfg.fitting_degree,
            samples=cfg.sliding_windows)
        return result.left,result.right

    def _crosscheck_lane_candidate(self,raw,mode,distance,component,warnings):
        if mode=='ROAD_ONLY':return raw,mode
        road=self._road_center(distance,component)
        if len(road)<self.config.minimum_path_points:return raw,mode
        overlap=(raw[:,0]>=road[:,0].min())&(raw[:,0]<=road[:,0].max())
        disagreement=(float('inf') if not np.any(overlap) else float(np.percentile(
            np.abs(raw[overlap,1]-np.interp(raw[overlap,0],road[:,0],road[:,1])),90)))
        lane_preview,_,_=self._resample(raw);road_preview,_,_=self._resample(road)
        curve=(abs(maximum_curvature(lane_preview)-maximum_curvature(road_preview))
               if len(lane_preview)>=3 and len(road_preview)>=3 else float('inf'))
        previous=(abs(maximum_curvature(lane_preview)-maximum_curvature(self.previous))
                  if self.previous is not None and len(lane_preview)>=3 else 0.)
        if (disagreement>self.config.road_center_gate_m or
                curve>self.config.temporal_curvature_gate_per_m or
                previous>self.config.temporal_curvature_gate_per_m):
            warnings.append('LANE_ROAD_CENTER_DISAGREEMENT');return road,'ROAD_ONLY'
        return raw,mode


def original_librealsense_path(lane, planner):
    """Literal 100x80/0.1m/2m-offset reference algorithm."""
    grid=np.zeros((100,80),np.uint8)
    rr,cc=np.nonzero(lane)
    metric=planner.grid_to_metric(rr,cc)
    r=np.rint(100-metric[:,0]/.1).astype(int)
    c=np.rint(40-metric[:,1]/.1).astype(int)
    keep=(r>=0)&(r<100)&(c>=0)&(c<80);grid[r[keep],c[keep]]=1
    histogram=np.sum(grid[50:],axis=0);mid=40
    if not np.any(histogram[:mid]) and not np.any(histogram[mid:]): return np.empty((0,2))
    bases=[int(np.argmax(histogram[:mid])),int(np.argmax(histogram[mid:])+mid)]
    nzr,nzc=grid.nonzero();tracks=[]
    for base in bases:
        current=base;chosen=[]
        for window in range(15):
            lo=100-(window+1)*int(100/15);hi=100-window*int(100/15)
            ids=np.flatnonzero((nzr>=lo)&(nzr<hi)&(nzc>=current-10)&(nzc<current+10))
            chosen.append(ids)
            if len(ids)>40: current=int(np.mean(nzc[ids]))
        ids=np.concatenate(chosen);tracks.append((nzr[ids],nzc[ids]))
    plot=np.linspace(99,60,25);fits=[]
    for rows,cols in tracks:
        roi=(rows>=60)&(rows<=99)
        fits.append(np.polyval(np.polyfit(rows[roi],cols[roi],2),plot)
                    if np.count_nonzero(roi)>100 else None)
    if fits[0] is not None and fits[1] is not None: center=.5*(fits[0]+fits[1])
    elif fits[0] is not None: center=fits[0]+20
    elif fits[1] is not None: center=fits[1]-20
    else:return np.empty((0,2))
    center += (40-center[0])*np.linspace(1.,0.,len(center))
    return np.column_stack(((100-plot)*.1,-(center-40)*.1))


def masks(result, shape):
    road=np.zeros(shape,np.uint8);lane=np.zeros(shape,np.uint8)
    if result.masks is None:return road,lane
    data=result.masks.data.cpu().numpy();classes=result.boxes.cls.cpu().numpy().astype(int)
    for mask,cls in zip(data,classes):
        resized=cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)>=.5
        if cls==0:road[resized]=1
        if cls in (1,2):lane[resized]=1
    return road,lane


def row(name,result,planner,road,lane,elapsed):
    points=result.points;diag=result.diagnostics
    near_y=(float(np.interp(planner.config.near_required_m,points[:,0],points[:,1]))
            if len(points) else float("nan"))
    return {"planner":name,"state":result.state,"mode":result.mode,
            "valid":result.valid,
            "path_point_count":len(points),"path_length_m":float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()) if len(points)>1 else 0.,
            "minimum_clearance_m":diag.get("minimum_clearance_m",0.),"safe_coverage":diag.get("safe_road_coverage",0.),
            "required_steering_deg":float(diag.get("required_steering_deg",0.) or 0.),
            "near_path_y_m":near_y,
            "fitting_residual":diag.get("fitting_residual_m",0.),"failure_reason":"|".join(diag.get("reasons",[])),
            "processing_ms":elapsed,"road_pixels":int(road.sum()),"lane_pixels":int(lane.sum())}


def bev_panel(component, safe, points, planner, title, state, reason="",
              steering=0., coverage=0., processing_ms=0.):
    canvas=np.dstack((component*45,safe*125,component*75)).astype(np.uint8)
    for point in np.asarray(points).reshape(-1,2):
        row,col=planner.metric_to_grid([point])[0]
        if 0<=row<canvas.shape[0] and 0<=col<canvas.shape[1]:
            cv2.circle(canvas,(int(col),int(row)),2,(0,255,255),-1)
    out=cv2.resize(canvas,(320,480),interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(out,(0,0),(319,122),(0,0,0),-1)
    cv2.putText(out,f"{title} {state}",(7,19),0,.48,(255,255,255),1)
    cv2.putText(out,reason[:42],(7,40),0,.38,(0,180,255),1)
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()) if len(points)>1 else 0.
    cv2.putText(out,f"length={length:.2f}m",(7,58),0,.38,(255,255,255),1)
    cv2.putText(out,f"points={len(points)}",(7,76),0,.38,(255,255,255),1)
    cv2.putText(out,f"steer={steering:+.2f}deg safe={coverage:.3f}",(7,94),0,.38,(255,255,255),1)
    cv2.putText(out,f"planner={processing_ms:.2f}ms {1000./max(.001,processing_ms):.1f} FPS",(7,112),0,.38,(255,255,255),1)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=100);ap.add_argument('--start-frame',type=int,default=-1,help='Use consecutive frames from this index; negative samples uniformly and independently.');ap.add_argument('--device',default='cpu');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=DirectBevConfig();camera=CameraModel(640,480,np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]]),np.zeros(5));mx,my=build_ground_remap(cfg,camera,rotation_matrix_rpy(0,-5,0),np.array([.32,0,.85]))
    cap=cv2.VideoCapture(str(a.video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=max(1.,cap.get(cv2.CAP_PROP_FPS));indices=(np.arange(a.start_frame,min(count,a.start_frame+a.frames),dtype=int) if a.start_frame>=0 else np.linspace(0,count-121,a.frames,dtype=int));model=YOLO(str(a.model),task='segment');rows=[]
    if a.start_frame >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start_frame))
    old_planner=LegacyDirectBevPlanner(DirectBevConfig())
    new_planner=AdaptedDirectBevPlanner(DirectBevConfig())
    safety_planner=DirectBevPlanner(DirectBevConfig())
    output_fps=min(30.,fps) if a.start_frame>=0 else 10.
    writer=cv2.VideoWriter(str(a.output/'bev_old_vs_librealsense_vs_new.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),output_fps,(1280,480))
    new_writer=cv2.VideoWriter(str(a.output/'bev_new_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),output_fps,(640,480))
    total=len(indices)
    for position,index in enumerate(indices):
        if a.start_frame < 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
        ok,frame=cap.read()
        if not ok:
            continue
        index=int(index)
        if a.start_frame < 0:
            # Uniform samples are independent scenes several seconds apart;
            # carrying a live-frame temporal prior across them is invalid.
            old_planner=LegacyDirectBevPlanner(DirectBevConfig())
            new_planner=AdaptedDirectBevPlanner(DirectBevConfig())
        pred=model.predict(frame,imgsz=640,conf=.25,device=a.device,verbose=False)[0];ri,li=masks(pred,frame.shape[:2]);road=project_mask_to_bev(ri,mx,my);lane=project_mask_to_bev(li,mx,my)
        planner_results={}
        for name,planner in (("old",old_planner),("new",new_planner)):
            t=time.perf_counter();result=planner.plan(road,lane,index/fps);elapsed=(time.perf_counter()-t)*1000;record=row(name,result,planner,road,lane,elapsed);record['frame_index']=index;rows.append(record);planner_results[name]=(planner,result,record)
        processed=safety_planner.preprocess(road,lane);component,safe=processed[2],processed[3]
        t=time.perf_counter();original=original_librealsense_path(lane,safety_planner);original_ms=(time.perf_counter()-t)*1000
        grid=safety_planner.metric_to_grid(original) if len(original) else np.empty((0,2),int);inside=len(grid)>0 and np.all((grid[:,0]>=0)&(grid[:,0]<safe.shape[0])&(grid[:,1]>=0)&(grid[:,1]<safe.shape[1]));coverage=float(np.mean(safe[grid[:,0],grid[:,1]])) if inside else 0.;original_record={'frame_index':index,'planner':'librealsense_original','state':'VALID' if coverage==1 else 'INVALID','mode':'REFERENCE','valid':coverage==1,'path_point_count':len(original),'path_length_m':float(np.linalg.norm(np.diff(original,axis=0),axis=1).sum()) if len(original)>1 else 0.,'minimum_clearance_m':0.,'safe_coverage':coverage,'required_steering_deg':0.,'near_path_y_m':float(np.interp(cfg.near_required_m,original[:,0],original[:,1])) if len(original) else float('nan'),'fitting_residual':0.,'failure_reason':'' if coverage==1 else ('LANE_POINTS_LOW' if not len(original) else 'OUTSIDE_SAFE_ROAD'),'processing_ms':original_ms,'road_pixels':int(road.sum()),'lane_pixels':int(lane.sum())};rows.append(original_record)
        rgb=frame.copy();overlay=np.zeros_like(rgb);overlay[ri>0]=(0,150,0);overlay[li>0]=(0,220,255);rgb=cv2.addWeighted(rgb,.7,overlay,.3,0);rgb=cv2.resize(rgb,(320,480));cv2.putText(rgb,f'frame={index} RGB+mask',(7,20),0,.45,(255,255,255),1)
        oldp,oldr,oldrow=planner_results['old'];newp,newr,newrow=planner_results['new']
        panels=[rgb,bev_panel(oldr.component,oldr.safe_road,oldr.points,oldp,'old',oldr.state,oldrow['failure_reason'],oldrow['required_steering_deg'],oldrow['safe_coverage'],oldrow['processing_ms']),bev_panel(component,safe,original,safety_planner,'librealsense original',original_record['state'],original_record['failure_reason'],original_record['required_steering_deg'],original_record['safe_coverage'],original_record['processing_ms']),bev_panel(newr.component,newr.safe_road,newr.points,newp,'new metric-adapted',newr.state,newrow['failure_reason'],newrow['required_steering_deg'],newrow['safe_coverage'],newrow['processing_ms'])]
        combined=np.hstack(panels);writer.write(combined);new_writer.write(np.hstack((rgb,panels[-1])))
        if total<=10 or position % max(1,total//10)==0:cv2.imwrite(str(a.output/f'representative_{index:06d}.png'),combined)
        if (position+1)%300==0 or position+1==total:
            print(f"processed {position+1}/{total}",flush=True)
    cap.release();writer.release();new_writer.release()
    with (a.output/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={}
    for name in sorted({r['planner'] for r in rows}):
        selected=[r for r in rows if r['planner']==name]
        steer=np.asarray([r['required_steering_deg'] for r in selected],float)
        lateral=np.asarray([r['near_path_y_m'] for r in selected],float)
        finite=np.isfinite(lateral)
        lateral_delta=np.abs(np.diff(lateral[finite])) if finite.sum()>1 else np.array([])
        nonzero=np.sign(steer[np.abs(steer)>.25])
        state_counts={state:sum(r['state']==state for r in selected)
                      for state in ('VALID','DEGRADED','INVALID')}
        summary[name]={'frames':len(selected),
                       'state_counts':state_counts,
                       'drivable':sum(bool(r['valid']) for r in selected),
                       'short_paths_lt_2m':sum(0<r['path_length_m']<2. for r in selected),
                       'mean_path_length_m':float(np.mean([r['path_length_m'] for r in selected])),
                       'safe_or_clearance_violations':sum(r['safe_coverage']<.999 and r['path_point_count']>0 for r in selected),
                       'steering_limit_violations':int(np.count_nonzero(np.abs(steer)>cfg.maximum_steering_deg+1e-6)),
                       'steering_sign_reversals':int(np.count_nonzero(nonzero[1:]!=nonzero[:-1])) if len(nonzero)>1 else 0,
                       'near_path_jumps_over_gate':int(np.count_nonzero(lateral_delta>cfg.temporal_lateral_gate_m)),
                       'mean_abs_steering_deg':float(np.mean(np.abs(steer))),
                       'mean_processing_ms':float(np.mean([r['processing_ms'] for r in selected]))}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
