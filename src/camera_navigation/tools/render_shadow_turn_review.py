#!/usr/bin/env python3
"""Render A0/A6 regressions and A6 predicted swept paths on selected turns."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from camera_navigation.direct_bev_controller import BevControllerConfig, DirectBevController
from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_navigation.metric_path_quality import maximum_curvature


def arc(points_x, steering_deg, wheelbase):
    k = math.tan(math.radians(steering_deg))/wheelbase
    y=[]; heading=[]
    for x in points_x:
        value=1-(k*x)**2
        if value < 0: y.append(math.nan);heading.append(math.nan)
        elif abs(k)<1e-9:y.append(0.);heading.append(0.)
        else:y.append((1-math.sqrt(value))/k);heading.append(math.asin(k*x))
    return np.asarray(y),np.asarray(heading)


def draw_metric(canvas, planner, points, color, thickness=2):
    points=np.asarray(points,float).reshape(-1,2);points=points[np.isfinite(points).all(axis=1)]
    if not len(points):return
    grid=planner.metric_to_grid(points);keep=(grid[:,0]>=0)&(grid[:,0]<planner.rows)&(grid[:,1]>=0)&(grid[:,1]<planner.cols)
    pixels=np.asarray([(c,r) for r,c in grid[keep]],np.int32)
    if len(pixels)>1:cv2.polylines(canvas,[pixels],False,color,thickness,cv2.LINE_AA)


def panel(result, planner, path_color=(255,0,255)):
    image=np.zeros((*result.road.shape,3),np.uint8)
    image[result.component>0]=(35,85,35);image[result.safe_road>0]=(45,155,45)
    draw_metric(image,planner,result.points,path_color,2)
    draw_metric(image,planner,[[planner.config.x_min_m,0],[planner.config.x_max_m,0]],(180,180,180),1)
    return image


def put_lines(image, lines):
    for n,line in enumerate(lines):cv2.putText(image,str(line)[:57],(7,22+n*28),0,.43,(255,255,255),1,cv2.LINE_AA)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--segments',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    segments=json.loads(a.segments.read_text())['segments'];meta=json.loads(a.summary.read_text());reg=meta['special_frames']['regressed']
    # Merge flickering regressed frames separated by less than the requested 2 s context.
    macro=[]
    for index in reg:
        if not macro or index-macro[-1][-1]>240:macro.append([index])
        else:macro[-1].append(index)
    reps=[group[len(group)//2] for group in macro]
    contexts=set(i for rep in reps for i in range(max(0,rep-120),min(14409,rep+121)))
    turn_frames=set(i for s in segments if s['class'] in ('left','right') for i in range(s['start_frame'],s['end_frame']+1))
    cfg=DirectBevConfig();pls=ablation_planners(cfg);p0,p6=pls['A0'],pls['A6'];c0=DirectBevController();c6=DirectBevController(BevControllerConfig(lookahead_from_path_start=True))
    cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v');turn_writer=cv2.VideoWriter(str(a.output/'left_right_swept_overlay.mp4'),fourcc,30,(1280,480));reg_writer=cv2.VideoWriter(str(a.output/'a6_regressed_contexts.mp4'),fourcc,30,(1280,480))
    swept={'frames':0,'float_required_safe':0,'quantized_wheel_safe':0,'quantized_target_error':[]};idx=0
    for cp in sorted(a.cache.glob('chunk_*.npz')):
      with np.load(cp) as d:
       for road,lane in zip(d['road'],d['lane']):
        ok,rgb=cap.read()
        if not ok:break
        r0=p0.plan(road,lane,idx/fps);r6=p6.plan(road,lane,idx/fps)
        q0=c0.command(r0.points,r0.confidence,r0.state=='DEGRADED',idx/fps) if r0.valid else c0.neutral();q6=c6.command(r6.points,r6.confidence,r6.state=='DEGRADED',idx/fps) if r6.valid else c6.neutral()
        if idx in turn_frames and r6.valid:
            base=panel(r6,p6);target=q6.get('target_point');
            if target:draw_metric(base,p6,[target],(255,255,0),4)
            x=np.linspace(max(cfg.x_min_m,float(r6.points[0,0])),float(r6.points[-1,0]),80)
            required=float(q6.get('required_steering_deg',0));wheel=int(q6.get('wheel',0));yf,hf=arc(x,required,cfg.wheelbase_m);yq,hq=arc(x,-wheel,cfg.wheelbase_m)
            float_points=np.column_stack((x,yf));quant_points=np.column_stack((x,yq));draw_metric(base,p6,float_points,(255,255,0),2);draw_metric(base,p6,quant_points,(0,140,255),2)
            half=cfg.vehicle_width_m/2+cfg.lateral_safety_margin_m;left=np.column_stack((x,yq+half*np.cos(hq)));right=np.column_stack((x,yq-half*np.cos(hq)));draw_metric(base,p6,left,(0,0,255),1);draw_metric(base,p6,right,(0,0,255),1)
            def inside(points):
                points=points[np.isfinite(points).all(axis=1)];g=p6.metric_to_grid(points);good=(g[:,0]>=0)&(g[:,0]<p6.rows)&(g[:,1]>=0)&(g[:,1]<p6.cols);return bool(np.all(good) and np.all(r6.safe_road[g[:,0],g[:,1]]>0))
            swept['frames']+=1;swept['float_required_safe']+=inside(float_points);swept['quantized_wheel_safe']+=inside(np.vstack((quant_points,left,right)))
            if target:
                pred=np.interp(float(target[0]),x,yq);swept['quantized_target_error'].append(float(pred-float(target[1])))
            p_rgb=cv2.resize(rgb,(320,480));p_bev=cv2.resize(base,(320,480),interpolation=cv2.INTER_NEAREST);p_path=cv2.resize(panel(r6,p6),(320,480),interpolation=cv2.INTER_NEAREST);text=np.zeros((480,320,3),np.uint8)
            label=next((s['class'] for s in segments if s['start_frame']<=idx<=s['end_frame']),'turn')
            put_lines(text,[f'{label} frame={idx}',f'k_max={maximum_curvature(r6.points):+.3f}/m',f'required={required:+.3f} deg',f'wheel={wheel:+d} deg',f'cyan=float arc orange=Int32 arc','red=swept footprint bounds',f'float safe={inside(float_points)}',f'quantized swept safe={inside(np.vstack((quant_points,left,right)))}',f'state={r6.state}','reason='+'|'.join(r6.diagnostics.get('reasons',[]))])
            if idx%2==0:turn_writer.write(np.hstack((p_rgb,p_path,p_bev,text)))
        if idx in contexts:
            p_rgb=cv2.resize(rgb,(320,480));a0=cv2.resize(panel(r0,p0,(0,0,255)),(320,480),interpolation=cv2.INTER_NEAREST);a6=cv2.resize(panel(r6,p6),(320,480),interpolation=cv2.INTER_NEAREST);text=np.zeros((480,320,3),np.uint8);put_lines(text,[f'regressed context frame={idx}',f'A0 {r0.state}: '+('|'.join(r0.diagnostics.get('reasons',[])) or 'OK'),f'A6 {r6.state}: '+('|'.join(r6.diagnostics.get('reasons',[])) or 'OK'),f'A0 road safe={r0.diagnostics.get("safe_road_coverage",0):.3f}',f'A6 fail-closed={not r6.valid}'])
            if idx%2==0:reg_writer.write(np.hstack((p_rgb,a0,a6,text)))
        idx+=1
    cap.release();turn_writer.release();reg_writer.release();err=swept.pop('quantized_target_error');swept['quantized_target_error_m']={'mean':float(np.mean(err)) if err else 0.,'p95_abs':float(np.percentile(np.abs(err),95)) if err else 0.,'max_abs':max(map(abs,err),default=0.)};swept['regressed_macro_clusters']=len(macro);swept['regressed_representatives']=reps;(a.output/'swept_path_summary.json').write_text(json.dumps(swept,indent=2)+'\n')


if __name__=='__main__':main()
