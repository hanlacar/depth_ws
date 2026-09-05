#!/usr/bin/env python3
"""Benchmark and compare PyTorch CUDA and TensorRT FP16 on one real frame."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from camera_yolo_inference.inference_backend import create_inference_backend
from camera_yolo_inference.mask_postprocessor import restore_masks_to_raw_shape


def stats(values):
    a=np.asarray(values,float);return {"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),"p95_ms":float(np.percentile(a,95)),"max_ms":float(a.max()),"fps_equivalent":1000./float(a.mean())}


def run(backend,image,warmup,count):
    backend.load_model();backend.warmup(warmup);values=[];last=None
    for _ in range(count):
        started=time.perf_counter();last=backend.infer(image);values.append((time.perf_counter()-started)*1000.)
    return last,stats(values)


def compare(left,right):
    def grouped(items):
        result={}
        for item in items:result.setdefault(int(item["class_id"]),[]).append(item)
        return result
    a,b=grouped(left),grouped(right);per_class={}
    for class_id in sorted(set(a)|set(b)):
        aa,bb=a.get(class_id,[]),b.get(class_id,[])
        masks=[]
        for group in (aa,bb):
            merged=np.zeros((480,640),bool)
            for item in group:merged|=restore_masks_to_raw_shape(item["mask"],(480,640))>=.5
            masks.append(merged)
        union=np.logical_or(*masks).sum();intersection=np.logical_and(*masks).sum()
        per_class[str(class_id)]={"pytorch_count":len(aa),"tensorrt_count":len(bb),
            "mask_iou":None if union==0 else float(intersection/union),
            "pytorch_confidence":[round(float(x["confidence"]),4) for x in aa],
            "tensorrt_confidence":[round(float(x["confidence"]),4) for x in bb]}
    return {"object_count_match":len(left)==len(right),"per_class":per_class}


def main():
    p=argparse.ArgumentParser();p.add_argument("--pt",type=Path,required=True);p.add_argument("--engine",type=Path,required=True);p.add_argument("--video",type=Path,required=True);p.add_argument("--frame",type=int,default=0);p.add_argument("--warmup",type=int,default=10);p.add_argument("--count",type=int,default=100);a=p.parse_args()
    cap=cv2.VideoCapture(str(a.video));cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,image=cap.read();cap.release()
    if not ok:raise RuntimeError("cannot read benchmark video")
    image=cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA)
    pt=create_inference_backend("pytorch",a.pt);engine=create_inference_backend("tensorrt",a.engine)
    pt_results,pt_stats=run(pt,image,a.warmup,a.count);trt_results,trt_stats=run(engine,image,a.warmup,a.count)
    print(json.dumps({"pytorch":pt_stats,"tensorrt":trt_stats,"comparison":compare(pt_results,trt_results)},indent=2))


if __name__=="__main__":main()
