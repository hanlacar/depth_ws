#!/usr/bin/env python3
"""Render two threshold overlays from one low-threshold YOLO pass."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COLORS = {"road": (20, 180, 20), "W_line": (255, 255, 255),
          "Y_line": (0, 220, 255), "stop": (0, 0, 255)}


def render(frame, result, names, threshold, frame_index, backend):
    output = frame.copy(); instances = []; road = np.zeros(frame.shape[:2], bool)
    if result.boxes is not None:
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        for i, conf in enumerate(result.boxes.conf.cpu().numpy()):
            if float(conf) < threshold: continue
            cls = int(result.boxes.cls[i]); name = names[cls]
            mask = (cv2.resize(masks[i], (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_LINEAR) >= .5) if i < len(masks) else None
            if mask is not None:
                color = np.asarray(COLORS.get(name, (255, 100, 200)), np.uint8)
                output[mask] = (output[mask].astype(np.uint16)*65//100 + color*35//100).astype(np.uint8)
                if name == "road": road |= mask
            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), COLORS.get(name, (255, 100, 200)), 1)
            cv2.putText(output, f"{name} {float(conf):.2f}", (box[0], max(62, box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, COLORS.get(name, (255, 100, 200)), 1)
            instances.append((name, float(conf)))
    cv2.rectangle(output, (0, 0), (640, 47), (0, 0, 0), -1)
    cv2.putText(output, f"frame={frame_index} backend={backend} conf={threshold:.2f}",
                (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    cv2.putText(output, f"instances={len(instances)} road_pixels={int(road.sum())}",
                (7, 39), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    return output


def main():
    p = argparse.ArgumentParser(); p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60); p.add_argument("--output-fps", type=float, default=15)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video)); source_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(source_fps/args.output_fps)); wanted = round(args.seconds*args.output_fps)
    frames=[]; indices=[]
    for index in range(0, round(args.seconds*source_fps), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=cap.read()
        if not ok: break
        frames.append(frame);indices.append(index)
    cap.release(); frames=frames[:wanted];indices=indices[:wanted]
    model=YOLO(str(args.model),task="segment")
    writers={}
    names={.25:"perception_pt_conf025.mp4",.10:"perception_pt_best_tested_conf.mp4"}
    for threshold,name in names.items():
        writers[threshold]=cv2.VideoWriter(str(args.output/name),cv2.VideoWriter_fourcc(*"mp4v"),
                                           args.output_fps,(640,480))
    representative={}
    for start in range(0,len(frames),32):
        batch=frames[start:start+32]
        results=model.predict(batch,imgsz=640,conf=.10,device="cpu",verbose=False)
        for offset,(frame,result) in enumerate(zip(batch,results)):
            index=indices[start+offset]
            for threshold,writer in writers.items():
                image=render(frame,result,model.names,threshold,index,"pytorch")
                writer.write(image)
                if start+offset==len(frames)//2: representative[threshold]=image
    for threshold,writer in writers.items():
        writer.release(); cv2.imwrite(str(args.output/f"representative_pt_conf{int(threshold*100):03d}.png"),representative[threshold])


if __name__ == "__main__": main()
