#!/usr/bin/env python3
"""Compare legacy CPU instance aggregation with GPU class aggregation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def mask_iou(left, right):
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=3000)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(args.frame))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {args.frame}")
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    result = YOLO(str(args.engine), task="segment").predict(
        frame, imgsz=640, conf=0.25, device="cuda:0", verbose=False)[0]
    masks = result.masks.data.detach()
    classes = result.boxes.cls.detach().to(dtype=torch.int64)
    comparison = {}
    for class_id in torch.unique(classes, sorted=True):
        selected = masks[classes == class_id]
        legacy = selected.cpu().numpy().max(axis=0) >= 0.5
        optimized = selected.amax(dim=0).cpu().numpy() >= 0.5
        comparison[int(class_id)] = {
            "instances": int(selected.shape[0]),
            "semantic_iou": mask_iou(legacy, optimized),
        }
    print(json.dumps({
        "frame": args.frame,
        "instance_masks_before": int(masks.shape[0]),
        "class_masks_after": len(comparison),
        "per_class": comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
