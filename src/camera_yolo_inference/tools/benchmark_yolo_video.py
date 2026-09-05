#!/usr/bin/env python3
"""Full-video PT-model benchmark with fixed-frame visual review evidence.

The tool deliberately evaluates PyTorch checkpoints directly.  It writes one
row per source frame/model, a compact summary, a 500-frame side-by-side review
video, and (when requested) chunked metric-BEV masks for deterministic planner
ablations.  No model or production configuration is modified.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_KEYS = ("road", "W_line", "Y_line")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_mask(mask, shape):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def frame_metrics(result, names, shape, inference_ms):
    detected = {name: 0 for name in CLASS_KEYS}
    confidences = {name: [] for name in CLASS_KEYS}
    road = np.zeros(shape, np.uint8)
    lane = np.zeros(shape, np.uint8)
    boxes = result.boxes
    masks = (result.masks.data.cpu().numpy()
             if result.masks is not None else np.empty((0, *shape), np.float32))
    if boxes is not None:
        for index, (class_id, confidence) in enumerate(zip(
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy().astype(float))):
            name = names[class_id]
            if name in detected:
                detected[name] = 1
                confidences[name].append(float(confidence))
            if index < len(masks) and name in CLASS_KEYS:
                binary = resize_mask(masks[index], shape) >= 0.5
                if name == "road":
                    road[binary] = 1
                elif name in ("W_line", "Y_line"):
                    lane[binary] = 1
    row = {f"{name}_detected": detected[name] for name in CLASS_KEYS}
    for name in CLASS_KEYS:
        values = confidences[name]
        row[f"{name}_confidence_mean"] = (
            float(statistics.fmean(values)) if values else 0.0)
        row[f"{name}_confidence_max"] = max(values) if values else 0.0
    row.update({"road_mask_pixels": int(road.sum()),
                "lane_mask_pixels": int(lane.sum()),
                "inference_ms": float(inference_ms)})
    return row, road, lane


def overlay(frame, road, lane, label, row):
    output = frame.copy()
    tint = np.zeros_like(output)
    tint[road > 0] = (20, 170, 20)
    tint[lane > 0] = (0, 220, 255)
    output = cv2.addWeighted(output, 0.68, tint, 0.32, 0.0)
    cv2.rectangle(output, (0, 0), (output.shape[1], 55), (0, 0, 0), -1)
    text = (f"{label} road={row['road_detected']} "
            f"W={row['W_line_detected']} Y={row['Y_line_detected']} "
            f"road_px={row['road_mask_pixels']}")
    cv2.putText(output, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"inference={row['inference_ms']:.1f} ms",
                (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def summarize(rows):
    summary = {}
    for label in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == label]
        item = {"frames": len(selected)}
        for name in CLASS_KEYS:
            item[f"{name}_detection_frames"] = sum(
                int(row[f"{name}_detected"]) for row in selected)
            values = [float(row[f"{name}_confidence_mean"])
                      for row in selected if int(row[f"{name}_detected"])]
            item[f"{name}_confidence_mean_detected"] = (
                float(statistics.fmean(values)) if values else 0.0)
        road_pixels = [int(row["road_mask_pixels"]) for row in selected]
        timing = [float(row["inference_ms"]) for row in selected]
        item["road_mask_pixels"] = {
            "mean": float(statistics.fmean(road_pixels)),
            "median": float(statistics.median(road_pixels)),
            "max": max(road_pixels),
        }
        item["inference_ms"] = {
            "mean": float(statistics.fmean(timing)),
            "p95": float(np.percentile(timing, 95)),
        }
        summary[label] = item
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2,
                        metavar=("LABEL", "PT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--review-frames", type=int, default=500)
    parser.add_argument("--cache-bev-label", default="")
    parser.add_argument("--cache-chunk", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    review_indices = set(np.linspace(
        0, frame_count - 1, min(args.review_frames, frame_count),
        dtype=int).tolist())

    inventory = []
    all_rows = []
    review = {}
    for label, text_path in args.model:
        path = Path(text_path).resolve()
        model = YOLO(str(path), task="segment")
        inventory.append({"label": label, "path": str(path),
                          "filename": path.name, "size": path.stat().st_size,
                          "mtime": path.stat().st_mtime,
                          "sha256": sha256(path), "task": model.task,
                          "class_names": model.names})
        capture = cv2.VideoCapture(str(args.video))
        index = 0
        cache_road, cache_lane, cache_start = [], [], 0
        while index < frame_count:
            frames = []
            for _ in range(min(args.batch, frame_count - index)):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            started = time.perf_counter()
            results = model.predict(frames, imgsz=640, conf=args.confidence,
                                    device=args.device, batch=args.batch,
                                    verbose=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(frames)
            for offset, (frame, result) in enumerate(zip(frames, results)):
                frame_index = index + offset
                metrics, road, lane = frame_metrics(
                    result, model.names, frame.shape[:2], elapsed_ms)
                row = {"frame_index": frame_index,
                       "timestamp_sec": frame_index / max(1.0, fps),
                       "model": label, **metrics}
                all_rows.append(row)
                if frame_index in review_indices:
                    review[(label, frame_index)] = overlay(
                        frame, road, lane, label, row)
                if label == args.cache_bev_label:
                    # Imported lazily so perception-only use has no navigation
                    # package dependency.
                    from camera_navigation.direct_bev_projection import project_mask_to_bev
                    cache_road.append(project_mask_to_bev(road, map_x, map_y))
                    cache_lane.append(project_mask_to_bev(lane, map_x, map_y))
                    if len(cache_road) >= args.cache_chunk:
                        target = args.output / "bev_mask_cache" / f"chunk_{cache_start:06d}.npz"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target, road=np.asarray(cache_road, np.uint8),
                                            lane=np.asarray(cache_lane, np.uint8))
                        cache_start += len(cache_road)
                        cache_road, cache_lane = [], []
            index += len(frames)
            if index % 1024 < len(frames) or index == frame_count:
                print(f"{label}: {index}/{frame_count}", flush=True)
        capture.release()
        if label == args.cache_bev_label and cache_road:
            target = args.output / "bev_mask_cache" / f"chunk_{cache_start:06d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, road=np.asarray(cache_road, np.uint8),
                                lane=np.asarray(cache_lane, np.uint8))

    fields = list(all_rows[0])
    with (args.output / "full_frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "model_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summarize(all_rows), indent=2) + "\n")
    settings = {"video": str(args.video.resolve()), "frames": frame_count,
                "fps": fps, "width": width, "height": height,
                "confidence_threshold": args.confidence,
                "mask_threshold": 0.5, "backend": "PyTorch .pt",
                "device": args.device, "batch": args.batch,
                "review_frames": len(review_indices),
                "review_indices": sorted(review_indices),
                "bev_cache_model": args.cache_bev_label}
    (args.output / "run_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n")

    labels = [label for label, _ in args.model]
    if len(labels) >= 2:
        writer = cv2.VideoWriter(
            str(args.output / "review_500_old_vs_new.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (2 * width, height))
        for frame_index in sorted(review_indices):
            panels = [review[(label, frame_index)] for label in labels[:2]]
            combined = np.hstack(panels)
            cv2.putText(combined, f"frame={frame_index}", (width - 145, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            writer.write(combined)
        writer.release()


if __name__ == "__main__":
    # Build the commissioned metric-BEV remap only when mask caching is used.
    import sys
    if "--cache-bev-label" in sys.argv:
        from camera_navigation.direct_bev_core import DirectBevConfig
        from camera_navigation.direct_bev_projection import CameraModel, build_ground_remap
        from camera_navigation.ground_plane_calibration import rotation_matrix_rpy
        config = DirectBevConfig()
        camera = CameraModel(640, 480, np.array(
            [[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]), np.zeros(5))
        map_x, map_y = build_ground_remap(
            config, camera, rotation_matrix_rpy(0., -5., 0.),
            np.array([.32, 0., .85]))
    main()
