#!/usr/bin/env python3
"""Reproducible fixed-frame YOLO segmentation root-cause benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from ultralytics import YOLO


THRESHOLDS = (0.25, 0.10, 0.01)
COLORS = {"road": (30, 180, 30), "W_line": (255, 255, 255),
          "Y_line": (0, 220, 255), "stop": (0, 0, 255)}


def frame_at(capture, index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index}")
    return frame


def stats(image):
    pixels = image.reshape(-1, image.shape[-1]).astype(np.float64)
    return {"shape": list(image.shape), "dtype": str(image.dtype),
            "mean_bgr": pixels.mean(axis=0).tolist(),
            "std_bgr": pixels.std(axis=0).tolist()}


def resize_mask(mask, shape):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def result_rows(result, names, frame_index, backend, threshold, shape):
    rows = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        for i, (cls, conf, box) in enumerate(zip(boxes.cls.cpu().numpy(),
                                                 boxes.conf.cpu().numpy(),
                                                 boxes.xyxy.cpu().numpy())):
            mask = resize_mask(masks[i], shape) if len(masks) else np.zeros(shape, np.float32)
            rows.append({"frame_index": frame_index, "backend": backend,
                         "threshold": threshold, "class_id": int(cls),
                         "class_name": names[int(cls)], "confidence": float(conf),
                         "mask_pixels": int(np.count_nonzero(mask >= 0.5)),
                         "box": " ".join(f"{float(v):.3f}" for v in box)})
    if not rows:
        rows.append({"frame_index": frame_index, "backend": backend,
                     "threshold": threshold, "class_id": -1,
                     "class_name": "NONE", "confidence": 0.0,
                     "mask_pixels": 0, "box": ""})
    return rows


def aggregate(rows):
    output = {}
    for backend in sorted({r["backend"] for r in rows}):
        for threshold in THRESHOLDS:
            selected = [r for r in rows if r["backend"] == backend and
                        abs(r["threshold"] - threshold) < 1e-9]
            real = [r for r in selected if r["class_id"] >= 0]
            classes = {}
            for name in sorted({r["class_name"] for r in real}):
                vals = [r for r in real if r["class_name"] == name]
                classes[name] = {"instances": len(vals),
                                 "frames": len({r["frame_index"] for r in vals}),
                                 "max_confidence": max(r["confidence"] for r in vals),
                                 "mean_confidence": statistics.fmean(r["confidence"] for r in vals)}
            road_pixels = [sum(r["mask_pixels"] for r in real
                               if r["frame_index"] == idx and r["class_name"] == "road")
                           for idx in sorted({r["frame_index"] for r in selected})]
            output[f"{backend}@{threshold:.2f}"] = {
                "road_detection_frames": len({r["frame_index"] for r in real if r["class_name"] == "road"}),
                "W_line_detection_frames": len({r["frame_index"] for r in real if r["class_name"] == "W_line"}),
                "Y_line_detection_frames": len({r["frame_index"] for r in real if r["class_name"] == "Y_line"}),
                "total_instances": len(real), "classes": classes,
                "road_mask_pixels": {"min": min(road_pixels), "median": statistics.median(road_pixels),
                                     "max": max(road_pixels), "mean": statistics.fmean(road_pixels)}}
    return output


def draw(frame, result, names, threshold, label):
    out = frame.copy()
    road_pixels = 0
    count = 0
    if result.boxes is not None and len(result.boxes):
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        for i, (cls, conf) in enumerate(zip(result.boxes.cls.cpu().numpy(),
                                            result.boxes.conf.cpu().numpy())):
            name = names[int(cls)]
            if i < len(masks):
                mask = resize_mask(masks[i], frame.shape[:2]) >= 0.5
                color = np.asarray(COLORS.get(name, (255, 100, 200)), np.uint8)
                out[mask] = (out[mask].astype(np.uint16) * 6 // 10 + color * 4 // 10).astype(np.uint8)
                if name == "road": road_pixels += int(mask.sum())
            count += 1
            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), COLORS.get(name, (255, 100, 200)), 1)
            cv2.putText(out, f"{name} {float(conf):.2f}", (box[0], max(45, box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, COLORS.get(name, (255, 100, 200)), 1)
    cv2.rectangle(out, (0, 0), (640, 38), (0, 0, 0), -1)
    cv2.putText(out, f"{label} conf={threshold:.2f} instances={count} road_pixels={road_pixels}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Some H.264 containers report undecodable delayed frames at the very end.
    # Excluding the final two seconds keeps seeking deterministic across FFmpeg builds.
    indices = np.linspace(0, max(0, count - 121), args.frames, dtype=int).tolist()
    frames = [(i, frame_at(cap, i)) for i in indices]
    cap.release()
    (args.output / "fixed_frame_indices.json").write_text(json.dumps(indices, indent=2) + "\n")

    input_rows = []
    montage_tiles = []
    for i, frame in frames:
        # Exact sensor_msgs/Image bgr8 memory contract: contiguous row-major bytes.
        ros_bytes = np.ascontiguousarray(frame).tobytes()
        ros = np.frombuffer(ros_bytes, np.uint8).reshape(frame.shape).copy()
        backend = ros.copy()
        row = {"frame_index": i, "A": stats(frame), "B": stats(ros), "C": stats(backend),
               "encoding": "bgr8", "A_B_MAE": float(np.abs(frame.astype(np.int16)-ros).mean()),
               "B_C_MAE": float(np.abs(ros.astype(np.int16)-backend).mean()),
               "A_B_swapped_MAE": float(np.abs(frame.astype(np.int16)-ros[..., ::-1].astype(np.int16)).mean()),
               "resize": False, "letterbox": False}
        input_rows.append(row)
        if len(montage_tiles) < 10:
            tile = np.hstack([frame, ros, backend])
            cv2.putText(tile, f"frame {i}: A direct | B ROS bgr8 | C backend", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            montage_tiles.append(cv2.resize(tile, (960, 240)))
    (args.output / "input_abc_metrics.json").write_text(json.dumps(input_rows, indent=2) + "\n")
    cv2.imwrite(str(args.output / "input_abc_montage.png"), np.vstack(montage_tiles))

    all_rows = []
    model_meta = []
    representative = []
    for label, path_text in args.model:
        path = Path(path_text).resolve()
        model = YOLO(str(path), task="segment")
        model_meta.append({"label": label, "path": str(path), "size": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "task": model.task, "class_names": model.names})
        for threshold in THRESHOLDS:
            results = model.predict([f for _, f in frames], imgsz=640, conf=threshold,
                                    device="cpu", verbose=False)
            for (idx, frame), result in zip(frames, results):
                all_rows.extend(result_rows(result, model.names, idx, label, threshold, frame.shape[:2]))
            if threshold == 0.25:
                representative.append(draw(frames[len(frames)//2][1], results[len(results)//2],
                                           model.names, threshold, label))
    with (args.output / "detections.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (args.output / "model_inventory.json").write_text(json.dumps(model_meta, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(aggregate(all_rows), indent=2) + "\n")
    cv2.imwrite(str(args.output / "representative_models.png"), np.vstack(representative))


if __name__ == "__main__":
    main()
