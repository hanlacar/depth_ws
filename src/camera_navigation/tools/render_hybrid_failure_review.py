#!/usr/bin/env python3
"""Render 20 temporal-failure sections as RGB/mask + A0/A1/A6 panels."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from camera_navigation.direct_bev_core import DirectBevConfig, DirectBevPlanner


def load_cache(cache):
    output = {}
    for path in sorted(cache.glob("chunk_*.npz")):
        start = int(path.stem.split("_")[-1])
        with np.load(path) as chunk:
            for offset, (road, lane) in enumerate(zip(chunk["road"], chunk["lane"])):
                output[start+offset] = (road.copy(), lane.copy())
    return output


def load_rows(path):
    with path.open() as stream:
        return {(int(row["frame_index"]), row["planner"]): row
                for row in csv.DictReader(stream)}


def camera_masks(result, names, shape):
    road = np.zeros(shape, np.uint8)
    lane = np.zeros(shape, np.uint8)
    if result.masks is None:
        return road, lane
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for mask, class_id in zip(masks, classes):
        name = names[class_id]
        resized = cv2.resize(mask, (shape[1], shape[0]),
                             interpolation=cv2.INTER_LINEAR) >= .5
        if name == "road":
            road[resized] = 1
        elif name in ("W_line", "Y_line"):
            lane[resized] = 1
    return road, lane


def rgb_panel(frame, road, lane, index):
    tint = np.zeros_like(frame)
    tint[road > 0] = (20, 170, 20)
    tint[lane > 0] = (0, 220, 255)
    output = cv2.addWeighted(frame, .68, tint, .32, 0.)
    output = cv2.resize(output, (320, 480))
    cv2.rectangle(output, (0, 0), (320, 50), (0, 0, 0), -1)
    cv2.putText(output, f"frame={index} RGB + semantic", (6, 20), 0,
                .46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"road_px={int(road.sum())} lane_px={int(lane.sum())}",
                (6, 41), 0, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def bev_panel(planner, road, lane, row, title):
    _, _, component, safe, _ = planner.preprocess(road, lane)
    canvas = np.dstack((component*45, safe*125, component*75)).astype(np.uint8)
    current = np.asarray(json.loads(row["path"]), float).reshape(-1, 2)
    raw = np.asarray(json.loads(row["raw_path"]), float).reshape(-1, 2)
    previous = np.asarray(json.loads(row["previous_accepted_path"]),
                          float).reshape(-1, 2)
    for points, color, radius in ((previous, (255, 180, 0), 1),
                                  (raw, (255, 0, 255), 1),
                                  (current, (0, 255, 255), 2)):
        for point in points:
            rr, cc = planner.metric_to_grid([point])[0]
            if 0 <= rr < planner.rows and 0 <= cc < planner.cols:
                cv2.circle(canvas, (int(cc), int(rr)), radius, color, -1)
    output = cv2.resize(canvas, (320, 480), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(output, (0, 0), (320, 118), (0, 0, 0), -1)
    lines = [
        f"{title} {row['state']} {row['attempted_mode']}",
        f"curv={float(row['curvature_per_m']):.2f} d={float(row['curvature_delta_per_m']):.2f}",
        f"steer={float(row['required_steering_deg']):+.2f} len={float(row['path_length_m']):.2f}",
        row["failure_reason"][:39],
        "yellow=path cyan=previous magenta=raw",
    ]
    for line_index, text in enumerate(lines):
        cv2.putText(output, text, (6, 19+line_index*22), 0, .38,
                    (255, 255, 255) if line_index != 3 else (0, 180, 255),
                    1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.ablation/"temporal_failure_summary.json").read_text())
    representatives = report["representative_frames"]
    rows = load_rows(args.ablation/"temporal_failure_context.csv")
    available = {index for index, _ in rows}
    indices = sorted({nearby for center in representatives
                      for nearby in range(max(0, center-5), center+4)
                      if nearby in available})
    cache = load_cache(args.cache)

    capture = cv2.VideoCapture(str(args.video))
    wanted = set(indices)
    frames = {}
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.remove(index)
        index += 1
    capture.release()
    model = YOLO(str(args.model), task="segment")
    ordered_frames = [frames[index] for index in indices]
    predictions = model.predict(ordered_frames, imgsz=640, conf=.25,
                                device="cpu", batch=16, verbose=False)
    perception = {}
    for index, frame, result in zip(indices, ordered_frames, predictions):
        perception[index] = camera_masks(result, model.names, frame.shape[:2])

    planner = DirectBevPlanner(DirectBevConfig())
    writer = cv2.VideoWriter(str(args.output/"temporal_failure_20_sections.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 10., (1280, 480))
    center_images = []
    for index in indices:
        camera_road, camera_lane = perception[index]
        bev_road, bev_lane = cache[index]
        panels = [rgb_panel(frames[index], camera_road, camera_lane, index)]
        for name, title in (("A0", "A0 production"),
                            ("A1", "A1 spatial"),
                            ("A6", "A6 hybrid")):
            panels.append(bev_panel(planner, bev_road, bev_lane,
                                    rows[(index, name)], title))
        combined = np.hstack(panels)
        writer.write(combined)
        if index in representatives:
            target = args.output/f"representative_{index:06d}.png"
            cv2.imwrite(str(target), combined)
            center_images.append(cv2.resize(combined, (640, 240)))
    writer.release()
    if center_images:
        sheets = []
        for start in range(0, len(center_images), 5):
            sheet = np.vstack(center_images[start:start+5])
            target = args.output/f"contact_sheet_{start//5:02d}.png"
            cv2.imwrite(str(target), sheet)
            sheets.append(str(target))
    (args.output/"review_manifest.json").write_text(json.dumps({
        "representative_frames": representatives,
        "context_indices": indices,
        "video": "temporal_failure_20_sections.mp4",
        "legend": {"yellow": "accepted/output path", "cyan": "previous path",
                   "magenta": "raw candidate"}}, indent=2)+"\n")


if __name__ == "__main__":
    main()
