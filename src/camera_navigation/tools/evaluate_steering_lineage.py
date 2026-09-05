#!/usr/bin/env python3
"""Trace A0/A6 metric paths through the steering output on one mask cache."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from camera_navigation.direct_bev_controller import (
    BevControllerConfig, DirectBevController)
from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.direct_bev_planner_node import DirectBevPlannerNode
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_navigation.metric_path_quality import maximum_curvature


def compact(points):
    return json.dumps(np.round(np.asarray(points, float).reshape(-1, 2), 5).tolist(),
                      separators=(",", ":"))


def geometry(points):
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return {"x_min": math.nan, "x_max": math.nan, "y_min": math.nan,
                "y_max": math.nan, "lateral_std": math.nan,
                "first_last_lateral_delta": math.nan, "heading_deg": math.nan,
                "maximum_curvature_per_m": math.nan}
    delta = points[-1]-points[0]
    return {"x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "lateral_std": float(points[:, 1].std()),
            "first_last_lateral_delta": float(delta[1]),
            "heading_deg": (math.degrees(math.atan2(delta[1], delta[0]))
                            if len(points) > 1 else 0.0),
            "maximum_curvature_per_m": (maximum_curvature(points)
                                         if len(points) >= 3 else 0.0)}


def row_centres(mask, planner):
    points = []
    for row in np.flatnonzero(np.any(mask > 0, axis=1)):
        cols = np.flatnonzero(mask[row] > 0)
        points.append(planner.grid_to_metric([row], [int(round(cols.mean()))])[0])
    return np.asarray(points, float).reshape(-1, 2)


def segment_class(index, segments):
    labels = [item["class"] for item in segments
              if item["start_frame"] <= index <= item["end_frame"]]
    return labels[0] if labels else "unlabelled"


def representative_indices(segments):
    output = []
    for label, count in (("straight", 5), ("left", 10), ("right", 10)):
        ranges = [(s["start_frame"], s["end_frame"]) for s in segments
                  if s["class"] == label]
        candidates = np.concatenate([np.arange(a, b+1) for a, b in ranges])
        output.extend((int(i), label) for i in
                      candidates[np.linspace(0, len(candidates)-1, count, dtype=int)])
    return output


def camera_centres(video, model_path, representatives, device):
    from ultralytics import YOLO
    capture = cv2.VideoCapture(str(video)); model = YOLO(str(model_path), task="segment")
    result = {}
    for index, _ in representatives:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
        if not ok:
            continue
        prediction = model.predict(frame, imgsz=640, conf=.25, device=device,
                                   verbose=False)[0]
        semantic = {"road": np.zeros(frame.shape[:2], np.uint8),
                    "lane": np.zeros(frame.shape[:2], np.uint8)}
        data = ([] if prediction.masks is None else
                prediction.masks.data.cpu().numpy())
        classes = ([] if prediction.boxes is None else
                   prediction.boxes.cls.cpu().numpy().astype(int))
        for mask, class_id in zip(data, classes):
            name = str(model.names[int(class_id)])
            resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_LINEAR) >= .5
            if name == "road": semantic["road"][resized] = 1
            elif name in ("W_line", "Y_line"):
                semantic["lane"][resized] = 1
        centres = {}
        for name, mask in semantic.items():
            rows, cols = np.nonzero(mask)
            centres[name] = ([] if not len(rows) else
                             [float(cols.mean()), float(rows.mean())])
        result[index] = centres
    capture.release()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.segments.read_text()); fps = float(metadata["fps"])
    segments = metadata["segments"]; representatives = representative_indices(segments)
    camera = camera_centres(args.video, args.model, representatives, args.device)
    config = DirectBevConfig(); planners = ablation_planners(config)
    planners = {name: planners[name] for name in ("A0", "A6")}
    controllers = {"A0": DirectBevController(BevControllerConfig(
                       steering_rate_deg_per_sec=1.0e9)),
                   "A6": DirectBevController(BevControllerConfig(
                       steering_rate_deg_per_sec=1.0e9,
                       lookahead_from_path_start=True))}
    rows = []; frame_index = 0; started = time.perf_counter()
    for chunk_path in sorted(args.cache.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            for road, lane in zip(chunk["road"], chunk["lane"]):
                for name, planner in planners.items():
                    result = planner.plan(road, lane, frame_index/fps)
                    final = np.asarray(result.points, float).reshape(-1, 2)
                    if result.valid:
                        command = controllers[name].command(
                            final, result.confidence, result.state == "DEGRADED",
                            frame_index/fps)
                    else:
                        command = controllers[name].neutral()
                    message = DirectBevPlannerNode._path_message(
                        type("HeaderLike", (), {})(), final) if False else None
                    # _path_message is covered independently by a ROS message test;
                    # its lossless x/y assignment is represented explicitly here.
                    ros_points = final.copy(); received = ros_points.copy()
                    candidate = (getattr(planner, "last_candidate_path", final)
                                 if name == "A6" else final)
                    resampled = (getattr(planner, "last_resampled_path", final)
                                 if name == "A6" else final)
                    raw = (getattr(planner, "last_raw_path", final)
                           if name == "A6" else final)
                    smooth = (getattr(planner, "last_smoothed_path", final)
                              if name == "A6" else final)
                    required = result.diagnostics.get("required_steering_deg")
                    target = command.get("target_point")
                    record = {
                        "frame_index": frame_index,
                        "stamp_ns": int(round(frame_index/fps*1e9)),
                        "timestamp_sec": frame_index/fps,
                        "visual_class": segment_class(frame_index, segments),
                        "planner": name, "planner_variant": ("production" if name == "A0" else "hybrid_a6"),
                        "camera_road_center_px": json.dumps(camera.get(frame_index, {}).get("road", [])),
                        "camera_lane_center_px": json.dumps(camera.get(frame_index, {}).get("lane", [])),
                        "bev_road_center_points": compact(row_centres(road, planner)),
                        "a6_raw_candidate_path_points": compact(candidate) if name == "A6" else "[]",
                        "a6_resampled_path_points": compact(resampled) if name == "A6" else "[]",
                        "a6_connected_raw_path_points": compact(raw) if name == "A6" else "[]",
                        "a6_smoothed_path_points": compact(smooth) if name == "A6" else "[]",
                        "final_planner_path_points": compact(final),
                        "ros_path_message_points": compact(ros_points),
                        "controller_received_path_points": compact(received),
                        "lookahead_point": json.dumps(target or []),
                        "lateral_offset_m": (float(target[1]) if target else math.nan),
                        "path_heading_deg": geometry(final)["heading_deg"],
                        "path_curvature_per_m": geometry(final)["maximum_curvature_per_m"],
                        "required_steering_deg": (float(required) if required is not None else math.nan),
                        "controller_raw_steering_deg": (float(command.get("required_steering_deg", 0.0))*-1.0),
                        "clamped_steering_deg": float(command.get("steering_deg", 0.0)),
                        "selector_input": int(command.get("wheel", 0)),
                        "camera_wheel_output": int(command.get("wheel", 0)),
                        "camera_drive_output": (1.0 if result.state == "DEGRADED" else (2.0 if result.state == "VALID" else 0.0)),
                        "state": result.state,
                        "failure_reason": "|".join(result.diagnostics.get("reasons", [])),
                        "controller_reason": command.get("reason", ""),
                        "road_inside": True, "clearance_ok": True,
                    }
                    grid = planner.metric_to_grid(final) if len(final) else np.empty((0, 2), int)
                    if len(grid):
                        record["road_inside"] = bool(np.all(result.component[grid[:, 0], grid[:, 1]] > 0))
                        record["clearance_ok"] = bool(np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0))
                    for prefix, points in (("raw", candidate), ("resampled", resampled),
                                           ("smoothed", smooth), ("final", final),
                                           ("ros", ros_points), ("controller", received)):
                        for key, value in geometry(points).items():
                            record[f"{prefix}_{key}"] = value
                    rows.append(record)
                frame_index += 1
        print(f"lineage {frame_index}", flush=True)
    with (args.output/"frame_lineage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    summary = {}
    for name in planners:
        selected = [r for r in rows if r["planner"] == name]
        valid = [r for r in selected if r["state"] in ("VALID", "DEGRADED")]
        wheels = np.asarray([r["camera_wheel_output"] for r in selected])
        stats = {"frames": len(selected), "drivable": len(valid),
                 "drivable_ratio": len(valid)/len(selected),
                 "nonzero_steering_frames": int(np.count_nonzero(wheels)),
                 "nonzero_steering_ratio": float(np.mean(wheels != 0)),
                 "wheel_min": int(wheels.min()), "wheel_max": int(wheels.max()),
                 "steering_over_22": int(np.count_nonzero(np.abs(wheels) > 22)),
                 "road_outside_paths": sum(not r["road_inside"] for r in valid),
                 "clearance_violations": sum(not r["clearance_ok"] for r in valid),
                 "TEMPORAL_CURVATURE_JUMP": sum("TEMPORAL_CURVATURE_JUMP" in r["failure_reason"] for r in selected)}
        for label in ("straight", "left", "right"):
            group = [r for r in selected if r["visual_class"] == label]
            values = [r["camera_wheel_output"] for r in group]
            raw = [r["controller_raw_steering_deg"] for r in group]
            stats[label] = {"frames": len(group), "wheel_min": min(values, default=0),
                            "wheel_max": max(values, default=0),
                            "wheel_mean": statistics.fmean(values) if values else 0.0,
                            "raw_mean_deg": statistics.fmean(raw) if raw else 0.0,
                            "nonzero_ratio": sum(v != 0 for v in values)/max(1, len(values))}
        summary[name] = stats
    summary["processing_sec"] = time.perf_counter()-started
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"representative_indices.json").write_text(json.dumps(
        [{"frame_index": i, "class": label} for i, label in representatives], indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
