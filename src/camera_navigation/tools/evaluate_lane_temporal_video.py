#!/usr/bin/env python3
"""One-inference-pass T0/T1/T2 + A0/A6 full-video evaluation."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from ultralytics import YOLO

from camera_navigation.direct_bev_controller import DirectBevController
from camera_navigation.direct_bev_core import DirectBevConfig, DirectBevPlanner
from camera_navigation.direct_bev_projection import (
    CameraModel, build_ground_remap, project_mask_to_bev)
from camera_navigation.ground_plane_calibration import rotation_matrix_rpy
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_yolo_inference.lane_temporal_tracker import (
    LaneMaskTemporalTracker, LaneTemporalConfig)


def resize_binary(mask, shape):
    resized = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return (resized >= 0.5).astype(np.uint8) * 255


def semantic_masks(result, names, shape):
    output = {name: np.zeros(shape, np.uint8)
              for name in ("road", "W_line", "Y_line")}
    masks = (result.masks.data.cpu().numpy() if result.masks is not None else [])
    if result.boxes is None:
        return output
    for index, class_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
        name = names[class_id]
        if name in output and index < len(masks):
            output[name] |= resize_binary(masks[index], shape)
    return output


def path_length(points):
    points = np.asarray(points, float).reshape(-1, 2)
    return (float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if len(points) > 1 else 0.0)


def metric_row(planner, result, frame_index, fps, road, lane, elapsed_ms):
    points = np.asarray(result.points, float).reshape(-1, 2)
    grid = planner.metric_to_grid(points) if len(points) else np.empty((0, 2), int)
    inside = (len(grid) == 0 or bool(np.all(
        (grid[:, 0] >= 0) & (grid[:, 0] < planner.rows) &
        (grid[:, 1] >= 0) & (grid[:, 1] < planner.cols))))
    road_inside = (inside and (len(grid) == 0 or
                   bool(np.all(result.component[grid[:, 0], grid[:, 1]] > 0))))
    clearance = (inside and (len(grid) == 0 or
                  bool(np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0))))
    angle = result.diagnostics.get("required_steering_deg")
    near_y = (float(np.interp(planner.config.near_required_m,
                              points[:, 0], points[:, 1]))
              if len(points) else float("nan"))
    return {"frame_index": frame_index, "timestamp_sec": frame_index/fps,
            "state": result.state, "valid": bool(result.valid),
            "failure_reason": "|".join(result.diagnostics.get("reasons", [])),
            "road_pixels": int(np.count_nonzero(road)),
            "lane_pixels": int(np.count_nonzero(lane)),
            "path_length_m": path_length(points), "near_path_y_m": near_y,
            "required_steering_deg": (float(angle) if angle is not None else float("nan")),
            "road_inside": bool(road_inside), "clearance_ok": bool(clearance),
            "processing_ms": float(elapsed_ms)}


def support_overlap(mask, road, margin=11):
    area = int(np.count_nonzero(mask))
    if not area:
        return 1.0
    support = cv2.dilate((road > 0).astype(np.uint8),
                         np.ones((2*margin+1, 2*margin+1), np.uint8))
    return float(np.count_nonzero((mask > 0) & (support > 0))/area)


def mask_iou(first, second):
    union = np.count_nonzero((first > 0) | (second > 0))
    return (0.0 if not union else
            float(np.count_nonzero((first > 0) & (second > 0))/union))


def bev_panel(result, planner, title, scale=2):
    road = (result.component > 0)
    safe = (result.safe_road > 0)
    image = np.zeros((*road.shape, 3), np.uint8)
    image[road] = (45, 100, 45)
    image[safe] = (50, 175, 50)
    grid = planner.metric_to_grid(result.points) if len(result.points) else []
    if len(grid):
        points = np.asarray([(int(col), int(row)) for row, col in grid], np.int32)
        cv2.polylines(image, [points], False, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, tuple(points[min(len(points)-1, len(points)//3)]),
                   3, (255, 80, 255), -1)
    state = result.state
    reason = "|".join(result.diagnostics.get("reasons", [])) or "OK"
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, f"{title} {state}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
    cv2.putText(image, reason[:26], (3, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .25, (220, 220, 220), 1)
    return cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)


def tracking_panel(frame, road, result, label):
    panel = cv2.resize(frame, (480, 360))
    shape = panel.shape[:2]
    masks = {
        "road": cv2.resize(road, (480, 360), interpolation=cv2.INTER_NEAREST),
        "raw_w": cv2.resize(result.raw_white_line, (480, 360), interpolation=cv2.INTER_NEAREST),
        "track_w": cv2.resize(result.tracked_white_line, (480, 360), interpolation=cv2.INTER_NEAREST),
        "raw_y": cv2.resize(result.raw_yellow_line, (480, 360), interpolation=cv2.INTER_NEAREST),
        "track_y": cv2.resize(result.tracked_yellow_line, (480, 360), interpolation=cv2.INTER_NEAREST),
    }
    tint = np.zeros((*shape, 3), np.uint8)
    tint[masks["road"] > 0] = (30, 100, 30)
    tint[masks["raw_w"] > 0] = (255, 255, 255)
    tint[masks["track_w"] > 0] = (255, 0, 255)
    tint[masks["raw_y"] > 0] = (0, 255, 255)
    tint[masks["track_y"] > 0] = (0, 80, 255)
    panel = cv2.addWeighted(panel, .7, tint, .45, 0)
    cv2.rectangle(panel, (0, 0), (480, 38), (0, 0, 0), -1)
    diag = result.diagnostics
    cv2.putText(panel, f"{label} W={diag['white_line_source']} Y={diag['yellow_line_source']}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
    cv2.putText(panel, f"scene={int(diag['scene_change_detected'])} reset={diag['reset_reason']}",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
    return panel


def summarize(rows, auxiliary, config):
    output = {}
    for name in ("T0", "T1", "T2"):
        selected = [row for row in rows if row["candidate"] == name]
        valid = [row for row in selected if row["valid"]]
        reversals = jumps = 0
        prior_sign = 0
        prior_near = prior_frame = None
        wheels = []
        for row in selected:
            if not row["valid"]:
                prior_sign = 0
                prior_near = prior_frame = None
                continue
            angle = float(row["required_steering_deg"])
            sign = (0 if not np.isfinite(angle) or abs(angle) <= .25 else
                    int(np.sign(angle)))
            if sign and prior_sign and sign != prior_sign:
                reversals += 1
            if sign:
                prior_sign = sign
            near = float(row["near_path_y_m"])
            if prior_frame is not None and row["frame_index"] == prior_frame+1:
                jumps += abs(near-prior_near) > config.temporal_lateral_gate_m
            prior_near, prior_frame = near, row["frame_index"]
            wheels.append(int(row["camera_wheel"]))
        aux = auxiliary[name]
        output[name] = {
            "frames": len(selected),
            "raw_white_frames": aux["raw_white_frames"],
            "effective_white_frames": aux["effective_white_frames"],
            "raw_yellow_frames": aux["raw_yellow_frames"],
            "effective_yellow_frames": aux["effective_yellow_frames"],
            "white_tracked_frames": aux["white_tracked_frames"],
            "yellow_tracked_frames": aux["yellow_tracked_frames"],
            "unsafe_retained_line_frames": aux["unsafe_retained_line_frames"],
            "cross_class_conflict_frames": aux["cross_class_conflict_frames"],
            "scene_change_count": aux["scene_change_count"],
            "scene_change_residual_frames": aux["scene_change_residual_frames"],
            "state_counts": {state: sum(row["state"] == state for row in selected)
                             for state in ("VALID", "DEGRADED", "INVALID")},
            "path_published_frames": len(valid),
            "drivable_ratio": len(valid)/max(1, len(selected)),
            "mean_path_length_m": (statistics.fmean(
                float(row["path_length_m"]) for row in valid) if valid else 0.0),
            "paths_lt_2m": sum(0 < float(row["path_length_m"]) < 2 for row in valid),
            "road_outside_paths": sum(not row["road_inside"] for row in valid),
            "clearance_violations": sum(not row["clearance_ok"] for row in valid),
            "steering_over_22": sum(abs(float(row["required_steering_deg"])) > 22
                                    for row in valid),
            "path_jumps": jumps,
            "steering_sign_reversals": reversals,
            "TEMPORAL_CURVATURE_JUMP": sum(
                "TEMPORAL_CURVATURE_JUMP" in row["failure_reason"] for row in selected),
            "camera_drive_counts": {str(value): sum(float(row["camera_drive"]) == value
                                                       for row in selected)
                                    for value in (0.0, 1.0, 2.0)},
            "camera_wheel_min": min(wheels, default=0),
            "camera_wheel_max": max(wheels, default=0),
            "mean_processing_ms": statistics.fmean(
                float(row["processing_ms"]) for row in selected),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    config = DirectBevConfig()
    camera = CameraModel(width, height, np.array(
        [[500., 0., width/2.], [0., 500., height/2.], [0., 0., 1.]]), np.zeros(5))
    map_x, map_y = build_ground_remap(
        config, camera, rotation_matrix_rpy(0., -5., 0.),
        np.array([.015, 0., .970]))
    trackers = {
        "T0": LaneMaskTemporalTracker(LaneTemporalConfig(mode="none")),
        "T1": LaneMaskTemporalTracker(LaneTemporalConfig(mode="hold")),
        "T2": LaneMaskTemporalTracker(LaneTemporalConfig(mode="flow")),
    }
    planners = {name: ablation_planners(config)["A6"] for name in trackers}
    controllers = {name: DirectBevController() for name in trackers}
    a0 = DirectBevPlanner(config)
    model = YOLO(str(args.model), task="segment")
    rows = []
    auxiliary = {name: {key: 0 for key in (
        "raw_white_frames", "effective_white_frames", "raw_yellow_frames",
        "effective_yellow_frames", "white_tracked_frames", "yellow_tracked_frames",
        "unsafe_retained_line_frames", "cross_class_conflict_frames",
        "scene_change_count", "scene_change_residual_frames")}
                 for name in trackers}
    tracking_writer = cv2.VideoWriter(
        str(args.output/"tracking_t0_t1_t2_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps/10.0), (1440, 360))
    bev_size = (planners["T2"].cols, planners["T2"].rows)
    a0a6_writer = cv2.VideoWriter(
        str(args.output/"a0_vs_a6_full_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (bev_size[0]*4, bev_size[1]*2))
    a6_writer = cv2.VideoWriter(
        str(args.output/"a6_full_bev_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (bev_size[0]*2, bev_size[1]*2))
    inference_ms = []
    frame_index = 0
    while frame_index < frames_total:
        batch = []
        for _ in range(min(args.batch, frames_total-frame_index)):
            ok, frame = capture.read()
            if not ok:
                break
            batch.append(frame)
        if not batch:
            break
        started = time.perf_counter()
        predictions = model.predict(batch, imgsz=640, conf=args.confidence,
                                    device=args.device, batch=args.batch,
                                    verbose=False)
        per_frame_inference = (time.perf_counter()-started)*1000.0/len(batch)
        inference_ms.extend([per_frame_inference]*len(batch))
        for offset, (frame, prediction) in enumerate(zip(batch, predictions)):
            index = frame_index+offset
            stamp = index/fps
            masks = semantic_masks(prediction, model.names, frame.shape[:2])
            road_bev = project_mask_to_bev(masks["road"], map_x, map_y)
            frame_results = {}
            planner_results = {}
            for name in ("T0", "T1", "T2"):
                item_started = time.perf_counter()
                tracked = trackers[name].update(
                    frame, masks["W_line"], masks["Y_line"], masks["road"], stamp)
                lane_camera = tracked.effective_white_line | tracked.effective_yellow_line
                lane_bev = project_mask_to_bev(lane_camera, map_x, map_y)
                result = planners[name].plan(road_bev, lane_bev, stamp)
                elapsed = (time.perf_counter()-item_started)*1000.0
                row = metric_row(planners[name], result, index, fps,
                                 road_bev, lane_bev, elapsed)
                command = (controllers[name].command(
                    result.points, result.confidence, result.state == "DEGRADED", stamp)
                    if result.valid else controllers[name].neutral())
                row.update({"candidate": name,
                            "camera_drive": 1.0 if result.state == "DEGRADED" else
                                            2.0 if result.state == "VALID" else 0.0,
                            "camera_wheel": int(command.get("wheel", 0)),
                            "white_line_source": tracked.diagnostics["white_line_source"],
                            "yellow_line_source": tracked.diagnostics["yellow_line_source"],
                            "scene_change": tracked.diagnostics["scene_change_detected"]})
                rows.append(row)
                aux = auxiliary[name]
                aux["raw_white_frames"] += int(bool(np.any(tracked.raw_white_line)))
                aux["effective_white_frames"] += int(bool(np.any(tracked.effective_white_line)))
                aux["raw_yellow_frames"] += int(bool(np.any(tracked.raw_yellow_line)))
                aux["effective_yellow_frames"] += int(bool(np.any(tracked.effective_yellow_line)))
                aux["white_tracked_frames"] += int(bool(np.any(tracked.tracked_white_line)))
                aux["yellow_tracked_frames"] += int(bool(np.any(tracked.tracked_yellow_line)))
                combined_track = tracked.tracked_white_line | tracked.tracked_yellow_line
                aux["unsafe_retained_line_frames"] += int(bool(
                    np.any(combined_track) and support_overlap(combined_track, masks["road"]) < .65))
                aux["cross_class_conflict_frames"] += int(bool(
                    mask_iou(tracked.tracked_white_line, tracked.raw_yellow_line) > .15 or
                    mask_iou(tracked.tracked_yellow_line, tracked.raw_white_line) > .15))
                aux["scene_change_count"] += int(bool(
                    tracked.diagnostics["scene_change_detected"]))
                aux["scene_change_residual_frames"] += int(bool(
                    tracked.diagnostics["scene_change_detected"] and np.any(combined_track)))
                frame_results[name] = tracked
                planner_results[name] = result
            a0_result = a0.plan(road_bev, project_mask_to_bev(
                masks["W_line"] | masks["Y_line"], map_x, map_y), stamp)
            left = bev_panel(a0_result, a0, "A0", 2)
            right = bev_panel(planner_results["T2"], planners["T2"], "A6+T2", 2)
            a0a6_writer.write(np.hstack((left, right)))
            a6_writer.write(right)
            if index % 10 == 0:
                tracking_writer.write(np.hstack([
                    tracking_panel(frame, masks["road"], frame_results[name], name)
                    for name in ("T0", "T1", "T2")]))
        frame_index += len(batch)
        if frame_index % 1024 < len(batch) or frame_index == frames_total:
            print(f"temporal evaluation: {frame_index}/{frames_total}", flush=True)
    capture.release(); tracking_writer.release(); a0a6_writer.release(); a6_writer.release()
    with (args.output/"t0_t1_t2_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output/"tracking_safety_counters.json").write_text(
        json.dumps(auxiliary, indent=2)+"\n")
    summary = summarize(rows, auxiliary, config)
    summary["run"] = {"video": str(args.video.resolve()), "model": str(args.model.resolve()),
                      "frames": frame_index, "source_fps": fps,
                      "inference_backend": "PyTorch .pt", "device": args.device,
                      "confidence": args.confidence,
                      "mean_inference_ms": statistics.fmean(inference_ms),
                      "semantic_fps_sequential_equivalent": 1000.0/statistics.fmean(inference_ms)}
    (args.output/"t0_t1_t2_summary.json").write_text(
        json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
