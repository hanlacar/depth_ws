#!/usr/bin/env python3
"""Render the required four-panel A6 review from immutable BEV mask cache."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from camera_navigation.direct_bev_controller import BevControllerConfig, DirectBevController
from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.direct_bev_projection import CameraModel, build_ground_remap
from camera_navigation.ground_plane_calibration import rotation_matrix_rpy
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_navigation.metric_path_quality import maximum_curvature


SIZE = (320, 480)


def title(image, value):
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, 28), (0, 0, 0), -1)
    cv2.putText(image, value, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    reps = {int(item["frame_index"]): item["class"] for item in
            json.loads(args.representatives.read_text())}
    capture = cv2.VideoCapture(str(args.video)); fps = capture.get(cv2.CAP_PROP_FPS)
    config = DirectBevConfig(); planner = ablation_planners(config)["A6"]
    controller = DirectBevController(BevControllerConfig(
        lookahead_from_path_start=True, steering_rate_deg_per_sec=1.0e9))
    camera = CameraModel(640, 480, np.array(
        [[386., 0., 320.], [0., 386., 240.], [0., 0., 1.]]), np.zeros(5),
        "plumb_bob")
    map_x, map_y = build_ground_remap(
        config, camera, rotation_matrix_rpy(0., -5., 0.),
        np.array([.015, 0., .970]))
    writer = cv2.VideoWriter(str(args.output/"hybrid_a6_four_panel.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps/2.,
                             (4*SIZE[0], SIZE[1]))
    index = 0
    for chunk_path in sorted(args.cache.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            for road, lane in zip(chunk["road"], chunk["lane"]):
                ok, frame = capture.read()
                if not ok: break
                result = planner.plan(road, lane, index/fps)
                command = (controller.command(result.points, result.confidence,
                           result.state == "DEGRADED", index/fps)
                           if result.valid else controller.neutral())
                if index % 2 and index not in reps:
                    index += 1; continue
                # Re-splat exact BEV samples into camera pixels for a compact
                # semantic-on-RGB review. Panel 2 remains the authoritative BEV.
                semantic = np.zeros_like(frame)
                for mask, color in ((road, (35, 180, 35)), (lane, (0, 220, 255))):
                    rows, cols = np.nonzero(mask)
                    xs = np.rint(map_x[rows, cols]).astype(int)
                    ys = np.rint(map_y[rows, cols]).astype(int)
                    keep = (xs >= 0) & (xs < 640) & (ys >= 0) & (ys < 480)
                    semantic[ys[keep], xs[keep]] = color
                semantic = cv2.dilate(semantic, np.ones((5, 5), np.uint8))
                first = cv2.addWeighted(frame, .78, semantic, .45, 0)
                first = cv2.resize(first, SIZE); title(first, "RGB + semantic mask")

                road_panel = np.zeros((*road.shape, 3), np.uint8)
                road_panel[road > 0] = (40, 150, 40); road_panel[lane > 0] = (0, 230, 255)
                second = cv2.resize(road_panel, SIZE, interpolation=cv2.INTER_NEAREST)
                title(second, "BEV road / lane")

                path_panel = np.zeros((*road.shape, 3), np.uint8)
                path_panel[result.component > 0] = (40, 90, 40)
                path_panel[result.safe_road > 0] = (50, 170, 50)
                grid = planner.metric_to_grid(result.points) if len(result.points) else []
                if len(grid):
                    pixels = np.asarray([(c, r) for r, c in grid], np.int32)
                    cv2.polylines(path_panel, [pixels], False, (255, 0, 255), 2)
                target = command.get("target_point")
                if target:
                    r, c = planner.metric_to_grid([target])[0]
                    cv2.circle(path_panel, (int(c), int(r)), 4, (255, 255, 0), -1)
                third = cv2.resize(path_panel, SIZE, interpolation=cv2.INTER_NEAREST)
                title(third, "A6 path + controller lookahead")

                points = np.asarray(result.points, float).reshape(-1, 2)
                heading = (math.degrees(math.atan2(points[-1, 1]-points[0, 1],
                                                   points[-1, 0]-points[0, 0]))
                           if len(points) > 1 else 0.0)
                lateral = ((float(points[:, 1].min()), float(points[:, 1].max()))
                           if len(points) else (0.0, 0.0))
                required = result.diagnostics.get("required_steering_deg", 0.) or 0.
                drive = 1.0 if result.state == "DEGRADED" else (2.0 if result.state == "VALID" else 0.0)
                lines = ["steering / drive / state", f"frame {index}  t={index/fps:.2f}s",
                         f"lateral [{lateral[0]:+.2f}, {lateral[1]:+.2f}] m",
                         f"heading {heading:+.2f} deg",
                         f"curvature {maximum_curvature(points) if len(points)>=3 else 0.:.3f} 1/m",
                         f"required {required:+.2f} deg",
                         f"/camera_wheel {int(command.get('wheel', 0)):+d}",
                         f"/camera_drive {drive:.1f}", f"state {result.state}",
                         "reason "+("|".join(result.diagnostics.get("reasons", [])) or "NONE")]
                fourth = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
                for line_no, line in enumerate(lines):
                    cv2.putText(fourth, line[:43], (8, 24+line_no*38),
                                cv2.FONT_HERSHEY_SIMPLEX, .48,
                                (255, 255, 255) if line_no else (0, 230, 255), 1,
                                cv2.LINE_AA)
                combined = np.hstack((first, second, third, fourth))
                if index % 2 == 0: writer.write(combined)
                if index in reps:
                    folder = args.output/("left" if reps[index] == "left" else
                                          "right" if reps[index] == "right" else "straight")
                    folder.mkdir(exist_ok=True)
                    cv2.imwrite(str(folder/f"frame_{index:06d}.png"), combined)
                index += 1
    capture.release(); writer.release()


if __name__ == "__main__":
    main()
