#!/usr/bin/env python3
"""Render a compact, sampled full-video review of the gated road edges."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import (
    HybridCandidateOptions, HybridDirectBevCandidate)


COLORS = {
    "left": (255, 0, 255), "right": (255, 255, 0),
    "path": (0, 255, 0), "lookahead": (0, 128, 255),
}


def planner():
    return HybridDirectBevCandidate(DirectBevConfig(), HybridCandidateOptions(
        temporal_smoothing=True, curvature_stabilization=True,
        mode_hysteresis_frames=3, fixed_resample_origin=True,
        fail_closed_hold=True, road_boundary_fallback="gated"))


def polyline(image, planner_instance, points, color, thickness=2):
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(points):
        return
    grid = planner_instance.metric_to_grid(points)
    pixels = np.column_stack((grid[:, 1], grid[:, 0])).astype(np.int32)
    cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)


def render_bev(planner_instance, result, lane):
    image = np.zeros((*result.road.shape, 3), np.uint8)
    image[result.road > 0] = (55, 55, 55)
    image[result.safe_road > 0] = (85, 45, 15)
    image[np.asarray(lane) > 0] = (230, 230, 230)
    diagnostics = result.diagnostics
    polyline(image, planner_instance,
             diagnostics.get("road_left_boundary", []), COLORS["left"])
    polyline(image, planner_instance,
             diagnostics.get("road_right_boundary", []), COLORS["right"])
    if len(result.points):
        # Thick translucent-looking contour represents the complete swept
        # footprint width already used to erode safe_road.
        clearance = (planner_instance.config.vehicle_width_m/2.0 +
                     planner_instance.config.lateral_safety_margin_m)
        thickness = max(1, int(round(2*clearance /
                                   planner_instance.config.resolution_m)))
        polyline(image, planner_instance, result.points, (40, 90, 40), thickness)
        polyline(image, planner_instance, result.points, COLORS["path"], 2)
    lookahead = diagnostics.get("target_point")
    if lookahead is not None:
        grid = planner_instance.metric_to_grid([lookahead])[0]
        cv2.circle(image, (int(grid[1]), int(grid[0])), 4,
                   COLORS["lookahead"], -1, cv2.LINE_AA)
    return cv2.resize(image, (600, 480), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=120)
    parser.add_argument("--input-fps", type=float, default=60.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    writer = cv2.VideoWriter(str(args.output),
                             cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                             (1240, 480))
    if not cap.isOpened() or not writer.isOpened():
        raise RuntimeError("video input/output could not be opened")
    p = planner()
    frame_index = 0
    rendered = 0
    try:
        for chunk_path in sorted(args.cache.glob("chunk_*.npz")):
            with np.load(chunk_path) as chunk:
                for road, lane in zip(chunk["road"], chunk["lane"]):
                    ok, source = cap.read()
                    if not ok:
                        raise RuntimeError(f"source video ended at {frame_index}")
                    result = p.plan(road, lane, frame_index/args.input_fps)
                    if frame_index % args.sample_every == 0:
                        source = cv2.resize(source, (640, 480))
                        bev = render_bev(p, result, lane)
                        canvas = np.hstack((source, bev))
                        text = (f"frame={frame_index} state={result.state} "
                                f"source={result.diagnostics.get('path_source')} "
                                f"wheel={-round(result.diagnostics.get('required_steering_deg', 0) or 0)}")
                        cv2.rectangle(canvas, (0, 0), (1240, 34), (0, 0, 0), -1)
                        cv2.putText(canvas, text, (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        writer.write(canvas)
                        rendered += 1
                    frame_index += 1
    finally:
        cap.release(); writer.release()
    print(f"frames={frame_index} rendered={rendered} output={args.output}")


if __name__ == "__main__":
    main()
