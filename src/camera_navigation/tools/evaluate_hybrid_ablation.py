#!/usr/bin/env python3
"""Run A0--A6 on one immutable chunked metric-BEV mask cache."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from camera_navigation.direct_bev_core import DirectBevConfig
from camera_navigation.hybrid_bev_candidate import ablation_planners
from camera_navigation.metric_path_quality import maximum_curvature


def path_json(points):
    return json.dumps(np.round(np.asarray(points, float), 5).tolist(),
                      separators=(",", ":"))


def path_length(points):
    points = np.asarray(points, float).reshape(-1, 2)
    return (float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if len(points) > 1 else 0.0)


def heading(points):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2:
        return float("nan")
    delta = points[1]-points[0]
    return math.degrees(math.atan2(float(delta[1]), float(delta[0])))


def steering(planner, points):
    angle, _, _, _ = planner._steering(points)
    return float(angle) if angle is not None else float("nan")


def reason(result):
    return "|".join(result.diagnostics.get("reasons", []))


def make_row(name, planner, result, frame_index, fps, road, lane, elapsed,
             previous_curvature):
    points = result.points
    raw = getattr(planner, "last_raw_path", points)
    smooth = getattr(planner, "last_smoothed_path", points)
    previous = getattr(planner, "last_previous_path", np.empty((0, 2)))
    curve = maximum_curvature(smooth if len(smooth) else points)
    curve_delta = (abs(curve-previous_curvature)
                   if previous_curvature is not None else 0.0)
    grid = planner.metric_to_grid(points) if len(points) else np.empty((0, 2), int)
    in_bounds = (len(grid) == 0 or np.all(
        (grid[:, 0] >= 0) & (grid[:, 0] < planner.rows) &
        (grid[:, 1] >= 0) & (grid[:, 1] < planner.cols)))
    in_component = (bool(in_bounds) and (len(grid) == 0 or
                    bool(np.all(result.component[grid[:, 0], grid[:, 1]] > 0))))
    in_safe = (bool(in_bounds) and (len(grid) == 0 or
               bool(np.all(result.safe_road[grid[:, 0], grid[:, 1]] > 0))))
    attempted_mode = getattr(planner, "last_candidate_mode", None)
    angle = result.diagnostics.get("required_steering_deg")
    if angle is None and len(points):
        angle = steering(planner, points)
    near_y = (float(np.interp(planner.config.near_required_m,
                              points[:, 0], points[:, 1]))
              if len(points) else float("nan"))
    row = {
        "frame_index": frame_index, "timestamp_sec": frame_index/fps,
        "planner": name, "state": result.state, "mode": result.mode,
        "attempted_mode": attempted_mode or result.mode,
        "valid": bool(result.valid), "road_pixels": int(road.sum()),
        "lane_pixels": int(lane.sum()), "path_point_count": len(points),
        "path_length_m": path_length(points), "heading_deg": heading(smooth),
        "near_path_y_m": near_y,
        "curvature_per_m": curve, "curvature_delta_per_m": curve_delta,
        "required_steering_deg": angle if angle is not None else float("nan"),
        "minimum_clearance_m": result.diagnostics.get("minimum_clearance_m", 0.0),
        "safe_coverage": result.diagnostics.get("safe_road_coverage", 0.0),
        "road_inside": in_component, "clearance_ok": in_safe,
        "failure_reason": reason(result), "processing_ms": elapsed,
        "path": path_json(points), "raw_path": path_json(raw),
        "smoothed_path": path_json(smooth),
        "previous_accepted_path": path_json(previous),
    }
    accepted = getattr(planner, "previous", None)
    return row, (maximum_curvature(accepted)
                 if accepted is not None and len(accepted) else previous_curvature)


def classify(row, prior, config):
    raw = np.asarray(json.loads(row["raw_path"]), float).reshape(-1, 2)
    previous = np.asarray(json.loads(row["previous_accepted_path"]),
                          float).reshape(-1, 2)
    if len(raw) >= 2 and np.any(np.diff(raw[:, 0]) < -1.0e-6):
        kind = "경로 점 순서 또는 방향 오류"
    elif len(raw) < config.minimum_path_points:
        kind = "근거리 point 부족"
    elif row["attempted_mode"] != prior.get("attempted_mode"):
        kind = "road-only/lane mode 전환"
    elif previous.size and abs(path_length(raw)-path_length(previous)) > 0.8:
        kind = "경로 길이 변화"
    elif prior and (abs(int(row["road_pixels"])-int(prior["road_pixels"])) > 5000 or
                    abs(int(row["lane_pixels"])-int(prior["lane_pixels"])) > 1000):
        kind = "mask의 순간 변화"
    elif float(row["curvature_delta_per_m"]) > config.temporal_curvature_gate_per_m:
        kind = "곡선 fitting 불안정"
    else:
        kind = "smoothing 부족"
    dangerous = (float(row["curvature_delta_per_m"]) >
                 3.0*config.temporal_curvature_gate_per_m)
    verdict = ("실제 위험한 곡률 변화" if dangerous else
               "temporal gate의 오판")
    return kind, verdict


def summary(rows, config):
    output = {}
    for name in sorted({row["planner"] for row in rows}):
        selected = [row for row in rows if row["planner"] == name]
        valid = [row for row in selected if row["valid"]]
        signs = []
        reversals = 0
        jumps = 0
        previous_near = None
        previous_frame = None
        for row in selected:
            if not row["valid"]:
                previous_near = None
                previous_frame = None
                continue
            angle = float(row["required_steering_deg"])
            sign = 0 if abs(angle) <= .25 else int(np.sign(angle))
            if signs and sign and signs[-1] and sign != signs[-1]:
                reversals += 1
            signs.append(sign)
            near = float(row["near_path_y_m"])
            if previous_frame is not None and row["frame_index"] == previous_frame+1:
                jumps += abs(near-previous_near) > config.temporal_lateral_gate_m
            previous_near, previous_frame = near, row["frame_index"]
        output[name] = {
            "frames": len(selected),
            "state_counts": {state: sum(row["state"] == state for row in selected)
                             for state in ("VALID", "DEGRADED", "INVALID")},
            "drivable": len(valid),
            "drivable_ratio": len(valid)/len(selected),
            "TEMPORAL_CURVATURE_JUMP": sum(
                "TEMPORAL_CURVATURE_JUMP" in row["failure_reason"]
                for row in selected),
            "mean_path_length_m_drivable": (statistics.fmean(
                float(row["path_length_m"]) for row in valid) if valid else 0.0),
            "paths_lt_2m": sum(0 < float(row["path_length_m"]) < 2
                               for row in valid),
            "road_outside_paths": sum(not row["road_inside"] for row in valid),
            "clearance_violations": sum(not row["clearance_ok"] for row in valid),
            "steering_over_22": sum(abs(float(row["required_steering_deg"])) > 22
                                    for row in valid),
            "steering_sign_reversals": reversals,
            "path_jumps": jumps,
            "mean_processing_ms": statistics.fmean(
                float(row["processing_ms"]) for row in selected),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = DirectBevConfig()
    planners = ablation_planners(config)
    rows = []
    detailed_by_frame = {}
    recent = []
    future_context = set()
    targets = []
    previous_curves = {name: None for name in planners}
    frame_index = 0
    for chunk_path in sorted(args.cache.glob("chunk_*.npz")):
        with np.load(chunk_path) as chunk:
            roads, lanes = chunk["road"], chunk["lane"]
            for road, lane in zip(roads, lanes):
                frame_rows = {}
                for name, planner in planners.items():
                    started = time.perf_counter()
                    result = planner.plan(road, lane, frame_index/args.fps)
                    elapsed = (time.perf_counter()-started)*1000.0
                    row, curve = make_row(
                        name, planner, result, frame_index, args.fps,
                        road, lane, elapsed, previous_curves[name])
                    frame_rows[name] = row
                    rows.append({key: value for key, value in row.items()
                                 if key not in ("path", "raw_path",
                                                "smoothed_path",
                                                "previous_accepted_path")})
                    previous_curves[name] = curve
                recent.append((frame_index, {name: frame_rows[name]
                                             for name in ("A0", "A1", "A6")}))
                recent = recent[-6:]
                a0, a1 = frame_rows["A0"], frame_rows["A1"]
                if (a0["valid"] and not a1["valid"] and
                        "TEMPORAL_CURVATURE_JUMP" in a1["failure_reason"]):
                    targets.append(frame_index)
                    for old_index, old_rows in recent:
                        detailed_by_frame[old_index] = old_rows
                    future_context.update(range(frame_index+1, frame_index+4))
                if frame_index in future_context:
                    detailed_by_frame[frame_index] = {
                        name: frame_rows[name] for name in ("A0", "A1", "A6")}
                frame_index += 1
                if frame_index % 1000 == 0:
                    print(f"ablations: {frame_index}", flush=True)
    with (args.output / "ablation_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = summary(rows, config)
    (args.output / "ablation_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n")

    contexts = sorted(detailed_by_frame)
    detail_rows = []
    category_counts = {}
    verdict_counts = {}
    for index in contexts:
        for name in ("A0", "A1", "A6"):
            row = dict(detailed_by_frame[index][name])
            prior = detailed_by_frame.get(index-1, {}).get(name, {})
            if name == "A1" and index in targets:
                kind, verdict = classify(row, prior, config)
                row["failure_type"] = kind
                row["gate_verdict"] = verdict
                category_counts[kind] = category_counts.get(kind, 0)+1
                verdict_counts[verdict] = verdict_counts.get(verdict, 0)+1
            else:
                row["failure_type"] = ""
                row["gate_verdict"] = ""
            detail_rows.append(row)
    if detail_rows:
        with (args.output / "temporal_failure_context.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
            writer.writeheader(); writer.writerows(detail_rows)
    clusters = []
    for index in targets:
        if not clusters or index-clusters[-1][-1] > 9:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    representatives = [cluster[len(cluster)//2] for cluster in clusters]
    if len(targets) >= 20 and len(representatives) != 20:
        representatives = [targets[i] for i in np.linspace(
            0, len(targets)-1, 20, dtype=int)]
    report = {"target_failure_frames": len(targets),
              "context_frames": len(contexts),
              "failure_type_counts": category_counts,
              "gate_verdict_counts": verdict_counts,
              "representative_frames": representatives,
              "production_changed": False,
              "temporal_curvature_gate_per_m": config.temporal_curvature_gate_per_m}
    (args.output / "temporal_failure_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
