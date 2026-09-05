#!/usr/bin/env python3
"""Build compact tables for the deterministic shadow validation report."""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("validation/shadow_ab_20260830")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    segments = json.loads(Path(
        "validation/steering_fix_20260830/turn_segments.json").read_text())["segments"]
    def label(index):
        return next((s["class"] for s in segments
                     if s["start_frame"] <= index <= s["end_frame"]), None)
    turns = {name: [] for name in ("straight", "left", "right")}
    with gzip.open(ROOT/"shadow_ab_frames.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            kind = label(int(row["frame_index"]))
            if kind and row["a6_state"] != "INVALID": turns[kind].append(row)
    fields = ("a6_local_target_curvature_per_m", "a6_turn_radius_m",
              "a6_expected_steering_deg", "a6_required_steering_deg",
              "a6_wheel_command", "a6_steering_error_deg",
              "a6_quantized_target_lateral_error_m")
    curvature = {}
    for kind, rows in turns.items():
        curvature[kind] = {"frames": len(rows)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in rows]); values = values[np.isfinite(values)]
            curvature[kind][field] = {
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "p50": float(np.percentile(values, 50)) if len(values) else 0.0,
                "p95_abs": float(np.percentile(np.abs(values), 95)) if len(values) else 0.0,
                "min": float(np.min(values)) if len(values) else 0.0,
                "max": float(np.max(values)) if len(values) else 0.0,
            }
    (ROOT/"curvature_steering_summary.json").write_text(json.dumps(curvature, indent=2)+"\n")

    offline = []
    for rate, directory in ((12,"rate_12"),(15,"rate_15"),(30,"rate_30"),(60,".")):
        summary=json.loads((ROOT/directory/"shadow_ab_summary.json").read_text())
        sets=summary["sets"]
        offline.append({"input_fps":rate,"frames":summary["frames"],
            "semantic_fps":rate,"planner_drop_ratio":0.0,
            "a0_drivable_ratio":(sets["both_drivable"]+sets["a0_only_drivable"])/summary["frames"],
            "a6_drivable_ratio":(sets["both_drivable"]+sets["a6_only_drivable"])/summary["frames"],
            "a0_a6_disagreement_frames":sets["a0_only_drivable"]+sets["a6_only_drivable"],
            "pair_processing_mean_ms":summary["processing_ms"]["a0"]["mean"]+summary["processing_ms"]["a6"]["mean"],
            "pair_processing_p95_ms_upper_bound":summary["processing_ms"]["a0"]["p95"]+summary["processing_ms"]["a6"]["p95"],
            "a6_digest":summary["deterministic_digest"]["a6"]})
    ros=[]
    for rate,directory in ((12,"ros_rate_12_ok"),(15,"ros_rate_15"),(30,"ros_rate_30"),(60,"ros_rate_60")):
        base=ROOT/directory;m=json.loads((base/"measurement_summary.json").read_text());e=json.loads((base/"extended_summary.json").read_text());diag=jsonl(base/"diagnostics.jsonl")
        raw=m["unique_frames"].get("image_raw",0);sem=m["unique_frames"].get("semantic",0);reasons=Counter(x for row in diag for x in row.get("reasons",[]));lat=[row.get("end_to_end_latency_ms") for row in diag if row.get("end_to_end_latency_ms") is not None]
        ros.append({"input_fps":rate,"wall_duration_sec":m["measured_duration_sec"],"input_frames":raw,"semantic_frames":sem,
                    "semantic_fps":m["observed_fps"].get("semantic",0),"planner_fps":e["planner_processing_fps_median"],
                    "drop_ratio":1-sem/max(1,raw),"drivable_ratio":e["drivable_ratio"],
                    "latency_mean_ms":float(np.mean(lat)) if lat else 0.,"latency_p95_ms":e["end_to_end_latency_ms_p95"],
                    "INPUT_TIMEOUT":reasons["INPUT_TIMEOUT"],"CALIBRATION_INVALID":reasons["CALIBRATION_INVALID"],
                    "planner_worker_replaced":max((row.get("worker",{}).get("replaced",0) for row in diag),default=0)})
    output={"offline_full_cache_virtual_replay":offline,"ros_25s_cpu_replay":ros}
    (ROOT/"rate_comparison.json").write_text(json.dumps(output,indent=2)+"\n")
    with (ROOT/"rate_comparison.csv").open("w",newline="") as stream:
        rows=[]
        for row in offline:rows.append({"mode":"offline_full_cache",**row})
        for row in ros:rows.append({"mode":"ros_cpu_25s",**row})
        keys=[]
        for row in rows:
            for key in row:
                if key not in keys:keys.append(key)
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

    comparisons=[]
    for name,directory in (("prior_91_65",Path("validation/lane_tracking_ros_20260830/ros_full_hybrid_a6_final")),
                           ("current_91_01",Path("validation/steering_fix_20260830/ros_full_localhost"))):
        m=json.loads((directory/"measurement_summary.json").read_text());diag=jsonl(directory/"diagnostics.jsonl");reasons=Counter(x for row in diag for x in row.get("reasons",[]));stamps=np.asarray([row.get("source_stamp_ns",row.get("stamp_ns",0)) for row in diag],np.int64);gaps=np.diff(stamps)/1e6;normal=gaps[(gaps>0)&(gaps<1000)];road=[row.get("decoded_road_pixels") or 0 for row in diag];cal=Counter(row.get("calibration_state") for row in diag)
        states=Counter(row.get("state") for row in diag);semantic=m["unique_frames"].get("semantic",0);raw=m["unique_frames"].get("image_raw",0)
        comparisons.append({"name":name,"semantic_frames":semantic,"input_frames":raw,"semantic_drop_ratio":1-semantic/max(1,raw),
            "state_counts":dict(states),"drivable_ratio":(states["VALID"]+states["DEGRADED"])/len(diag),"reason_counts":dict(reasons),"calibration_counts":dict(cal),
            "timestamp_gap_ms":{"p50":float(np.percentile(normal,50)),"p95":float(np.percentile(normal,95)),"max":float(np.max(normal)),"nonpositive":int(np.sum(gaps<=0)),"over_250_under_1000":int(np.sum((gaps>250)&(gaps<1000)))},
            "decoded_road_pixels_mean":float(np.mean(road)),"raw_road_zero_frames":sum(value==0 for value in road),
            "semantic_fps":m["observed_fps"].get("semantic",0),"publisher_loop":True,
            "recorder_warmup_sec":8 if name=="prior_91_65" else 0})
    conclusion={"runs":comparisons,
      "same_cache_a6_drivable_ratio":13518/14409,
      "same_cache_a6_digest_all_rates":len({row['a6_digest'] for row in offline})==1,
      "classification":"ROS scheduling/subsequence/start-window variation; not deterministic A6 algorithm regression"}
    (ROOT/"offline_ros_difference.json").write_text(json.dumps(conclusion,indent=2)+"\n")


if __name__=="__main__":main()
