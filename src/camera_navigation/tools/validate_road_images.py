#!/usr/bin/env python3
"""Offline production-pipeline validation for road-image ZIPs/directories."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import tempfile
import zipfile

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import yaml
from race_interfaces.msg import ImagePath, ImagePathPoint

from camera_navigation.camera_pixel_controller_node import (
    PixelController,
    PixelControllerConfig,
)
from camera_navigation.image_path_planner import (
    BOTH_BOUNDARIES,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY,
    ROAD_CENTER,
    TEMPORAL_FALLBACK,
    ImagePathPlanner,
    PlannerConfig,
)
from camera_navigation.pixel_lateral_control import lookahead_offset_px
from camera_yolo_inference.class_mapper import SemanticClassMapper
from camera_yolo_inference.inference_backend import create_inference_backend
from camera_yolo_inference.mask_postprocessor import build_semantic_masks
from camera_yolo_inference.model_manifest import load_manifest
from camera_yolo_inference.perception_outputs import SEMANTIC_MASK_TOPICS


SUPPORTED_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def parse_arguments():
    nav_share = Path(get_package_share_directory("camera_navigation"))
    yolo_share = Path(get_package_share_directory("camera_yolo_inference"))
    parser = argparse.ArgumentParser(
        description="Validate production perception/path/controller on road images")
    parser.add_argument("images", type=Path, help="image directory or ZIP archive")
    parser.add_argument("--output", type=Path,
                        help="output directory (default: a new temporary directory)")
    parser.add_argument("--planner-config", type=Path,
                        default=nav_share / "config/image_path.yaml")
    parser.add_argument("--controller-config", type=Path,
                        default=nav_share / "config/camera_pixel_controller.yaml")
    parser.add_argument("--model", type=Path,
                        default=yolo_share / "models/hanla_yolo11n_seg_best.engine")
    parser.add_argument("--manifest", type=Path,
                        default=yolo_share / "config/class_manifest.yaml")
    parser.add_argument("--backend", default="tensorrt",
                        choices=("tensorrt", "pytorch"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=1)
    return parser.parse_args()


def ros_parameters(path, node_name):
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(document[node_name]["ros__parameters"])


def collect_images(root):
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def capture_times(paths):
    raw = []
    for index, path in enumerate(paths):
        try:
            value = float(path.stem)
        except ValueError:
            value = index/20.0
        raw.append(value)
    origin = raw[0] if raw else 0.0
    return raw, [1.0+value-origin for value in raw]


def make_planner(values):
    selected = {name: values[name] for name in PlannerConfig.__dataclass_fields__
                if name in values}
    flat_polygon = list(selected.get("ego_exclusion_polygon", ()))
    if flat_polygon and not isinstance(flat_polygon[0], (list, tuple)):
        if len(flat_polygon) % 2:
            raise ValueError("ego_exclusion_polygon must have an even number of values")
        selected["ego_exclusion_polygon"] = tuple(
            (float(flat_polygon[index]), float(flat_polygon[index+1]))
            for index in range(0, len(flat_polygon), 2))
    return ImagePathPlanner(PlannerConfig(**selected))


def make_controller(values):
    selected = {
        name: values[name]
        for name in PixelControllerConfig.__dataclass_fields__
        if name in values
    }
    return PixelController(PixelControllerConfig(**selected))


def command_for(controller, result, width, now):
    stamp_ns = int(round(now*1.0e9))
    source_codes = {
        BOTH_BOUNDARIES: ImagePathPoint.BOTH_BOUNDARIES,
        LEFT_BOUNDARY: ImagePathPoint.LEFT_BOUNDARY,
        RIGHT_BOUNDARY: ImagePathPoint.RIGHT_BOUNDARY,
        ROAD_CENTER: ImagePathPoint.ROAD_CENTER,
        TEMPORAL_FALLBACK: ImagePathPoint.TEMPORAL_FALLBACK,
    }
    state_codes = {
        "VALID": ImagePath.STATE_VALID,
        "DEGRADED": ImagePath.STATE_DEGRADED,
        "INVALID": ImagePath.STATE_INVALID,
        "INACTIVE": ImagePath.STATE_INACTIVE,
    }
    controller.ingest_path(
        stamp_ns, width, result.points, result.valid, result.confidence, now,
        [source_codes[source] for source in result.sources],
        state_codes[result.state])
    requested = controller.step(now+0.001, stamp_ns+1_000_000)
    return controller.finalize_output(requested, now+0.20)


def bilateral_center_errors(planner, result):
    errors = []
    for point, source in zip(result.raw, result.sources):
        if source != BOTH_BOUNDARIES:
            continue
        y = point[1]
        left = planner._at_y(
            result.left, y, planner.config.sample_interval_px/2)
        right = planner._at_y(
            result.right, y, planner.config.sample_interval_px/2)
        if left is not None and right is not None:
            errors.append(abs(float(point[0])-(left+right)/2.0))
    return errors


def single_geometry_checks(result, minimum_clearance_m):
    checks = []
    for detail in result.virtual_details or ():
        delta = detail["virtual"]-detail["boundary"]
        has_normal = "tangent" in detail
        clearance = detail.get("clearance_m")
        checks.append({
            "source": detail["source"],
            "width_source": detail.get("lane_width_source"),
            "method": detail.get("method", "normal_offset"),
            "clearance_m": (float(clearance)
                            if clearance is not None else None),
            "normal_error": (abs(float(np.dot(detail["tangent"], delta)))
                             if has_normal else 0.0),
            "geometry_ok": bool(
                np.isfinite(delta).all() and
                (not has_normal or
                 abs(float(np.dot(detail["tangent"], delta))) <= 1e-5)),
            "clearance_ok": (clearance is None or
                             float(clearance)+1e-6 >= minimum_clearance_m),
        })
    return checks


def analyze_frame(planner, controller, path, image, masks, result, now):
    command = command_for(controller, result, image.shape[1], now)
    diagnostics = result.diagnostics or {}
    offset = lookahead_offset_px(
        result.points, image.shape[1], controller.config.lookahead_y_ratio)
    finite = bool(
        np.isfinite(result.points).all() and
        math.isfinite(result.confidence) and
        math.isfinite(command.drive) and math.isfinite(command.wheel))
    expected_drive = (controller.drive_for_path(controller.path, command.wheel)
                      if result.valid else 0.0)
    sign_ok = (not result.valid or command.wheel == 0 or offset is None or
               abs(offset) < 1e-9 or
               int(math.copysign(1, command.wheel)) ==
               int(math.copysign(1, offset)))
    return {
        "file": path.name,
        "processed": True,
        "path_valid": bool(result.valid),
        "path_state": result.state,
        "confidence": float(result.confidence),
        "point_count": int(len(result.points)),
        "near_x": float(result.points[0, 0]) if len(result.points) else None,
        "middle_x": (float(result.points[len(result.points)//2, 0])
                     if len(result.points) else None),
        "far_x": float(result.points[-1, 0]) if len(result.points) else None,
        "lookahead_offset_px": float(offset) if offset is not None else None,
        "camera_wheel": int(command.wheel),
        "camera_drive": float(command.drive),
        "command_reason": command.reason,
        "finite": finite,
        "wheel_range_ok": -22 <= command.wheel <= 22,
        "drive_rule_ok": (command.drive == expected_drive if result.valid
                          else command.drive == 0.0),
        "invalid_fail_safe_ok": result.valid or
                                (command.drive == 0.0 and command.wheel == 0),
        "steering_sign_ok": sign_ok,
        "both_boundary_rows": int(diagnostics.get("both_boundary_rows", 0)),
        "boundary_mode": diagnostics.get("boundary_mode"),
        "lane_width_profile_rows": int(
            diagnostics.get("lane_width_profile_rows", 0)),
        "road_containment_ratio": diagnostics.get("road_containment_ratio"),
        "vehicle_containment_ok": bool(
            diagnostics.get("vehicle_containment_ok", True)),
        "vehicle_clearance_rejections": int(
            diagnostics.get("rejections", {}).get("VEHICLE_CLEARANCE", 0)),
        "source_counts": dict(Counter(result.sources)),
        "bilateral_center_errors_px": bilateral_center_errors(planner, result),
        "single_geometry": single_geometry_checks(
            result, planner.config.minimum_boundary_clearance_m),
        "mask_pixels": {name: int(np.count_nonzero(mask))
                        for name, mask in masks.items()
                        if name in ("road", "white_line", "yellow_line")},
    }


def draw_overlay(image, result, row, index):
    overlay = image.copy()
    road_layer = np.zeros_like(overlay)
    road_layer[result.road_component > 0] = (0, 100, 0)
    overlay = cv2.addWeighted(overlay, 1.0, road_layer, 0.35, 0.0)
    for points, color in ((result.left, (255, 170, 0)),
                          (result.right, (0, 220, 255))):
        for point in points:
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 3, color, -1)
    if len(result.points) > 1:
        cv2.polylines(overlay, [np.rint(result.points).astype(np.int32)],
                      False, (0, 0, 255), 4, cv2.LINE_AA)
    center_x = image.shape[1]//2
    cv2.line(overlay, (center_x, image.shape[0]-1), (center_x, 120),
             (255, 255, 0), 2)
    lines = (
        f"{index:02d} {row['path_state']} conf={row['confidence']:.2f} "
        f"pts={row['point_count']}",
        f"wheel={row['camera_wheel']:+d} drive={row['camera_drive']:.1f}",
        f"mode={row['boundary_mode']} width={row['lane_width_profile_rows']}",
    )
    for line_index, text in enumerate(lines):
        cv2.putText(overlay, text, (8, 22+line_index*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255),
                    2, cv2.LINE_AA)
    return overlay


def run_pass(frames, paths, raw_times, relative_times, planner_values,
             controller_values, independent, overlay_dir=None):
    planner = make_planner(planner_values)
    controller = make_controller(controller_values)
    controller.finalize_output(PixelController.stop("initial_stop"), 0.0)
    rows = []
    for index, (path, frame, raw_time, now) in enumerate(zip(
            paths, frames, raw_times, relative_times)):
        if independent:
            planner = make_planner(planner_values)
            controller = make_controller(controller_values)
            controller.finalize_output(PixelController.stop("initial_stop"), now-1.0)
        image, masks = frame
        result = planner.plan(
            masks["road"], masks["white_line"], masks["yellow_line"],
            masks["words"], masks["stop_line"], masks["c_line"],
            timestamp_sec=raw_time)
        row = analyze_frame(planner, controller, path, image, masks, result, now)
        rows.append(row)
        if overlay_dir is not None:
            overlay = draw_overlay(image, result, row, index)
            cv2.imwrite(str(overlay_dir/f"{index:03d}_{path.stem}.jpg"), overlay)
    return rows


def temporal_dropout_checks(frames, raw_times, planner_values):
    checks = []
    for index, ((_, masks), timestamp) in enumerate(zip(frames, raw_times)):
        planner = make_planner(planner_values)
        result = planner.plan(
            masks["road"], masks["white_line"], masks["yellow_line"],
            masks["words"], masks["stop_line"], masks["c_line"],
            timestamp_sec=timestamp)
        if not result.valid:
            continue
        empty = np.zeros_like(masks["road"])
        fallback = planner.plan(
            empty, empty, empty, empty, empty, empty,
            timestamp_sec=timestamp+0.05)
        checks.append({
            "frame": index,
            "fallback_valid": bool(fallback.valid),
            "same_path": bool(
                result.points.shape == fallback.points.shape and
                np.allclose(result.points, fallback.points)),
            "safe_stop": bool(not fallback.valid and not len(fallback.points)),
        })
    return checks


def summarize(rows, temporal_checks, images):
    all_single = [detail for row in rows for detail in row["single_geometry"]]
    bilateral_errors = [error for row in rows
                        for error in row["bilateral_center_errors_px"]]
    valid_rows = [row for row in rows if row["path_valid"]]
    similar_pairs = []
    for previous, current, previous_image, current_image in zip(
            rows, rows[1:], images, images[1:]):
        difference = float(np.mean(cv2.absdiff(previous_image, current_image))/255.0)
        if difference <= 0.08 and previous["path_valid"] and current["path_valid"]:
            jump = abs((current["lookahead_offset_px"] or 0.0)-
                       (previous["lookahead_offset_px"] or 0.0))
            signs = (int(np.sign(previous["camera_wheel"])),
                     int(np.sign(current["camera_wheel"])))
            similar_pairs.append({"from": previous["file"], "to": current["file"],
                                  "image_difference": difference,
                                  "lookahead_jump_px": jump,
                                  "steering_sign_flip": signs[0]*signs[1] < 0})
    summary = {
        "image_count": len(rows),
        "processed_count": sum(row["processed"] for row in rows),
        "path_valid_count": len(valid_rows),
        "path_invalid_count": len(rows)-len(valid_rows),
        "finite_count": sum(row["finite"] for row in rows),
        "bilateral_frame_count": sum(row["both_boundary_rows"] > 0 for row in rows),
        "single_boundary_valid_count": sum(
            row["path_valid"] and row["boundary_mode"] in
            (LEFT_BOUNDARY, RIGHT_BOUNDARY) for row in rows),
        "learned_width_single_detail_count": sum(
            detail["width_source"] == "width_profile" for detail in all_single),
        "road_fallback_single_detail_count": sum(
            detail["width_source"] == "road_corridor" for detail in all_single),
        "minimum_clearance_m": min(
            (detail["clearance_m"] for detail in all_single
             if detail["clearance_m"] is not None), default=None),
        "maximum_bilateral_center_error_px": max(bilateral_errors, default=0.0),
        "wheel_min": min((row["camera_wheel"] for row in rows), default=0),
        "wheel_max": max((row["camera_wheel"] for row in rows), default=0),
        "similar_pair_count": len(similar_pairs),
        "maximum_similar_pair_jump_px": max(
            (pair["lookahead_jump_px"] for pair in similar_pairs), default=0.0),
        "similar_pair_steering_sign_flips": sum(
            pair["steering_sign_flip"] for pair in similar_pairs),
        "temporal_dropout_checks": len(temporal_checks),
        "temporal_dropout_passed": sum(
            check["safe_stop"]
            for check in temporal_checks),
    }
    checks = {
        "all_images_processed": summary["processed_count"] == summary["image_count"],
        "all_outputs_finite": summary["finite_count"] == summary["image_count"],
        "wheel_range": all(row["wheel_range_ok"] for row in rows),
        "drive_rule": all(row["drive_rule_ok"] for row in rows),
        "invalid_fail_safe": all(row["invalid_fail_safe_ok"] for row in rows),
        "steering_sign": all(row["steering_sign_ok"] for row in rows),
        "bilateral_center": bool(bilateral_errors) and
                            summary["maximum_bilateral_center_error_px"] <= 1.0,
        "single_boundary_geometry": bool(all_single) and all(
            detail["geometry_ok"] for detail in all_single),
        "minimum_1_5m_clearance": bool(all_single) and all(
            detail["clearance_ok"] for detail in all_single),
        "learned_width_precedence":
            summary["learned_width_single_detail_count"] > 0,
        "road_corridor_fallback":
            summary["road_fallback_single_detail_count"] > 0,
        # Rejected candidate rows are evidence that the gate worked; only the
        # accepted final path's containment result determines this check.
        "vehicle_body_containment": all(
            not row["path_valid"] or
            row["vehicle_containment_ok"]
            for row in rows),
        "temporal_dropout": bool(temporal_checks) and all(
            check["safe_stop"]
            for check in temporal_checks),
        "similar_frame_stability": all(
            pair["lookahead_jump_px"] <= 55.0 and
            not pair["steering_sign_flip"] for pair in similar_pairs),
    }
    summary["checks"] = checks
    summary["pass"] = all(checks.values())
    return summary, similar_pairs


def main():
    args = parse_arguments()
    output = args.output or Path(tempfile.mkdtemp(prefix="camera-stage2-output-"))
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output/"overlays"
    overlay_dir.mkdir(exist_ok=True)

    source_context = nullcontext(args.images)
    if args.images.is_file() and args.images.suffix.lower() == ".zip":
        temporary = tempfile.TemporaryDirectory(prefix="camera-stage2-images-")
        source_context = temporary
    with source_context as source:
        if isinstance(source, str):
            image_root = Path(source)
            with zipfile.ZipFile(args.images) as archive:
                archive.extractall(image_root)
        else:
            image_root = Path(source)
        paths = collect_images(image_root)
        if not paths:
            raise RuntimeError("no supported images found")

        backend = create_inference_backend(
            args.backend, args.model, args.device, args.input_size,
            args.confidence, True)
        backend.load_model()
        mapper = SemanticClassMapper(load_manifest(args.manifest))
        mapping = mapper.resolve_model_classes(backend.get_model_names())
        backend.warmup(args.warmup_count)
        role_ids = {role: tuple(mapping.get(role, ()))
                    for role in SEMANTIC_MASK_TOPICS}

        frames = []
        failures = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                failures.append({"file": path.name, "error": "decode_failed"})
                continue
            try:
                instances = backend.infer(image)
                masks = build_semantic_masks(
                    instances, role_ids, image.shape[:2], args.mask_threshold)
                frames.append((image, masks))
            except Exception as error:
                failures.append({"file": path.name, "error": repr(error)})

        if failures:
            report = {"summary": {"image_count": len(paths),
                                   "processed_count": len(frames),
                                   "pass": False},
                      "processing_failures": failures}
        else:
            raw_times, relative_times = capture_times(paths)
            planner_values = ros_parameters(
                args.planner_config, "camera_image_path_node")
            controller_values = ros_parameters(
                args.controller_config, "camera_pixel_controller_node")
            independent = run_pass(
                frames, paths, raw_times, relative_times, planner_values,
                controller_values, True)
            sequential = run_pass(
                frames, paths, raw_times, relative_times, planner_values,
                controller_values, False, overlay_dir)
            temporal = temporal_dropout_checks(
                frames, raw_times, planner_values)
            summary, similar_pairs = summarize(
                sequential, temporal, [frame[0] for frame in frames])
            report = {
                "summary": summary,
                "processing_failures": [],
                "sequential_frames": sequential,
                "independent_frames": independent,
                "temporal_dropout_checks": temporal,
                "similar_frame_pairs": similar_pairs,
                "configuration": {
                    "images": str(args.images),
                    "model": str(args.model),
                    "planner_config": str(args.planner_config),
                    "controller_config": str(args.controller_config),
                    "backend": args.backend,
                    "device": args.device,
                },
            }
        report_path = output/"report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"report={report_path}")
        print(f"overlays={overlay_dir}")
        return 0 if report["summary"].get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
