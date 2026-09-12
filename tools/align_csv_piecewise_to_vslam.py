#!/usr/bin/env python3
"""Build and audit a six-zone piecewise-SE(2) active-A map route.

The source CSV and RTAB-Map database are never modified.  Zone fits use only
the ordered, distinctive trajectory blocks established by the single-SE(2)
audit.  Ambiguous intersections and repeated loops are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/depth_hybrid_slam"))

from depth_hybrid_slam.piecewise_alignment import (  # noqa: E402
    AlignmentZone, apply_piecewise,
)
from align_csv_to_vslam import (  # noqa: E402
    ACTIVE_ORDER, MAP0_RANGES, THRESHOLDS, cumulative_distance,
    error_metrics, geometry_features, make_blocks, read_database,
    read_network, resample, robust_section_fit, sha256, transform, xy,
)


ZONE_DEFINITIONS = (
    ("ZONE_01_START", "START", ("START_A",), (1, 2)),
    ("ZONE_02_FIRST_CORNER", "RIGHT_ANGLE", ("START_A",), (3,)),
    ("ZONE_03_NARROW", "NARROW", ("COMMON_1",), (4,)),
    ("ZONE_04_S_CURVE", "S_MODE5", ("COMMON_1",), (5,)),
    ("ZONE_05_LONG_CORRIDOR", "LONG", ("COMMON_2",), (8,)),
    ("ZONE_06_END", "END", ("END_common", "END_AA"), (11,)),
)
OUTPUT_COLUMNS = (
    "segment_id", "segment_type", "point_index", "latitude", "longitude",
    "x_m", "y_m", "direction", "mode", "drive_level", "event",
    "from_node", "to_node", "source_x_m", "source_y_m", "map_yaw_rad",
    "alignment_zone",
)


def active_rows(segments):
    return [row for name in ACTIVE_ORDER for row in segments[name]]


def active_masks(rows, names, modes):
    allowed_names, allowed_modes = set(names), set(modes)
    return np.asarray([
        row["segment_id"] in allowed_names and int(row["mode"]) in allowed_modes
        for row in rows
    ])


def make_zones(rows, distances, blocks, initial):
    fits, anchor_ranges = [], []
    by_name = {block[0]: block for block in blocks}
    independent = {}
    for zone_id, block_name, names, modes in ZONE_DEFINITIONS:
        fitted, metrics, _ = robust_section_fit([by_name[block_name]], initial)
        mask = active_masks(rows, names, modes)
        if not mask.any():
            raise RuntimeError(f"empty anchor source for {zone_id}")
        indexes = np.flatnonzero(mask)
        anchor_range = (float(distances[indexes[0]]), float(distances[indexes[-1]]))
        anchor_ranges.append(anchor_range)
        fits.append(fitted)
        independent[zone_id] = {
            "anchor_id": block_name,
            "anchor_source": f"{'+'.join(names)} modes={list(modes)}",
            "anchor_start_s_m": anchor_range[0],
            "anchor_end_s_m": anchor_range[1],
            "theta_rad": math.radians(fitted[0]),
            "theta_deg": fitted[0], "tx_m": fitted[1], "ty_m": fitted[2],
            **metrics[block_name],
        }
    boundaries = [
        (anchor_ranges[index][1] + anchor_ranges[index + 1][0]) / 2.0
        for index in range(len(anchor_ranges) - 1)
    ]
    edges = [0.0, *boundaries, float(distances[-1])]
    zones = tuple(
        AlignmentZone(
            definition[0], edges[index], edges[index + 1],
            math.radians(fits[index][0]), fits[index][1], fits[index][2],
        )
        for index, definition in enumerate(ZONE_DEFINITIONS)
    )
    return zones, independent, anchor_ranges


def reference_tree(trajectories, map_route):
    references = [*trajectories.values(), resample(map_route, 0.05, 20000)]
    return cKDTree(np.vstack(references))


def route_metrics(rows, aligned, tree):
    errors = tree.query(aligned)[0]
    result = {"OVERALL": error_metrics(errors)}
    for name in ACTIVE_ORDER:
        mask = np.asarray([row["segment_id"] == name for row in rows])
        result[name] = error_metrics(errors[mask])
    end = np.asarray([row["segment_id"] in {"END_common", "END_AA"}
                      for row in rows])
    result["END"] = error_metrics(errors[end])
    before = np.flatnonzero(np.asarray([row["segment_id"] == "COMMON_1"
                                       for row in rows]))[-50:]
    after = np.flatnonzero(np.asarray([row["segment_id"] == "COMMON_2"
                                      for row in rows]))[:50]
    result["BEFORE_INTERSECTION_reference"] = error_metrics(errors[before])
    result["AFTER_INTERSECTION_reference"] = error_metrics(errors[after])
    return result, errors


def ordered_anchor_metrics(rows, aligned, blocks):
    output = {}
    for definition, block in zip(ZONE_DEFINITIONS, blocks):
        zone_id, block_name, names, modes = definition
        mask = active_masks(rows, names, modes)
        source = resample(aligned[mask])
        target = block[2]
        if block[3] == "target":
            errors = cKDTree(source).query(target)[0]
        else:
            errors = cKDTree(target).query(source)[0]
        output[zone_id] = {
            "anchor_id": block_name,
            "query_direction": block[3],
            **error_metrics(errors),
        }
    return output


def s_landmark_indices(points):
    distance, _, curvature = geometry_features(points)
    ratio = distance / distance[-1]
    first_range = np.flatnonzero((ratio >= 0.04) & (ratio <= 0.28))
    first = int(first_range[np.argmin(curvature[first_range])])
    second_range = np.flatnonzero((ratio >= 0.28) & (ratio <= 0.65))
    second = int(second_range[np.argmax(curvature[second_range])])
    between = np.arange(first + 1, second)
    changes = between[
        np.signbit(curvature[between - 1]) != np.signbit(curvature[between])]
    switch = int(changes[0] if len(changes) else
                 between[np.argmin(np.abs(curvature[between]))])
    return [0, first, switch, second, len(points) - 1]


def s_metrics(rows, aligned, trajectories):
    mask = active_masks(rows, ("COMMON_1",), (5,))
    indexes = np.flatnonzero(mask)
    source = aligned[indexes]
    lo, hi = MAP0_RANGES["S_MODE5"]
    target = resample(trajectories[0][lo:hi], 0.05, 10000)
    tree = cKDTree(target)
    errors = tree.query(source)[0]
    labels = ("entry", "first_curve_peak", "curvature_switch",
              "second_curve_peak", "exit")
    landmarks = s_landmark_indices(xy([rows[index] for index in indexes]))
    return error_metrics(errors), [
        {"name": label, "active_index": int(indexes[local]),
         "error_m": float(errors[local])}
        for label, local in zip(labels, landmarks)
    ]


def body_yaws(rows, points):
    headings = np.zeros(len(points))
    for index, row in enumerate(rows):
        if (index + 1 < len(points) and
                rows[index + 1]["direction"] == row["direction"] and
                rows[index + 1]["segment_id"] == row["segment_id"]):
            delta = points[index + 1] - points[index]
        elif (index > 0 and rows[index - 1]["direction"] == row["direction"] and
              rows[index - 1]["segment_id"] == row["segment_id"]):
            delta = points[index] - points[index - 1]
        else:
            delta = np.asarray([1.0, 0.0])
        yaw = math.atan2(delta[1], delta[0])
        if int(row["direction"]) < 0:
            yaw += math.pi
        headings[index] = math.atan2(math.sin(yaw), math.cos(yaw))
    return headings


def motion_chunks(rows):
    start = 0
    for index in range(1, len(rows) + 1):
        if (index == len(rows) or
                (rows[index]["segment_id"], rows[index]["direction"]) !=
                (rows[index - 1]["segment_id"], rows[index - 1]["direction"])):
            yield start, index
            start = index


def steering_demand(rows, points, wheelbase_m=0.73):
    """Measure physical curvature after 0.20 m resampling and 2 m smoothing.

    Segments and forward/reverse blocks are handled independently so STOP-like
    coincident boundary points are not misclassified as an infinite turn.
    """
    demands = []
    for start, end in motion_chunks(rows):
        section = points[start:end]
        if len(section) < 5 or cumulative_distance(section)[-1] < 3.0:
            continue
        sampled = resample(section, 0.20, 100000)
        window = min(11, len(sampled) if len(sampled) % 2 else len(sampled) - 1)
        if window < 5:
            continue
        smooth = np.c_[
            savgol_filter(sampled[:, 0], window, 3),
            savgol_filter(sampled[:, 1], window, 3),
        ]
        distance = cumulative_distance(smooth)
        heading = np.unwrap(np.arctan2(
            np.gradient(smooth[:, 1]), np.gradient(smooth[:, 0])))
        curvature = savgol_filter(
            np.gradient(heading) / np.maximum(np.gradient(distance), 1e-6),
            window, 3)
        edge = min(window // 2, len(sampled) // 4)
        selected = curvature[edge:len(curvature) - edge]
        demands.extend(np.abs(np.degrees(np.arctan(wheelbase_m * selected))))
    values = np.asarray(demands)
    return {
        "method": "per-motion-block 0.20m resample, 2.0m Savitzky-Golay",
        "max_abs_required_steering_deg": float(values.max()),
        "p99_abs_required_steering_deg": float(np.percentile(values, 99)),
        "limit_deg": 22.0,
        "passed": bool(values.max() <= 22.0),
    }


def continuity_metrics(rows, points, distances, zones, schedule):
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    yaw = body_yaws(rows, points)
    yaw_delta = np.abs(np.arctan2(
        np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
    same_motion = np.asarray([
        rows[index]["segment_id"] == rows[index + 1]["segment_id"] and
        rows[index]["direction"] == rows[index + 1]["direction"]
        for index in range(len(rows) - 1)
    ])
    boundary_gaps, boundary_yaw = [], []
    for left in zones[:-1]:
        index = int(np.searchsorted(distances, left.end_s_m))
        index = min(max(index, 1), len(points) - 1)
        boundary_gaps.append(float(gaps[index - 1]))
        if 1 <= index < len(points) - 1:
            before = math.atan2(*(points[index] - points[index - 1])[::-1])
            after = math.atan2(*(points[index + 1] - points[index])[::-1])
            boundary_yaw.append(abs(math.atan2(
                math.sin(after - before), math.cos(after - before))))
    transforms = np.asarray([entry[:3] for entry in schedule])
    theta_step = np.abs(np.arctan2(
        np.sin(np.diff(transforms[:, 0])), np.cos(np.diff(transforms[:, 0]))))
    translation_step = np.linalg.norm(np.diff(transforms[:, 1:3], axis=0), axis=1)
    return {
        "max_waypoint_gap_m": float(gaps.max()),
        "max_segment_boundary_gap_m": float(max(
            gap for index, gap in enumerate(gaps) if not same_motion[index])),
        "max_transform_boundary_gap_m": float(max(boundary_gaps)),
        "max_yaw_jump_deg": float(np.degrees(yaw_delta[same_motion]).max()),
        "max_transform_boundary_yaw_jump_deg": float(np.degrees(max(boundary_yaw))),
        "max_schedule_theta_step_deg": float(np.degrees(theta_step.max())),
        "max_schedule_translation_step_m": float(translation_step.max()),
    }


def write_aligned_csv(path, rows, source, aligned, yaws, schedule):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row, raw, mapped, yaw, state in zip(rows, source, aligned, yaws, schedule):
            output = {name: row[name] for name in OUTPUT_COLUMNS[:13]}
            output.update({
                "x_m": f"{mapped[0]:.9f}", "y_m": f"{mapped[1]:.9f}",
                "source_x_m": f"{raw[0]:.9f}",
                "source_y_m": f"{raw[1]:.9f}",
                "map_yaw_rad": f"{yaw:.12f}", "alignment_zone": state[3],
            })
            writer.writerow(output)


def zone_records(zones, independent, anchor_metrics):
    records = []
    for zone in zones:
        fitted, metrics = independent[zone.zone_id], anchor_metrics[zone.zone_id]
        records.append({
            "id": zone.zone_id,
            "start_s_m": zone.start_s_m, "end_s_m": zone.end_s_m,
            "theta_rad": zone.theta_rad,
            "theta_deg": math.degrees(zone.theta_rad),
            "tx_m": zone.tx_m, "ty_m": zone.ty_m,
            "anchor_source": fitted["anchor_source"],
            "anchor_start_s_m": fitted["anchor_start_s_m"],
            "anchor_end_s_m": fitted["anchor_end_s_m"],
            "rmse_m": metrics["rmse_m"], "median_error_m": metrics["median_m"],
            "p95_error_m": metrics["p95_m"], "max_error_m": metrics["max_m"],
            "validated": bool(
                metrics["rmse_m"] <= THRESHOLDS["rmse_m"] and
                metrics["median_m"] <= THRESHOLDS["median_m"] and
                metrics["p95_m"] <= THRESHOLDS["p95_m"] and
                metrics["max_m"] <= THRESHOLDS["distinct_max_m"]),
        })
    return records


def plot_overview(path, trajectories, map_route, raw, single, piecewise, zones,
                  distances):
    fig, axis = plt.subplots(figsize=(13, 10))
    for trajectory in trajectories.values():
        axis.plot(*trajectory.T, color="0.78", linewidth=0.6)
    axis.plot(*trajectories[0].T, color="black", linewidth=1.0,
              label="VSLAM map_id=0 trajectory")
    axis.plot(*map_route.T, "o-", color="tab:green", markersize=1.5,
              linewidth=0.7, label="v10 map route")
    axis.plot(*single.T, color="tab:orange", linewidth=0.8,
              label="single SE(2)")
    axis.plot(*piecewise.T, color="tab:blue", linewidth=1.0,
              label="piecewise SE(2)")
    for zone in zones[:-1]:
        index = int(np.searchsorted(distances, zone.end_s_m))
        axis.scatter(*piecewise[index], color="red", s=24, zorder=4)
        axis.text(*piecewise[index], zone.zone_id.replace("ZONE_", "Z"),
                  color="red", fontsize=7)
    axis.axis("equal")
    axis.grid(True)
    axis.legend(fontsize=8)
    axis.set_title("Single vs six-zone piecewise alignment (red: zone boundaries)")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_focus(path, rows, trajectories, single, piecewise):
    selections = (
        ("START", active_masks(rows, ("START_A",), (1, 2))),
        ("RIGHT ANGLE", active_masks(rows, ("START_A",), (3,))),
        ("NARROW", active_masks(rows, ("COMMON_1",), (4,))),
        ("S CURVE", active_masks(rows, ("COMMON_1",), (5,))),
        ("COMMON_2 LONG", active_masks(rows, ("COMMON_2",), (8,))),
        ("END", active_masks(rows, ("END_common", "END_AA"), (11,))),
    )
    targets = [
        trajectories[0][slice(*MAP0_RANGES["START"])],
        trajectories[0][slice(*MAP0_RANGES["RIGHT_ANGLE"])],
        trajectories[0][slice(*MAP0_RANGES["NARROW"])],
        trajectories[0][slice(*MAP0_RANGES["S_MODE5"])],
        trajectories[0][slice(*MAP0_RANGES["LONG"])], trajectories[5],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, (name, mask), target in zip(axes.ravel(), selections, targets):
        axis.plot(*target.T, "k.-", markersize=2, linewidth=1.0, label="VSLAM")
        axis.plot(*single[mask].T, color="tab:orange", linewidth=0.9,
                  label="single")
        axis.plot(*piecewise[mask].T, color="tab:blue", linewidth=1.0,
                  label="piecewise")
        axis.set_title(name)
        axis.axis("equal")
        axis.grid(True)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_one_focus(path, title, mask, target, single, piecewise):
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.plot(*target.T, "k.-", markersize=2, linewidth=1.1,
              label="ordered VSLAM reference")
    axis.plot(*single[mask].T, color="tab:orange", linewidth=1.0,
              label="single SE(2)")
    axis.plot(*piecewise[mask].T, color="tab:blue", linewidth=1.1,
              label="piecewise SE(2)")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    axis.set_title(title)
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error(path, distances, single_errors, piecewise_errors, zones):
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(distances, single_errors, color="tab:orange", linewidth=0.6,
              label="single SE(2)")
    axis.plot(distances, piecewise_errors, color="tab:blue", linewidth=0.6,
              label="piecewise SE(2)")
    for zone in zones[:-1]:
        axis.axvline(zone.end_s_m, color="red", alpha=0.35, linewidth=0.8)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                 label="1.0 m corridor")
    axis.set_ylim(bottom=0.0)
    axis.grid(True)
    axis.legend()
    axis.set_title("Alignment error vs active-A route distance")
    axis.set_xlabel("source route distance s [m]")
    axis.set_ylabel("nearest validation-reference error [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_schedule(path, distances, schedule, zones):
    values = np.asarray([entry[:3] for entry in schedule])
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ((np.degrees(values[:, 0]), "theta [deg]"),
              (values[:, 1], "tx [m]"), (values[:, 2], "ty [m]"))
    for axis, (series, label) in zip(axes, labels):
        axis.plot(distances, series, color="tab:blue", linewidth=0.9)
        for zone in zones[:-1]:
            axis.axvline(zone.end_s_m, color="red", alpha=0.35, linewidth=0.8)
        axis.grid(True)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("source route distance s [m]")
    fig.suptitle("Piecewise transform schedule with smoothstep boundary blends")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path,
                        default=ROOT / "routes/network/route_network_segmented.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT /
                        "routes/network/route_network_segmented.metadata.yaml")
    parser.add_argument("--database", type=Path, default=ROOT /
                        "maps/merged_competition_level_aligned_v10/rtabmap.db")
    parser.add_argument("--map-route", type=Path,
                        default=ROOT / "routes/v10/START_A.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT /
                        "routes/network/route_network_segmented_aligned.csv")
    parser.add_argument("--output-metadata", type=Path, default=ROOT /
                        "routes/network/route_network_segmented_aligned.metadata.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT /
                        "analysis/csv_vslam_piecewise_alignment")
    parser.add_argument("--blend-distance-m", type=float, default=8.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    _, segments = read_network(args.route)
    rows = active_rows(segments)
    source = xy(rows)
    distances = cumulative_distance(source)
    trajectories = read_database(args.database)
    blocks = make_blocks(segments, trajectories)
    source_metadata = yaml.safe_load(args.metadata.read_text(encoding="utf-8"))
    configured = source_metadata["csv_to_map"]
    initial = (float(configured["yaw_deg"]), float(configured["x_m"]),
               float(configured["y_m"]))
    zones, independent, _ = make_zones(rows, distances, blocks, initial)
    piecewise, schedule = apply_piecewise(
        source, distances, zones, args.blend_distance_m)
    single = transform(source, *initial)
    yaws = body_yaws(rows, piecewise)

    with args.map_route.open(newline="", encoding="utf-8") as stream:
        map_route = np.asarray([(float(row["x"]), float(row["y"]))
                                for row in csv.DictReader(stream)])
    tree = reference_tree(trajectories, map_route)
    blend_diagnostics = {}
    for trial_distance in (4.0, 6.0, 8.0):
        trial_points, _ = apply_piecewise(source, distances, zones, trial_distance)
        trial_errors = tree.query(trial_points)[0]
        trial_steering = steering_demand(rows, trial_points)
        blend_diagnostics[f"{trial_distance:.1f}m"] = {
            "rmse_m": error_metrics(trial_errors)["rmse_m"],
            "max_abs_required_steering_deg":
                trial_steering["max_abs_required_steering_deg"],
            "steering_passed": trial_steering["passed"],
        }
    single_metrics, single_errors = route_metrics(rows, single, tree)
    piecewise_metrics, piecewise_errors = route_metrics(rows, piecewise, tree)
    anchor_metrics = ordered_anchor_metrics(rows, piecewise, blocks)
    zone_values = zone_records(zones, independent, anchor_metrics)
    s_summary, s_landmarks = s_metrics(rows, piecewise, trajectories)
    continuity = continuity_metrics(rows, piecewise, distances, zones, schedule)
    steering = steering_demand(rows, piecewise)

    checks = {
        "overall_median": piecewise_metrics["OVERALL"]["median_m"] <= 0.30,
        "overall_rmse": piecewise_metrics["OVERALL"]["rmse_m"] <= 0.50,
        "overall_p95": piecewise_metrics["OVERALL"]["p95_m"] <= 0.80,
        "all_distinct_zones": all(zone["validated"] for zone in zone_values),
        "s_section": (s_summary["rmse_m"] <= 0.50 and
                      s_summary["median_m"] <= 0.30 and
                      s_summary["p95_m"] <= 0.80 and
                      s_summary["max_m"] <= 1.00),
        "waypoint_continuity": continuity["max_waypoint_gap_m"] <= 1.25,
        "steering_limit": steering["passed"],
        "improves_single_rmse": (piecewise_metrics["OVERALL"]["rmse_m"] <
                                 single_metrics["OVERALL"]["rmse_m"]),
    }
    accepted = all(checks.values())

    write_aligned_csv(args.output_csv, rows, source, piecewise, yaws, schedule)
    output_hash = sha256(args.output_csv)
    metadata = {
        "format_version": 4,
        "finalized": True,
        "route_name": "competition_active_a_piecewise_candidate",
        "route_csv": str(args.output_csv.resolve()),
        "route_sha256": output_hash,
        "rtabmap_db": str(args.database.resolve()),
        "rtabmap_db_sha256": sha256(args.database),
        "route_coordinate_frame": "map",
        "source_coordinate_frame": "gps_local_enu",
        "frame_id": "map", "units": "m",
        "alignment": {
            "method": "piecewise_se2_ordered_distinctive_sections",
            "required_for_runtime": True,
            "validated": accepted,
            "route_length_m": float(distances[-1]),
            "blend_distance_m": args.blend_distance_m,
            "interpolation": "centered cubic smoothstep; shortest angular distance",
            "rmse_m": piecewise_metrics["OVERALL"]["rmse_m"],
            "median_error_m": piecewise_metrics["OVERALL"]["median_m"],
            "p95_error_m": piecewise_metrics["OVERALL"]["p95_m"],
            "max_error_m": piecewise_metrics["OVERALL"]["max_m"],
            "zones": zone_values,
            "approval_checks": checks,
        },
        "selection": {
            "default_branch": "A", "external_branch_topic": False,
            "active_segments": list(ACTIVE_ORDER),
            "excluded_b_segments": ["START_B", "T_B", "V_B", "END_AB"],
        },
        "continuity": continuity,
        "vehicle_geometry": steering,
    }
    args.output_metadata.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    plot_overview(args.output_dir / "piecewise_overview.png", trajectories,
                  map_route, source, single, piecewise, zones, distances)
    plot_focus(args.output_dir / "zone_zooms.png", rows, trajectories,
               single, piecewise)
    focus_definitions = (
        ("start_zoom.png", "START", active_masks(rows, ("START_A",), (1, 2)),
         trajectories[0][slice(*MAP0_RANGES["START"])]),
        ("right_angle_zoom.png", "RIGHT ANGLE",
         active_masks(rows, ("START_A",), (3,)),
         trajectories[0][slice(*MAP0_RANGES["RIGHT_ANGLE"])]),
        ("s_curve_zoom.png", "MODE-5 S CURVE",
         active_masks(rows, ("COMMON_1",), (5,)),
         trajectories[0][slice(*MAP0_RANGES["S_MODE5"])]),
        ("end_zoom.png", "END",
         active_masks(rows, ("END_common", "END_AA"), (11,)), trajectories[5]),
    )
    for filename, title, mask, target in focus_definitions:
        plot_one_focus(args.output_dir / filename, title, mask, target,
                       single, piecewise)
    plot_error(args.output_dir / "error_vs_distance.png", distances,
               single_errors, piecewise_errors, zones)
    plot_schedule(args.output_dir / "transform_vs_distance.png", distances,
                  schedule, zones)

    report = {
        "decision": "PASS" if accepted else "NEED_MAP_FRAME_ROUTE_RERECORD",
        "database_read_only": True,
        "source": {
            "csv": str(args.route.resolve()), "csv_sha256": sha256(args.route),
            "database": str(args.database.resolve()),
            "database_sha256": sha256(args.database),
            "v10_map_route": str(args.map_route.resolve()),
            "usable_synchronized_map_gps_recording_found": False,
            "bag_audit": "camera probe/stationary MCAP only; no route map-pose recording",
        },
        "method": {
            "zone_count": len(zones), "blend_distance_m": args.blend_distance_m,
            "fit": "per-zone 15% trimmed Huber ordered-section SE(2)",
            "interpolation": "centered cubic smoothstep and shortest-angle rotation",
            "whole_route_nearest_neighbor_used_for_fit": False,
            "occupancy_wall_icp_used": False,
            "blend_diagnostics": blend_diagnostics,
        },
        "single_se2": {"transform": {
            "theta_deg": initial[0], "theta_rad": math.radians(initial[0]),
            "tx_m": initial[1], "ty_m": initial[2]}, "metrics": single_metrics},
        "piecewise": {
            "metrics": piecewise_metrics, "zones": zone_values,
            "anchor_metrics": anchor_metrics,
            "s_curve": {"metrics": s_summary, "landmarks": s_landmarks},
            "continuity": continuity, "vehicle_geometry": steering,
        },
        "approval_checks": checks,
        "metadata_validated": accepted,
        "reason": (None if accepted else
                   "Piecewise fitting still misses overall/zone criteria; a synchronized "
                   "map-frame route recording is required."),
        "artifacts": {
            "aligned_csv": str(args.output_csv.resolve()),
            "aligned_metadata": str(args.output_metadata.resolve()),
            "report_markdown": str((args.output_dir / "REPORT.md").resolve()),
            "overview": str((args.output_dir / "piecewise_overview.png").resolve()),
            "zone_zooms": str((args.output_dir / "zone_zooms.png").resolve()),
            "start_zoom": str((args.output_dir / "start_zoom.png").resolve()),
            "right_angle_zoom": str((args.output_dir /
                                     "right_angle_zoom.png").resolve()),
            "s_curve_zoom": str((args.output_dir / "s_curve_zoom.png").resolve()),
            "end_zoom": str((args.output_dir / "end_zoom.png").resolve()),
            "error_vs_distance": str((args.output_dir / "error_vs_distance.png").resolve()),
            "transform_vs_distance": str((args.output_dir /
                                          "transform_vs_distance.png").resolve()),
        },
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "single": single_metrics["OVERALL"],
        "piecewise": piecewise_metrics["OVERALL"],
        "s_curve": s_summary,
        "continuity": continuity,
        "vehicle_geometry": steering,
        "approval_checks": checks,
        "report": str(report_path),
    }, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
