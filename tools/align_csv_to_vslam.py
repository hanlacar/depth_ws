#!/usr/bin/env python3
"""Audit GPS-local CSV to RTAB-Map alignment using ordered route sections.

The database is opened read-only.  This intentionally does not fit the CSV to
occupancy-grid walls and does not run unconstrained whole-route nearest-neighbor
ICP.  Each source block is paired with a temporally contiguous VSLAM trajectory
block whose geometry and traversal order were independently identified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
import yaml


ACTIVE_ORDER = (
    "START_A", "COMMON_1", "T_foword", "T_A", "COMMON_2", "V_A",
    "END_common", "END_AA",
)

# Indices are in map_id=0 timestamp order, not nearest-neighbor discoveries.
# The ranges were verified from route order and curvature-sign sequences.  A
# little endpoint overlap is deliberate so projection at a boundary is stable.
MAP0_RANGES = {
    "START": (42, 152),
    "RIGHT_ANGLE": (151, 332),
    "NARROW": (331, 417),
    "S_MODE5": (415, 534),
    "LONG": (723, 998),
}

THRESHOLDS = {
    "median_m": 0.30,
    "rmse_m": 0.50,
    "p95_m": 0.80,
    "distinct_max_m": 1.00,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_from_blob(blob: bytes | None) -> np.ndarray | None:
    if blob is None or len(blob) != 48:
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(struct.unpack("<12f", blob)).reshape(3, 4)
    return matrix


def read_database(path: Path) -> dict[int, np.ndarray]:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("RTAB-Map database integrity check failed")
    grouped: dict[int, list[tuple[float, float]]] = {}
    query = "SELECT map_id,pose FROM Node ORDER BY map_id,stamp,id"
    for map_id, blob in connection.execute(query):
        matrix = matrix_from_blob(blob)
        if matrix is not None:
            grouped.setdefault(int(map_id), []).append(
                (float(matrix[0, 3]), float(matrix[1, 3])))
    connection.close()
    return {key: np.asarray(value) for key, value in grouped.items()}


def read_network(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    segments: dict[str, list[dict]] = {}
    for row in rows:
        segments.setdefault(row["segment_id"], []).append(row)
    missing = [name for name in ("AAA_BASE",) + ACTIVE_ORDER
               if name not in segments]
    if missing:
        raise ValueError("missing route segments: " + ", ".join(missing))
    return rows, segments


def xy(rows: list[dict]) -> np.ndarray:
    return np.asarray([(float(row["x_m"]), float(row["y_m"]))
                       for row in rows], dtype=np.float64)


def select(segments: dict[str, list[dict]], names, modes=None) -> np.ndarray:
    if isinstance(names, str):
        names = (names,)
    selected = []
    allowed = None if modes is None else set(modes)
    for name in names:
        selected.extend(row for row in segments[name]
                        if allowed is None or int(row["mode"]) in allowed)
    return xy(selected)


def transform(points: np.ndarray, theta_deg: float, tx: float,
              ty: float, scale: float = 1.0) -> np.ndarray:
    theta = math.radians(theta_deg)
    rotation = np.asarray([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ])
    return scale * points @ rotation + np.asarray([tx, ty])


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]


def resample(points: np.ndarray, spacing: float = 0.15,
             maximum_count: int = 200) -> np.ndarray:
    distance = cumulative_distance(points)
    unique, indices = np.unique(distance, return_index=True)
    points = points[indices]
    if unique[-1] <= 0.0:
        return points[:1]
    query = np.arange(0.0, unique[-1] + spacing * 0.1, spacing)
    result = np.c_[np.interp(query, unique, points[:, 0]),
                   np.interp(query, unique, points[:, 1])]
    if len(result) > maximum_count:
        indexes = np.linspace(0, len(result) - 1, maximum_count).astype(int)
        result = result[indexes]
    return result


def fit_rigid(source: np.ndarray, target: np.ndarray,
              weights: np.ndarray | None = None,
              similarity: bool = False) -> tuple[float, float, float, float]:
    if weights is None:
        weights = np.ones(len(source))
    weights = weights / weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    left, singular, right = np.linalg.svd(
        (source_zero * weights[:, None]).T @ target_zero)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    scale = 1.0
    if similarity:
        denominator = np.sum(weights[:, None] * source_zero * source_zero)
        scale = float(singular.sum() / denominator)
    translation = target_center - scale * source_center @ rotation
    theta = math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
    return theta, float(translation[0]), float(translation[1]), scale


def error_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(errors.max()),
        "inside_1m_fraction": float(np.mean(errors <= 1.0)),
    }


def block_correspondences(block, candidate):
    name, source, target, query_side = block
    projected_source = transform(source, *candidate[:3])
    if query_side == "source":
        errors, indexes = cKDTree(target).query(projected_source)
        matched_source, matched_target = source, target[indexes]
    else:
        errors, indexes = cKDTree(projected_source).query(target)
        matched_source, matched_target = source[indexes], target
    return name, matched_source, matched_target, errors


def robust_section_fit(blocks, initial, iterations: int = 30):
    candidate = (*initial, 1.0)
    for _ in range(iterations):
        all_source, all_target, all_weights = [], [], []
        for block in blocks:
            _, source, target, errors = block_correspondences(block, candidate)
            cutoff = np.percentile(errors, 85)
            robust_scale = max(float(np.median(errors) * 1.4826), 0.10)
            weights = np.minimum(1.0, robust_scale / np.maximum(errors, 1e-9))
            weights *= errors <= cutoff
            weights /= max(float(weights.sum()), 1e-12)  # equal block weight
            all_source.append(source)
            all_target.append(target)
            all_weights.append(weights)
        updated = fit_rigid(
            np.vstack(all_source), np.vstack(all_target),
            np.concatenate(all_weights))
        delta = np.linalg.norm(np.asarray(updated[:3]) - np.asarray(candidate[:3]))
        candidate = updated
        if delta < 1e-10:
            break
    metrics = {}
    pairs = []
    for block in blocks:
        name, source, target, errors = block_correspondences(block, candidate)
        metrics[name] = error_metrics(errors)
        pairs.append((source, target, errors))
    return candidate, metrics, pairs


def make_blocks(segments, trajectories):
    map0 = trajectories[0]
    end_target = trajectories[5]
    definitions = (
        ("START", select(segments, "START_A", (1, 2)),
         map0[slice(*MAP0_RANGES["START"])], "source"),
        ("RIGHT_ANGLE", select(segments, "START_A", (3,)),
         map0[slice(*MAP0_RANGES["RIGHT_ANGLE"])], "source"),
        ("NARROW", select(segments, "COMMON_1", (4,)),
         map0[slice(*MAP0_RANGES["NARROW"])], "source"),
        ("S_MODE5", select(segments, "COMMON_1", (5,)),
         map0[slice(*MAP0_RANGES["S_MODE5"])], "source"),
        ("LONG", select(segments, "COMMON_2", (8,)),
         map0[slice(*MAP0_RANGES["LONG"])], "source"),
        # map_id=5 is an independent, temporally ordered recording of only a
        # subset of the A ending, so target-to-source distance is intentional.
        ("END", select(segments, ("END_common", "END_AA")),
         end_target, "target"),
    )
    return [(name, resample(source), resample(target), query)
            for name, source, target, query in definitions]


def geometry_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = cumulative_distance(points)
    count = len(points)
    window = min(31, count if count % 2 else count - 1)
    if window < 5:
        smooth = points
    else:
        smooth = np.c_[savgol_filter(points[:, 0], window, 3),
                       savgol_filter(points[:, 1], window, 3)]
    dx = np.gradient(smooth[:, 0])
    dy = np.gradient(smooth[:, 1])
    heading = np.unwrap(np.arctan2(dy, dx))
    ds = np.maximum(np.gradient(distance), 1e-6)
    curvature = np.gradient(heading) / ds
    if window >= 5:
        curvature = savgol_filter(curvature, window, 3)
    return distance, heading, curvature


def write_active_features(path: Path, segments) -> None:
    fields = ("active_index", "x", "y", "heading_rad", "curvature_1pm",
              "s_m", "segment_id", "mode", "direction", "event")
    output = []
    global_s = 0.0
    previous = None
    active_index = 0
    for name in ACTIVE_ORDER:
        rows = segments[name]
        points = xy(rows)
        local_s, heading, curvature = geometry_features(points)
        if previous is not None:
            global_s += float(np.linalg.norm(points[0] - previous))
        for index, row in enumerate(rows):
            output.append({
                "active_index": active_index,
                "x": float(points[index, 0]), "y": float(points[index, 1]),
                "heading_rad": float(heading[index]),
                "curvature_1pm": float(curvature[index]),
                "s_m": global_s + float(local_s[index]),
                "segment_id": name, "mode": int(row["mode"]),
                "direction": int(row["direction"]), "event": row["event"],
            })
            active_index += 1
        global_s += float(local_s[-1])
        previous = points[-1]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def detect_geometry(segments) -> dict:
    """Detect long low-curvature runs, corners, and mode-5 sign changes."""
    straight_runs = []
    corners = []
    global_offset = 0.0
    for name in ACTIVE_ORDER:
        rows = segments[name]
        points = xy(rows)
        distance, heading, curvature = geometry_features(points)
        straight = np.abs(curvature) <= 0.015
        start = None
        for index in range(len(points) + 1):
            enabled = index < len(points) and bool(straight[index])
            if enabled and start is None:
                start = index
            if not enabled and start is not None:
                end = index - 1
                length = float(distance[end] - distance[start])
                if length >= 8.0:
                    delta = points[end] - points[start]
                    straight_runs.append({
                        "segment_id": name,
                        "mode": int(rows[(start + end) // 2]["mode"]),
                        "start_point_index": int(rows[start]["point_index"]),
                        "end_point_index": int(rows[end]["point_index"]),
                        "s_start_m": global_offset + float(distance[start]),
                        "s_end_m": global_offset + float(distance[end]),
                        "length_m": length,
                        "heading_deg": math.degrees(math.atan2(delta[1], delta[0])),
                    })
                start = None
        candidates = np.flatnonzero(np.abs(curvature) >= 0.08)
        ranked = sorted(candidates, key=lambda i: abs(curvature[i]), reverse=True)
        chosen = []
        for index in ranked:
            if all(abs(distance[index] - distance[other]) >= 5.0
                   for other in chosen):
                chosen.append(int(index))
        for index in chosen[:4]:
            corners.append({
                "segment_id": name, "mode": int(rows[index]["mode"]),
                "point_index": int(rows[index]["point_index"]),
                "s_m": global_offset + float(distance[index]),
                "curvature_1pm": float(curvature[index]),
                "heading_deg": math.degrees(float(heading[index])),
            })
        global_offset += float(distance[-1])
    mode5 = select(segments, "COMMON_1", (5,))
    mode5_s, _, mode5_k = geometry_features(mode5)
    signs = np.signbit(mode5_k)
    changes = np.flatnonzero(signs[:-1] != signs[1:]) + 1
    sign_changes = [{
        "point_offset": int(index), "s_in_mode_m": float(mode5_s[index]),
        "curvature_before_1pm": float(mode5_k[index - 1]),
        "curvature_after_1pm": float(mode5_k[index]),
    } for index in changes]
    return {
        "long_straights": sorted(straight_runs, key=lambda item: item["length_m"],
                                 reverse=True),
        "large_corners": sorted(corners, key=lambda item: abs(item["curvature_1pm"]),
                                reverse=True),
        "mode5_curvature_sign_changes": sign_changes,
    }


def s_landmarks(source: np.ndarray) -> tuple[list[str], np.ndarray]:
    distance, _, curvature = geometry_features(source)
    ratio = distance / distance[-1]
    # Exclude the filter boundary so "entry" and the first curvature peak can
    # never collapse to the same point.
    first_range = np.flatnonzero((ratio >= 0.04) & (ratio <= 0.28))
    first = int(first_range[np.argmin(curvature[first_range])])
    second_range = np.flatnonzero((ratio >= 0.28) & (ratio <= 0.65))
    second = int(second_range[np.argmax(curvature[second_range])])
    between = np.arange(first + 1, second)
    changes = between[np.signbit(curvature[between - 1]) !=
                      np.signbit(curvature[between])]
    switch = int(changes[0] if len(changes) else
                 between[np.argmin(np.abs(curvature[between]))])
    indexes = [0, first, switch, second, len(source) - 1]
    labels = ["entry", "first_curve_peak", "curvature_switch",
              "second_curve_peak", "exit"]
    return labels, source[indexes]


def mapping_diagnostics(pairs) -> tuple[list[dict], dict]:
    source = np.vstack([item[0] for item in pairs])
    target = np.vstack([item[1] for item in pairs])
    errors = np.concatenate([item[2] for item in pairs])
    keep = errors <= np.percentile(errors, 85)
    source, target = source[keep], target[keep]
    mappings = {
        "(x,y)": ((1, 0), (0, 1)),
        "(x,-y)": ((1, 0), (0, -1)),
        "(-x,y)": ((-1, 0), (0, 1)),
        "(-x,-y)": ((-1, 0), (0, -1)),
        "(y,x)": ((0, 1), (1, 0)),
        "(y,-x)": ((0, 1), (-1, 0)),
        "(-y,x)": ((0, -1), (1, 0)),
        "(-y,-x)": ((0, -1), (-1, 0)),
    }
    results = []
    for name, values in mappings.items():
        axis = np.asarray(values, dtype=np.float64)
        changed = source @ axis.T
        fitted = fit_rigid(changed, target)
        predicted = transform(changed, *fitted[:3])
        results.append({
            "mapping": name, "determinant": float(np.linalg.det(axis)),
            **error_metrics(np.linalg.norm(predicted - target, axis=1)),
        })
    similarity = fit_rigid(source, target, similarity=True)
    sim_error = np.linalg.norm(transform(source, *similarity) - target, axis=1)
    return results, {
        "theta_deg": similarity[0], "tx_m": similarity[1],
        "ty_m": similarity[2], "scale": similarity[3],
        **error_metrics(sim_error),
    }


def evaluation_metrics(segments, trajectories, candidate, map_route=None):
    references = list(trajectories.values())
    if map_route is not None:
        references.append(resample(map_route, 0.05, 20000))
    all_target = np.vstack(references)
    tree = cKDTree(all_target)
    result = {}
    all_errors = []
    for name in ACTIVE_ORDER:
        errors = tree.query(transform(select(segments, name), *candidate[:3]))[0]
        result[name] = error_metrics(errors)
        all_errors.append(errors)
    result["END"] = error_metrics(np.r_[
        tree.query(transform(select(segments, "END_common"), *candidate[:3]))[0],
        tree.query(transform(select(segments, "END_AA"), *candidate[:3]))[0],
    ])
    before = select(segments, "COMMON_1")[-50:]
    after = select(segments, "COMMON_2")[:50]
    result["BEFORE_INTERSECTION_reference"] = error_metrics(
        tree.query(transform(before, *candidate[:3]))[0])
    result["AFTER_INTERSECTION_reference"] = error_metrics(
        tree.query(transform(after, *candidate[:3]))[0])
    s_source = select(segments, "COMMON_1", (5,))
    lo, hi = MAP0_RANGES["S_MODE5"]
    s_target = resample(trajectories[0][lo:hi], 0.05, 10000)
    result["S_MODE5"] = error_metrics(cKDTree(s_target).query(
        transform(s_source, *candidate[:3]))[0])
    result["OVERALL"] = error_metrics(np.concatenate(all_errors))
    return result


def plot_overview(path, segments, trajectories, map_route, current, candidate,
                  blocks):
    active = np.vstack([select(segments, name) for name in ACTIVE_ORDER])
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].plot(trajectories[0][:, 0], trajectories[0][:, 1], color="black",
                 linewidth=1.0, label="VSLAM map_id=0 trajectory")
    axes[0].plot(active[:, 0], active[:, 1], color="tab:purple",
                 linewidth=0.8, label="raw GPS-local active A (numeric axes)")
    axes[0].set_title("Before SE(2): coordinate frames are intentionally separate")
    for points in trajectories.values():
        axes[1].plot(points[:, 0], points[:, 1], color="0.75", linewidth=0.7)
    axes[1].plot(trajectories[0][:, 0], trajectories[0][:, 1], color="black",
                 linewidth=1.0, label="VSLAM map_id=0")
    if map_route is not None:
        axes[1].plot(map_route[:, 0], map_route[:, 1], "o-", color="tab:green",
                     markersize=2, linewidth=0.8, label="v10 map route")
    current_points = transform(active, *current)
    candidate_points = transform(active, *candidate[:3])
    axes[1].plot(current_points[:, 0], current_points[:, 1], color="tab:orange",
                 linewidth=0.9, label="retained provisional")
    axes[1].plot(candidate_points[:, 0], candidate_points[:, 1], "--",
                 color="tab:cyan", linewidth=0.9, label="rejected robust candidate")
    for name, _, target, _ in blocks:
        center = target[len(target) // 2]
        axes[1].scatter(*center, s=22, color="red")
        axes[1].text(*center, name, fontsize=7, color="red")
    axes[1].set_title("VSLAM trajectories, ordered anchors, and two transforms")
    for axis in axes:
        axis.axis("equal")
        axis.grid(True)
        axis.legend(fontsize=8)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_s_zoom(path, segments, trajectories, current, candidate,
                landmark_report):
    source = select(segments, "COMMON_1", (5,))
    lo, hi = MAP0_RANGES["S_MODE5"]
    target = trajectories[0][lo:hi]
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.plot(target[:, 0], target[:, 1], "k.-", markersize=2,
              linewidth=1.2, label="VSLAM ordered S section")
    axis.plot(*transform(source, *current).T, color="tab:orange",
              linewidth=1.2, label="retained provisional")
    axis.plot(*transform(source, *candidate[:3]).T, "--", color="tab:cyan",
              linewidth=1.2, label="rejected robust candidate")
    _, landmarks = s_landmarks(source)
    shown = transform(landmarks, *candidate[:3])
    axis.scatter(shown[:, 0], shown[:, 1], color="red", zorder=3)
    for point, item in zip(shown, landmark_report):
        axis.text(point[0], point[1], f"{item['name']} {item['error_m']:.2f}m",
                  fontsize=8, color="red")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()
    axis.set_title("Mode-5 S section: curvature landmarks and cross-track errors")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_anchor_zooms(path, blocks, current, candidate, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, block in zip(axes.ravel(), blocks):
        name, source, target, _ = block
        axis.plot(target[:, 0], target[:, 1], "k.-", markersize=2,
                  linewidth=1.2, label="VSLAM ordered section")
        axis.plot(*transform(source, *current).T, color="tab:orange",
                  linewidth=1.0, label="retained provisional")
        axis.plot(*transform(source, *candidate[:3]).T, "--", color="tab:cyan",
                  linewidth=1.0, label="rejected candidate")
        value = metrics[name]
        axis.set_title(
            f"{name}: RMSE {value['rmse_m']:.2f}, p95 {value['p95_m']:.2f} m")
        axis.axis("equal")
        axis.grid(True)
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path,
                        default=Path("routes/network/route_network_segmented.csv"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("routes/network/route_network_segmented.metadata.yaml"))
    parser.add_argument("--database", type=Path,
                        default=Path("maps/merged_competition_level_aligned_v10/rtabmap.db"))
    parser.add_argument("--map-route", type=Path,
                        default=Path("routes/v10/START_A.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/csv_vslam_alignment"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, segments = read_network(args.route)
    trajectories = read_database(args.database)
    if 0 not in trajectories or 5 not in trajectories:
        raise RuntimeError("expected ordered RTAB-Map trajectories map_id 0 and 5")
    metadata = yaml.safe_load(args.metadata.read_text(encoding="utf-8"))
    configured = metadata["csv_to_map"]
    current = (float(configured["yaw_deg"]), float(configured["x_m"]),
               float(configured["y_m"]))

    blocks = make_blocks(segments, trajectories)
    candidate, candidate_sections, candidate_pairs = robust_section_fit(
        blocks, current)
    independent = {}
    for block in blocks:
        fitted, metrics, _ = robust_section_fit([block], current)
        independent[block[0]] = {
            "theta_deg": fitted[0], "tx_m": fitted[1], "ty_m": fitted[2],
            "scale": fitted[3], **metrics[block[0]],
        }

    map_route = None
    if args.map_route.is_file():
        with args.map_route.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        map_route = np.asarray([(float(row["x"]), float(row["y"]))
                                for row in rows])

    axis_results, similarity = mapping_diagnostics(candidate_pairs)
    current_metrics = evaluation_metrics(
        segments, trajectories, (*current, 1.0), map_route)
    candidate_metrics = evaluation_metrics(
        segments, trajectories, candidate, map_route)

    s_source = select(segments, "COMMON_1", (5,))
    labels, landmarks = s_landmarks(s_source)
    lo, hi = MAP0_RANGES["S_MODE5"]
    s_tree = cKDTree(resample(trajectories[0][lo:hi], 0.05, 10000))
    landmark_errors = s_tree.query(transform(landmarks, *candidate[:3]))[0]
    landmark_report = [
        {"name": name, "error_m": float(error)}
        for name, error in zip(labels, landmark_errors)
    ]

    distinct_errors = np.concatenate([
        block_correspondences(block, candidate)[3] for block in blocks])
    approval = {
        "overall_median": candidate_metrics["OVERALL"]["median_m"] <=
                          THRESHOLDS["median_m"],
        "overall_rmse": candidate_metrics["OVERALL"]["rmse_m"] <=
                        THRESHOLDS["rmse_m"],
        "overall_p95": candidate_metrics["OVERALL"]["p95_m"] <=
                       THRESHOLDS["p95_m"],
        "distinct_max": float(distinct_errors.max()) <=
                        THRESHOLDS["distinct_max_m"],
        "s_section": (candidate_sections["S_MODE5"]["rmse_m"] <=
                      THRESHOLDS["rmse_m"] and
                      candidate_sections["S_MODE5"]["p95_m"] <=
                      THRESHOLDS["p95_m"]),
    }
    accepted = all(approval.values())

    write_active_features(args.output_dir / "active_a_features.csv", segments)
    plot_overview(args.output_dir / "alignment_overview.png", segments,
                  trajectories, map_route, current, candidate, blocks)
    plot_s_zoom(args.output_dir / "s_curve_zoom.png", segments, trajectories,
                current, candidate, landmark_report)
    plot_anchor_zooms(args.output_dir / "anchor_zooms.png", blocks, current,
                      candidate, candidate_sections)

    report = {
        "decision": "PASS" if accepted else "FAIL",
        "database_read_only": True,
        "source": {
            "route": str(args.route.resolve()), "route_sha256": sha256(args.route),
            "database": str(args.database.resolve()),
            "database_sha256": sha256(args.database),
            "map_route": str(args.map_route.resolve()),
            "map_route_length_m": (float(cumulative_distance(map_route)[-1])
                                   if map_route is not None else None),
            "gps_base_length_m": float(cumulative_distance(
                select(segments, "AAA_BASE"))[-1]),
            "vslam_map0_length_m": float(cumulative_distance(
                trajectories[0])[-1]),
            "trajectory_counts": {str(key): int(len(value))
                                  for key, value in trajectories.items()},
        },
        "method": {
            "name": "ordered_partial_sections_equal_weight_huber_trimmed_se2",
            "occupancy_wall_icp_used": False,
            "whole_route_nearest_neighbor_used_for_fit": False,
            "trim_percent": 15,
            "validation_reference": "union of RTAB-Map node poses and dense v10 route",
        },
        "detected_geometry": detect_geometry(segments),
        "retained_provisional": {
            "theta_deg": current[0], "theta_rad": math.radians(current[0]),
            "tx_m": current[1], "ty_m": current[2], "scale": 1.0,
            "metadata_validated": bool(metadata["alignment"]["validated"]),
            "metadata_symmetric_rms_m": metadata["alignment"].get(
                "symmetric_rms_m"),
            "metrics": current_metrics,
        },
        "new_candidate": {
            "theta_deg": candidate[0], "theta_rad": math.radians(candidate[0]),
            "tx_m": candidate[1], "ty_m": candidate[2], "scale": 1.0,
            "metrics": candidate_metrics,
            "anchor_metrics": candidate_sections,
            "accepted": accepted,
        },
        "independent_candidates": independent,
        "s_curve_landmarks": landmark_report,
        "axis_mapping_diagnostics": axis_results,
        "selected_axis_mapping": "(x,y) followed by proper SE(2) rotation",
        "similarity_diagnostic": similarity,
        "thresholds": THRESHOLDS,
        "approval_checks": approval,
        "metadata_action": "retain validated=false and retain provisional transform",
        "rejection_reasons": [
            "Independent section transforms disagree materially.",
            "The mode-5 S section exceeds the corridor validation thresholds.",
            "The full active-A route exceeds median, RMSE, p95, and max targets.",
            "A global scale cannot correct the section-dependent deformation.",
        ] if not accepted else [],
        "artifacts": {
            "features_csv": str((args.output_dir / "active_a_features.csv").resolve()),
            "overview_plot": str((args.output_dir / "alignment_overview.png").resolve()),
            "s_zoom_plot": str((args.output_dir / "s_curve_zoom.png").resolve()),
            "anchor_zooms_plot": str((args.output_dir / "anchor_zooms.png").resolve()),
            "rviz_live_capture": (
                str((args.output_dir / "rviz_alignment_live.png").resolve())
                if (args.output_dir / "rviz_alignment_live.png").is_file()
                else None),
        },
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "retained_provisional": report["retained_provisional"],
        "new_candidate": report["new_candidate"],
        "similarity_diagnostic": similarity,
        "approval_checks": approval,
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
