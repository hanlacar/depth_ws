#!/usr/bin/env python3
"""Generate a visualization-only map overlay for all segmented-route rows.

The approved-for-nothing piecewise schedule is reused exactly.  Active-A rows
retain the existing candidate coordinates byte-for-value; alternate branches
use the matching A stage's normalized route-distance schedule.  Inputs are
read-only and the output metadata always remains alignment.validated=false.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/depth_hybrid_slam"))

from depth_hybrid_slam.piecewise_alignment import (  # noqa: E402
    parse_alignment_zones, transform_at_distance,
)
from depth_hybrid_slam.stop_editor_network import (  # noqa: E402
    CHOICE_SEGMENTS, generate_route_cases,
)


OUTPUT_COLUMNS = (
    "segment_id", "segment_type", "point_index", "latitude", "longitude",
    "x_m", "y_m", "direction", "mode", "drive_level", "event",
    "from_node", "to_node", "source_x_m", "source_y_m", "map_yaw_rad",
    "alignment_zone",
)
ACTIVE_A_ORDER = (
    "START_A", "COMMON_1", "T_foword", "T_A", "COMMON_2", "V_A",
    "END_common", "END_AA",
)
REFERENCE_FOR = {
    "START_B": "START_A", "T_B": "T_A", "V_B": "V_A",
    "END_AB": "END_AA",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cumulative(rows):
    output = [0.0]
    for before, after in zip(rows, rows[1:]):
        output.append(output[-1] + math.hypot(
            float(after["x_m"]) - float(before["x_m"]),
            float(after["y_m"]) - float(before["y_m"])))
    return output


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mapped_yaws(rows, mapped):
    output = {}
    by_segment = {}
    for row in rows:
        by_segment.setdefault(row["segment_id"], []).append(row)
    for segment_rows in by_segment.values():
        for index, row in enumerate(segment_rows):
            other_index = index + 1 if index + 1 < len(segment_rows) else index - 1
            key = (row["segment_id"], int(row["point_index"]))
            other = segment_rows[max(0, other_index)]
            other_key = (other["segment_id"], int(other["point_index"]))
            x, y = mapped[key]
            ox, oy = mapped[other_key]
            yaw = (math.atan2(oy - y, ox - x) if other_index > index else
                   math.atan2(y - oy, x - ox))
            if int(row["direction"]) < 0:
                yaw += math.pi
            output[key] = math.atan2(math.sin(yaw), math.cos(yaw))
    return output


def generate(source_path, active_aligned_path, active_metadata_path,
             output_path, output_metadata_path):
    source_path, active_aligned_path = Path(source_path), Path(active_aligned_path)
    active_metadata_path = Path(active_metadata_path)
    output_path, output_metadata_path = Path(output_path), Path(output_metadata_path)
    source = load_csv(source_path)
    aligned = load_csv(active_aligned_path)
    metadata = yaml.safe_load(active_metadata_path.read_text(encoding="utf-8")) or {}
    if metadata.get("alignment", {}).get("validated") is not False:
        raise ValueError("source piecewise alignment must remain explicitly unvalidated")
    if metadata.get("frame_id") != "map" or metadata.get("route_coordinate_frame") != "map":
        raise ValueError("source piecewise overlay must be in map frame")

    source_keys = [(row["segment_id"], int(row["point_index"])) for row in source]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("source contains duplicate segment_id/point_index keys")
    source_by_key = dict(zip(source_keys, source))
    segments = {}
    for row in source:
        segments.setdefault(row["segment_id"], []).append(row)
    cases = generate_route_cases(segments)

    aligned_keys = [(row["segment_id"], int(row["point_index"])) for row in aligned]
    expected_active = [
        (row["segment_id"], int(row["point_index"]))
        for segment in ACTIVE_A_ORDER for row in segments[segment]
    ]
    if aligned_keys != expected_active:
        raise ValueError("active-A aligned input does not exactly match source topology")
    aligned_by_key = dict(zip(aligned_keys, aligned))

    active_source_rows = [source_by_key[key] for key in aligned_keys]
    active_s = cumulative(active_source_rows)
    s_by_key = dict(zip(aligned_keys, active_s))
    segment_s_range = {}
    for segment in ACTIVE_A_ORDER:
        keys = [(row["segment_id"], int(row["point_index"]))
                for row in segments[segment]]
        segment_s_range[segment] = (s_by_key[keys[0]], s_by_key[keys[-1]])

    zones = parse_alignment_zones(metadata)
    blend = float(metadata["alignment"]["blend_distance_m"])
    route_length = float(metadata["alignment"]["route_length_m"])
    mapped, zone_names = {}, {}
    for row, key in zip(source, source_keys):
        segment = row["segment_id"]
        if key in aligned_by_key:
            overlay = aligned_by_key[key]
            mapped[key] = float(overlay["x_m"]), float(overlay["y_m"])
            zone_names[key] = overlay["alignment_zone"]
            continue
        local_rows = segments[segment]
        local_s = cumulative(local_rows)
        position = int(row["point_index"])
        fraction = (local_s[position] / local_s[-1]) if local_s[-1] else 0.0
        if segment == "AAA_BASE":
            route_s = fraction * route_length
        else:
            reference = REFERENCE_FOR.get(segment)
            if reference is None:
                raise ValueError(f"no piecewise schedule reference for segment {segment}")
            start_s, end_s = segment_s_range[reference]
            route_s = start_s + fraction * (end_s - start_s)
        theta, tx, ty, zone = transform_at_distance(route_s, zones, blend)
        x, y = float(row["x_m"]), float(row["y_m"])
        mapped[key] = (
            tx + math.cos(theta) * x - math.sin(theta) * y,
            ty + math.sin(theta) * x + math.cos(theta) * y,
        )
        zone_names[key] = zone

    if set(mapped) != set(source_keys):
        raise ValueError("generated display keys do not cover the complete source")
    yaws = mapped_yaws(source, mapped)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row, key in zip(source, source_keys):
            output = {name: row[name] for name in OUTPUT_COLUMNS[:13]}
            output.update({
                "x_m": f"{mapped[key][0]:.9f}",
                "y_m": f"{mapped[key][1]:.9f}",
                "source_x_m": f"{float(row['x_m']):.9f}",
                "source_y_m": f"{float(row['y_m']):.9f}",
                "map_yaw_rad": f"{yaws[key]:.12f}",
                "alignment_zone": zone_names[key],
            })
            writer.writerow(output)

    values = {
        "format_version": 1,
        "finalized": True,
        "route_name": "competition_all_branches_stop_editor_display_only",
        "route_csv": str(output_path.resolve()),
        "route_sha256": sha256(output_path),
        "source_route_csv": str(source_path.resolve()),
        "source_route_sha256": sha256(source_path),
        "active_a_overlay_csv": str(active_aligned_path.resolve()),
        "active_a_overlay_sha256": sha256(active_aligned_path),
        "route_coordinate_frame": "map",
        "source_coordinate_frame": "gps_local_enu",
        "frame_id": "map",
        "units": "m",
        "visualization_only": True,
        "alignment": {
            "method": "piecewise_se2_stage_schedule_extended_to_all_branches",
            "source_method": metadata["alignment"].get("method"),
            "validated": False,
            "required_for_runtime": False,
            "decision": "VISUALIZATION_ONLY_DO_NOT_USE_FOR_PRODUCTION",
            "route_length_m": route_length,
            "blend_distance_m": blend,
            "zones": metadata["alignment"]["zones"],
        },
        "network": {
            "source_rows": len(source),
            "display_rows": len(mapped),
            "unique_segments": list(segments),
            "reference_segments_not_in_cases": ["AAA_BASE"],
            "route_case_count": len(cases),
            "route_cases": {name: list(value) for name, value in cases.items()},
            "duplicate_keys": 0,
            "missing_keys": 0,
            "ambiguous_keys": 0,
        },
    }
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return values


def main():
    network = ROOT / "routes/network"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=network / "route_network_segmented.csv")
    parser.add_argument("--active-aligned", type=Path,
                        default=network / "route_network_segmented_aligned.csv")
    parser.add_argument("--active-metadata", type=Path,
                        default=network / "route_network_segmented_aligned.metadata.yaml")
    parser.add_argument("--output", type=Path, default=network /
                        "route_network_segmented_all_branches_display_aligned.csv")
    parser.add_argument("--output-metadata", type=Path, default=network /
                        "route_network_segmented_all_branches_display_aligned.metadata.yaml")
    args = parser.parse_args()
    values = generate(args.source, args.active_aligned, args.active_metadata,
                      args.output, args.output_metadata)
    print(yaml.safe_dump({"output": str(args.output), **values["network"]},
                         sort_keys=False))


if __name__ == "__main__":
    main()
