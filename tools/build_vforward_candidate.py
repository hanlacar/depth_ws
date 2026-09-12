#!/usr/bin/env python3
"""Build a depth_ws-only V_foword candidate from a pinned reference CSV."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import yaml


MODE10_SEGMENTS = ("V_foword", "V_A", "V_B")
REPLACED_SEGMENTS = frozenset(("V_A", "V_B"))
STOP_POINTS = {
    # p107 is only 8 mm after p106 in the pinned reference recording.  A
    # branch decision there asks the depth follower for >22 degrees on one
    # branch.  p105 is the latest unchanged common waypoint from which both
    # branch approaches satisfy the production corridor/steering envelope.
    "V_foword": frozenset((105,)),
    "V_A": frozenset((11, 66)),
    "V_B": frozenset((0, 70)),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return tuple(reader.fieldnames), list(reader)


def write_rows(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_reference_rows(source_rows, reference_rows):
    source_first = next(row for row in source_rows
                        if row["segment_id"] == "AAA_BASE" and
                        int(row["point_index"]) == 0)
    reference_first = next(row for row in reference_rows
                           if row["segment_id"] == "AAA_BASE" and
                           int(row["point_index"]) == 0)
    longitude_delta = (float(source_first["longitude"])-
                       float(reference_first["longitude"]))
    if abs(longitude_delta-1.694e-7) > 1.0e-10:
        raise RuntimeError("unexpected depth/reference longitude-origin delta")

    output = []
    for row in reference_rows:
        segment = row["segment_id"]
        if segment not in MODE10_SEGMENTS:
            continue
        item = dict(row)
        item["longitude"] = f"{float(row['longitude'])+longitude_delta:.10f}"
        point_index = int(item["point_index"])
        item["event"] = (
            "STOP_LINE" if point_index in STOP_POINTS[segment] else "NONE")
        output.append(item)
    counts = {segment: sum(row["segment_id"] == segment for row in output)
              for segment in MODE10_SEGMENTS}
    if counts != {"V_foword": 108, "V_A": 132, "V_B": 154}:
        raise RuntimeError(f"unexpected latest Mode-10 point counts: {counts}")
    return output, longitude_delta


def candidate_rows(source_rows, mode10_rows):
    output = []
    inserted = False
    for row in source_rows:
        if row["segment_id"] in REPLACED_SEGMENTS:
            if not inserted:
                output.extend(mode10_rows)
                inserted = True
            continue
        output.append(dict(row))
    if not inserted:
        raise RuntimeError("source CSV has no replaceable V_A/V_B segments")
    original_other = [row for row in source_rows
                      if row["segment_id"] not in REPLACED_SEGMENTS]
    candidate_other = [row for row in output
                       if row["segment_id"] not in MODE10_SEGMENTS]
    if original_other != candidate_other:
        raise RuntimeError("a non-Mode-10 source row changed")
    return output


def body_yaws(rows):
    result = {}
    by_segment = {}
    for row in rows:
        by_segment.setdefault(row["segment_id"], []).append(row)
    for segment, points in by_segment.items():
        for index, point in enumerate(points):
            neighbour = None
            before = False
            direction = int(point["direction"])
            if index+1 < len(points) and int(points[index+1]["direction"]) == direction:
                neighbour = points[index+1]
            elif index and int(points[index-1]["direction"]) == direction:
                neighbour = points[index-1]
                before = True
            if neighbour is None:
                motion_yaw = 0.0
            elif before:
                motion_yaw = math.atan2(
                    float(point["y_m"])-float(neighbour["y_m"]),
                    float(point["x_m"])-float(neighbour["x_m"]))
            else:
                motion_yaw = math.atan2(
                    float(neighbour["y_m"])-float(point["y_m"]),
                    float(neighbour["x_m"])-float(point["x_m"]))
            yaw = motion_yaw if direction > 0 else motion_yaw+math.pi
            result[(segment, int(point["point_index"]))] = math.atan2(
                math.sin(yaw), math.cos(yaw))
    return result


def display_rows(source_rows, original_display_rows, display_metadata):
    preserved = {(row["segment_id"], int(row["point_index"])): row
                 for row in original_display_rows
                 if row["segment_id"] not in REPLACED_SEGMENTS}
    zone = next(item for item in display_metadata["alignment"]["zones"]
                if item["id"] == "ZONE_06_END")
    theta, tx, ty = (float(zone["theta_rad"]), float(zone["tx_m"]),
                     float(zone["ty_m"]))
    cosine, sine = math.cos(theta), math.sin(theta)
    yaws = body_yaws(source_rows)
    output = []
    for row in source_rows:
        key = (row["segment_id"], int(row["point_index"]))
        if row["segment_id"] not in MODE10_SEGMENTS:
            if key not in preserved:
                raise RuntimeError(f"display row missing for {key}")
            output.append(dict(preserved[key]))
            continue
        x, y = float(row["x_m"]), float(row["y_m"])
        item = dict(row)
        item.update({
            "x_m": f"{tx+cosine*x-sine*y:.9f}",
            "y_m": f"{ty+sine*x+cosine*y:.9f}",
            "source_x_m": f"{x:.9f}",
            "source_y_m": f"{y:.9f}",
            "map_yaw_rad": f"{math.atan2(math.sin(yaws[key]+theta), math.cos(yaws[key]+theta)):.12f}",
            "alignment_zone": "ZONE_06_END",
        })
        output.append(item)
    return output


def write_metadata(source, output, reference, reference_commit,
                   longitude_delta, display=False):
    source_metadata = source.with_suffix(".metadata.yaml")
    with source_metadata.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    metadata["route_csv"] = str(output.resolve())
    metadata["route_sha256"] = digest(output)
    if not display:
        active = metadata["selection"]["active_segments"]
        active.insert(active.index("V_A"), "V_foword")
    metadata["vforward_integration"] = {
        "reference_repository": "https://github.com/hanlacar/mmission_ws.git",
        "reference_commit": reference_commit,
        "reference_csv": "routes/route_network_segmented_10.csv",
        "source_csv": str(source.resolve()),
        "source_sha256": digest(source),
        "longitude_origin_delta_deg": longitude_delta,
        "geometry_coordinates_modified": False,
        "mode10_segments_replaced": list(MODE10_SEGMENTS),
        "explicit_stop_points": {
            key: sorted(value) for key, value in STOP_POINTS.items()},
    }
    with output.with_suffix(".metadata.yaml").open(
            "w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-source", type=Path, required=True)
    parser.add_argument("--display-output", type=Path, required=True)
    args = parser.parse_args()

    fields, source_rows = read_rows(args.source)
    reference_fields, reference_rows = read_rows(args.reference)
    if fields != reference_fields:
        raise RuntimeError("source/reference CSV schemas differ")
    mode10, longitude_delta = normalized_reference_rows(
        source_rows, reference_rows)
    rows = candidate_rows(source_rows, mode10)
    write_rows(args.output, fields, rows)
    write_metadata(args.source, args.output, args.reference,
                   args.reference_commit, longitude_delta)

    display_fields, original_display = read_rows(args.display_source)
    with args.display_source.with_suffix(".metadata.yaml").open(
            encoding="utf-8") as stream:
        display_metadata = yaml.safe_load(stream)
    rendered = display_rows(rows, original_display, display_metadata)
    write_rows(args.display_output, display_fields, rendered)
    write_metadata(args.display_source, args.display_output, args.reference,
                   args.reference_commit, longitude_delta, display=True)


if __name__ == "__main__":
    main()
