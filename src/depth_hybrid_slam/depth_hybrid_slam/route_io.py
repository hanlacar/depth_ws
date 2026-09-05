"""Load and validate map-frame route CSV files for offline display."""

import csv
import math


def load_map_route(path, include_invalid=False):
    points = []
    previous_stamp = None
    previous_distance = -1.0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_sec", "timestamp_nanosec", "route_index",
                    "map_x_m", "map_y_m", "map_z_m", "yaw_rad",
                    "cumulative_distance_m", "valid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("route CSV missing fields: " + ", ".join(sorted(missing)))
        for row in reader:
            stamp = int(row["timestamp_sec"]) * 1_000_000_000 + int(row["timestamp_nanosec"])
            distance = float(row["cumulative_distance_m"])
            values = tuple(float(row[name]) for name in
                           ("map_x_m", "map_y_m", "map_z_m", "yaw_rad"))
            if not all(math.isfinite(value) for value in values + (distance,)):
                raise ValueError("route CSV contains NaN or Inf")
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError("route CSV timestamps are not strictly increasing")
            if distance < previous_distance:
                raise ValueError("route CSV cumulative distance decreased")
            valid = str(row["valid"]).lower() in ("1", "true", "yes")
            if valid or include_invalid:
                points.append({
                    "stamp_ns": stamp, "index": int(row["route_index"]),
                    "x": values[0], "y": values[1], "z": values[2],
                    "yaw": values[3], "valid": valid,
                })
            previous_stamp, previous_distance = stamp, distance
    return points
