#!/usr/bin/env python3
"""Run the requested test-only CSV closed-loop lateral-offset matrix."""

import argparse
import json
from pathlib import Path

from depth_hybrid_slam.csv_only_branching import load_csv_only_route_case
from depth_hybrid_slam.csv_only_recovery import simulate_lateral_recovery


OFFSETS_M = (0.20, -0.20, 0.50, -0.50, 0.80, -0.80,
             1.20, -1.20, 1.80, -1.80, 2.10, -2.10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--case", default="AAAA")
    parser.add_argument("--segment", default="COMMON_1")
    parser.add_argument("--point", type=int, default=300)
    args = parser.parse_args()
    metadata = args.metadata or args.route.with_suffix(".metadata.yaml")
    route = load_csv_only_route_case(
        args.route, metadata, args.case.strip().upper())
    start = next(point.index for point in route
                 if point.segment_id == args.segment and
                 point.point_index == args.point)
    reports = [simulate_lateral_recovery(route, start, value).dictionary()
               for value in OFFSETS_M]
    print(json.dumps({
        "route": str(args.route.resolve()), "case": args.case.upper(),
        "segment": args.segment, "point": args.point,
        "normal_corridor_m": 1.0, "results": reports,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
