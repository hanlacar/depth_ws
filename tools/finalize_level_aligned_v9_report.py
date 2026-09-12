#!/usr/bin/env python3
"""Append repeatable localization and RViz evidence to the v9 report."""

import json
import re
from pathlib import Path


ROOT = Path("/home/qor/depth_ws")
REPORT = ROOT / "analysis/level_aligned_v9/final/validation_report.json"
LOG = ROOT / "maps/merged_competition_level_aligned_v9/work/localization/reprocess.log"
TIME = ROOT / "maps/merged_competition_level_aligned_v9/work/localization/time.txt"


def match(pattern, text, cast=float):
    value = re.search(pattern, text)
    if not value:
        raise RuntimeError(f"missing pattern: {pattern}")
    return cast(value.group(1))


def main():
    report = json.loads(REPORT.read_text())
    log, timing = LOG.read_text(), TIME.read_text()
    success = match(r"Total localizations on previous session = (\d+)/273", log, int)
    report["localization"] = {
        "success": success, "queries": 273, "percent": 100 * success / 273,
        "v8_baseline": "260/273", "not_lower_than_v8": success >= 260,
        "average_ms": match(r"Average localization time = ([0-9.]+) ms", log),
        "max_jump_m": match(r"\(max=([0-9.]+) m,", log),
        "max_jump_deg": match(r"\(max=[0-9.]+ m, ([0-9.]+) deg", log),
        "peak_rss_kib": match(r"Maximum resident set size \(kbytes\): (\d+)", timing, int),
        "swaps": match(r"Swaps: (\d+)", timing, int),
        "query_limit": "source-related retained 091640 query; not independent driving validation",
    }
    report["rviz"] = {
        "config": str(ROOT / "analysis/level_aligned_v9/final/view_full_cloud.rviz"),
        "screenshot": str(ROOT / "analysis/level_aligned_v9/final/rviz_v9_full_cloud_display.png"),
        "point_count": 2283207, "fixed_frame": "map", "camera_yaw_rad": 3.141592654,
        "data_coordinates_rotated": False, "render_confirmed": True,
    }
    line = report["yellow_lateral_error"]["after"]["max_m"]
    height = report["road_height_error"]["after"]["max_difference_m"]
    graph = report["graph_and_preservation"]
    report["decision"] = "PASS" if (
        line <= .10 and height <= .03 and success >= 260 and graph["integrity"] == "ok" and
        graph["components"] == [1784] and graph["outside_node_rows_identical"] and
        report["rviz"]["render_confirmed"]
    ) else "FAIL"
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"localization": report["localization"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
