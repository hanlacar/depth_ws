#!/usr/bin/env python3
"""Reject all remaining v8 scan candidates after region validation failed."""

import json
import sqlite3
from pathlib import Path

ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
TARGET = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
REPORT = ROOT / "maps/merged_competition_level_aligned_v8/work/v8_build_report.json"
IDS = [1046, 1047, 1049, 1057, 1058, 1059, 1549, 1659]

source = sqlite3.connect(f"file:{SOURCE.resolve()}?mode=ro&immutable=1", uri=True)
target = sqlite3.connect(TARGET)
rows = list(source.execute(
    f"SELECT id,scan_info FROM Data WHERE id IN ({','.join('?' for _ in IDS)})", IDS))
target.executemany("UPDATE Data SET scan_info=? WHERE id=?", [(blob, node_id) for node_id, blob in rows])
target.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL")
target.commit()
integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
target.close(); source.close()

report = json.loads(REPORT.read_text())
report["changed_scan_info_nodes"] = []
report["rejected_after_validation"] = {
    "nodes": IDS,
    "reason": "node 1659 plane was curb/flowerbed, not road; candidate increased intersection max height step; other candidates were outside requested regions and lacked independent validation",
}
report["final_integrity_after_revert"] = integrity
report["problem_region_change"] = {"intersection": [], "straight": [], "arrival": []}
REPORT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"reverted_unvalidated_scan_nodes": IDS, "integrity": integrity}, indent=2))
