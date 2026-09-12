#!/usr/bin/env python3
"""Restore ramp scan transforms mistakenly included in the first v8 pass."""

import json
import sqlite3
from pathlib import Path

ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
TARGET = ROOT / "maps/merged_competition_level_aligned_v8/rtabmap.db"
REPORT = ROOT / "maps/merged_competition_level_aligned_v8/work/v8_build_report.json"
RAMP_IDS = [119, 120, 156, 157, 1381, 1403, 1404, 1405, 1406]

source = sqlite3.connect(f"file:{SOURCE.resolve()}?mode=ro&immutable=1", uri=True)
target = sqlite3.connect(TARGET)
rows = list(source.execute(
    f"SELECT id,scan_info FROM Data WHERE id IN ({','.join('?' for _ in RAMP_IDS)})", RAMP_IDS))
target.executemany("UPDATE Data SET scan_info=? WHERE id=?", [(blob, node_id) for node_id, blob in rows])
target.execute("UPDATE Admin SET opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL")
target.commit()
integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
target.close(); source.close()

report = json.loads(REPORT.read_text())
report["changed_scan_info_nodes"] = [x for x in report["changed_scan_info_nodes"] if x["node"] not in RAMP_IDS]
report["ramp_outliers_deliberately_unchanged"] = RAMP_IDS
report["ramp_restore_integrity"] = integrity
REPORT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"restored_ramp_scan_info_nodes": RAMP_IDS, "integrity": integrity}, indent=2))
