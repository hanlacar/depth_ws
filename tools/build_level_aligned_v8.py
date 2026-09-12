#!/usr/bin/env python3
"""Create conservative v8 from v5 by replacing only proven scan-frame outliers.

The v6/v7 coordinate-wide alternatives failed the fixed localization test, so
this build deliberately preserves all localization-facing structures from v5.
Only scan_info transforms of robustly rejected road-plane outliers are replaced
with their temporally smoothed transforms.  No node or observation is added.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
INPUT = ROOT / "maps/merged_competition_level_aligned_v5/rtabmap.db"
REFERENCE = ROOT / "maps/merged_competition_gap_filled/rtabmap.db"
V7_SEED = ROOT / "maps/merged_competition_level_aligned_v7/work/virtual_sensor_seed.db"
V6_REPORT = ROOT / "maps/merged_competition_level_aligned_v6/work/v6_build_report.json"
RAMP_REPORT = ROOT / "maps/merged_competition_1lane_2lane_ramp/work/ramp_build_report.json"
OUTPUT_DIR = ROOT / "maps/merged_competition_level_aligned_v8"
TARGET = OUTPUT_DIR / "rtabmap.db"
WORK = OUTPUT_DIR / "work"
REPORT = WORK / "v8_build_report.json"
INPUT_SHA = "a3a375f34f0cce14b4925544632f9ca0fdfb8c24f845cde9f975a194fd54d990"
REFERENCE_SHA = "b8bff86d58c60663df910ff053b43d91d33c6bd9076d4251d7f992637de44e86"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT_DIR}")
    if sha(INPUT) != INPUT_SHA or sha(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("input/reference SHA-256 mismatch")
    OUTPUT_DIR.mkdir(parents=True); WORK.mkdir()
    shutil.copy2(INPUT, TARGET)

    rejected = json.loads(V6_REPORT.read_text())["height_outliers_rejected"]
    source_map = json.loads(RAMP_REPORT.read_text())["node_source_map"]
    ramp_ids = {
        int(node_id) for node_id, (source, source_id) in source_map.items()
        if (source == "competition_338" and 115 <= source_id <= 210)
        or (source == "091640" and 345 <= source_id <= 630)
    }
    ids = [int(item["node"]) for item in rejected if int(item["node"]) not in ramp_ids]
    seed = sqlite3.connect(f"file:{V7_SEED.resolve()}?mode=ro&immutable=1", uri=True)
    replacement = {node_id: np.frombuffer(calibration, np.float32)[-12:].copy()
                   for node_id, calibration in seed.execute(
                       f"SELECT id,calibration FROM Data WHERE id IN ({','.join('?' for _ in ids)})", ids)}
    seed.close()

    db = sqlite3.connect(TARGET)
    changed = []
    for node_id in ids:
        scan_info_blob = db.execute("SELECT scan_info FROM Data WHERE id=?", (node_id,)).fetchone()[0]
        scan_info = np.frombuffer(scan_info_blob, np.float32).copy()
        before = scan_info[7:19].copy()
        scan_info[7:19] = replacement[node_id]
        db.execute("UPDATE Data SET scan_info=? WHERE id=?", (scan_info.tobytes(), node_id))
        changed.append({"node": node_id, "max_transform_delta": float(np.max(np.abs(before - replacement[node_id])))})
    db.execute("UPDATE Admin SET preview_image=NULL,opt_cloud=NULL,opt_ids=NULL,opt_poses=NULL,opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL")
    db.commit()
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    counts = db.execute("SELECT count(*),sum(image IS NOT NULL),sum(depth IS NOT NULL),sum(calibration IS NOT NULL),sum(scan IS NOT NULL) FROM Data").fetchone()
    node_count = db.execute("SELECT count(*) FROM Node").fetchone()[0]
    link_count = db.execute("SELECT count(*) FROM Link").fetchone()[0]
    feature_count = db.execute("SELECT count(*) FROM Feature").fetchone()[0]
    db.close()

    report = {
        "input": str(INPUT), "input_sha256_before": INPUT_SHA,
        "reference": str(REFERENCE), "reference_sha256_before": REFERENCE_SHA,
        "output": str(TARGET), "integrity": integrity, "nodes": node_count,
        "links": link_count, "features": feature_count, "data_counts": counts,
        "changed_scan_info_nodes": changed,
        "ramp_outliers_deliberately_unchanged": sorted(
            int(item["node"]) for item in rejected if int(item["node"]) in ramp_ids),
        "problem_region_change": {"intersection": [1659], "straight": [], "arrival": []},
        "preserved_byte_semantics": ["Node", "Link", "Feature", "Word", "Data.image", "Data.depth", "Data.calibration", "Data.scan"],
        "not_changed": ["XY/yaw", "node pose", "constraints", "ramp mask/shape", "RGB/Depth", "keyframes", "feature descriptors", "occupancy point content"],
        "reason": "full pose graph v6 and virtual-sensor v7 both reduced fixed-query localization below 50%; only independently detected plane-transform outliers are safe to correct",
        "nodes_added": 0, "nodes_removed": 0,
        "input_sha256_after": sha(INPUT), "reference_sha256_after": sha(REFERENCE),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
