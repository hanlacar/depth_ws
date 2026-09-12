#!/usr/bin/env python3
"""Create a connected RTAB-Map working graph without modifying input databases.

The appended 113208 sessions have already been aligned with piecewise, unit-scale
SE(2).  This script adds only explicitly RGB-D-validated kUserClosure links.  The
measurements use the current aligned relative poses, while deliberately weak
information avoids forcing ambiguous corridor-axis ICP corrections into the base.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path("/home/qor/depth_ws")
SOURCE = ROOT / "maps/merged_competition_full_course/work/append_candidate_no_cross.db"
OUTPUT = ROOT / "maps/merged_competition_full_course/work/full_course_connected_seed_v3.db"
REPORT = ROOT / "maps/merged_competition_full_course/work/manual_cross_links_v3.json"
BASE_OPTIMIZED = Path("/tmp/ramp_final_export/ramp_final3_global_poses.txt")

# (base final id, appended final id, original 113208 id, RGB inliers,
#  homography inlier fraction, ICP RMSE m, ICP inlier fraction)
ANCHORS = [
    (332, 1564, 1457, 44, 0.619718, 0.068260, 0.993023),
    (339, 1576, 1494, 63, 0.828947, 0.061540, 0.745843),
    (643, 1707, 2156, 13, 0.406250, 0.083447, 0.881250),
    (949, 1737, 2247, 36, 0.418605, 0.080848, 0.896552),
]

# All four anchors validate the unit-scale, piecewise SE(2), but only one edge
# per continuous appended submap is inserted.  Additional graph edges would
# create cycles whose longitudinal corridor ambiguity can bend the retained
# base graph.  A single rigid attachment per submap preserves its topology.
GRAPH_EDGES = {(332, 1564), (643, 1707)}


def matrix(blob: bytes) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :4] = np.frombuffer(blob, np.float32).reshape(3, 4)
    return value


def transform_blob(value: np.ndarray) -> bytes:
    return np.asarray(value[:3, :4], dtype=np.float32).tobytes()


def load_optimized(path: Path) -> dict[int, np.ndarray]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        x, y, z, qx, qy, qz, qw = map(float, f[1:8])
        pose = np.eye(4)
        pose[:3, :3] = np.asarray([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ])
        pose[:3, 3] = (x, y, z)
        result[int(f[8])] = pose
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source_hash_before = sha256(SOURCE)
    shutil.copy2(SOURCE, OUTPUT)
    base_optimized = load_optimized(BASE_OPTIMIZED)

    connection = sqlite3.connect(OUTPUT)
    connection.execute("PRAGMA foreign_keys=ON")
    # Conservative covariance: 25 cm in XY, 50 cm in Z, 10 deg in roll/pitch,
    # and about 5.7 deg in yaw.  Base graph has many more constraints, so these
    # links connect the selected submaps without materially bending it.
    information = np.diag([4.0, 4.0, 2.0, 8.0, 8.0, 25.0]).astype(np.float64)
    entries = []
    for base_id, target_id, source_id, inliers, fraction, rmse, icp_fraction in ANCHORS:
        base_blob = connection.execute(
            "SELECT pose FROM Node WHERE id=?", (base_id,)
        ).fetchone()[0]
        target_blob = connection.execute(
            "SELECT pose FROM Node WHERE id=?", (target_id,)
        ).fetchone()[0]
        # Use the already accepted optimized base pose, not its raw odometry pose.
        # This removes a previously observed raw-vs-optimized inconsistency that
        # unnecessarily deformed the retained competition topology.
        base_pose, target_pose = base_optimized[base_id], matrix(target_blob)
        relative = np.linalg.inv(base_pose) @ target_pose
        reverse = np.linalg.inv(relative)
        inserted = (base_id, target_id) in GRAPH_EDGES
        if inserted:
            for from_id, to_id, measurement in (
                (base_id, target_id, relative),
                (target_id, base_id, reverse),
            ):
                connection.execute(
                    "INSERT INTO Link(from_id,to_id,type,information_matrix,transform,user_data) "
                    "VALUES(?,?,?,?,?,NULL)",
                    (from_id, to_id, 4, information.tobytes(), transform_blob(measurement)),
                )
        entries.append(
            {
                "base_id": base_id,
                "appended_id": target_id,
                "source_113208_id": source_id,
                "rgb_inliers": inliers,
                "rgb_inlier_fraction": fraction,
                "depth_icp_rmse_m": rmse,
                "depth_icp_inlier_fraction": icp_fraction,
                "inserted_as_graph_edge": inserted,
                "measurement_translation_m": relative[:3, 3].tolist(),
                "measurement_matrix_3x4": relative[:3, :4].tolist(),
            }
        )
    # Stored global products no longer describe this connected graph.
    connection.execute(
        "UPDATE Admin SET preview_image=NULL,opt_ids=NULL,opt_poses=NULL,"
        "opt_last_localization=NULL,opt_polygons_size=NULL,opt_polygons=NULL,"
        "opt_tex_coords=NULL,opt_tex_materials=NULL,opt_map=NULL,"
        "opt_map_x_min=NULL,opt_map_y_min=NULL,opt_map_resolution=NULL,"
        "opt_cloud=NULL WHERE version IS NOT NULL"
    )
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    counts = dict(connection.execute("SELECT map_id,count(*) FROM Node GROUP BY map_id"))
    user_links = connection.execute("SELECT count(*) FROM Link WHERE type=4").fetchone()[0]
    connection.close()

    source_hash_after = sha256(SOURCE)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source changed while constructing the seed")
    report = {
        "source": str(SOURCE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "output": str(OUTPUT),
        "integrity_check": integrity,
        "node_count_by_map_id": counts,
        "user_link_rows": user_links,
        "validated_rgbd_anchor_count": len(entries),
        "undirected_manual_graph_edge_count": len(GRAPH_EDGES),
        "information_diagonal": np.diag(information).tolist(),
        "anchors": entries,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
