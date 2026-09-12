#!/usr/bin/env python3
"""Assemble a global XY occupancy cache from RTAB-Map node-local grids."""

from __future__ import annotations

import argparse
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np


def load_poses(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        x, y, z = map(float, f[1:4])
        qx, qy, qz, qw = map(float, f[4:8])
        rotation = np.asarray([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
        result[int(f[8])] = (rotation, np.asarray([x, y, z]))
    return result


def points(blob: bytes | None):
    if not blob:
        return np.empty((0, 3), np.float32)
    obj = zlib.decompressobj()
    raw = obj.decompress(blob)+obj.flush()
    if len(obj.unused_data) != 12:
        raise ValueError("RTAB-Map compressed matrix trailer missing")
    rows, cols, cv_type = struct.unpack("<3i", obj.unused_data)
    if cv_type != 29 or len(raw) != rows*cols*16:
        raise ValueError(f"unexpected local grid format rows={rows} cols={cols} type={cv_type}")
    return np.frombuffer(raw, np.float32).reshape(-1, 4)[:, :3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("poses", type=Path)
    parser.add_argument("--pgm", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    args = parser.parse_args()
    poses = load_poses(args.poses)
    resolution = args.resolution
    translations = np.asarray([value[1] for value in poses.values()])
    margin = 6.0
    x_min = math.floor((translations[:, 0].min()-margin)/resolution)*resolution
    y_min = math.floor((translations[:, 1].min()-margin)/resolution)*resolution
    x_max = math.ceil((translations[:, 0].max()+margin)/resolution)*resolution
    y_max = math.ceil((translations[:, 1].max()+margin)/resolution)*resolution
    width = int(round((x_max-x_min)/resolution))
    height = int(round((y_max-y_min)/resolution))
    ground_count = np.zeros((height, width), np.uint16)
    obstacle_count = np.zeros((height, width), np.uint16)

    connection = sqlite3.connect(args.database)
    cursor = connection.execute(
        "SELECT id,ground_cells,obstacle_cells FROM Data ORDER BY id")
    used_ground = used_obstacle = 0
    for node_id, ground_blob, obstacle_blob in cursor:
        if node_id not in poses:
            continue
        rotation, translation = poses[node_id]
        for blob, counts, kind in ((ground_blob, ground_count, "ground"),
                                   (obstacle_blob, obstacle_count, "obstacle")):
            local = points(blob)
            if not len(local):
                continue
            world = local@rotation.T+translation
            ix = np.floor((world[:, 0]-x_min)/resolution).astype(np.int32)
            iy = np.floor((world[:, 1]-y_min)/resolution).astype(np.int32)
            valid = (ix >= 0)&(ix < width)&(iy >= 0)&(iy < height)
            flat = np.unique(iy[valid]*width+ix[valid])
            values = counts.ravel()
            values[flat] = np.minimum(values[flat].astype(np.uint32)+1, 65535)
            if kind == "ground":
                used_ground += len(flat)
            else:
                used_obstacle += len(flat)
    grid = np.full((height, width), -1, np.int8)
    grid[ground_count > 0] = 0
    grid[obstacle_count > 0] = 100
    compressed = zlib.compress(grid.tobytes(), 6)+struct.pack("<3i", width, height, 1)
    connection.execute(
        "UPDATE Admin SET opt_map=?,opt_map_x_min=?,opt_map_y_min=?,opt_map_resolution=?",
        (compressed, x_min, y_min, resolution))
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()

    image = np.full((height, width), 127, np.uint8)
    image[grid == 0] = 254
    image[grid == 100] = 0
    args.pgm.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.pgm), np.flipud(image))
    print(f"grid={width}x{height} resolution={resolution} origin=({x_min},{y_min})")
    print(f"free={np.count_nonzero(grid==0)} occupied={np.count_nonzero(grid==100)} "
          f"unknown={np.count_nonzero(grid<0)} compressed={len(compressed)}")
    print(f"local_cell_observations ground={used_ground} obstacle={used_obstacle}")
    print(f"integrity={integrity} pgm={args.pgm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
