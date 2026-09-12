"""Create a Nav2 trinary occupancy map from the stored v10 scan PLY."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ply(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line:
                raise ValueError("invalid PLY header")
            header += line
        offset = stream.tell()
    lines = header.decode().splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("only the RTAB-Map binary little-endian PLY is supported")
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex")))
    fields = [line.split()[2] for line in lines if line.startswith("property ")]
    required = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "curvature"]
    if fields[:10] != required:
        raise ValueError(f"unexpected PLY vertex schema: {fields[:10]}")
    dtype = np.dtype({
        "names": ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "curvature"],
        "formats": ["<f4"] * 6 + ["u1"] * 3 + ["<f4"],
        "offsets": [0, 4, 8, 12, 16, 20, 24, 25, 26, 27], "itemsize": 31,
    })
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def export_map(ply: Path, output_dir: Path, name: str, resolution: float,
               padding: float) -> dict:
    points = load_ply(ply)
    finite = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
    x = points["x"][finite].astype(np.float64)
    y = points["y"][finite].astype(np.float64)
    z = points["z"][finite].astype(np.float64)
    nz = points["nz"][finite].astype(np.float64)
    xmin = math.floor((float(x.min()) - padding) / resolution) * resolution
    ymin = math.floor((float(y.min()) - padding) / resolution) * resolution
    xmax = math.ceil((float(x.max()) + padding) / resolution) * resolution
    ymax = math.ceil((float(y.max()) + padding) / resolution) * resolution
    width = int(round((xmax - xmin) / resolution)) + 1
    height = int(round((ymax - ymin) / resolution)) + 1
    ix = np.clip(((x - xmin) / resolution).astype(np.int64), 0, width - 1)
    iy = np.clip(((y - ymin) / resolution).astype(np.int64), 0, height - 1)
    flat = iy * width + ix

    # Ground classification uses surface orientation, not global Z alone, so
    # the existing ramp remains traversable. No unknown cell is fabricated.
    ground_mask = (np.abs(nz) >= 0.75) & (z >= -0.30) & (z <= 2.20)
    obstacle_mask = (np.abs(nz) < 0.75) & (z >= -0.10) & (z <= 2.50)
    cells = width * height
    ground_votes = np.bincount(flat[ground_mask], minlength=cells)
    obstacle_votes = np.bincount(flat[obstacle_mask], minlength=cells)
    free = ground_votes >= 1
    occupied = (obstacle_votes >= 2) & (obstacle_votes >= np.maximum(2, ground_votes // 4))

    values = np.full(cells, 205, dtype=np.uint8)
    values[free] = 254
    values[occupied] = 0
    image = values.reshape(height, width)[::-1, :]
    output_dir.mkdir(parents=True, exist_ok=True)
    pgm = output_dir / f"{name}.pgm"
    yaml_path = output_dir / f"{name}.yaml"
    if not cv2.imwrite(str(pgm), image):
        raise RuntimeError(f"failed to write {pgm}")
    config = {
        "image": pgm.name,
        "mode": "trinary",
        "resolution": float(resolution),
        "origin": [float(xmin), float(ymin), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "source_ply": str(ply.resolve()), "source_ply_sha256": sha256(ply),
        "method": "surface-normal trinary projection; no interpolation or unknown-area fill",
        "resolution_m": resolution, "origin": config["origin"],
        "width": width, "height": height,
        "counts": {"free": int(free.sum()), "occupied": int(occupied.sum()),
                   "unknown": int(cells - np.count_nonzero(free | occupied))},
        "areas_m2": {"free": float(free.sum() * resolution ** 2),
                     "occupied": float(occupied.sum() * resolution ** 2)},
        "ramp_preservation": "normal-based ground mask; elevated ramp points are not classified by height alone",
        "outputs": {"pgm": str(pgm.resolve()), "yaml": str(yaml_path.resolve())},
    }
    (output_dir / f"{name}.report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name", default="v10_nav2")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(export_map(args.ply, args.output_dir, args.name,
                                args.resolution, args.padding), indent=2))


if __name__ == "__main__":
    main()
