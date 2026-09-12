#!/usr/bin/env python3
"""Stream RGB blobs into compact DCT/color embeddings for cross-session retrieval."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def embed(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    parts = []
    for channel in cv2.split(lab):
        values = channel.astype(np.float32)
        values = (values - values.mean()) / (values.std() + 1e-6)
        dct = cv2.dct(values)[:8, :12].reshape(-1)[1:]
        dct /= np.linalg.norm(dct) + 1e-6
        parts.append(dct)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
    histogram /= np.linalg.norm(histogram) + 1e-6
    result = np.concatenate(parts + [histogram]).astype(np.float32)
    result /= np.linalg.norm(result) + 1e-6
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    connection = sqlite3.connect(
        f"file:{args.database.resolve()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    ids, stamps, vectors = [], [], []
    cursor = connection.execute(
        "SELECT Node.id,Node.stamp,Data.image FROM Node JOIN Data USING(id) "
        "ORDER BY Node.stamp,Node.id")
    for index, (node_id, stamp, blob) in enumerate(cursor):
        if index % args.stride:
            continue
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        ids.append(node_id)
        stamps.append(stamp)
        vectors.append(embed(image))
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), stamps=np.asarray(stamps),
                        embeddings=np.asarray(vectors, dtype=np.float32))
    print(f"{args.output}: {len(ids)} embeddings, dim={len(vectors[0]) if vectors else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
