#!/usr/bin/env python3
"""Find cross-DB visual node correspondences with compact retrieval + geometry."""

from __future__ import annotations

import argparse
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def feature_loader(connection: sqlite3.Connection):
    @lru_cache(maxsize=1800)
    def load(node_id: int):
        points, descriptors = [], []
        rows = connection.execute(
            "SELECT pos_x,pos_y,descriptor FROM Feature WHERE node_id=?", (node_id,))
        for x, y, descriptor in rows:
            if descriptor and len(descriptor) == 32:
                points.append((x, y))
                descriptors.append(np.frombuffer(descriptor, np.uint8))
        if not descriptors:
            return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
        return np.asarray(points, np.float32), np.asarray(descriptors, np.uint8)
    return load


def candidate_indices(similarities: np.ndarray, count: int, separation: int) -> list[int]:
    order = np.argpartition(similarities, -min(len(similarities), count*8))[
        -min(len(similarities), count*8):]
    order = order[np.argsort(similarities[order])[::-1]]
    selected = []
    for index in order:
        if all(abs(int(index)-other) >= separation for other in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return selected


def verify(points_a, descriptors_a, points_b, descriptors_b) -> dict:
    if len(descriptors_a) < 8 or len(descriptors_b) < 8:
        return {"good": 0, "inliers": 0}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
    good = [first for first, second in matches if first.distance < 0.80 * second.distance]
    if len(good) < 8:
        return {"good": len(good), "inliers": 0}
    source = np.asarray([points_a[match.queryIdx] for match in good])
    target = np.asarray([points_b[match.trainIdx] for match in good])
    _, mask = cv2.findFundamentalMat(source, target, cv2.FM_RANSAC, 2.0, 0.995)
    inliers = int(np.count_nonzero(mask)) if mask is not None else 0
    distances = [float(match.distance) for match in good]
    return {
        "good": len(good), "inliers": inliers,
        "inlier_fraction": inliers / len(good),
        "distance_median": float(np.median(distances)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_database", type=Path)
    parser.add_argument("target_database", type=Path)
    parser.add_argument("base_embeddings", type=Path)
    parser.add_argument("target_embeddings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=12)
    args = parser.parse_args()
    base_npz = np.load(args.base_embeddings)
    target_npz = np.load(args.target_embeddings)
    base_ids, base_vectors = base_npz["ids"], base_npz["embeddings"]
    target_ids, target_vectors = target_npz["ids"], target_npz["embeddings"]
    base_connection = open_read_only(args.base_database)
    target_connection = open_read_only(args.target_database)
    load_base = feature_loader(base_connection)
    load_target = feature_loader(target_connection)
    output = []
    for target_index in range(0, len(target_ids), args.stride):
        similarities = base_vectors @ target_vectors[target_index]
        target_id = int(target_ids[target_index])
        target_points, target_descriptors = load_target(target_id)
        for base_index in candidate_indices(similarities, args.candidates, separation=8):
            base_id = int(base_ids[base_index])
            base_points, base_descriptors = load_base(base_id)
            result = verify(base_points, base_descriptors, target_points, target_descriptors)
            if result["inliers"] >= 8:
                output.append({
                    "base_id": base_id, "target_id": target_id,
                    "retrieval_similarity": float(similarities[base_index]), **result,
                })
    base_connection.close()
    target_connection.close()
    output.sort(key=lambda row: (row["inliers"], row["inlier_fraction"]), reverse=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(output)} verified pairs")
    for row in output[:40]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
