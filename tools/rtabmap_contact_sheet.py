#!/usr/bin/env python3
"""Create a sparse contact sheet from RTAB-Map RGB blobs without loading a DB at once."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--ids", default="")
    args = parser.parse_args()

    uri = f"file:{args.database.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    all_ids = [row[0] for row in connection.execute("SELECT id FROM Node ORDER BY stamp,id")]
    if args.ids:
        wanted = [int(value) for value in args.ids.split(",")]
    elif len(all_ids) <= args.count:
        wanted = all_ids
    else:
        indices = np.linspace(0, len(all_ids)-1, args.count).round().astype(int)
        wanted = [all_ids[index] for index in indices]

    width, height = 240, 180
    columns = 6
    rows = (len(wanted) + columns - 1) // columns
    sheet = np.full((rows*height, columns*width, 3), 30, np.uint8)
    for index, node_id in enumerate(wanted):
        row = connection.execute("SELECT image FROM Data WHERE id=?", (node_id,)).fetchone()
        if not row or not row[0]:
            continue
        image = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (105, 24), (0, 0, 0), -1)
        cv2.putText(image, f"id {node_id}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y, x = divmod(index, columns)
        sheet[y*height:(y+1)*height, x*width:(x+1)*width] = image
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"cannot write {args.output}")
    print(f"{args.output}: {len(wanted)} frames, {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
