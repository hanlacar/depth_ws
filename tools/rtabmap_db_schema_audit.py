#!/usr/bin/env python3
"""Read-only, low-memory schema/integrity inventory for RTAB-Map SQLite DBs."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def main() -> int:
    for raw_path in sys.argv[1:]:
        path = Path(raw_path).resolve()
        uri = f"file:{path}?mode=ro&immutable=1"
        print(f"DB\t{path}\t{path.stat().st_size}")
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only=ON")
            result = connection.execute("PRAGMA quick_check").fetchone()[0]
            print(f"QUICK_CHECK\t{result}")
            rows = connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
            for name, sql in rows:
                count = connection.execute(
                    f"SELECT count(*) FROM {quote_identifier(name)}"
                ).fetchone()[0]
                compact_sql = " ".join((sql or "").split())
                print(f"TABLE\t{name}\t{count}\t{compact_sql}")
        print("END_DB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
