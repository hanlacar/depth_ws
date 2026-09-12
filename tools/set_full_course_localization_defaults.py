#!/usr/bin/env python3
"""Set the verified localization registration strategy on the final output DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


DB = Path("/home/qor/depth_ws/maps/merged_competition_full_course/rtabmap.db")
REPORT = DB.parent / "work/localization_parameter_update.json"


def parse(text: str) -> dict[str, str]:
    result = {}
    for field in text.split(";"):
        if ":" in field:
            key, value = field.split(":", 1)
            result[key] = value
    return result


def main() -> None:
    connection = sqlite3.connect(DB)
    rows = connection.execute("SELECT rowid,parameters FROM Info").fetchall()
    changes = []
    for rowid, parameters in rows:
        before = parse(parameters).get("Reg/Strategy")
        fields = parameters.split(";")
        replaced = False
        for index, field in enumerate(fields):
            if field.startswith("Reg/Strategy:"):
                fields[index] = "Reg/Strategy:0"
                replaced = True
        if not replaced:
            fields.append("Reg/Strategy:0")
        updated = ";".join(fields)
        connection.execute("UPDATE Info SET parameters=? WHERE rowid=?", (updated, rowid))
        changes.append({"rowid": rowid, "before": before, "after": "0"})
    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    verified = [
        {"rowid": rowid, "Reg/Strategy": parse(parameters).get("Reg/Strategy")}
        for rowid, parameters in connection.execute("SELECT rowid,parameters FROM Info")
    ]
    connection.close()
    report = {"database": str(DB), "changes": changes,
              "verified": verified, "integrity_check": integrity}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
