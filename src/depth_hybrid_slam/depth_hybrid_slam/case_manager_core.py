"""Four-case map/route bundle creation and fail-closed verification."""

import csv
import hashlib
from pathlib import Path
import re
import shutil

import yaml


CASE_PATTERN = re.compile(r"^case_[1-4]$")
REQUIRED_METADATA = {
    "case_id", "map_id", "route_id", "created_at", "rtabmap_db_sha256",
    "route_sha256", "d456_serial", "camera_mount", "sensor_profiles",
    "vehicle", "ros_domain_id", "rmw_implementation", "depth_ws_commit",
    "camera_ws_commit", "mcu_commit", "start_pose", "end_pose",
    "route_point_count", "loop_closure_count", "odometry_reset_count",
    "map_session_count", "localization_quality", "user_notes",
    "mission_markers",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_case(root, case_id):
    if not CASE_PATTERN.fullmatch(case_id):
        raise ValueError("case_id must be exactly case_1, case_2, case_3, or case_4")
    return Path(root).resolve()/case_id


def initialize_cases(root):
    """Create only the four empty case directories; never invent course data."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for number in range(1, 5):
        directory = _safe_case(root, f"case_{number}")
        directory.mkdir(exist_ok=True)
        result.append(directory)
    return result


def seal_case(root, case_id, database, route, metadata):
    """Copy a completed map/route into a new immutable-style verified bundle."""
    directory = _safe_case(root, case_id)
    database, route = Path(database).resolve(), Path(route).resolve()
    if not database.is_file() or not route.is_file():
        raise ValueError("database and route must both exist")
    directory.mkdir(parents=True, exist_ok=True)
    protected = [directory/name for name in
                 ("rtabmap.db", "route.csv", "route.yaml", "metadata.yaml",
                  "checksums.sha256")]
    if any(path.exists() for path in protected):
        raise FileExistsError("case already contains sealed data; no overwrite is allowed")
    values = dict(metadata)
    values["case_id"] = case_id
    generated = {"case_id", "rtabmap_db_sha256", "route_sha256", "route_point_count"}
    missing = REQUIRED_METADATA - (set(values) | generated)
    if missing:
        raise ValueError("metadata is missing: "+", ".join(sorted(missing)))
    shutil.copy2(database, directory/"rtabmap.db")
    shutil.copy2(route, directory/"route.csv")
    db_sum = sha256(directory/"rtabmap.db")
    route_sum = sha256(directory/"route.csv")
    values["rtabmap_db_sha256"] = db_sum
    values["route_sha256"] = route_sum
    with open(directory/"route.csv", newline="", encoding="utf-8") as stream:
        values["route_point_count"] = sum(1 for _ in csv.DictReader(stream))
    (directory/"route.yaml").write_text(yaml.safe_dump({
        "case_id": case_id, "map_id": values.get("map_id"),
        "route_id": values.get("route_id"),
    }, sort_keys=True), encoding="utf-8")
    (directory/"metadata.yaml").write_text(
        yaml.safe_dump(values, sort_keys=True, allow_unicode=True), encoding="utf-8")
    (directory/"checksums.sha256").write_text(
        f"{db_sum}  rtabmap.db\n{route_sum}  route.csv\n", encoding="utf-8")
    valid, problems, _ = verify_case(root, case_id)
    if not valid:
        raise ValueError("sealed case failed validation: "+", ".join(problems))
    return directory


def verify_case(root, case_id):
    directory = _safe_case(root, case_id)
    paths = {name: directory/name for name in
             ("rtabmap.db", "route.csv", "route.yaml", "metadata.yaml",
              "checksums.sha256")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return False, ["MISSING:"+item for item in missing], None
    with open(paths["metadata.yaml"], encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    problems = ["METADATA_MISSING:"+key for key in sorted(REQUIRED_METADATA-set(metadata))]
    if metadata.get("case_id") != case_id:
        problems.append("CASE_ID_MISMATCH")
    with open(paths["route.yaml"], encoding="utf-8") as stream:
        route_meta = yaml.safe_load(stream) or {}
    if metadata.get("map_id") != route_meta.get("map_id"):
        problems.append("MAP_ID_MISMATCH")
    if metadata.get("route_id") != route_meta.get("route_id"):
        problems.append("ROUTE_ID_MISMATCH")
    db_sum, route_sum = sha256(paths["rtabmap.db"]), sha256(paths["route.csv"])
    if metadata.get("rtabmap_db_sha256") != db_sum:
        problems.append("DB_CHECKSUM_MISMATCH")
    if metadata.get("route_sha256") != route_sum:
        problems.append("ROUTE_CHECKSUM_MISMATCH")
    expected = {db_sum: "rtabmap.db", route_sum: "route.csv"}
    actual = {}
    for line in paths["checksums.sha256"].read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            actual[parts[0]] = parts[1].lstrip(" *")
    if expected != actual:
        problems.append("CHECKSUM_FILE_MISMATCH")
    try:
        with open(paths["route.csv"], newline="", encoding="utf-8") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if count != int(metadata.get("route_point_count", -1)):
            problems.append("ROUTE_POINT_COUNT_MISMATCH")
    except (OSError, ValueError):
        problems.append("ROUTE_INVALID")
    return not problems, problems, metadata
