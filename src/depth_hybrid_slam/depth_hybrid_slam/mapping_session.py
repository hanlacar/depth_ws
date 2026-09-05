"""Non-overwriting mapping target protection and post-stop finalization."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time

import yaml

from .route_finalizer import realign_route


WORKSPACE = Path("/home/qor/depth_ws")
MAPS_ROOT = WORKSPACE / "maps"
ROUTES_ROOT = WORKSPACE / "routes"
BAGS_ROOT = WORKSPACE / "bags"
REPORTS_ROOT = WORKSPACE / "reports"
AUTO_SESSION_ATTEMPTS = 100
SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PROTECTED_MAP_DIRS = (
    MAPS_ROOT / "classroom_test",
    MAPS_ROOT / "corridor_hand_test",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class MappingSession:
    session_id: str
    map_path: Path
    route_path: Path
    final_route_path: Path
    graph_path: Path
    bag_path: Path
    report_path: Path
    metadata_path: Path
    created_at_local: str
    created_at_utc: str
    timezone: str
    utc_offset: str
    automatic_name: bool

    def metadata(self):
        return {
            "session_id": self.session_id,
            "created_at_local": self.created_at_local,
            "created_at_utc": self.created_at_utc,
            "timezone": self.timezone,
            "utc_offset": self.utc_offset,
            "map_path": str(self.map_path),
            "route_path": str(self.route_path),
            "raw_route_path": str(self.route_path),
            "final_route_path": str(self.final_route_path),
            "graph_path": str(self.graph_path),
            "bag_path": str(self.bag_path),
            "report_path": str(self.report_path),
            "automatic_name": self.automatic_name,
        }


def validate_session_id(session_id):
    value = str(session_id).strip()
    if not SAFE_SESSION_ID.fullmatch(value):
        raise ValueError(f"unsafe session_id: {value!r}")
    return value


def _session_paths(session_id, maps_root, routes_root, bags_root, reports_root):
    session_id = validate_session_id(session_id)
    maps_root = Path(maps_root).expanduser().resolve()
    routes_root = Path(routes_root).expanduser().resolve()
    bags_root = Path(bags_root).expanduser().resolve()
    reports_root = Path(reports_root).expanduser().resolve()
    return {
        "map_dir": maps_root / session_id,
        "map_path": maps_root / session_id / "rtabmap.db",
        "route_dir": routes_root / session_id,
        "route_path": routes_root / session_id / "route.csv",
        "final_route_path": routes_root / session_id / "route_final.csv",
        "graph_path": routes_root / session_id / "rtabmap_graph.json",
        "route_metadata": routes_root / session_id / "route_metadata.yaml",
        "bag_path": bags_root / session_id,
        "report_dir": reports_root / session_id,
        "report_path": reports_root / session_id / "mapping_quality.json",
        "session_metadata": reports_root / session_id / "session_metadata.yaml",
    }


def _output_conflicts(paths, allow_map_directory=False):
    candidates = [
        paths["map_path"], Path(str(paths["map_path"]) + "-wal"),
        Path(str(paths["map_path"]) + "-shm"),
        paths["map_dir"] / "checksums.sha256", paths["route_path"],
        paths["route_metadata"], paths["bag_path"], paths["report_path"],
        paths["final_route_path"], paths["graph_path"],
        paths["session_metadata"], paths["route_dir"], paths["report_dir"],
    ]
    if not allow_map_directory:
        candidates.append(paths["map_dir"])
    return [str(path) for path in candidates if path.exists()]


def _timestamp_fields(now=None):
    if now is None:
        local = datetime.now().astimezone()
    elif now.tzinfo is None:
        local = now.astimezone()
    else:
        local = now
    utc = local.astimezone(timezone.utc)
    return {
        "created_at_local": local.isoformat(timespec="seconds"),
        "created_at_utc": utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "timezone": local.tzname() or str(local.tzinfo),
        "utc_offset": local.strftime("%z"),
        "base_name": local.strftime("map_%Y%m%d_%H%M%S"),
    }


def create_mapping_session(map_path="", session_id="", auto_session_name=True,
                           now=None, maps_root=MAPS_ROOT, routes_root=ROUTES_ROOT,
                           bags_root=BAGS_ROOT, reports_root=REPORTS_ROOT,
                           max_attempts=AUTO_SESSION_ATTEMPTS):
    """Resolve all output paths once and atomically reserve a new map directory."""
    stamp = _timestamp_fields(now)
    raw_map = str(map_path).strip()
    raw_session = str(session_id).strip()
    roots = (maps_root, routes_root, bags_root, reports_root)

    if raw_map:
        target = ensure_new_mapping_target(raw_map, maps_root)
        # An explicit map_path has first priority.  Derive the shared identity
        # from that path even if a lower-priority session_id was also supplied.
        selected_id = validate_session_id(target.parent.name)
        paths = _session_paths(selected_id, *roots)
        paths["map_dir"] = target.parent
        paths["map_path"] = target
        conflicts = _output_conflicts(paths, allow_map_directory=True)
        if conflicts:
            raise RuntimeError("session outputs already exist: " + ", ".join(conflicts))
        target.parent.mkdir(parents=True, exist_ok=True)
        automatic = False
    else:
        if raw_session:
            candidates = [validate_session_id(raw_session)]
            automatic = False
        elif as_bool(auto_session_name):
            attempts = max(1, int(max_attempts))
            candidates = [stamp["base_name"]] + [
                f"{stamp['base_name']}_{index:02d}" for index in range(1, attempts)]
            automatic = True
        else:
            raise ValueError(
                "map_path and session_id are empty while auto_session_name=false")

        Path(maps_root).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        selected_id = None
        paths = None
        for candidate in candidates:
            proposed = _session_paths(candidate, *roots)
            target_problems = mapping_target_conflicts(
                proposed["map_path"], maps_root)
            if target_problems:
                if not automatic:
                    raise RuntimeError(
                        "new mapping target rejected: " + ", ".join(target_problems))
                continue
            if _output_conflicts(proposed):
                continue
            try:
                proposed["map_dir"].mkdir(exist_ok=False)
            except FileExistsError:
                continue
            # The map directory is now our atomic reservation. Recheck all
            # other roots in case another process created an output meanwhile.
            if _output_conflicts(proposed, allow_map_directory=True):
                continue
            selected_id, paths = candidate, proposed
            break
        if selected_id is None:
            raise RuntimeError(
                f"unable to reserve a unique mapping session after {len(candidates)} attempts")

    return MappingSession(
        session_id=selected_id,
        map_path=paths["map_path"],
        route_path=paths["route_path"],
        final_route_path=paths["final_route_path"],
        graph_path=paths["graph_path"],
        bag_path=paths["bag_path"],
        report_path=paths["report_path"],
        metadata_path=paths["session_metadata"],
        created_at_local=stamp["created_at_local"],
        created_at_utc=stamp["created_at_utc"],
        timezone=stamp["timezone"],
        utc_offset=stamp["utc_offset"],
        automatic_name=automatic,
    )


def mapping_target_conflicts(map_path, maps_root=MAPS_ROOT):
    if not str(map_path).strip():
        return ["MAP_PATH_EMPTY"]
    root = Path(maps_root).expanduser().resolve()
    path = Path(map_path).expanduser().resolve()
    problems = []
    try:
        path.relative_to(root)
    except ValueError:
        problems.append("MAP_PATH_OUTSIDE_ALLOWED_ROOT")
    if path.name != "rtabmap.db":
        problems.append("MAP_FILENAME_MUST_BE_RTABMAP_DB")
    for protected in PROTECTED_MAP_DIRS:
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            problems.append("PROTECTED_MAP_PATH")
            break
    guarded = (
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        path.parent / "checksums.sha256",
    )
    problems.extend("TARGET_EXISTS:" + item.name for item in guarded if item.exists())
    return problems


def ensure_new_mapping_target(map_path, maps_root=MAPS_ROOT):
    problems = mapping_target_conflicts(map_path, maps_root)
    if problems:
        raise RuntimeError("new mapping target rejected: " + ", ".join(problems))
    return Path(map_path).expanduser().resolve()


def mapping_graph_conflicts(node_names, odometry_publishers):
    names = set(node_names)
    conflicts = []
    if "/rtabmap/rtabmap" in names:
        conflicts.append("RTABMAP_NODE_ALREADY_RUNNING")
    if "/localization_fusion" in names:
        conflicts.append("LOCALIZATION_FUSION_ALREADY_RUNNING")
    if int(odometry_publishers) != 1:
        conflicts.append(f"ODOMETRY_PUBLISHER_COUNT:{int(odometry_publishers)}")
    return conflicts


def current_mapping_graph(discovery_seconds=1.0):
    """Return node names and cuVSLAM odometry publisher count."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    context = Context()
    # Launch arguments such as ``start_rviz:=false`` are not ROS remaps.  Keep
    # them out of rclpy parsing while this short-lived discovery context runs.
    rclpy.init(args=[], context=context)
    node = rclpy.create_node("mapping_start_guard", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        deadline = time.monotonic() + float(discovery_seconds)
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        names = [
            (namespace.rstrip("/") + "/" + name).replace("//", "/")
            for name, namespace in node.get_node_names_and_namespaces()
        ]
        publishers = len(node.get_publishers_info_by_topic(
            "/depth_slam/cuvslam/odometry"))
        return names, publishers
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def load_mapping_profile(config_path, high_density=False):
    with open(config_path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    name = "high_density" if bool(high_density) else "default"
    values = document.get("profiles", {}).get(name)
    if not isinstance(values, dict):
        raise ValueError(f"missing mapping profile: {name}")
    if not all(isinstance(value, str) for value in values.values()):
        raise TypeError("RTAB-Map core parameters must be represented as strings")
    return dict(values)


def rtabmap_argument_string(profile):
    ordered = (
        "Rtabmap/DetectionRate", "RGBD/LinearUpdate",
        "RGBD/AngularUpdate", "Vis/MinInliers",
    )
    result = []
    for name in ordered:
        result.extend((f"--{name}", profile[name]))
    result.extend((
        "--RGBD/CreateOccupancyGrid", "true",
        "--Mem/NotLinkedNodesKept", "false",
        "--Mem/LocalizationDataSaved", "false",
    ))
    return " ".join(result)


def _atomic_yaml(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, sort_keys=True, allow_unicode=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_session_metadata(metadata_path, values):
    """Create session metadata exactly once before any mapping node starts."""
    path = Path(metadata_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, sort_keys=False, allow_unicode=True)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Hard-link creation is atomic and, unlike os.replace(), cannot
        # overwrite a file another mapping process created first.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def validate_route_csv(route_path):
    previous_stamp = None
    previous_distance = -1.0
    count = 0
    with open(route_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["timestamp_sec"]) * 1_000_000_000 + int(row["timestamp_nanosec"])
            distance = float(row["cumulative_distance_m"])
            numeric = [float(row[key]) for key in
                       ("map_x_m", "map_y_m", "map_z_m", "yaw_rad", "yaw_deg")]
            if not all(math.isfinite(value) for value in numeric + [distance]):
                raise ValueError("route contains NaN or Inf")
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError("route timestamps are not strictly increasing")
            if distance < previous_distance:
                raise ValueError("route cumulative distance decreased")
            previous_stamp, previous_distance = stamp, distance
            count += 1
    return count, max(previous_distance, 0.0)


def finalize_mapping_session(map_path, route_path, metadata_path,
                             session_id, quality_report="", graph_path="",
                             final_route_path="", route_spacing_m=0.05):
    database = Path(map_path).expanduser().resolve()
    route = Path(route_path).expanduser().resolve()
    metadata = Path(metadata_path).expanduser().resolve()
    checksum = database.parent / "checksums.sha256"
    if not database.is_file() or not route.is_file() or not metadata.is_file():
        raise FileNotFoundError("map, route, and preliminary route metadata are required")
    if checksum.exists():
        raise FileExistsError(f"refusing to overwrite checksum: {checksum}")
    sidecars = [Path(str(database) + suffix) for suffix in ("-wal", "-shm")]
    if any(path.exists() for path in sidecars):
        raise RuntimeError("mapping DB still has SQLite sidecars; stop RTAB-Map first")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        result = connection.execute("pragma integrity_check").fetchone()[0]
    finally:
        connection.close()
    if result != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {result}")
    final_route = (Path(final_route_path).expanduser().resolve()
                   if str(final_route_path) else
                   (route.with_name("route_final.csv") if str(graph_path) else route))
    alignment = None
    if str(graph_path):
        resolved_graph = Path(graph_path).expanduser().resolve()
        with open(resolved_graph, encoding="utf-8") as stream:
            graph_metadata = json.load(stream)
        if graph_metadata.get("session_id", "") not in ("", session_id):
            raise ValueError("graph snapshot session ID does not match")
        alignment = realign_route(
            route, resolved_graph, final_route,
            spacing_m=float(route_spacing_m))
    point_count, distance = validate_route_csv(final_route)
    with open(metadata, encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if values.get("session_id") not in (None, session_id):
        raise ValueError("session ID does not match preliminary metadata")
    values.update({
        "session_id": session_id,
        "finalized": True,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "map_db_path": str(database),
        "map_db_sha256": sha256(database),
        "raw_route_path": str(route),
        "raw_route_csv_sha256": sha256(route),
        "route_path": str(final_route),
        "route_csv_sha256": sha256(final_route),
        "final_route_csv_sha256": sha256(final_route),
        "point_count": point_count,
        "total_distance_m": distance,
    })
    if alignment:
        values["route_alignment"] = alignment
    if quality_report:
        report = Path(quality_report).expanduser().resolve()
        if not report.is_file():
            raise FileNotFoundError(f"mapping quality report is missing: {report}")
        with open(report, encoding="utf-8") as stream:
            quality = json.load(stream)
        reset_increase = int(quality.get("reset_increase", 0))
        final_state = str(quality.get("final_state", "UNKNOWN"))
        values.update({
            "mapping_quality_report": str(report),
            "mapping_quality_report_sha256": sha256(report),
            "tracking_true_percent": quality.get("tracking_true_percent"),
            "reset_increase": reset_increase,
            "quality_verdict": ("FAIL" if
                                quality.get("invalid_session") or reset_increase > 0
                                else quality.get("quality_verdict", final_state)),
        })
    _atomic_yaml(metadata, values)
    lines = [
        f"{sha256(database)}  rtabmap.db\n",
        f"{sha256(route)}  {route}\n",
        f"{sha256(final_route)}  {final_route}\n",
        f"{sha256(metadata)}  {metadata}\n",
    ]
    with open(checksum, "x", encoding="utf-8") as stream:
        stream.writelines(lines)
        stream.flush()
        os.fsync(stream.fileno())
    return checksum


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--route-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--quality-report", default="")
    parser.add_argument("--graph-path", default="")
    parser.add_argument("--final-route-path", default="")
    parser.add_argument("--route-spacing-m", type=float, default=0.05)
    parsed = parser.parse_args(args)
    result = finalize_mapping_session(
        parsed.map_path, parsed.route_path, parsed.metadata_path,
        parsed.session_id, parsed.quality_report, parsed.graph_path,
        parsed.final_route_path, parsed.route_spacing_m)
    print(f"FINALIZED_CHECKSUM={result}")
