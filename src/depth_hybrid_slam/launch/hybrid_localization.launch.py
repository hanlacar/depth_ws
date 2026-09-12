"""Filesystem-read-only RTAB-Map localization and high-rate correction fusion."""

import hashlib
from pathlib import Path
import shlex
import shutil

from depth_hybrid_slam.case_manager_core import verify_case
from depth_hybrid_slam.launch_common import common_arguments, rtabmap_include
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _read_only_prefix(source):
    """Return a process prefix that blocks DB and SQLite sidecar writes."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError(
            "bwrap is required: RTAB-Map 0.22.1 has no Mem/LocalizationReadOnly")
    directory = str(source.parent)
    return " ".join(shlex.quote(item) for item in (
        bwrap, "--bind", "/", "/", "--ro-bind", directory, directory,
        "--die-with-parent", "--"))


def _prepare(context):
    source = Path(LaunchConfiguration("map_path").perform(context)).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"localization map does not exist: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    case_id = LaunchConfiguration("case_id").perform(context)
    matched, metadata = False, {}
    if case_id:
        route = Path(LaunchConfiguration("route_path").perform(context)).expanduser().resolve()
        valid, problems, metadata = verify_case(source.parent.parent, case_id)
        expected_route = source.parent/"route.csv"
        valid = valid and source == source.parent/"rtabmap.db" and route == expected_route
        if not valid:
            raise RuntimeError("case verification failed: "+", ".join(problems or [
                "MAP_OR_ROUTE_PATH_DOES_NOT_MATCH_CASE"]))
        matched = True
    # Version 0.22.1 predates Mem/LocalizationReadOnly. Bind only the DB
    # directory read-only in the RTAB-Map process mount namespace instead of
    # making a physical copy or trusting localization mode not to write.
    return [rtabmap_include(True, _read_only_prefix(source)), Node(
        package="depth_hybrid_slam", executable="localization_fusion",
        name="localization_fusion", output="screen",
        parameters=[{"map_id": str(metadata.get("map_id", digest)),
                     "route_id": str(metadata.get("route_id", "UNVERIFIED")),
                     "localization_mode": True,
                     "map_route_match": matched,
                     "require_route_match": bool(case_id)}])]


def generate_launch_description():
    root = "/home/qor/depth_ws"
    return LaunchDescription(common_arguments({
        "mapping_mode": "false", "localization_mode": "true",
        "map_path": root +
        "/maps/merged_competition_level_aligned_v10/rtabmap.db",
        "route_path": root + "/routes/network/route_network_segmented.csv",
        "route_metadata_path": root +
        "/routes/network/route_network_segmented.metadata.yaml",
    })+[OpaqueFunction(function=_prepare)])
