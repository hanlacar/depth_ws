"""Read-only saved-map viewer plus non-destructive segmented STOP editor."""

from pathlib import Path
import math
import os
import shlex
import shutil
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import offline_rtabmap_include
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_ROOT = "/home/qor/depth_ws"


def _read_only_prefix(source, bwrap=""):
    """Block DB and SQLite sidecar writes without copying the database."""
    bwrap = bwrap or shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap executable is unavailable")
    directory = str(source.parent)
    return " ".join(shlex.quote(item) for item in (
        bwrap, "--bind", "/", "/", "--ro-bind", directory, directory,
        "--die-with-parent", "--"))


def _probe_bwrap(timeout_s=5.0, apparmor_gate=Path(
        "/proc/sys/kernel/apparmor_restrict_unprivileged_userns")):
    """Return availability only when an actual unprivileged sandbox starts."""
    executable = shutil.which("bwrap")
    if not executable:
        return False, "NOT_INSTALLED", ""
    apparmor_gate = Path(apparmor_gate)
    try:
        if apparmor_gate.is_file() and apparmor_gate.read_text().strip() == "1":
            return False, "APPARMOR_RESTRICT_UNPRIVILEGED_USERNS", executable
    except OSError:
        pass
    command = [executable, "--ro-bind", "/", "/", "--proc", "/proc",
               "--dev", "/dev", "true"]
    try:
        result = subprocess.run(
            command, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=float(timeout_s))
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"PROBE_ERROR:{error}", executable
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or
                  f"exit code {result.returncode}").strip().splitlines()[0]
        return False, reason, executable
    return True, "PROBE_OK", executable


def _temporary_database(source, temporary_root="/tmp"):
    """Copy a DB only after a 1.5x free-space check; clean partial failures."""
    source = Path(source)
    source_size = source.stat().st_size
    free_bytes = shutil.disk_usage(temporary_root).free
    required_bytes = int(math.ceil(source_size * 1.5))
    if free_bytes < required_bytes:
        raise RuntimeError(
            "INSUFFICIENT_TEMP_SPACE "
            f"db_size={source_size} free={free_bytes} required={required_bytes}")
    directory = tempfile.mkdtemp(
        prefix=f"depth_stop_editor.{os.getpid()}.", dir=temporary_root)
    destination = Path(directory) / "rtabmap.db"
    try:
        shutil.copy2(source, destination)
        if destination.stat().st_size != source_size:
            raise OSError("temporary RTAB-Map DB size mismatch")
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return destination, directory, source_size, free_bytes, required_bytes


def _cleanup(_context, temporary_directory):
    if temporary_directory:
        shutil.rmtree(temporary_directory, ignore_errors=True)
    return []


def _shutdown_on_required_process_exit(event, context, label, failure_only=False):
    if context.is_shutdown:
        return []
    if failure_only and event.returncode == 0:
        return []
    reason = f"{label}_FAILED" if event.returncode else f"{label}_CLOSED"
    return [
        LogInfo(msg=f"{reason} exit_code={event.returncode}"),
        EmitEvent(event=Shutdown(reason=reason)),
    ]


def _resolved_output(context, name):
    return Path(LaunchConfiguration(name).perform(context)).expanduser().resolve()


def _prepare(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)
    source = Path(value("map_path")).expanduser().resolve()
    route = Path(value("route_path")).expanduser().resolve()
    metadata_text = value("route_metadata_path").strip()
    metadata = (Path(metadata_text).expanduser().resolve() if metadata_text else
                route.with_suffix(".metadata.yaml"))
    visualization_text = value("visualization_route_path").strip()
    visualization = (Path(visualization_text).expanduser().resolve()
                     if visualization_text else
                     route.with_name(
                         route.stem + "_all_branches_display_aligned.csv"))
    visualization_metadata_text = value("visualization_metadata_path").strip()
    visualization_metadata = (
        Path(visualization_metadata_text).expanduser().resolve()
        if visualization_metadata_text else
        visualization.with_suffix(".metadata.yaml"))
    required = (source, route, metadata, visualization, visualization_metadata)
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            "map, source route/metadata, and aligned visualization route/metadata "
            "must exist")
    output = _resolved_output(context, "output_path")
    output_metadata = _resolved_output(context, "output_metadata_path")
    protected = set(required)
    if output in protected or output_metadata in protected:
        raise RuntimeError("STOP editor outputs cannot overwrite map, route, or metadata")
    if output == output_metadata:
        raise RuntimeError("CSV and metadata output paths must be different")
    requested_backend = value("read_only_backend").strip().upper()
    if requested_backend not in ("AUTO", "BWRAP", "TEMP_COPY"):
        raise RuntimeError("read_only_backend must be AUTO, BWRAP, or TEMP_COPY")
    available, probe_reason, bwrap = _probe_bwrap(
        float(value("bwrap_probe_timeout_s")))
    if requested_backend == "BWRAP" and not available:
        raise RuntimeError("BWRAP_UNAVAILABLE: " + probe_reason)
    backend = ("BWRAP" if available else "TEMP_COPY") if requested_backend == "AUTO" \
        else requested_backend
    temporary_directory = ""
    source_size, free_bytes, required_bytes = source.stat().st_size, 0, 0
    database, prefix = source, ""
    if backend == "BWRAP":
        prefix = _read_only_prefix(source, bwrap)
    else:
        database, temporary_directory, source_size, free_bytes, required_bytes = \
            _temporary_database(source, value("temporary_root"))
    try:
        package = Path(get_package_share_directory("depth_hybrid_slam"))
        rtabmap = offline_rtabmap_include(str(database), prefix)
        editor = Node(package="depth_hybrid_slam", executable="stop_editor",
             name="depth_stop_editor", output="screen", parameters=[{
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "visualization_route_path": str(visualization),
                 "visualization_metadata_path": str(visualization_metadata),
                 "output_path": str(output),
                 "output_metadata_path": str(output_metadata),
                 "maximum_click_distance_m": float(value(
                     "maximum_click_distance_m")),
                 "initial_case": value("initial_case"),
             }])
        requester = Node(
            package="depth_hybrid_slam", executable="offline_map_requester",
            name="offline_map_requester", output="screen", parameters=[{
                "startup_timeout_s": float(value("rtabmap_startup_timeout_s")),
                "map_delivery_timeout_s": float(value(
                    "rtabmap_map_delivery_timeout_s")),
            }])
        map_to_base = Node(
            package="tf2_ros", executable="static_transform_publisher",
             name="stop_editor_map_to_base", output="screen",
             arguments=["0", "0", "0", "0", "0", "0", "map", "base_link"])
        rviz = Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(package / "config" / "stop_editor_vslam.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz")))
        actions = [
            LogInfo(msg=f"BWRAP_AVAILABLE={str(available).lower()} "
                        f"reason={probe_reason}"),
            LogInfo(msg=f"READ_ONLY_BACKEND={backend}"),
            LogInfo(msg=(f"DB_SIZE_BYTES={source_size} "
                         f"TEMP_FREE_BYTES={free_bytes} "
                         f"TEMP_REQUIRED_BYTES={required_bytes}")),
            rtabmap, editor, TimerAction(period=1.0, actions=[requester]),
            map_to_base, rviz,
            RegisterEventHandler(OnProcessExit(
                target_action=requester,
                on_exit=lambda event, ctx: _shutdown_on_required_process_exit(
                    event, ctx, "RTABMAP_START", failure_only=True))),
            RegisterEventHandler(OnProcessExit(
                target_action=rviz,
                on_exit=lambda event, ctx: _shutdown_on_required_process_exit(
                    event, ctx, "RVIZ"))),
        ]
        if temporary_directory:
            actions.append(RegisterEventHandler(OnShutdown(
                on_shutdown=[OpaqueFunction(function=lambda ctx: _cleanup(
                    ctx, temporary_directory))])))
        return actions
    except BaseException:
        _cleanup(context, temporary_directory)
        raise


def generate_launch_description():
    network = DEFAULT_ROOT + "/routes/network"
    return LaunchDescription([
        DeclareLaunchArgument(
            "map_path", default_value=DEFAULT_ROOT +
            "/maps/merged_competition_level_aligned_v10/rtabmap.db"),
        DeclareLaunchArgument(
            "route_path", default_value=network + "/route_network_segmented.csv"),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument("visualization_route_path", default_value=""),
        DeclareLaunchArgument("visualization_metadata_path", default_value=""),
        DeclareLaunchArgument(
            "output_path", default_value=network +
            "/route_network_segmented_stop_edited.csv"),
        DeclareLaunchArgument(
            "output_metadata_path", default_value=network +
            "/route_network_segmented_stop_edited.metadata.yaml"),
        DeclareLaunchArgument("maximum_click_distance_m", default_value="0.50"),
        DeclareLaunchArgument("initial_case", default_value="ALL"),
        DeclareLaunchArgument("read_only_backend", default_value="AUTO"),
        DeclareLaunchArgument("temporary_root", default_value="/tmp"),
        DeclareLaunchArgument("bwrap_probe_timeout_s", default_value="5.0"),
        DeclareLaunchArgument("rtabmap_startup_timeout_s", default_value="15.0"),
        DeclareLaunchArgument(
            "rtabmap_map_delivery_timeout_s", default_value="900.0"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        OpaqueFunction(function=_prepare),
    ])
