"""Camera-free saved RTAB-Map and vehicle-reference-route viewer."""

from pathlib import Path
import shlex
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments, offline_rtabmap_include
from depth_hybrid_slam.mapping_session import as_bool, current_mapping_graph
from launch import LaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _cleanup(_context, temporary_directory):
    if temporary_directory:
        shutil.rmtree(temporary_directory, ignore_errors=True)
    return []


def _read_only_prefix(source):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bwrap is required when use_temporary_db_copy=false")
    directory = str(source.parent)
    return " ".join(shlex.quote(value) for value in (
        bwrap, "--bind", "/", "/", "--ro-bind", directory, directory,
        "--die-with-parent", "--"))


def _prepare(context):
    source = Path(LaunchConfiguration("map_path").perform(context)).expanduser().resolve()
    route = Path(LaunchConfiguration("route_path").perform(context)).expanduser().resolve()
    if not source.is_file() or not route.is_file():
        raise RuntimeError("map_path and route_path must both exist")
    names, _publishers = current_mapping_graph()
    if "/rtabmap/rtabmap" in names or "/localization_fusion" in names:
        raise RuntimeError("offline view cannot run with live localization/mapping")
    temporary_directory = ""
    prefix = ""
    database = source
    if as_bool(LaunchConfiguration("use_temporary_db_copy").perform(context)):
        temporary_directory = tempfile.mkdtemp(prefix="depth_map_view.", dir="/tmp")
        database = Path(temporary_directory) / "rtabmap.db"
        shutil.copy2(source, database)
    else:
        prefix = _read_only_prefix(source)
    package = Path(get_package_share_directory("depth_hybrid_slam"))
    maximum = as_bool(LaunchConfiguration("highest_density_view").perform(context))
    rviz_config = package / "config" / (
        "map_route_max_density.rviz" if maximum else "map_route_high_density.rviz")
    return [
        offline_rtabmap_include(str(database), prefix),
        Node(package="depth_hybrid_slam", executable="route_path_publisher",
             name="route_path_publisher", output="screen",
             parameters=[{"route_path": str(route)}]),
        TimerAction(period=3.0, actions=[Node(
            package="depth_hybrid_slam", executable="offline_map_requester",
            name="offline_map_requester", output="screen")]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="offline_map_to_base_link", output="screen",
             arguments=["0", "0", "0", "0", "0", "0", "map", "base_link"],
             condition=IfCondition(LaunchConfiguration("publish_map_tf_for_view"))),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(rviz_config)],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
            function=lambda context: _cleanup(context, temporary_directory))])),
    ]


def generate_launch_description():
    return LaunchDescription(common_arguments({
        "start_rviz": "true",
        "use_temporary_db_copy": "true",
        "publish_map_tf_for_view": "true",
        "highest_density_view": "false",
    }) + [OpaqueFunction(function=_prepare)])
