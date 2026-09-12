"""Test-only CSV follower loop with the external real manager and virtual bridge."""

from pathlib import Path
import shutil
import tempfile

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import offline_rtabmap_include
from depth_hybrid_slam.route_io import load_segmented_route
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_ROOT = "/home/qor/depth_ws"


def _cleanup(_context, temporary_directory):
    shutil.rmtree(temporary_directory, ignore_errors=True)
    return []


def _prepare(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)
    source = Path(value("map_path")).expanduser().resolve()
    route = Path(value("route_path")).expanduser().resolve()
    metadata_text = value("route_metadata_path").strip()
    metadata = (Path(metadata_text).expanduser().resolve() if metadata_text else
                route.with_suffix(".metadata.yaml"))
    branch_command = value("branch_command").strip().upper()
    if branch_command not in ("", "A", "B"):
        raise RuntimeError("branch_command must be empty, A, or B")
    start_mode, end_mode = int(value("start_mode")), int(value("end_mode"))
    if not source.is_file() or not route.is_file() or not metadata.is_file():
        raise RuntimeError("map, segmented route, and route metadata must exist")
    route_info = load_segmented_route(
        route, metadata, branch="B" if branch_command == "B" else "A")
    selected = [point for point in route_info.points
                if start_mode <= point.mode <= end_mode]
    if not selected:
        raise RuntimeError("selected mode interval has no route points")
    first = selected[0]
    command_prefix = value("manager_command_prefix").rstrip("/")
    if not command_prefix or command_prefix == "/mcu":
        raise RuntimeError(
            "closed-loop manager_command_prefix must not use production /mcu")
    temporary_directory = tempfile.mkdtemp(prefix="depth_closed_loop.", dir="/tmp")
    database = Path(temporary_directory) / "rtabmap.db"
    shutil.copy2(source, database)
    package = Path(get_package_share_directory("depth_hybrid_slam"))
    visualization = Path(DEFAULT_ROOT) / "routes/network" / \
        "route_network_segmented_all_branches_display_aligned.csv"
    visualization_metadata = visualization.with_suffix(".metadata.yaml")
    if not visualization.is_file() or not visualization_metadata.is_file():
        raise RuntimeError("full-network STOP-editor visualization overlay is missing")
    common_route = {
        "route_path": str(route), "route_metadata_path": str(metadata),
        "map_path": str(source), "start_mode": start_mode, "end_mode": end_mode,
    }
    return [
        offline_rtabmap_include(str(database)),
        Node(package="depth_hybrid_slam", executable="stop_editor",
             name="depth_stop_editor", output="screen", parameters=[{
                 "route_path": str(route), "route_metadata_path": str(metadata),
                 "visualization_route_path": str(visualization),
                 "visualization_metadata_path": str(visualization_metadata),
                 "output_path": value("output_path"),
                 "output_metadata_path": value("output_metadata_path"),
             }]),
        Node(package="depth_hybrid_slam", executable="branch_selector",
             name="depth_route_branch_selector", output="screen"),
        Node(package="depth_hybrid_slam", executable="route_follower",
             name="route_follower", output="screen", parameters=[common_route, {
                 "enable_control": True, "dry_run": False,
                 "user_approved": True,
                 "prehardware_test_override_alignment": True,
                 "pose_timeout_s": 0.5, "localization_stability_s": 0.5,
             }]),
        Node(package="depth_hybrid_slam", executable="route_local_path",
             name="depth_route_local_path_connector", output="screen",
             parameters=[{"pose_timeout_s": 0.5, "index_timeout_s": 0.5}]),
        Node(package="depth_hybrid_slam", executable="csv_road_validator",
             name="csv_road_validator", output="screen"),
        Node(package="depth_hybrid_slam", executable="behavior_selector",
             name="depth_behavior_selector", output="screen",
             parameters=[{"armed": True}]),
        Node(package="depth_hybrid_slam", executable="mcu_source_adapter",
             name="depth_slam_mcu_source_adapter", output="screen",
             parameters=[{"armed": True}]),
        Node(package="depth_hybrid_slam", executable="closed_loop_support",
             name="depth_closed_loop_support", output="screen",
             parameters=[{"branch_command": branch_command}]),
        Node(package="depth_hybrid_slam", executable="virtual_mcu_bridge",
             name="depth_virtual_mcu_bridge", output="screen", parameters=[
                 str(package / "config" / "virtual_vehicle.yaml"), {
                 "origin_x_m": first.x, "origin_y_m": first.y,
                 "origin_yaw_rad": first.yaw,
                 "simulation_speedup": float(value("simulation_speedup")),
                 "input_drive_topic": command_prefix + "/mcu/cmd_drive",
                 "input_wheel_topic": command_prefix + "/mcu/cmd_wheel",
                 "input_stop_topic": command_prefix + "/mcu/cmd_stop",
                 "prehardware_test_only": True,
             }]),
        TimerAction(period=3.0, actions=[Node(
            package="depth_hybrid_slam", executable="offline_map_requester",
            name="offline_map_requester", output="screen")]),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(package / "config" /
                                  "prehardware_closed_loop.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
            function=lambda context: _cleanup(context, temporary_directory))])),
    ]


def generate_launch_description():
    network = DEFAULT_ROOT + "/routes/network"
    return LaunchDescription([
        DeclareLaunchArgument(
            "map_path", default_value=DEFAULT_ROOT +
            "/maps/merged_competition_level_aligned_v10/rtabmap.db"),
        DeclareLaunchArgument(
            "route_path", default_value=network + "/route_network_segmented.csv"),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument(
            "output_path", default_value=network +
            "/route_network_segmented_stop_edited.csv"),
        DeclareLaunchArgument(
            "output_metadata_path", default_value=network +
            "/route_network_segmented_stop_edited.metadata.yaml"),
        DeclareLaunchArgument("branch_command", default_value=""),
        DeclareLaunchArgument("start_mode", default_value="1"),
        DeclareLaunchArgument("end_mode", default_value="11"),
        DeclareLaunchArgument("simulation_speedup", default_value="10.0"),
        DeclareLaunchArgument(
            "manager_command_prefix", default_value="/prehardware"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        OpaqueFunction(function=_prepare),
    ])
