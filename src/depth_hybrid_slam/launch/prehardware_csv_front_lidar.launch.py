"""One real front A2M12 plus route tracking and TEST ONLY ODOM."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    route = LaunchConfiguration("route_path")
    metadata = LaunchConfiguration("route_metadata_path")
    route_loop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            share/"launch"/"prehardware_csv_only_closed_loop.launch.py")),
        launch_arguments={
            "route_path": route,
            "route_metadata_path": metadata,
            "spawn_branch": LaunchConfiguration("spawn_branch"),
            "start_mode": LaunchConfiguration("start_mode"),
            "end_mode": LaunchConfiguration("end_mode"),
        }.items())
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            share/"launch"/"dual_rplidar.launch.py")),
        launch_arguments={
            "launch_lidar_drivers": "true",
            "launch_rear_lidar_driver": "false",
            "front_serial_port": LaunchConfiguration("front_serial_port"),
        }.items())
    return LaunchDescription([
        DeclareLaunchArgument(
            "route_path", default_value=(
                "/home/qor/depth_ws/routes/network/"
                "route_network_segmented_stop_edited_vforward.csv")),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument("spawn_branch", default_value="A"),
        DeclareLaunchArgument("start_mode", default_value="1"),
        DeclareLaunchArgument("end_mode", default_value="11"),
        DeclareLaunchArgument("front_serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        route_loop,
        lidar,
        Node(package="depth_hybrid_slam", executable="mission_manager",
             name="depth_mission_manager", output="screen",
             parameters=[str(share/"config"/"mission.yaml")]),
        Node(package="depth_hybrid_slam", executable="lidar_perception",
             name="depth_lidar_perception", output="screen",
             parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(package="depth_hybrid_slam", executable="lidar_rejoin_validator",
             name="depth_lidar_rejoin_validator", output="screen",
             parameters=[str(share/"config"/"lidar_autonomy.yaml"), {
                 "route_path": route,
                 "route_metadata_path": metadata,
             }]),
        Node(package="depth_hybrid_slam", executable="maneuver_manager",
             name="depth_maneuver_manager", output="screen",
             parameters=[str(share/"config"/"lidar_autonomy.yaml"), {
                 "prehardware_case_commands": True,
                 "rear_lidar_enabled": False,
             }]),
        Node(package="depth_hybrid_slam", executable="command_arbiter",
             name="depth_command_arbiter", output="screen",
             remappings=[("/depth_slam/route/branch_stop",
                          "/depth_slam/csv_only/branch_stop")],
             parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(share/"config"/
                                  "prehardware_csv_front_lidar.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
    ])
