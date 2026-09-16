"""Real D456/RPLIDAR HIL with the explicitly isolated TEST ONLY ODOM."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.workspace_paths import workspace_root
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    root = workspace_root()
    camera_share = Path(get_package_share_directory("camera_bringup"))
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
            "controller_hz": "100.0",
        }.items())
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            camera_share/"launch"/"d456_production.launch.py")),
        launch_arguments={
            "launch_camera": "true",
            "serial_no": LaunchConfiguration("camera_serial"),
            "device": LaunchConfiguration("device"),
            "require_cuda": LaunchConfiguration("require_cuda"),
        }.items())
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            share/"launch"/"dual_rplidar.launch.py")),
        launch_arguments={
            "launch_lidar_drivers": "true",
            "launch_rear_lidar_driver": LaunchConfiguration("use_rear_lidar"),
            "front_serial_port": LaunchConfiguration("front_serial_port"),
            "rear_serial_port": LaunchConfiguration("rear_serial_port"),
        }.items())
    return LaunchDescription([
        DeclareLaunchArgument(
            "route_path", default_value=(
                str(root/"routes"/"network"/
                    "route_network_segmented_stop_edited_vforward.csv"))),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument("spawn_branch", default_value="A"),
        DeclareLaunchArgument("start_mode", default_value="4"),
        DeclareLaunchArgument("end_mode", default_value="5"),
        DeclareLaunchArgument("camera_serial", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("front_serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("rear_serial_port", default_value="/dev/ttyUSB1"),
        DeclareLaunchArgument("use_rear_lidar", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        route_loop,
        camera,
        lidar,
        Node(package="depth_hybrid_slam", executable="route_local_path",
             name="depth_route_local_path_connector", output="screen"),
        Node(package="depth_hybrid_slam", executable="csv_road_validator",
             name="csv_road_validator", output="screen", parameters=[
                 str(camera_share/"config"/"camera_mount.yaml"),
                 str(share/"config"/"csv_road_validator.yaml")]),
        Node(package="depth_hybrid_slam", executable="mission_manager",
             name="depth_mission_manager", output="screen",
             parameters=[str(share/"config"/"mission.yaml")]),
        Node(package="depth_hybrid_slam", executable="signal_exit",
             name="depth_signal_exit", output="screen",
             parameters=[str(share/"config"/"signal_exit.yaml")]),
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
                 "rear_lidar_enabled": ParameterValue(
                     LaunchConfiguration("use_rear_lidar"), value_type=bool),
             }]),
        Node(package="depth_hybrid_slam", executable="command_arbiter",
             name="depth_command_arbiter", output="screen",
             remappings=[("/depth_slam/route/branch_stop",
                          "/depth_slam/csv_only/branch_stop")],
             parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(share/"config"/"lidar_roi_debug.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
    ])
