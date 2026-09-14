"""Production CSV + D456 advisory + canonical LiDAR + sole command arbiter."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    camera_bringup = Path(get_package_share_directory("camera_bringup"))
    camera = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
        camera_bringup/"launch"/"d456_production.launch.py")),
        launch_arguments={
            "launch_camera": LaunchConfiguration("use_camera"),
            "serial_no": LaunchConfiguration("camera_serial"),
        }.items())
    lidar_drivers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            share/"launch"/"dual_rplidar.launch.py")),
        launch_arguments={
            "launch_lidar_drivers": LaunchConfiguration("use_lidar"),
            "launch_rear_lidar_driver": LaunchConfiguration("use_rear_lidar"),
            "front_serial_port": LaunchConfiguration("front_serial_port"),
            "rear_serial_port": LaunchConfiguration("rear_serial_port"),
        }.items())
    follower_parameters = [str(share/"config"/"vehicle_navigation.yaml"), {
        "route_path": LaunchConfiguration("route_path"),
        "route_metadata_path": LaunchConfiguration("route_metadata_path"),
        "map_path": LaunchConfiguration("map_path"),
        "enable_control": ParameterValue(
            LaunchConfiguration("enable_control"), value_type=bool),
        "dry_run": ParameterValue(
            LaunchConfiguration("dry_run"), value_type=bool),
        "user_approved": ParameterValue(
            LaunchConfiguration("user_approved"), value_type=bool),
    }]
    root = "/home/qor/depth_ws"
    return LaunchDescription(common_arguments({
        "use_camera": "true",
        "use_lidar": "true",
        "use_rear_lidar": "true",
        "route_path": root+"/routes/network/route_network_segmented_stop_edited_vforward.csv",
        "route_metadata_path": root+"/routes/network/route_network_segmented_stop_edited_vforward.metadata.yaml",
        "map_path": root+"/maps/merged_competition_level_aligned_v10/rtabmap.db",
    })+[
        camera,
        lidar_drivers,
        Node(package="depth_hybrid_slam", executable="odom_localization",
             name="odom_localization", output="screen"),
        Node(package="depth_hybrid_slam", executable="route_follower",
             name="route_follower", output="screen", parameters=follower_parameters),
        Node(package="depth_hybrid_slam", executable="branch_selector",
             name="depth_route_branch_selector", output="screen",
             # Mode11ExitGate commits at 5.0 s. The selector's extra transport
             # grace prevents its timeout timer from winning the same tick.
             parameters=[{"mode_11_wait_s": 5.5}]),
        Node(package="depth_hybrid_slam", executable="route_local_path",
             name="depth_route_local_path_connector", output="screen"),
        Node(package="depth_hybrid_slam", executable="csv_road_validator",
             name="csv_road_validator", output="screen", parameters=[
                 str(camera_bringup/"config"/"camera_mount.yaml"),
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
                 "route_path": LaunchConfiguration("route_path"),
                 "route_metadata_path": LaunchConfiguration(
                     "route_metadata_path"),
             }]),
        Node(package="depth_hybrid_slam", executable="maneuver_manager",
             name="depth_maneuver_manager", output="screen",
             parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(package="depth_hybrid_slam", executable="command_arbiter",
             name="depth_command_arbiter", output="screen",
             parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(package="depth_hybrid_slam", executable="safety_monitor",
             name="depth_safety_monitor", output="screen", parameters=[{
                 "enable_control": ParameterValue(
                     LaunchConfiguration("enable_control"), value_type=bool),
                 "dry_run": ParameterValue(
                     LaunchConfiguration("dry_run"), value_type=bool),
                 "user_approved": ParameterValue(
                     LaunchConfiguration("user_approved"), value_type=bool),
                 "require_tracking": True,
                 "require_map_route_match": True,
                 "require_within_map": False,
             }]),
    ])
