"""Validation-only semantic road containment; starts no controller."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    own = Path(get_package_share_directory("depth_hybrid_slam"))
    camera = Path(get_package_share_directory("camera_bringup"))
    return LaunchDescription([
        DeclareLaunchArgument("synthetic_path", default_value="false"),
        DeclareLaunchArgument("lateral_offset_m", default_value="0.0"),
        DeclareLaunchArgument("yaw_offset_deg", default_value="0.0"),
        Node(package="depth_hybrid_slam", executable="csv_road_validator",
             name="csv_road_validator", output="screen",
             parameters=[str(camera/"config"/"camera_mount.yaml"),
                         str(own/"config"/"csv_road_validator.yaml")]),
        Node(package="depth_hybrid_slam",
             executable="csv_validation_path_publisher",
             name="csv_validation_path_publisher", output="screen",
             condition=IfCondition(LaunchConfiguration("synthetic_path")),
             parameters=[{
                 "lateral_offset_m": LaunchConfiguration("lateral_offset_m"),
                 "yaw_offset_deg": LaunchConfiguration("yaw_offset_deg")}]),
    ])
