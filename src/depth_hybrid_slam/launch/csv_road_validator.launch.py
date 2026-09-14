"""Validation-only semantic road containment; starts no controller."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    own = Path(get_package_share_directory("depth_hybrid_slam"))
    camera = Path(get_package_share_directory("camera_bringup"))
    return LaunchDescription([
        Node(package="depth_hybrid_slam", executable="csv_road_validator",
             name="csv_road_validator", output="screen",
             parameters=[str(camera/"config"/"camera_mount.yaml"),
                         str(own/"config"/"csv_road_validator.yaml")]),
    ])
