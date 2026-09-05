from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory("camera_navigation"))/"config"/"image_path.yaml")
    return LaunchDescription([
        # Standalone image-path execution does not require a mission mode
        # publisher. The package config remains fail-closed by default for
        # mission-integrated launches.
        DeclareLaunchArgument("require_control_mode", default_value="false"),
        Node(package="camera_navigation", executable="adaptive_non_bev_node",
             name="camera_image_path_node", output="screen",
             parameters=[config, str(Path(get_package_share_directory("camera_navigation"))/"config"/"adaptive_non_bev.yaml"), {
                 "require_control_mode": LaunchConfiguration("require_control_mode"),
             }]),
    ])
