"""Create base_link metric path and wait for an external odom TF to adapt it."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    navigation = Path(get_package_share_directory("camera_navigation"))
    bringup = Path(get_package_share_directory("camera_bringup"))
    return LaunchDescription([
        Node(
            package="camera_navigation",
            executable="camera_metric_path_node",
            name="camera_metric_path_node",
            output="screen",
            parameters=[str(bringup / "config" / "camera_mount.yaml")],
        ),
        Node(
            package="camera_navigation",
            executable="camera_reference_path_adapter_node",
            name="camera_reference_path_adapter_node",
            output="screen",
            parameters=[str(
                navigation / "config" / "camera_reference_path_adapter.yaml"),
                {"metric_path_topic": "/camera/path"}],
        ),
    ])
