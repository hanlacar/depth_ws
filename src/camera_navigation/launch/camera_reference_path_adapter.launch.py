"""Consume a future odom->base_link TF and publish the stitched reference."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("camera_navigation"))
        / "config" / "camera_reference_path_adapter.yaml")
    return LaunchDescription([
        DeclareLaunchArgument(
            "metric_path_topic", default_value="/camera/bev/path"),
        DeclareLaunchArgument(
            "reference_path_topic",
            default_value="/avoidance/route/reference_path"),
        DeclareLaunchArgument(
            "mode_topic", default_value="/mcu/current_mode"),
        Node(
            package="camera_navigation",
            executable="camera_reference_path_adapter_node",
            name="camera_reference_path_adapter_node",
            output="screen",
            parameters=[config, {
                "metric_path_topic": LaunchConfiguration("metric_path_topic"),
                "reference_path_topic": LaunchConfiguration(
                    "reference_path_topic"),
                "mode_topic": LaunchConfiguration("mode_topic"),
            }],
        ),
    ])
