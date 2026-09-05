"""Start only the fail-closed pixel-space camera controller."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("camera_navigation"))
        / "config"
        / "camera_pixel_controller.yaml"
    )
    return LaunchDescription([
        Node(
            package="camera_navigation",
            executable="camera_pixel_controller_node",
            name="camera_pixel_controller_node",
            output="screen",
            parameters=[config],
        ),
        Node(
            package="camera_navigation",
            executable="camera_command_selector_node",
            name="camera_command_selector_node",
            output="screen",
            parameters=[{
                "candidate_drive_topic": "/camera/candidate/pixel/drive",
                "candidate_wheel_topic": "/camera/candidate/pixel/wheel",
            }],
        ),
    ])
