"""Opt-in advisory mission perception; never starts a controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package = get_package_share_directory("camera_navigation")
    return LaunchDescription([
        DeclareLaunchArgument("debug_overlay", default_value="false"),
        DeclareLaunchArgument("front_axle_frame", default_value="front_axle"),
        DeclareLaunchArgument("allow_offset_fallback", default_value="false"),
        Node(
            package="camera_navigation",
            executable="camera_mission_perception_node",
            name="camera_mission_perception_node",
            output="screen",
            parameters=[
                os.path.join(package, "config", "mission_perception.yaml"),
                {"debug_overlay_enabled": LaunchConfiguration("debug_overlay"),
                 "front_axle_frame": LaunchConfiguration("front_axle_frame"),
                 "allow_camera_to_front_axle_fallback": LaunchConfiguration(
                     "allow_offset_fallback")},
            ],
        ),
    ])
