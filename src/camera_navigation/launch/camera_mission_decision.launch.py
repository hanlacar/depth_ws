"""Opt-in advisory mission decision node; never owns a control topic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("camera_navigation")
    return LaunchDescription([
        DeclareLaunchArgument("debug_overlay", default_value="false"),
        DeclareLaunchArgument("mode_topic", default_value="/mcu/current_mode"),
        DeclareLaunchArgument("section_topic",
                              default_value="/camera/mission/section"),
        Node(
            package="camera_navigation",
            executable="camera_mission_decision_node",
            name="camera_mission_decision_node",
            output="screen",
            parameters=[
                os.path.join(share, "config", "mission_decision.yaml"),
                {"debug_overlay_enabled": LaunchConfiguration("debug_overlay"),
                 "mode_topic": LaunchConfiguration("mode_topic"),
                 "section_topic": LaunchConfiguration("section_topic")},
            ],
        ),
    ])
