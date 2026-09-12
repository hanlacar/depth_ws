"""Explicitly armed route-follower to current T870 MCU GPS source boundary."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("armed", default_value="false"),
        Node(package="depth_hybrid_slam", executable="mcu_source_adapter",
             name="depth_slam_mcu_source_adapter", output="screen",
             parameters=[{"armed": LaunchConfiguration("armed")}]),
    ])
