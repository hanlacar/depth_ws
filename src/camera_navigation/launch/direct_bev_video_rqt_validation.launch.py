"""Five-hertz A6 validation preset with an RQT semantic overlay."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    navigation = Path(get_package_share_directory("camera_navigation"))
    return LaunchDescription([
        DeclareLaunchArgument("planner_variant", default_value="hybrid_a6"),
        DeclareLaunchArgument("line_track_mode", default_value="none"),
        DeclareLaunchArgument("debug_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("mask_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("overlay_publish_hz", default_value="5.0"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                navigation / "launch" /
                "direct_bev_video_validation.launch.py")),
            launch_arguments={
                "active_planner": "none",
                "planner_variant": LaunchConfiguration("planner_variant"),
                "line_track_mode": LaunchConfiguration("line_track_mode"),
                "debug_image_publish_hz": LaunchConfiguration(
                    "debug_image_publish_hz"),
                "mask_image_publish_hz": LaunchConfiguration(
                    "mask_image_publish_hz"),
                "overlay_publish_hz": LaunchConfiguration(
                    "overlay_publish_hz"),
            }.items()),
        Node(package="rqt_image_view", executable="rqt_image_view",
             name="direct_bev_validation_rqt",
             arguments=["/camera/perception_overlay_image"]),
    ])
