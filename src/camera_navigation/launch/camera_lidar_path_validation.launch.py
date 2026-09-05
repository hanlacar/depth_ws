"""Video-only camera path adapter validation; no LiDAR or fake odom/TF."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    navigation = get_package_share_directory("camera_navigation")
    return LaunchDescription([
        DeclareLaunchArgument(
            "video_path",
            default_value="/home/qor/urrc_hanla/20260829_170118.mp4"),
        DeclareLaunchArgument("planner_variant", default_value="hybrid_a6"),
        DeclareLaunchArgument("start_video", default_value="true"),
        DeclareLaunchArgument("start_fake_imu", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                navigation, "launch", "direct_bev_video_rqt_validation.launch.py")),
            launch_arguments={
                "video_path": LaunchConfiguration("video_path"),
                "planner_variant": LaunchConfiguration("planner_variant"),
                "start_video": LaunchConfiguration("start_video"),
                "start_fake_imu": LaunchConfiguration("start_fake_imu"),
            }.items()),
        Node(
            package="camera_navigation",
            executable="camera_reference_path_adapter_node",
            name="camera_reference_path_adapter_node",
            output="screen",
            parameters=[
                os.path.join(navigation, "config", "camera_reference_path_adapter.yaml"),
                {"metric_path_topic": "/camera/bev/path",
                 "reference_path_topic": "/avoidance/route/reference_path",
                 "mode_topic": "/mcu/current_mode"},
            ]),
    ])
