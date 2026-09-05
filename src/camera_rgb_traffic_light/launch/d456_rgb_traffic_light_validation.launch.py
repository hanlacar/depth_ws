"""Real D456 plus CPU traffic-light validation; no YOLO or control nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = get_package_share_directory("camera_bringup")
    package = get_package_share_directory("camera_rgb_traffic_light")
    default_config = os.path.join(package, "config", "rgb_traffic_light.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("launch_camera", default_value="true"),
        DeclareLaunchArgument("launch_rqt", default_value="false"),
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("input_image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("config_file", default_value=default_config),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                camera, "launch", "d456_bringup.launch.py")),
            condition=IfCondition(LaunchConfiguration("launch_camera")),
            launch_arguments={
                "serial_no": LaunchConfiguration("serial_no"),
            }.items()),
        Node(
            package="camera_rgb_traffic_light",
            executable="rgb_traffic_light_node",
            name="rgb_traffic_light_node",
            output="screen",
            parameters=[LaunchConfiguration("config_file"), {
                "input_image_topic": LaunchConfiguration("input_image_topic"),
            }]),
        Node(
            package="rqt_image_view", executable="rqt_image_view",
            name="rgb_traffic_light_rqt", output="screen",
            condition=IfCondition(LaunchConfiguration("launch_rqt")),
            arguments=["/camera/traffic_light_rgb/overlay_image"]),
    ])
