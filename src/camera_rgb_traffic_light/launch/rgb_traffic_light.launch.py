"""Start only the YOLO-independent RGB traffic-light node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("camera_rgb_traffic_light"),
        "config", "rgb_traffic_light.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("input_image_topic", default_value="/camera/image_raw"),
        Node(
            package="camera_rgb_traffic_light",
            executable="rgb_traffic_light_node",
            name="rgb_traffic_light_node",
            output="screen",
            parameters=[LaunchConfiguration("config_file"), {
                "input_image_topic": LaunchConfiguration("input_image_topic"),
            }]),
    ])
