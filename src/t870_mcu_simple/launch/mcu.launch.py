from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("t870_mcu_simple"))
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="auto"),
        Node(
            package="t870_mcu_simple", executable="bridge",
            name="t870_mcu_simple_bridge", output="screen",
            parameters=[str(share/"config"/"mcu.yaml"), {
                "port": LaunchConfiguration("port"),
            }],
        ),
    ])
