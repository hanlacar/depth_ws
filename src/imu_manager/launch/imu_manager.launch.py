from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("imu_manager")) / "config" / "imu_manager.yaml"
    return LaunchDescription([
        Node(package="imu_manager", executable="imu_manager_node", name="imu_manager_node",
             output="screen", parameters=[str(config)], respawn=False)
    ])
