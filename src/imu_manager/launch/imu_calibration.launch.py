from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("imu_manager")) / "config" / "imu_manager.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("calibration_mode", default_value="LEVEL"),
        DeclareLaunchArgument("duration_sec", default_value="10.0"),
        DeclareLaunchArgument(
            "result_file",
            default_value=str(
                Path.home() / ".config" / "imu_manager" / "imu_calibration_result.yaml"),
        ),
        Node(
            package="imu_manager",
            executable="imu_calibration_node",
            name="imu_calibration_node",
            output="screen",
            parameters=[str(config), {
                "calibration_mode": LaunchConfiguration("calibration_mode"),
                "duration_sec": LaunchConfiguration("duration_sec"),
                "result_file": LaunchConfiguration("result_file"),
            }],
        ),
    ])
