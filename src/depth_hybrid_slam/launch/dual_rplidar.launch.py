"""Optional single-owner front/rear RPLIDAR A2M12 production bringup."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    enabled = LaunchConfiguration("launch_lidar_drivers")
    rear_enabled = LaunchConfiguration("launch_rear_lidar_driver")
    return LaunchDescription([
        DeclareLaunchArgument("launch_lidar_drivers", default_value="false"),
        DeclareLaunchArgument(
            "launch_rear_lidar_driver", default_value="false"),
        DeclareLaunchArgument(
            "front_serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument(
            "rear_serial_port", default_value="/dev/ttyUSB1"),
        Node(
            package="rplidar_ros", executable="rplidar_node",
            name="front_rplidar_node", output="screen",
            condition=IfCondition(enabled),
            remappings=[("scan", "/front/scan")],
            parameters=[str(share/"config"/"rplidar_dual.yaml"), {
                "serial_port": LaunchConfiguration("front_serial_port"),
            }]),
        Node(
            package="rplidar_ros", executable="rplidar_node",
            name="rear_rplidar_node", output="screen",
            condition=IfCondition(PythonExpression([
                "'", enabled, "' == 'true' and '",
                rear_enabled, "' == 'true'"
            ])),
            remappings=[("scan", "/rear/scan")],
            parameters=[str(share/"config"/"rplidar_dual.yaml"), {
                "serial_port": LaunchConfiguration("rear_serial_port"),
            }]),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_to_front_laser_tf", condition=IfCondition(enabled),
            arguments=[
                "--x", "0.730", "--y", "0.0", "--z", "0.105",
                # rplidar_dual.yaml normalizes the lead-rearward A2 scan with
                # flip_x_axis, so logical front_laser +X equals base_link +X.
                "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
                "--frame-id", "base_link", "--child-frame-id", "front_laser",
            ]),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_to_rear_laser_tf",
            condition=IfCondition(PythonExpression([
                "'", enabled, "' == 'true' and '",
                rear_enabled, "' == 'true'"
            ])),
            arguments=[
                "--x", "-0.680", "--y", "0.0", "--z", "0.155",
                "--roll", "0.0", "--pitch", "0.0",
                "--yaw", "3.14159265359",
                "--frame-id", "base_link", "--child-frame-id", "rear_laser",
            ]),
    ])
