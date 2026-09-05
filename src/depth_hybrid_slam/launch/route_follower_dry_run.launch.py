"""Fail-closed 30 Hz route follower; never enables MCU control."""

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_config = (get_package_share_directory("depth_hybrid_slam") +
                      "/config/vehicle_navigation.yaml")
    follower = Node(package="depth_hybrid_slam", executable="route_follower",
                    name="route_follower", output="screen",
                    parameters=[vehicle_config, {
                        "route_path": LaunchConfiguration("route_path"),
                        "map_path": LaunchConfiguration("map_path"),
                        "enable_control": False, "dry_run": True,
                        "user_approved": False}])
    return LaunchDescription(common_arguments()+[follower])
