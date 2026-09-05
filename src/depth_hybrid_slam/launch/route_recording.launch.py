"""Record a route while the hybrid mapping graph is already running."""

from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    recorder = Node(package="depth_hybrid_slam", executable="route_recorder",
                    name="route_recorder", output="screen",
                    parameters=[{
                        "route_path": LaunchConfiguration("route_path"),
                        "map_path": LaunchConfiguration("map_path"),
                        "session_id": LaunchConfiguration("session_id"),
                        "high_density": LaunchConfiguration("high_density"),
                        "route_min_distance_m": LaunchConfiguration(
                            "route_min_distance_m"),
                        "route_min_angle_rad": LaunchConfiguration(
                            "route_min_angle_rad"),
                        "graph_snapshot_path": LaunchConfiguration(
                            "graph_snapshot_path"),
                    }])
    return LaunchDescription(common_arguments({"mapping_mode": "true"})+[recorder])
