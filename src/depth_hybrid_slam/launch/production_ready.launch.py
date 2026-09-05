"""Safety/mission/follower graph. Control remains off unless every gate is set."""

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mission_config = get_package_share_directory("depth_hybrid_slam")+"/config/mission.yaml"
    vehicle_config = get_package_share_directory("depth_hybrid_slam")+"/config/vehicle_navigation.yaml"
    shared = {"enable_control": LaunchConfiguration("enable_control"),
              "dry_run": LaunchConfiguration("dry_run"),
              "user_approved": LaunchConfiguration("user_approved")}
    mission = Node(package="depth_hybrid_slam", executable="mission_manager",
                   name="mission_manager", output="screen", parameters=[mission_config])
    safety = Node(package="depth_hybrid_slam", executable="safety_monitor",
                  name="safety_monitor", output="screen", parameters=[shared])
    follower = Node(package="depth_hybrid_slam", executable="route_follower",
                    name="route_follower", output="screen",
                    parameters=[vehicle_config, shared,
                                {"route_path": LaunchConfiguration("route_path"),
                                 "map_path": LaunchConfiguration("map_path")}])
    return LaunchDescription(common_arguments()+[mission, safety, follower])
