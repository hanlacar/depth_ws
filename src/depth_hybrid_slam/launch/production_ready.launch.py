"""Safety/mission/follower graph. Control remains off unless every gate is set."""

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.conditions import IfCondition
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
                                 "map_path": LaunchConfiguration("map_path"),
                                     "route_metadata_path": LaunchConfiguration(
                                     "route_metadata_path")}])
    branch = Node(
        package="depth_hybrid_slam", executable="branch_selector",
        name="depth_route_branch_selector", output="screen")
    local_path = Node(
        package="depth_hybrid_slam", executable="route_local_path",
        name="depth_route_local_path_connector", output="screen")
    validator = Node(
        package="depth_hybrid_slam", executable="csv_road_validator",
        name="csv_road_validator", output="screen",
        parameters=[get_package_share_directory("camera_bringup")+
                    "/config/camera_mount.yaml",
                    get_package_share_directory("depth_hybrid_slam")+
                    "/config/csv_road_validator.yaml"])
    behavior = Node(
        package="depth_hybrid_slam", executable="behavior_selector",
        name="depth_behavior_selector", output="screen",
        parameters=[{"armed": LaunchConfiguration("enable_mcu_adapter")}])
    lidar = Node(
        package="depth_hybrid_slam", executable="lidar_source",
        name="depth_lidar_source", output="screen",
        condition=IfCondition(LaunchConfiguration("use_lidar")))
    adapter = Node(
        package="depth_hybrid_slam", executable="mcu_source_adapter",
        name="depth_slam_mcu_source_adapter", output="screen",
        condition=IfCondition(LaunchConfiguration("enable_mcu_adapter")),
        parameters=[{"armed": LaunchConfiguration("enable_mcu_adapter")}])
    return LaunchDescription(common_arguments()+[
        mission, safety, branch, follower, local_path, validator, behavior,
        lidar, adapter])
