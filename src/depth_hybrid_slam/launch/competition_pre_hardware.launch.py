"""Single depth_ws pre-hardware graph with optional sensor/runtime layers."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, launch_file, condition=None, arguments=None):
    share = Path(get_package_share_directory(package))
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(share/"launch"/launch_file)),
        condition=condition,
        launch_arguments=(arguments or {}).items())


def generate_launch_description():
    camera = include(
        "camera_bringup", "d456_production.launch.py",
        IfCondition(LaunchConfiguration("use_camera")), {
            "launch_camera": "true",
            "serial_no": LaunchConfiguration("camera_serial")})
    cuvslam = include(
        "depth_hybrid_slam", "cuvslam_only.launch.py",
        IfCondition(LaunchConfiguration("use_vslam")), {
            "publish_camera_mount_tf": LaunchConfiguration(
                "publish_camera_mount_tf")})
    localization = include(
        "depth_hybrid_slam", "hybrid_localization.launch.py",
        IfCondition(LaunchConfiguration("use_vslam")), {
            "map_path": LaunchConfiguration("map_path"),
            "route_path": LaunchConfiguration("route_path"),
            "route_metadata_path": LaunchConfiguration(
                "route_metadata_path")})
    autonomy = include(
        "depth_hybrid_slam", "production_ready.launch.py", arguments={
            "route_path": LaunchConfiguration("route_path"),
            "map_path": LaunchConfiguration("map_path"),
            "route_metadata_path": LaunchConfiguration(
                "route_metadata_path"),
            "enable_control": LaunchConfiguration("enable_control"),
            "dry_run": LaunchConfiguration("dry_run"),
            "user_approved": LaunchConfiguration("user_approved"),
            "enable_mcu_adapter": LaunchConfiguration("enable_mcu_adapter"),
            "use_lidar": LaunchConfiguration("use_lidar")})
    return LaunchDescription(common_arguments()+[
        camera, cuvslam, localization, autonomy])
