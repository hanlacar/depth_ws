"""Self-contained D456 perception/advisory stack; never owns vehicle motion."""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _mount_tf(bringup):
    """Build the sole base mount TF from the canonical calibration file."""
    source = yaml.safe_load(
        (bringup/"config"/"camera_mount.yaml").read_text(encoding="utf-8"))
    mount = source["/**"]["ros__parameters"]["camera_mount"]
    return Node(package="tf2_ros", executable="static_transform_publisher",
                name="d456_base_mount_tf", arguments=[
                    "--x", str(mount["position_x_m"]),
                    "--y", str(mount["position_y_m"]),
                    "--z", str(mount["height_z_m"]),
                    "--roll", str(math.radians(mount["reference_roll_deg"])),
                    "--pitch", str(math.radians(
                        mount["reference_pitch_deg"])),
                    "--yaw", str(math.radians(mount["reference_yaw_deg"])),
                    "--frame-id", "base_link",
                    "--child-frame-id", "camera_link"])


def generate_launch_description():
    bringup = Path(get_package_share_directory("camera_bringup"))
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    nav = Path(get_package_share_directory("camera_navigation"))
    rgb = Path(get_package_share_directory("camera_rgb_traffic_light"))
    imu = Path(get_package_share_directory("imu_manager"))
    return LaunchDescription([
        DeclareLaunchArgument("launch_camera", default_value="true"),
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
            bringup/"launch"/"d456_bringup.launch.py")),
            launch_arguments={"serial_no": LaunchConfiguration("serial_no")}.items(),
            condition=IfCondition(LaunchConfiguration("launch_camera"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
            imu/"launch"/"imu_manager.launch.py"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
            yolo/"launch"/"yolo_inference.launch.py")), launch_arguments={
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda")}.items()),
        _mount_tf(bringup),
        Node(package="camera_navigation", executable="camera_mission_perception_node",
             name="camera_mission_perception_node", parameters=[str(
                 nav/"config"/"mission_perception.yaml")]),
        Node(package="camera_rgb_traffic_light", executable="rgb_traffic_light_node",
             name="rgb_traffic_light_node", parameters=[str(
                 rgb/"config"/"rgb_traffic_light.yaml")]),
        Node(package="camera_navigation", executable="traffic_light_fusion_node",
             name="traffic_light_fusion_node", parameters=[str(
                 nav/"config"/"traffic_light_fusion.yaml")]),
    ])
