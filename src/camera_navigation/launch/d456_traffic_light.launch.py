"""D456 traffic detector/fusion plus the real IMU used by Mode 2."""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def generate_launch_description():
    camera = Path(get_package_share_directory("camera_bringup"))
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    rgb = Path(get_package_share_directory("camera_rgb_traffic_light"))
    navigation = Path(get_package_share_directory("camera_navigation"))
    imu = Path(get_package_share_directory("imu_manager"))
    mount = yaml.safe_load((camera/"config"/"camera_mount.yaml").read_text(
        encoding="utf-8"))["/**"]["ros__parameters"]["camera_mount"]
    return LaunchDescription([
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("enable_vslam", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                camera/"launch"/"d456_bringup.launch.py")),
            launch_arguments={
                "serial_no": LaunchConfiguration("serial_no"),
                "enable_depth": LaunchConfiguration("enable_vslam"),
                "enable_imu": "true",
                "enable_vslam": LaunchConfiguration("enable_vslam"),
            }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                imu/"launch"/"imu_manager.launch.py"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                yolo/"launch"/"yolo_inference.launch.py")),
            launch_arguments={
                "input_image_topic": "/camera/image_raw",
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda"),
                "enable_depth_assist": "false",
                "line_track_mode": "flow",
                "line_track_max_hold_sec": "0.20",
            }.items()),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="d456_base_mount_tf", arguments=[
                "--x", str(mount["position_x_m"]),
                "--y", str(mount["position_y_m"]),
                "--z", str(mount["height_z_m"]),
                "--roll", str(math.radians(mount["reference_roll_deg"])),
                "--pitch", str(math.radians(mount["reference_pitch_deg"])),
                "--yaw", str(math.radians(mount["reference_yaw_deg"])),
                "--frame-id", "base_link", "--child-frame-id", "camera_link",
            ]),
        Node(
            package="camera_rgb_traffic_light",
            executable="rgb_traffic_light_node",
            name="rgb_traffic_light_node", output="screen",
            parameters=[str(rgb/"config"/"rgb_traffic_light.yaml")]),
        Node(
            package="camera_navigation", executable="traffic_light_fusion_node",
            name="traffic_light_fusion_node", output="screen",
            parameters=[str(navigation/"config"/"traffic_light_fusion.yaml")]),
    ])
