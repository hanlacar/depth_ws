"""Validation-only A6 video stack with rate-limited RQT Image outputs.

Production defaults are untouched: this launch explicitly opts into
hybrid_a6 and may optionally start the recorded-video and stationary-IMU
test publishers.  Set both start_* arguments false when using separate
terminals for easier fault injection.
"""

import hashlib
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = get_package_share_directory("camera_bringup")
    yolo = get_package_share_directory("camera_yolo_inference")
    navigation = get_package_share_directory("camera_navigation")
    model = os.path.join(yolo, "models", "hanla_yolo11n_seg_best.pt")
    model_sha = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    video_tool = os.path.join(yolo, "tools", "video_publisher.py")
    imu_tool = os.path.join(navigation, "tools", "fake_imu_publisher.py")

    return LaunchDescription([
        DeclareLaunchArgument(
            "video_path", default_value="/home/qor/urrc_hanla/20260829_170118.mp4"),
        DeclareLaunchArgument("video_fps", default_value="60.0"),
        DeclareLaunchArgument("video_loop", default_value="true"),
        DeclareLaunchArgument("start_video", default_value="true"),
        DeclareLaunchArgument("start_fake_imu", default_value="true"),
        DeclareLaunchArgument("planner_variant", default_value="hybrid_a6"),
        DeclareLaunchArgument("road_boundary_fallback", default_value="none"),
        DeclareLaunchArgument("line_track_mode", default_value="none"),
        DeclareLaunchArgument("active_planner", default_value="bev"),
        DeclareLaunchArgument("debug_images_enabled", default_value="true"),
        DeclareLaunchArgument("debug_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("mask_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("overlay_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("control_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("start_debug_mosaic", default_value="false"),
        DeclareLaunchArgument("debug_mosaic_hz", default_value="4.0"),

        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("start_video")),
            cmd=[FindExecutable(name="python3"), video_tool, "--ros-args",
                 "-p", ["video_path:=", LaunchConfiguration("video_path")],
                 "-p", ["fps:=", LaunchConfiguration("video_fps")],
                 "-p", ["loop:=", LaunchConfiguration("video_loop")]],
            output="screen"),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("start_fake_imu")),
            cmd=[FindExecutable(name="python3"), imu_tool], output="screen"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(yolo, "launch", "yolo_inference.launch.py")),
            launch_arguments={
                "backend": "pytorch",
                "segmentation_model_path": model,
                "expected_model_sha256": model_sha,
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda"),
                "line_track_mode": LaunchConfiguration("line_track_mode"),
                "publish_debug_image": LaunchConfiguration("debug_images_enabled"),
                "publish_optional_masks": LaunchConfiguration("debug_images_enabled"),
                "perception_overlay_max_fps": LaunchConfiguration(
                    "overlay_publish_hz"),
                "detections_image_max_fps": LaunchConfiguration(
                    "debug_image_publish_hz"),
                "mask_image_publish_hz": LaunchConfiguration(
                    "mask_image_publish_hz"),
            }.items()),
        Node(
            package="camera_navigation", executable="direct_bev_planner_node",
            name="direct_bev_planner_node", output="screen",
            parameters=[
                os.path.join(camera, "config", "camera_mount.yaml"),
                os.path.join(navigation, "config", "bev_path.yaml"),
                {"debug_enabled": LaunchConfiguration("debug_images_enabled"),
                 "debug_publish_rate_hz": LaunchConfiguration(
                     "debug_image_publish_hz"),
                 "planner_variant": LaunchConfiguration("planner_variant"),
                 "road_boundary_fallback": LaunchConfiguration(
                     "road_boundary_fallback"),
                 "camera_overlay_publish_rate_hz": LaunchConfiguration(
                     "overlay_publish_hz"),
                 "bev_overlay_max_fps": LaunchConfiguration(
                     "overlay_publish_hz")},
            ]),
        Node(
            package="camera_navigation", executable="direct_bev_controller_node",
            name="direct_bev_controller_node", output="screen",
            parameters=[
                os.path.join(navigation, "config", "bev_controller.yaml"),
                {"control_rate_hz": LaunchConfiguration("control_rate_hz")},
            ]),
        Node(
            package="camera_navigation", executable="bev_wheel_selector_node",
            name="bev_wheel_selector_node", output="screen",
            parameters=[{"active_planner": LaunchConfiguration("active_planner")}]),
        Node(
            package="camera_navigation", executable="direct_bev_drive_node",
            name="direct_bev_drive_node", output="screen",
            parameters=[{"drive_rate_hz": LaunchConfiguration("control_rate_hz")}]),
        Node(
            condition=IfCondition(LaunchConfiguration("start_debug_mosaic")),
            package="camera_navigation", executable="debug_mosaic_node",
            name="camera_debug_mosaic", output="screen",
            parameters=[{"publish_hz": LaunchConfiguration("debug_mosaic_hz")}]),
    ])
