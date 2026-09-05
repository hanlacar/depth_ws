"""One YOLO inference feeding independent direct-BEV and non-BEV planners."""

import os
import hashlib
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    camera = get_package_share_directory("camera_bringup")
    yolo = get_package_share_directory("camera_yolo_inference")
    navigation = get_package_share_directory("camera_navigation")
    model = os.path.join(yolo, "models", "hanla_yolo11n_seg_best.pt")
    backend = "pytorch"
    expected_sha256 = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    active = LaunchConfiguration("active_planner")
    return LaunchDescription([
        DeclareLaunchArgument("active_planner", default_value="none"),
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("backend", default_value=backend),
        DeclareLaunchArgument("segmentation_model_path", default_value=model),
        DeclareLaunchArgument("expected_model_sha256", default_value=expected_sha256),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("fixed_output_rate_enabled", default_value="false"),
        DeclareLaunchArgument("output_rate_hz", default_value="60.0"),
        DeclareLaunchArgument("control_rate_hz", default_value="20.0"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(camera, "launch", "d456_bringup.launch.py"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(yolo, "launch", "yolo_inference.launch.py")),
            launch_arguments={
                "backend": LaunchConfiguration("backend"),
                "segmentation_model_path": LaunchConfiguration(
                    "segmentation_model_path"),
                "expected_model_sha256": LaunchConfiguration(
                    "expected_model_sha256"),
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda"),
            }.items()),
        Node(package="camera_navigation", executable="adaptive_non_bev_node",
             name="camera_image_path_node", output="screen",
             parameters=[os.path.join(navigation, "config", "image_path.yaml"),
                         os.path.join(navigation, "config", "adaptive_non_bev.yaml"),
                         {"require_control_mode": False}]),
        Node(package="camera_navigation", executable="direct_bev_planner_node",
             name="direct_bev_planner_node", output="screen",
             parameters=[os.path.join(camera, "config", "camera_mount.yaml"),
                         os.path.join(navigation, "config", "bev_path.yaml"),
                         {"debug_enabled": LaunchConfiguration("debug"),
                          "fixed_output_rate_enabled": LaunchConfiguration(
                              "fixed_output_rate_enabled"),
                          "output_rate_hz": LaunchConfiguration(
                              "output_rate_hz")}]),
        Node(package="camera_navigation", executable="direct_bev_controller_node",
             name="direct_bev_controller_node", output="screen",
             parameters=[os.path.join(navigation, "config", "bev_controller.yaml"),
                         {"control_rate_hz": LaunchConfiguration(
                              "control_rate_hz"),
                          "fixed_output_rate_enabled": LaunchConfiguration(
                              "fixed_output_rate_enabled")}]),
        Node(package="camera_navigation", executable="adaptive_pixel_controller_node",
             name="camera_pixel_controller_node", output="screen",
             condition=IfCondition(PythonExpression(["'", active, "' == 'non_bev'"])),
             parameters=[os.path.join(navigation, "config", "camera_pixel_controller.yaml"),
                         os.path.join(navigation, "config", "adaptive_non_bev.yaml")]),
        Node(package="camera_navigation", executable="bev_wheel_selector_node",
             name="bev_wheel_selector_node", output="screen",
             parameters=[{"active_planner": active}]),
    ])
