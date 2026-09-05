"""Run transplanted YOLO/RGB/fusion on one external D456 without control."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.mapping_session import as_bool
from depth_hybrid_slam.traffic_light_live_guard import ensure_safe_live_graph
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _guard(context):
    if os.environ.get("ROS_DOMAIN_ID") != "41":
        raise RuntimeError("live traffic-light test requires ROS_DOMAIN_ID=41")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise RuntimeError(
            "live traffic-light test requires RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
    if not as_bool(LaunchConfiguration("graph_guard").perform(context)):
        return [LogInfo(msg="WARNING: live traffic-light graph guard disabled")]
    graph = ensure_safe_live_graph()
    return [LogInfo(msg=f"Live D456/cuVSLAM graph accepted: {graph}")]


def generate_launch_description():
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    rgb = Path(get_package_share_directory("camera_rgb_traffic_light"))
    navigation = Path(get_package_share_directory("camera_navigation"))
    declarations = [
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("graph_guard", default_value="true"),
        DeclareLaunchArgument("start_image_view", default_value="true"),
    ]
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            yolo / "launch" / "yolo_inference.launch.py")),
        launch_arguments={
            "input_image_topic": "/camera/camera/color/image_raw",
            "input_camera_info_topic": "/camera/camera/color/camera_info",
            "device": LaunchConfiguration("device"),
            "require_cuda": LaunchConfiguration("require_cuda"),
            "enable_depth_assist": "false",
        }.items())
    rgb_node = Node(
        package="camera_rgb_traffic_light", executable="rgb_traffic_light_node",
        name="rgb_traffic_light_node", output="screen",
        parameters=[str(rgb / "config" / "rgb_traffic_light.yaml"), {
            "input_image_topic": "/camera/camera/color/image_raw",
        }])
    fusion_node = Node(
        package="camera_navigation", executable="traffic_light_fusion_node",
        name="traffic_light_fusion_node", output="screen",
        parameters=[str(navigation / "config" / "traffic_light_fusion.yaml"), {
            "route_mode_topic": "/depth_slam/traffic_light/unused_route_mode",
        }])
    overlay = Node(
        package="depth_hybrid_slam", executable="traffic_light_live_overlay",
        name="traffic_light_live_overlay", output="screen")
    image = Node(
        package="rqt_image_view", executable="rqt_image_view",
        name="traffic_light_live_image_view", output="screen",
        arguments=["/depth_slam/traffic_light/debug_overlay"],
        condition=IfCondition(LaunchConfiguration("start_image_view")))
    return LaunchDescription(declarations + [
        OpaqueFunction(function=_guard), yolo_launch, rgb_node,
        fusion_node, overlay, image,
    ])
