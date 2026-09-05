"""Existing camera perception plus live VSLAM projection and two-view display."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.mapping_session import as_bool
from depth_hybrid_slam.perception_integration_guard import (
    ensure_safe_integration_graph,
)
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
        raise RuntimeError("integration requires ROS_DOMAIN_ID=41")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        raise RuntimeError("integration requires RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
    if not as_bool(LaunchConfiguration("graph_guard").perform(context)):
        return [LogInfo(msg="WARNING: perception/VSLAM graph guard disabled")]
    graph = ensure_safe_integration_graph()
    return [LogInfo(msg=(
        "Perception/VSLAM graph accepted: one external D456 RGB publisher, "
        "cuVSLAM and RTAB-Map present, no forbidden control publishers. "
        f"Snapshot={graph}"))]


def generate_launch_description():
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    rgb = Path(get_package_share_directory("camera_rgb_traffic_light"))
    navigation = Path(get_package_share_directory("camera_navigation"))
    depth = Path(get_package_share_directory("depth_hybrid_slam"))
    declarations = [
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("graph_guard", default_value="true"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("start_image_view", default_value="true"),
    ]
    # D456 is deliberately absent. scripts/run_d456_host.sh remains the one
    # and only RealSense owner for the integrated process graph.
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(yolo / "launch" /
                                          "yolo_inference.launch.py")),
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
            "route_mode_topic": "/depth_slam/perception/unused_route_mode",
        }])
    mission_node = Node(
        package="camera_navigation", executable="camera_mission_perception_node",
        name="camera_mission_perception_node", output="screen",
        parameters=[str(navigation / "config" / "mission_perception.yaml"), {
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "color_topic": "/camera/camera/color/image_raw",
            "imu_topic": "/depth_slam/perception/unused_orientation_imu",
            "imu_valid_topic": "/depth_slam/perception/unused_imu_valid",
            "debug_overlay_enabled": True,
            "publish_stop_line_tf": True,
        }])
    validator = Node(
        package="depth_hybrid_slam", executable="perception_vslam_validator",
        name="perception_vslam_validator", output="screen")
    overlay = Node(
        package="depth_hybrid_slam", executable="perception_vslam_overlay",
        name="perception_vslam_overlay", output="screen")
    rviz = Node(
        package="rviz2", executable="rviz2", output="screen",
        arguments=["-d", str(depth / "config" / "perception_vslam.rviz")],
        condition=IfCondition(LaunchConfiguration("start_rviz")))
    image = Node(
        package="rqt_image_view", executable="rqt_image_view",
        name="perception_vslam_image_view", output="screen",
        arguments=["/depth_slam/perception/debug_overlay"],
        condition=IfCondition(LaunchConfiguration("start_image_view")))
    return LaunchDescription(declarations + [
        OpaqueFunction(function=_guard), yolo_launch, rgb_node, fusion_node,
        mission_node, validator, overlay, rviz, image,
    ])
