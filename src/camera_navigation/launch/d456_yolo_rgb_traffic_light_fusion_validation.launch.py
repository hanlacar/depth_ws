"""Run D456, existing YOLO, RGB detector, and fusion in parallel."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = Path(get_package_share_directory("camera_bringup"))
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    rgb = Path(get_package_share_directory("camera_rgb_traffic_light"))
    navigation = Path(get_package_share_directory("camera_navigation"))
    declarations = [
        DeclareLaunchArgument("launch_camera", default_value="true"),
        DeclareLaunchArgument("launch_yolo", default_value="true"),
        DeclareLaunchArgument("launch_rgb", default_value="true"),
        DeclareLaunchArgument("launch_fusion", default_value="true"),
        DeclareLaunchArgument("launch_rqt", default_value="false"),
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
    ]
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            camera / "launch" / "d456_bringup.launch.py")),
        launch_arguments={"serial_no": LaunchConfiguration("serial_no")}.items(),
        condition=IfCondition(LaunchConfiguration("launch_camera")))
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            yolo / "launch" / "yolo_inference.launch.py")),
        launch_arguments={
            "input_image_topic": "/camera/image_raw",
            "device": LaunchConfiguration("device"),
            "require_cuda": LaunchConfiguration("require_cuda"),
            "enable_depth_assist": "false",
        }.items(), condition=IfCondition(LaunchConfiguration("launch_yolo")))
    rgb_node = Node(
        package="camera_rgb_traffic_light", executable="rgb_traffic_light_node",
        name="rgb_traffic_light_node", output="screen",
        parameters=[str(rgb / "config" / "rgb_traffic_light.yaml"),
                    {"input_image_topic": "/camera/image_raw"}],
        condition=IfCondition(LaunchConfiguration("launch_rgb")))
    fusion_node = Node(
        package="camera_navigation", executable="traffic_light_fusion_node",
        name="traffic_light_fusion_node", output="screen",
        parameters=[str(navigation / "config" / "traffic_light_fusion.yaml")],
        condition=IfCondition(LaunchConfiguration("launch_fusion")))
    rqt = Node(
        package="rqt_image_view", executable="rqt_image_view",
        name="traffic_light_fusion_rqt", output="screen",
        arguments=["/camera/traffic_light_rgb/overlay_image"],
        condition=IfCondition(LaunchConfiguration("launch_rqt")))
    return LaunchDescription(declarations + [
        camera_launch, yolo_launch, rgb_node, fusion_node, rqt])
