"""Self-contained depth_ws camera perception, mission, and command stack."""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
        Node(package="camera_navigation", executable="camera_image_path_node",
             name="camera_image_path_node", parameters=[str(
                 nav/"config"/"image_path.yaml")]),
        Node(package="camera_navigation", executable="camera_metric_path_node",
             name="camera_metric_path_node", parameters=[str(
                 bringup/"config"/"camera_mount.yaml")]),
        Node(package="camera_navigation", executable="camera_path_controller_node",
             name="camera_path_controller_node", parameters=[str(
                 nav/"config"/"camera_path_controller.yaml")]),
        Node(package="camera_navigation", executable="camera_mission_perception_node",
             name="camera_mission_perception_node", parameters=[str(
                 nav/"config"/"mission_perception.yaml")]),
        Node(package="camera_rgb_traffic_light", executable="rgb_traffic_light_node",
             name="rgb_traffic_light_node", parameters=[str(
                 rgb/"config"/"rgb_traffic_light.yaml")]),
        Node(package="camera_navigation", executable="traffic_light_fusion_node",
             name="traffic_light_fusion_node", parameters=[str(
                 nav/"config"/"traffic_light_fusion.yaml")]),
        Node(package="camera_navigation", executable="camera_mission_decision_node",
             name="camera_mission_decision_node", parameters=[str(
                 nav/"config"/"mission_decision.yaml")]),
        Node(package="camera_navigation", executable="camera_command_selector_node",
             name="camera_command_selector_node", parameters=[str(
                 nav/"config"/"camera_command_selector.yaml")]),
    ])
