"""BEV (metric) camera driving stack.

Pipeline:
  D456 -> YOLO seg -> camera_image_path_node (pixel path)
       -> camera_metric_path_node (BEV projection, needs camera_mount.yaml +
          boot-time IMU attitude lock)
       -> camera_path_controller_node (metric Pure Pursuit)
       -> internal candidate topics -> camera_command_selector_node

Requires an accurate camera_mount.yaml (height, pitch, forward offset). The
metric path only publishes while calibration is valid; otherwise the
controller fail-safes to STOP.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    camera_share = get_package_share_directory("camera_bringup")
    imu_share = get_package_share_directory("imu_manager")
    yolo_share = get_package_share_directory("camera_yolo_inference")
    nav_share = get_package_share_directory("camera_navigation")

    camera = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(camera_share, "launch", "d456_bringup.launch.py")))
    imu = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(imu_share, "launch", "imu_manager.launch.py")))
    yolo = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(yolo_share, "launch", "yolo_inference.launch.py")))

    image_path = Node(
        package="camera_navigation", executable="camera_image_path_node",
        name="camera_image_path_node", output="screen",
        parameters=[os.path.join(nav_share, "config", "image_path.yaml")])
    metric_path = Node(
        package="camera_navigation", executable="camera_metric_path_node",
        name="camera_metric_path_node", output="screen",
        parameters=[os.path.join(camera_share, "config", "camera_mount.yaml")])
    controller = Node(
        package="camera_navigation", executable="camera_path_controller_node",
        name="camera_path_controller_node", output="screen",
        parameters=[os.path.join(nav_share, "config", "camera_path_controller.yaml")])
    selector = Node(
        package="camera_navigation", executable="camera_command_selector_node",
        name="camera_command_selector_node", output="screen",
        parameters=[os.path.join(nav_share, "config", "camera_command_selector.yaml")])

    return LaunchDescription([
        LogInfo(msg="BEV camera stack: needs camera_mount.yaml + IMU lock"),
        camera, imu, yolo, image_path, metric_path, controller, selector,
    ])
