"""Non-BEV (pixel-space) camera driving stack.

Pipeline:
  D456 -> YOLO seg -> camera_image_path_node (pixel path)
       -> camera_pixel_controller_node (PD on pixel offset)
       -> internal pixel candidate topics

Deliberately does NOT launch camera_metric_path_node and needs NO
camera_mount.yaml or IMU attitude lock for steering. The existing imu_manager
is launched only to provide the filtered /imu/slope and /imu/valid inputs for
the independent one-shot uphill longitudinal stop.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_share = get_package_share_directory("camera_bringup")
    yolo_share = get_package_share_directory("camera_yolo_inference")
    nav_share = get_package_share_directory("camera_navigation")
    imu_share = get_package_share_directory("imu_manager")

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(camera_share, "launch", "d456_bringup.launch.py")
        ),
        launch_arguments={"serial_no": LaunchConfiguration("serial_no")}.items(),
    )
    yolo = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(yolo_share, "launch", "yolo_inference.launch.py")))
    imu = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(imu_share, "launch", "imu_manager.launch.py")))

    image_path = Node(
        package="camera_navigation", executable="adaptive_non_bev_node",
        name="camera_image_path_node", output="screen",
        parameters=[os.path.join(nav_share, "config", "image_path.yaml"),
                    os.path.join(nav_share, "config", "adaptive_non_bev.yaml"),
                    {"require_control_mode": False}])
    controller = Node(
        package="camera_navigation", executable="adaptive_pixel_controller_node",
        name="camera_pixel_controller_node", output="screen",
        parameters=[os.path.join(nav_share, "config", "camera_pixel_controller.yaml"),
                    os.path.join(nav_share, "config", "adaptive_non_bev.yaml")])
    selector = Node(
        package="camera_navigation", executable="camera_command_selector_node",
        name="camera_command_selector_node", output="screen",
        parameters=[{"candidate_drive_topic": "/camera/candidate/pixel/drive",
                     "candidate_wheel_topic": "/camera/candidate/pixel/wheel"}])

    return LaunchDescription([
        DeclareLaunchArgument(
            "serial_no",
            default_value="",
            description="Optional D456 serial; empty uses automatic discovery",
        ),
        LogInfo(msg="Pixel camera stack: IMU uphill state enabled; no BEV/extrinsics"),
        camera, imu, yolo, image_path, controller, selector,
    ])
