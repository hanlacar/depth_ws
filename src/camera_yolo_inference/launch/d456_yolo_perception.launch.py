"""Perception-only D456 + CUDA YOLO-Seg validation profile."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_share = Path(get_package_share_directory("camera_bringup"))
    yolo_share = Path(get_package_share_directory("camera_yolo_inference"))
    declarations = [
        DeclareLaunchArgument("launch_camera", default_value="true"),
        DeclareLaunchArgument("launch_rqt", default_value="false"),
        DeclareLaunchArgument(
            "serial_no", default_value="",
            description="Optional D456 serial; empty uses automatic discovery"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("traffic_light_confirmation_frames", default_value="3"),
        DeclareLaunchArgument("traffic_light_lost_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("traffic_light_minimum_confidence", default_value="0.5"),
        DeclareLaunchArgument(
            "rmw_implementation",
            default_value="rmw_cyclonedds_cpp",
            description="Validated RMW for sustained D456 raw image delivery",
        ),
    ]
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(camera_share / "launch" / "d456_bringup.launch.py")),
        launch_arguments={"serial_no": LaunchConfiguration("serial_no")}.items(),
        condition=IfCondition(LaunchConfiguration("launch_camera")))
    inference = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(yolo_share / "launch" / "yolo_inference.launch.py")),
        launch_arguments={"device": LaunchConfiguration("device"),
                          "require_cuda": LaunchConfiguration("require_cuda"),
                          "traffic_light_confirmation_frames":
                              LaunchConfiguration("traffic_light_confirmation_frames"),
                          "traffic_light_lost_timeout_sec":
                              LaunchConfiguration("traffic_light_lost_timeout_sec"),
                          "traffic_light_minimum_confidence":
                              LaunchConfiguration("traffic_light_minimum_confidence"),
                          "enable_depth_assist": "false"}.items())
    rqt = Node(package="rqt_image_view", executable="rqt_image_view",
               arguments=["/camera/perception_overlay_image"],
               condition=IfCondition(LaunchConfiguration("launch_rqt")), output="screen")
    middleware = SetEnvironmentVariable(
        "RMW_IMPLEMENTATION", LaunchConfiguration("rmw_implementation")
    )
    return LaunchDescription(declarations + [middleware, camera, inference, rqt])
