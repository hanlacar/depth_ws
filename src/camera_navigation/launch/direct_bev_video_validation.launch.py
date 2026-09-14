"""Safe camera semantic validation graph; it publishes no drive command."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera = Path(get_package_share_directory("camera_bringup"))
    yolo = Path(get_package_share_directory("camera_yolo_inference"))
    declarations = [
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("active_planner", default_value="none"),
        DeclareLaunchArgument("planner_variant", default_value="production"),
        DeclareLaunchArgument("line_track_mode", default_value="none"),
        DeclareLaunchArgument("debug_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("mask_image_publish_hz", default_value="5.0"),
        DeclareLaunchArgument("overlay_publish_hz", default_value="5.0"),
    ]
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            camera / "launch" / "d456_bringup.launch.py")),
        launch_arguments={
            "serial_no": LaunchConfiguration("serial_no"),
            "enable_depth": "true", "enable_imu": "true",
        }.items())
    inference = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            yolo / "launch" / "yolo_inference.launch.py")),
        launch_arguments={
            "input_image_topic": "/camera/image_raw",
            "device": LaunchConfiguration("device"),
            "require_cuda": LaunchConfiguration("require_cuda"),
            "line_track_mode": LaunchConfiguration("line_track_mode"),
            "mask_image_publish_hz": LaunchConfiguration(
                "mask_image_publish_hz"),
            "detections_image_max_fps": LaunchConfiguration(
                "debug_image_publish_hz"),
            "perception_overlay_max_fps": LaunchConfiguration(
                "overlay_publish_hz"),
        }.items())
    return LaunchDescription(declarations + [bringup, inference])
