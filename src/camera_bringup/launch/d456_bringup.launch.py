"""Start exactly one D456 wrapper with stable depth_ws public topics."""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    width = int(LaunchConfiguration("color_width").perform(context))
    height = int(LaunchConfiguration("color_height").perform(context))
    fps = int(LaunchConfiguration("color_fps").perform(context))
    if min(width, height, fps) <= 0:
        raise RuntimeError("D456 width, height, and FPS must be positive")
    serial = LaunchConfiguration("serial_no").perform(context).strip()
    def enabled(name):
        return LaunchConfiguration(name).perform(context).strip().lower() in (
            "1", "true", "yes", "on")

    depth = enabled("enable_depth")
    imu = enabled("enable_imu")
    vslam = enabled("enable_vslam")
    overrides = {
        "rgb_camera.color_profile": f"{width}x{height}x{fps}",
        "enable_depth": depth or vslam,
        "align_depth.enable": depth or vslam,
        "enable_sync": depth or vslam,
        # Production RTAB-Map uses aligned RGB-D. Keep unused stereo streams
        # off so the shared USB hub is not saturated.
        "enable_infra1": False,
        "enable_infra2": False,
        "depth_module.infra_profile": "640x480x60",
        "enable_gyro": imu or vslam,
        "enable_accel": imu or vslam,
        "gyro_fps": 400 if vslam else 200,
        "accel_fps": 400 if vslam else 100,
        "unite_imu_method": 2 if vslam else 0,
    }
    if serial:
        overrides["serial_no"] = serial
    config = Path(get_package_share_directory("camera_bringup"))/"config"/"d456.yaml"
    return [Node(
        package="realsense2_camera", executable="realsense2_camera_node",
        namespace="camera", name="camera", output="screen",
        parameters=[str(config), overrides],
        remappings=[
            ("/camera/camera/color/image_raw", "/camera/image_raw"),
            ("/camera/camera/color/camera_info", "/camera/camera_info"),
            ("/camera/camera/aligned_depth_to_color/image_raw",
             "/camera/aligned_depth_to_color/image_raw")])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("serial_no", default_value=""),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="480"),
        DeclareLaunchArgument("color_fps", default_value="60"),
        DeclareLaunchArgument("enable_depth", default_value="true"),
        DeclareLaunchArgument("enable_imu", default_value="true"),
        DeclareLaunchArgument("enable_vslam", default_value="false"),
        OpaqueFunction(function=_setup)])
