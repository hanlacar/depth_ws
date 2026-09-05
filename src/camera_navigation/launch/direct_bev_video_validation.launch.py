"""Direct BEV stack for mp4-video validation -- same nodes as
camera_bev_standalone.launch.py, but WITHOUT d456_bringup.launch.py.

Why a separate launch file instead of reusing camera_bev_standalone.launch.py:
that launch unconditionally IncludeLaunchDescription()s d456_bringup, which
starts realsense2_camera_node and registers an OnProcessExit handler that
calls Shutdown() on the WHOLE launch the moment that node exits. With no
physical D456 attached, the RealSense driver fails to find a device and
exits almost immediately -- which would tear down yolo_inference and the
BEV nodes too, before a single video frame is even published. There is no
launch argument to skip d456_bringup in the standalone file, so composing a
new launch (option (a) from the investigation) is the clean path rather than
fighting camera_bev_standalone's fixed structure (option (b)).

Feed /camera/image_raw and /camera/camera_info yourself (tools/video_publisher.py)
and /camera/camera/accel|gyro/sample yourself (tools/fake_imu_publisher.py) --
this launch starts neither.

active_planner defaults to "none" on purpose, matching
camera_bev_standalone.launch.py: bev_wheel_selector_node only relays
/camera/bev/wheel and /camera/bev/path onto /camera_wheel and /camera/path
when active_planner=="bev". Pass active_planner:=bev explicitly.

Also starts direct_bev_drive_node (this package's new thin /camera_drive
publisher -- see its module docstring for why camera_path_controller_node
cannot be reused here). Its drive_rate_hz is wired to the same
control_rate_hz launch argument as direct_bev_controller_node so
/camera_drive and /camera/bev/wheel (-> /camera_wheel) stay at the same
fixed rate; control_rate_hz's default is 30.0 (was 20.0 in
camera_bev_standalone.launch.py) to match the 30fps validation target.

valid_min_safe_coverage defaults to "0.999" (bev_path.yaml's own value, left
untouched on disk) but can be overridden at launch time, e.g.
valid_min_safe_coverage:=0.7 -- direct_bev_planner_node declares it via a
plain declare_parameter(name, default) loop over DirectBevConfig's
dataclass fields (see its __init__), so a parameter dict placed AFTER
bev_path.yaml in this Node's `parameters=[...]` list overrides it exactly
like debug_enabled/fixed_output_rate_enabled/output_rate_hz already do
above -- no copied yaml file needed.
"""
import os
import hashlib
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera = get_package_share_directory("camera_bringup")
    yolo = get_package_share_directory("camera_yolo_inference")
    navigation = get_package_share_directory("camera_navigation")
    model = os.path.join(yolo, "models", "hanla_yolo11n_seg_best.pt")
    backend = "pytorch"
    expected_sha256 = hashlib.sha256(Path(model).read_bytes()).hexdigest()
    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("active_planner", default_value="none"),
        DeclareLaunchArgument("planner_variant", default_value="production"),
        DeclareLaunchArgument("road_boundary_fallback", default_value="none"),
        DeclareLaunchArgument("backend", default_value=backend),
        DeclareLaunchArgument("segmentation_model_path", default_value=model),
        DeclareLaunchArgument("expected_model_sha256", default_value=expected_sha256),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("fixed_output_rate_enabled", default_value="false"),
        DeclareLaunchArgument("output_rate_hz", default_value="60.0"),
        DeclareLaunchArgument("control_rate_hz", default_value="30.0"),
        DeclareLaunchArgument("valid_min_safe_coverage", default_value="0.999"),
        DeclareLaunchArgument("line_track_mode", default_value="none"),
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
                "line_track_mode": LaunchConfiguration("line_track_mode"),
            }.items()),
        Node(package="camera_navigation", executable="direct_bev_planner_node",
             name="direct_bev_planner_node", output="screen",
             parameters=[os.path.join(camera, "config", "camera_mount.yaml"),
                         os.path.join(navigation, "config", "bev_path.yaml"),
                         {"debug_enabled": LaunchConfiguration("debug"),
                          "planner_variant": LaunchConfiguration(
                              "planner_variant"),
                          "road_boundary_fallback": LaunchConfiguration(
                              "road_boundary_fallback"),
                          "fixed_output_rate_enabled": LaunchConfiguration(
                              "fixed_output_rate_enabled"),
                          "output_rate_hz": LaunchConfiguration(
                              "output_rate_hz"),
                          "valid_min_safe_coverage": LaunchConfiguration(
                              "valid_min_safe_coverage")}]),
        Node(package="camera_navigation", executable="direct_bev_controller_node",
             name="direct_bev_controller_node", output="screen",
             parameters=[os.path.join(navigation, "config", "bev_controller.yaml"),
                         {"control_rate_hz": LaunchConfiguration(
                              "control_rate_hz"),
                          "fixed_output_rate_enabled": LaunchConfiguration(
                              "fixed_output_rate_enabled")}]),
        Node(package="camera_navigation", executable="bev_wheel_selector_node",
             name="bev_wheel_selector_node", output="screen",
             parameters=[{"active_planner": LaunchConfiguration("active_planner")}]),
        Node(package="camera_navigation", executable="direct_bev_drive_node",
             name="direct_bev_drive_node", output="screen",
             parameters=[{"drive_rate_hz": LaunchConfiguration("control_rate_hz")}]),
    ])
