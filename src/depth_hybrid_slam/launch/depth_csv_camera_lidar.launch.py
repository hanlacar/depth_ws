"""Production CSV, D456 and one-front-LiDAR stack; MCU ODOM attaches later."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.workspace_paths import workspace_root
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    root = workspace_root()
    route = str(root/"routes"/"network"/
                "route_network_segmented_stop_edited_vforward.csv")
    return LaunchDescription([
        DeclareLaunchArgument("route_path", default_value=route),
        DeclareLaunchArgument(
            "route_metadata_path",
            default_value=route.replace(".csv", ".metadata.yaml")),
        DeclareLaunchArgument("start_branch", default_value="A"),
        DeclareLaunchArgument("start_mode", default_value="1"),
        DeclareLaunchArgument("end_mode", default_value="11"),
        DeclareLaunchArgument("enable_vslam", default_value="true"),
        DeclareLaunchArgument("enable_parking_slam", default_value="false"),
        DeclareLaunchArgument(
            "map_path", default_value=(
                str(root/"maps"/"merged_competition_level_aligned_v10"/
                    "rtabmap.db"))),
        DeclareLaunchArgument("front_serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("camera_serial", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("enable_control", default_value="true"),
        DeclareLaunchArgument("user_approved", default_value="true"),
        DeclareLaunchArgument("enable_rosbag", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                share/"launch"/"depth_real_vehicle.launch.py")),
            launch_arguments={
                "route_path": LaunchConfiguration("route_path"),
                "route_metadata_path": LaunchConfiguration(
                    "route_metadata_path"),
                "start_branch": LaunchConfiguration("start_branch"),
                "start_mode": LaunchConfiguration("start_mode"),
                "end_mode": LaunchConfiguration("end_mode"),
                "enable_vslam": LaunchConfiguration("enable_vslam"),
                "enable_parking_slam": LaunchConfiguration(
                    "enable_parking_slam"),
                "map_path": LaunchConfiguration("map_path"),
                "front_serial_port": LaunchConfiguration(
                    "front_serial_port"),
                "camera_serial": LaunchConfiguration("camera_serial"),
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda"),
                "use_lidar": "true",
                "use_camera": "true",
                "enable_control": LaunchConfiguration("enable_control"),
                "user_approved": LaunchConfiguration("user_approved"),
                "enable_rosbag": LaunchConfiguration("enable_rosbag"),
            }.items()),
    ])
