"""Record manual-drive localization poses directly in the VSLAM map frame."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = get_package_share_directory("depth_hybrid_slam") + \
        "/config/classroom_localization.rviz"
    arguments = [
        DeclareLaunchArgument("output_directory", default_value="/home/qor/depth_ws/routes/recorded_map"),
        DeclareLaunchArgument("map_path", default_value="/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db"),
        DeclareLaunchArgument("gps_reference_path", default_value="/home/qor/depth_ws/routes/network/route_network_segmented.csv"),
        DeclareLaunchArgument("resample_spacing_m", default_value="0.10"),
        DeclareLaunchArgument("stationary_duplicate_m", default_value="0.01"),
        DeclareLaunchArgument("recovery_jump_m", default_value="0.75"),
        DeclareLaunchArgument("pose_timeout_s", default_value="0.50"),
        DeclareLaunchArgument("use_drive_command_direction", default_value="false"),
        DeclareLaunchArgument("drive_command_topic", default_value="/slam_drive"),
        DeclareLaunchArgument("start_rviz", default_value="false"),
    ]
    recorder = Node(
        package="depth_hybrid_slam", executable="map_route_recorder",
        name="map_route_recorder", output="screen", parameters=[{
            "output_directory": LaunchConfiguration("output_directory"),
            "map_path": LaunchConfiguration("map_path"),
            "gps_reference_path": LaunchConfiguration("gps_reference_path"),
            "resample_spacing_m": LaunchConfiguration("resample_spacing_m"),
            "stationary_duplicate_m": LaunchConfiguration("stationary_duplicate_m"),
            "recovery_jump_m": LaunchConfiguration("recovery_jump_m"),
            "pose_timeout_s": LaunchConfiguration("pose_timeout_s"),
            "use_drive_command_direction": LaunchConfiguration("use_drive_command_direction"),
            "drive_command_topic": LaunchConfiguration("drive_command_topic"),
            "wheelbase_m": 0.73, "max_steering_deg": 22.0,
        }])
    rviz = Node(
        package="rviz2", executable="rviz2", output="screen",
        arguments=["-d", config],
        condition=IfCondition(LaunchConfiguration("start_rviz")))
    return LaunchDescription(arguments + [recorder, rviz])
