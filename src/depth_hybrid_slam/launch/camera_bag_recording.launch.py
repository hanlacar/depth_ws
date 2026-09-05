"""Launch the fail-closed host competition recorder (never vehicle control)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("case_id"),
        DeclareLaunchArgument("purpose", default_value="competition"),
        DeclareLaunchArgument("duration", default_value="0"),
        DeclareLaunchArgument("profile", default_value="camera_slam"),
        ExecuteProcess(cmd=[
            "/home/qor/depth_ws/scripts/record_competition_bag.sh",
            "--case", LaunchConfiguration("case_id"),
            "--purpose", LaunchConfiguration("purpose"),
            "--duration", LaunchConfiguration("duration"),
            "--profile", LaunchConfiguration("profile"),
        ], output="screen"),
    ])
