"""Launch the fail-closed host competition recorder (never vehicle control)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from depth_hybrid_slam.workspace_paths import workspace_path


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("case_id"),
        DeclareLaunchArgument("purpose", default_value="competition"),
        DeclareLaunchArgument("duration", default_value="0"),
        DeclareLaunchArgument("profile", default_value="camera_slam"),
        ExecuteProcess(cmd=[
            str(workspace_path("scripts", "record_competition_bag.sh")),
            "--case", LaunchConfiguration("case_id"),
            "--purpose", LaunchConfiguration("purpose"),
            "--duration", LaunchConfiguration("duration"),
            "--profile", LaunchConfiguration("profile"),
        ], output="screen"),
    ])
