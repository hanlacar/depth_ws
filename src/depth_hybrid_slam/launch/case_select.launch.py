"""Case verifier. Selection is read-only and fails closed on any mismatch."""

from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    verify = ExecuteProcess(
        cmd=["ros2", "run", "depth_hybrid_slam", "case_manager",
             "verify", LaunchConfiguration("case_id"), "--root", "maps"],
        output="screen")
    return LaunchDescription(common_arguments()+[verify])
