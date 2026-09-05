"""Read-only performance probe for an already running cuVSLAM graph."""

from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    probe = ExecuteProcess(
        cmd=["ros2", "run", "depth_hybrid_slam", "performance_probe",
             "--duration", "60", "/visual_slam/tracking/odometry",
             "/depth_slam/cuvslam/odometry"], output="screen")
    return LaunchDescription(common_arguments()+[probe])
