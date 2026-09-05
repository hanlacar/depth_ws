"""Camera-topic mission state machine with all propulsion disabled."""

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = get_package_share_directory("depth_hybrid_slam")+"/config/mission.yaml"
    mission = Node(package="depth_hybrid_slam", executable="mission_manager",
                   name="mission_manager", output="screen", parameters=[config])
    return LaunchDescription(common_arguments()+[mission])
