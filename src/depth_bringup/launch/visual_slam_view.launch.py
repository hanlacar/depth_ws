from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory('depth_bringup')) /
                 'config' / 'rviz_visual_slam.rviz')
    return LaunchDescription([
        DeclareLaunchArgument('use_rtabmap_viz', default_value='false'),
        Node(package='rviz2', executable='rviz2', name='depth_slam_rviz',
             output='screen', arguments=['-d', config],
             condition=IfCondition(LaunchConfiguration('use_rtabmap_viz'))),
    ])
