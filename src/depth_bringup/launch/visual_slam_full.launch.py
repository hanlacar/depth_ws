from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(package, filename, arguments=None, condition=None):
    path = str(Path(get_package_share_directory(package)) / 'launch' / filename)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=(arguments or {}).items(),
                                    condition=condition)


def generate_launch_description():
    defaults = {
        'serial_no': "''", 'color_profile': '640,480,60',
        'depth_profile': '640,480,30', 'gyro_fps': '0', 'accel_fps': '0',
        'use_mcu_odom': 'true', 'odom_topic': '/odom',
        'database_path': '~/depth_ws/maps/rtabmap.db', 'delete_db': 'false',
        'publish_mount_tf': 'true',
        'start_rviz': 'false', 'overlay_enabled': 'true',
    }
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    camera_args = {k: LaunchConfiguration(k) for k in
                   ['serial_no', 'color_profile', 'depth_profile', 'gyro_fps', 'accel_fps']}
    slam_args = {k: LaunchConfiguration(k) for k in
                 ['use_mcu_odom', 'odom_topic', 'database_path', 'delete_db']}
    monitor = Node(
        package='depth_monitor', executable='fps_monitor', name='depth_slam_monitor',
        output='screen', parameters=[{
            'visual_odom_topic': '/visual_odom',
            'overlay_enabled': LaunchConfiguration('overlay_enabled'),
        }],
    )
    return LaunchDescription(declarations + [
        include('depth_description', 'd456_mount.launch.py',
                condition=IfCondition(LaunchConfiguration('publish_mount_tf'))),
        include('depth_bringup', 'd456_60fps.launch.py', camera_args),
        include('depth_bringup', 'visual_slam_mapping.launch.py', slam_args),
        monitor,
        include('depth_bringup', 'visual_slam_view.launch.py',
                {'use_rtabmap_viz': 'true'}, IfCondition(LaunchConfiguration('start_rviz'))),
    ])
