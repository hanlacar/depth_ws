from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory('t870_mcu_simple')
    cfg = os.path.join(share, 'config', 'mcu.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='auto'),
        Node(
            package='t870_mcu_simple',
            executable='bridge',
            name='t870_mcu_simple_bridge',
            output='screen',
            parameters=[cfg, {'port': LaunchConfiguration('port')}],
        )
    ])
