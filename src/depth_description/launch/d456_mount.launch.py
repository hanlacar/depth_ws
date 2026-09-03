from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    xacro_file = str(Path(get_package_share_directory('depth_description')) /
                     'urdf' / 'd456_mount.urdf.xacro')
    names = {
        'parent_frame': 'base_link', 'camera_frame': 'camera_link',
        'x': '0.32', 'y': '0.0', 'z': '0.85',
        'roll': '0.0', 'pitch': '-0.0872664626', 'yaw': '0.0',
    }
    declarations = [DeclareLaunchArgument(k, default_value=v) for k, v in names.items()]
    command = ['xacro ', xacro_file]
    for name in names:
        command += [' ', name, ':=', LaunchConfiguration(name)]
    robot_description = ParameterValue(Command(command), value_type=str)
    return LaunchDescription(declarations + [Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='d456_mount_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description}],
    )])

