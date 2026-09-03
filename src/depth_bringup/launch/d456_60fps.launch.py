from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    rs_launch = str(Path(get_package_share_directory('realsense2_camera')) /
                    'launch' / 'rs_launch.py')
    arguments = {
        'serial_no': "''",
        'color_profile': '640,480,60',
        # This conservative fallback is not a D456 capability claim. Override only
        # with a profile shown by rs-enumerate-devices for the connected unit.
        'depth_profile': '640,480,30',
        'gyro_fps': '0',
        'accel_fps': '0',
        'align_depth': 'true',
        'enable_pointcloud': 'false',
        'publish_camera_tf': 'true',
    }
    declared = [DeclareLaunchArgument(name, default_value=value)
                for name, value in arguments.items()]
    rs_args = {
        'camera_name': 'camera',
        'camera_namespace': 'camera',
        'serial_no': LaunchConfiguration('serial_no'),
        'enable_color': 'true',
        'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
        'rgb_camera.color_format': 'RGB8',
        'enable_depth': 'true',
        'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
        'depth_module.depth_format': 'Z16',
        'enable_infra': 'false',
        'enable_infra1': 'false',
        'enable_infra2': 'false',
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'gyro_fps': LaunchConfiguration('gyro_fps'),
        'accel_fps': LaunchConfiguration('accel_fps'),
        'unite_imu_method': '2',
        'enable_sync': 'true',
        'align_depth.enable': LaunchConfiguration('align_depth'),
        'pointcloud.enable': LaunchConfiguration('enable_pointcloud'),
        'colorizer.enable': 'false',
        'decimation_filter.enable': 'false',
        'spatial_filter.enable': 'false',
        'temporal_filter.enable': 'false',
        'hole_filling_filter.enable': 'false',
        'publish_tf': LaunchConfiguration('publish_camera_tf'),
        'tf_publish_rate': '0.0',
        'base_frame_id': 'camera_link',
        'diagnostics_period': '1.0',
    }
    return LaunchDescription(declared + [
        LogInfo(msg=['D456 requested profiles: color=', LaunchConfiguration('color_profile'),
                     ', depth=', LaunchConfiguration('depth_profile'),
                     '. Confirm against rs-enumerate-devices before use.']),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(rs_launch),
                                 launch_arguments=rs_args.items()),
    ])
