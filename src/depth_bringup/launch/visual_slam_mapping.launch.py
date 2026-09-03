import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    cfg = str(Path(get_package_share_directory('depth_bringup')) /
              'config' / 'rtabmap_mapping.yaml')
    use_mcu = LaunchConfiguration('use_mcu_odom')
    rgb = LaunchConfiguration('rgb_topic')
    depth = LaunchConfiguration('depth_topic')
    info = LaunchConfiguration('camera_info_topic')
    imu = LaunchConfiguration('imu_topic')
    mcu_odom = LaunchConfiguration('odom_topic')
    graph_odom = (mcu_odom if LaunchConfiguration('use_mcu_odom').perform(context).lower()
                  in ('1', 'true', 'yes') else '/visual_odom')
    common_remaps = [('rgb/image', rgb), ('depth/image', depth),
                     ('rgb/camera_info', info), ('imu', imu)]
    delete_db = LaunchConfiguration('delete_db').perform(context).lower() in ('1', 'true', 'yes')
    database = os.path.expanduser(LaunchConfiguration('database_path').perform(context))
    mcu_enabled = LaunchConfiguration('use_mcu_odom').perform(context).lower()
    actions = [LogInfo(msg=(
        'Using MCU /odom; rgbd_odometry TF disabled.'
        if mcu_enabled in ('1', 'true', 'yes')
        else 'Using camera-only RGB-D odometry on /odom.'))]
    actions.append(Node(
        package='rtabmap_odom', executable='rgbd_odometry', name='rgbd_odometry',
        output='screen', parameters=[cfg, {'publish_tf': True, 'odom_frame_id': 'odom'}],
        remappings=common_remaps + [('odom', '/visual_odom')],
        condition=UnlessCondition(use_mcu),
    ))
    actions.append(Node(
        package='rtabmap_odom', executable='rgbd_odometry', name='rgbd_odometry',
        output='screen', parameters=[cfg, {
            'publish_tf': False, 'odom_frame_id': 'visual_odom'}],
        remappings=common_remaps + [('odom', '/visual_odom')],
        condition=IfCondition(use_mcu),
    ))
    rtab_args = ['--delete_db_on_start'] if delete_db else []
    actions.append(Node(
        package='rtabmap_slam', executable='rtabmap', name='rtabmap', namespace='rtabmap',
        output='screen', parameters=[cfg, {'database_path': database}], arguments=rtab_args,
        remappings=common_remaps + [('odom', graph_odom)],
    ))
    return actions


def generate_launch_description():
    default_db = os.path.expanduser('~/depth_ws/maps/rtabmap.db')
    args = {
        'use_mcu_odom': ('true', 'Use existing MCU odometry and odom->base_link TF.'),
        'odom_topic': ('/odom', 'Odometry input/output topic.'),
        'rgb_topic': ('/camera/camera/color/image_raw', 'RGB image topic.'),
        'depth_topic': (
            '/camera/camera/aligned_depth_to_color/image_raw',
            'Aligned depth image topic.'),
        'camera_info_topic': ('/camera/camera/color/camera_info', 'RGB camera info topic.'),
        'imu_topic': ('/camera/camera/imu', 'Unified IMU topic.'),
        'database_path': (default_db, 'RTAB-Map database path.'),
        'delete_db': (
            'false', 'Pass --delete_db_on_start to RTAB-Map for a new database.'),
    }
    declarations = [DeclareLaunchArgument(k, default_value=v[0], description=v[1])
                    for k, v in args.items()]
    return LaunchDescription(declarations + [OpaqueFunction(function=_nodes)])
