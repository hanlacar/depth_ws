"""Shared launch arguments and RTAB-Map wiring for the hybrid stack."""

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


COMMON_DEFAULTS = {
    "camera_serial": "338122302896",
    "case_id": "",
    "map_path": "",
    "route_path": "",
    "mapping_mode": "false",
    "localization_mode": "false",
    "delete_test_db": "false",
    "use_gpu_vslam": "true",
    "start_rviz": "false",
    "start_image_view": "false",
    "start_monitor": "true",
    "enable_control": "false",
    "dry_run": "true",
    "user_approved": "false",
    "high_density": "false",
    "session_id": "",
    "auto_session_name": "true",
    "record_bag": "false",
    "quality_report": "",
    "route_min_distance_m": "0.05",
    "route_min_angle_rad": "0.03",
    "graph_snapshot_path": "",
    "final_route_path": "",
    "use_temporary_db_copy": "true",
    "publish_map_tf_for_view": "true",
    "highest_density_view": "false",
}


def common_arguments(overrides=None):
    values = dict(COMMON_DEFAULTS)
    values.update(overrides or {})
    return [DeclareLaunchArgument(name, default_value=value) for name, value in values.items()]


def rtabmap_include(localization, launch_prefix="", rtabmap_args=None,
                    database_path=None):
    launch_file = (get_package_share_directory("rtabmap_launch") +
                   "/launch/rtabmap.launch.py")
    # RTAB-Map consumes the bridge odometry. Its own rgbd/stereo odometry nodes
    # are explicitly disabled, so only cuVSLAM owns odom->base_link.
    arguments = {
        "localization": "true" if localization else "false",
        "database_path": (database_path if database_path is not None else
                          LaunchConfiguration("map_path")),
        "frame_id": "base_link",
        "map_frame_id": "map",
        # cuVSLAM's Odometry message has zero XYZ covariance. In localization
        # mode RTAB-Map 0.22.1 can turn that into an invalid zero-information
        # link, so use the equivalent cuVSLAM TF with explicit variances.
        "odom_frame_id": "odom" if localization else "",
        "odom_topic": "/depth_slam/cuvslam/odometry",
        "visual_odometry": "false",
        "icp_odometry": "false",
        "publish_tf_odom": "false",
        "publish_tf_map": "true",
        "odom_tf_linear_variance": "0.001",
        "odom_tf_angular_variance": "0.01",
        "depth": "true",
        "subscribe_rgb": "true",
        "rgbd_sync": "false",
        "approx_sync": "true",
        "approx_sync_max_interval": "0.025",
        "rgb_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
        # The D456 raw/united IMU has no orientation. cuVSLAM already consumes it;
        # feeding it to RTAB-Map would be ignored at 400 Hz and flood diagnostics.
        "imu_topic": "/depth_slam/unused_orientation_imu",
        "qos": "2",
        "qos_odom": "2",
        "topic_queue_size": "30",
        "sync_queue_size": "30",
        "rtabmap_viz": LaunchConfiguration("start_rviz"),
        "rviz": LaunchConfiguration("start_rviz"),
        "launch_prefix": launch_prefix,
        "args": rtabmap_args or (
            "--Rtabmap/DetectionRate 5 --RGBD/LinearUpdate 0.05 "
            "--RGBD/AngularUpdate 0.03 --RGBD/CreateOccupancyGrid true "
            "--Vis/MinInliers 15 --Mem/NotLinkedNodesKept false "
            "--Mem/LocalizationDataSaved false"),
    }
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments=arguments.items())


def offline_rtabmap_include(database_path, launch_prefix=""):
    """Load a saved database without camera, odometry, or persistence writes."""
    launch_file = (get_package_share_directory("rtabmap_launch") +
                   "/launch/rtabmap.launch.py")
    arguments = {
        "localization": "true",
        "database_path": database_path,
        "frame_id": "base_link",
        "map_frame_id": "map",
        "odom_frame_id": "",
        "visual_odometry": "false",
        "icp_odometry": "false",
        "publish_tf_odom": "false",
        "publish_tf_map": "false",
        "depth": "false",
        "subscribe_rgb": "false",
        "subscribe_rgbd": "false",
        "subscribe_stereo": "false",
        "rtabmap_viz": "false",
        "rviz": "false",
        "launch_prefix": launch_prefix,
        "args": ("--Mem/IncrementalMemory false --Mem/InitWMWithAllNodes true "
                 "--Mem/LocalizationDataSaved false"),
    }
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file),
        launch_arguments=arguments.items())
