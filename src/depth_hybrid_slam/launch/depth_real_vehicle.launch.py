"""T870 final real-vehicle graph: measured ODOM, one LiDAR, D456, one arbiter."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from depth_hybrid_slam.rosbag_rotation import prepare_bag_path
from depth_hybrid_slam.route_follower_core import validate_mode_range


ROOT = "/home/qor/depth_ws"


def _runtime(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    route_path = Path(value("route_path")).expanduser().resolve()
    metadata_path = Path(value("route_metadata_path")).expanduser().resolve()
    branch = value("start_branch").strip().upper()
    if branch not in ("A", "B"):
        raise RuntimeError("start_branch must be A or B")
    try:
        start_mode, end_mode = validate_mode_range(
            value("start_mode"), value("end_mode"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    vslam_enabled = value("enable_vslam").strip().lower() in (
        "1", "true", "yes", "on")
    camera_enabled = value("use_camera").strip().lower() in (
        "1", "true", "yes", "on")
    if vslam_enabled and not camera_enabled:
        raise RuntimeError("enable_vslam=true requires use_camera=true")
    if not route_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("final route CSV and metadata are required")
    share = Path(get_package_share_directory("depth_hybrid_slam"))
    camera = Path(get_package_share_directory("camera_navigation"))
    enabled = ParameterValue(
        LaunchConfiguration("enable_control"), value_type=bool)
    approved = ParameterValue(
        LaunchConfiguration("user_approved"), value_type=bool)
    follower = [str(share/"config"/"vehicle_navigation.yaml"), {
        "route_path": str(route_path),
        "route_metadata_path": str(metadata_path),
        "map_path": "",
        "allow_odom_route_origin": True,
        "enable_control": enabled,
        "dry_run": False,
        "user_approved": approved,
        "start_mode": start_mode,
        "end_mode": end_mode,
    }]
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                share/"launch"/"dual_rplidar.launch.py")),
            launch_arguments={
                "launch_lidar_drivers": LaunchConfiguration("use_lidar"),
                "launch_rear_lidar_driver": "false",
                "front_serial_port": LaunchConfiguration("front_serial_port"),
            }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                camera/"launch"/"d456_traffic_light.launch.py")),
            launch_arguments={
                "serial_no": LaunchConfiguration("camera_serial"),
                "device": LaunchConfiguration("device"),
                "require_cuda": LaunchConfiguration("require_cuda"),
                "enable_vslam": LaunchConfiguration("enable_vslam"),
            }.items(), condition=IfCondition(LaunchConfiguration("use_camera"))),
        Node(
            package="depth_hybrid_slam", executable="odom_localization",
            name="odom_localization", output="screen",
            parameters=[str(share/"config"/"vehicle_navigation.yaml"), {
                "enable_vslam": vslam_enabled,
            }]),
        Node(
            package="depth_hybrid_slam", executable="route_follower",
            name="route_follower", output="screen", parameters=follower),
        Node(
            package="depth_hybrid_slam", executable="route_local_path",
            name="depth_route_local_path_connector", output="screen"),
        Node(
            package="depth_hybrid_slam", executable="csv_road_validator",
            name="csv_road_validator", output="screen", parameters=[
                str(Path(get_package_share_directory("camera_bringup")) /
                    "config"/"camera_mount.yaml"),
                str(share/"config"/"csv_road_validator.yaml")],
            condition=IfCondition(LaunchConfiguration("use_camera"))),
        Node(
            package="depth_hybrid_slam", executable="camera_correction",
            name="depth_camera_correction", output="screen",
            parameters=[str(share/"config"/"vehicle_navigation.yaml")],
            condition=IfCondition(LaunchConfiguration("use_camera"))),
        Node(
            package="depth_hybrid_slam", executable="branch_selector",
            name="depth_route_branch_selector", output="screen",
            parameters=[{
                "initial_branch": branch, "mode_11_wait_s": 5.5,
            }]),
        Node(
            package="depth_hybrid_slam",
            executable="csv_only_network_visualizer",
            name="depth_real_route_network_visualizer", output="screen",
            parameters=[{
                "route_path": str(route_path),
                "route_metadata_path": str(metadata_path),
            }]),
        Node(
            package="depth_hybrid_slam", executable="mission_manager",
            name="depth_mission_manager", output="screen",
            parameters=[str(share/"config"/"mission.yaml"), {
                "end_mode": end_mode,
            }]),
        Node(
            package="depth_hybrid_slam", executable="signal_exit",
            name="depth_signal_exit", output="screen",
            parameters=[str(share/"config"/"signal_exit.yaml")]),
        Node(
            package="depth_hybrid_slam", executable="lidar_perception",
            name="depth_lidar_perception", output="screen",
            parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(
            package="depth_hybrid_slam", executable="lidar_rejoin_validator",
            name="depth_lidar_rejoin_validator", output="screen",
            parameters=[str(share/"config"/"lidar_autonomy.yaml"), {
                "route_path": str(route_path),
                "route_metadata_path": str(metadata_path),
            }]),
        Node(
            package="depth_hybrid_slam", executable="maneuver_manager",
            name="depth_maneuver_manager", output="screen",
            parameters=[str(share/"config"/"lidar_autonomy.yaml"), {
                "route_path": str(route_path),
                "route_metadata_path": str(metadata_path),
                "rear_lidar_enabled": False,
            }]),
        Node(
            package="depth_hybrid_slam", executable="command_arbiter",
            name="depth_command_arbiter", output="screen",
            parameters=[str(share/"config"/"lidar_autonomy.yaml")]),
        Node(
            package="depth_hybrid_slam", executable="safety_monitor",
            name="depth_safety_monitor", output="screen", parameters=[{
                "enable_control": enabled, "dry_run": False,
                "user_approved": approved, "require_tracking": True,
                "require_map_route_match": False,
                "require_within_map": False,
                "localization_mode": (
                    "VSLAM" if vslam_enabled else "ODOM_ONLY"),
            }]),
        Node(
            package="depth_hybrid_slam", executable="runtime_monitor",
            name="depth_runtime_monitor", output="screen", parameters=[{
                "enable_vslam": vslam_enabled,
            }]),
    ]
    if vslam_enabled:
        actions.insert(2, IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(
                share/"launch"/"hybrid_localization.launch.py")),
            launch_arguments={
                "map_path": LaunchConfiguration("map_path"),
                "route_path": str(route_path),
                "route_metadata_path": str(metadata_path),
                "use_vehicle_odom": "true",
                "publish_camera_mount_tf": "false",
                "start_rviz": "false",
            }.items()))
    if value("enable_rosbag").strip().lower() in ("1", "true", "yes", "on"):
        bag_path = prepare_bag_path(ROOT+"/rosbags", max_bags=3)
        topics = (
            "/odom", "/tf", "/tf_static", "/cmd_drive", "/cmd_wheel",
            "/mcu/steer_deg", "/drive_mode",
            "/depth_slam/route/active_index",
            "/depth_slam/route/active_segment",
            "/depth_slam/route/mode_status", "/depth_slam/path_owner",
            "/depth_slam/csv_validation/local_path",
            "/depth_slam/lidar/local_path",
            "/depth_slam/camera/correction_path",
            "/depth_slam/lidar/perception",
            "/depth_slam/lidar/safety_event",
            "/depth_slam/camera/csv_validation",
            "/depth_slam/camera/correction_state",
            "/camera/traffic_light_fused/state",
            "/camera/exit_branch_signal",
            "/depth_slam/vslam/tracking_valid",
            "/depth_slam/vslam/confidence",
            "/depth_slam/vslam/evidence", "/imu/pitch_deg", "/imu/valid",
            "/depth_slam/runtime/watchdog",
            "/depth_slam/localization/watchdog",
            "/depth_slam/mission/state", "/depth_slam/mission/event",
            "/depth_slam/mission/mode_status",
            "/depth_slam/command_arbiter/state")
        actions.append(ExecuteProcess(
            cmd=["ros2", "bag", "record", "--output", str(bag_path),
                 *topics], output="screen"))
    return actions


def generate_launch_description():
    route = ROOT+"/routes/network/route_network_segmented_stop_edited_vforward.csv"
    return LaunchDescription([
        DeclareLaunchArgument("route_path", default_value=route),
        DeclareLaunchArgument(
            "route_metadata_path",
            default_value=ROOT+"/routes/network/route_network_segmented_stop_edited_vforward.metadata.yaml"),
        DeclareLaunchArgument("start_branch", default_value="A"),
        DeclareLaunchArgument("start_mode", default_value="1"),
        DeclareLaunchArgument("end_mode", default_value="11"),
        DeclareLaunchArgument("enable_vslam", default_value="true"),
        DeclareLaunchArgument(
            "map_path", default_value=ROOT +
            "/maps/merged_competition_level_aligned_v10/rtabmap.db"),
        DeclareLaunchArgument("front_serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("camera_serial", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("require_cuda", default_value="true"),
        DeclareLaunchArgument("use_lidar", default_value="true"),
        DeclareLaunchArgument("use_camera", default_value="true"),
        DeclareLaunchArgument("enable_control", default_value="false"),
        DeclareLaunchArgument("user_approved", default_value="false"),
        DeclareLaunchArgument("enable_rosbag", default_value="true"),
        OpaqueFunction(function=_runtime),
    ])
