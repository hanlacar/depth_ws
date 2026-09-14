"""TEST ONLY ODOM route loop; no fake sensor or vehicle runtime."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.csv_only_branching import (
    csv_only_network_segments, load_csv_only_route_case)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_ROOT = "/home/qor/depth_ws"


def _prepare(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    route = Path(value("route_path")).expanduser().resolve()
    metadata_text = value("route_metadata_path").strip()
    metadata = (Path(metadata_text).expanduser().resolve()
                if metadata_text else route.with_suffix(".metadata.yaml"))
    if not route.is_file() or not metadata.is_file():
        raise RuntimeError("canonical route and metadata must exist")
    segments = csv_only_network_segments(route, metadata)
    for required in ("START_A", "START_B", "T_A", "T_B", "V_A", "V_B",
                     "END_AA", "END_AB"):
        if required not in segments:
            raise RuntimeError(f"route network is missing {required}")
    branch = value("spawn_branch").strip().upper()
    start_mode, end_mode = int(value("start_mode")), int(value("end_mode"))
    if branch not in ("A", "B"):
        raise RuntimeError("spawn_branch must be A or B")
    if not 1 <= start_mode <= end_mode <= 11:
        raise RuntimeError("start_mode/end_mode must satisfy 1 <= start <= end <= 11")
    first = next(point for point in load_csv_only_route_case(
        route, metadata, branch+"AAA") if int(point.mode) == start_mode)
    display = Path(value("display_route_path")).expanduser().resolve()
    display_metadata = display.with_suffix(".metadata.yaml")
    common = {
        "route_path": str(route),
        "route_metadata_path": str(metadata),
        "map_path": str(route),
        "enable_control": True,
        "dry_run": False,
        "user_approved": True,
        "prehardware_test_override_alignment": True,
        "prehardware_csv_only_case_selection": True,
        "pose_timeout_s": 0.50,
        "localization_stability_s": 0.20,
        "start_mode": start_mode,
        "end_mode": end_mode,
        "wheelbase_m": 0.73,
        "max_steering_deg": 22.0,
        "controller_hz": float(value("controller_hz")),
    }
    return [
        LogInfo(msg=("TEST ONLY ODOM HIL: real encoder/steering, "
                     f"modes={start_mode}-{end_mode}, branch={branch}")),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="test_map_to_odom_tf", arguments=[
                 "--x", str(first.x), "--y", str(first.y), "--z", "0.0",
                 "--roll", "0.0", "--pitch", "0.0", "--yaw", str(first.yaw),
                 "--frame-id", "map", "--child-frame-id", "odom"]),
        Node(package="depth_hybrid_slam", executable="test_odom_publisher",
             name="test_odom_publisher", output="screen",
             parameters=[{"test_only_acknowledged": True}]),
        Node(package="depth_hybrid_slam", executable="odom_localization",
             name="odom_localization", output="screen"),
        Node(package="depth_hybrid_slam", executable="csv_only_branch_selector",
             name="depth_csv_only_branch_selector", output="screen",
             parameters=[{
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "start_mode": start_mode,
                 "spawn_branch": branch,
                 "prehardware_test_only": True,
             }]),
        Node(package="depth_hybrid_slam",
             executable="csv_only_network_visualizer",
             name="depth_csv_only_network_visualizer", output="screen",
             parameters=[{
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "display_route_path": str(display),
                 "display_metadata_path": str(display_metadata),
                 "prehardware_test_only": True,
             }]),
        Node(package="depth_hybrid_slam", executable="route_follower",
             name="route_follower", output="screen", parameters=[common]),
        Node(package="depth_hybrid_slam", executable="safety_monitor",
             name="depth_safety_monitor", output="screen", parameters=[{
                 "enable_control": True,
                 "dry_run": False,
                 "user_approved": True,
                 "require_tracking": True,
                 "require_map_route_match": False,
                 "require_within_map": False,
             }]),
    ]


def generate_launch_description():
    network = DEFAULT_ROOT+"/routes/network"
    return LaunchDescription([
        DeclareLaunchArgument(
            "route_path", default_value=network+
            "/route_network_segmented_stop_edited_vforward.csv"),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument(
            "display_route_path", default_value=network+
            "/route_network_segmented_all_branches_display_aligned_vforward.csv"),
        DeclareLaunchArgument("spawn_branch", default_value="A"),
        DeclareLaunchArgument("start_mode", default_value="1"),
        DeclareLaunchArgument("end_mode", default_value="11"),
        DeclareLaunchArgument("controller_hz", default_value="100.0"),
        OpaqueFunction(function=_prepare),
    ])
