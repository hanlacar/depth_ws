"""Full-network CSV follower loop with an isolated test-only virtual vehicle."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.csv_only_branching import csv_only_network_segments
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


DEFAULT_ROOT = "/home/qor/depth_ws"


def _prepare(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    route = Path(value("route_path")).expanduser().resolve()
    metadata_text = value("route_metadata_path").strip()
    metadata = (Path(metadata_text).expanduser().resolve() if metadata_text else
                route.with_suffix(".metadata.yaml"))
    if not route.is_file() or not metadata.is_file():
        raise RuntimeError("edited segmented route and metadata must exist")
    display_route = Path(value("display_route_path")).expanduser().resolve()
    display_metadata_text = value("display_metadata_path").strip()
    display_metadata = (
        Path(display_metadata_text).expanduser().resolve()
        if display_metadata_text else display_route.with_suffix(".metadata.yaml"))
    if not display_route.is_file() or not display_metadata.is_file():
        raise RuntimeError("all-branches display route and metadata must exist")
    segments = csv_only_network_segments(route, metadata)
    for required in (
            "START_A", "START_B", "T_A", "T_B", "V_A", "V_B",
            "END_AA", "END_AB"):
        if required not in segments:
            raise RuntimeError(f"CSV-only network is missing {required}")
    spawn_branch = value("spawn_branch").strip().upper()
    if spawn_branch not in ("A", "B"):
        raise RuntimeError("spawn_branch must be A or B")

    package = Path(get_package_share_directory("depth_hybrid_slam"))
    calibration_file = package / "config" / "virtual_vehicle.yaml"
    calibration = yaml.safe_load(calibration_file.read_text(encoding="utf-8"))[
        "depth_virtual_mcu_bridge"]["ros__parameters"]
    first = segments[f"START_{spawn_branch}"][0]
    follower_parameters = {
        "route_path": str(route),
        "route_metadata_path": str(metadata),
        # Existing explicit test override requires a readable sentinel path.
        # It is never opened as a world model in FOLLOW_ROUTE operation.
        "map_path": str(route),
        "enable_control": True,
        "dry_run": False,
        "user_approved": True,
        "prehardware_test_override_alignment": True,
        "prehardware_csv_only_case_selection": True,
        "pose_timeout_s": 0.50,
        "localization_stability_s": 0.20,
        "start_mode": 1,
        "end_mode": 11,
        "wheelbase_m": 0.73,
        "max_steering_deg": 22.0,
        "controller_hz": float(value("controller_hz")),
    }
    return [
        LogInfo(msg=(
            "PREHARDWARE_CSV_ONLY full_network=true "
            f"spawn=START_{spawn_branch}:{first.point_index} "
            f"pose=({first.x:.6f},{first.y:.6f},{first.yaw:.6f})")),
        Node(package="depth_hybrid_slam", executable="csv_only_branch_selector",
             name="depth_csv_only_branch_selector", output="screen",
             parameters=[{
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "prehardware_test_only": True,
             }]),
        Node(package="depth_hybrid_slam", executable="csv_only_network_visualizer",
             name="depth_csv_only_network_visualizer", output="screen",
             parameters=[{
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "display_route_path": str(display_route),
                 "display_metadata_path": str(display_metadata),
                 "prehardware_test_only": True,
             }]),
        Node(package="depth_hybrid_slam", executable="route_follower",
             name="depth_csv_only_route_follower", output="screen",
             parameters=[follower_parameters]),
        Node(package="depth_hybrid_slam",
             executable="csv_only_localization_source",
             name="depth_csv_only_localization_source", output="screen",
             parameters=[{"prehardware_test_only": True}]),
        Node(package="depth_hybrid_slam",
             executable="csv_only_virtual_vehicle",
             name="depth_csv_only_virtual_vehicle", output="screen",
             parameters=[calibration, {
                 "route_path": str(route),
                 "route_metadata_path": str(metadata),
                 "spawn_branch": spawn_branch,
                 "prehardware_test_only": True,
                 "simulation_speedup": float(value("simulation_speedup")),
                 "publish_hz": float(value("vehicle_publish_hz")),
             }]),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(package / "config" /
                                  "prehardware_csv_only_closed_loop.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
    ]


def generate_launch_description():
    network = DEFAULT_ROOT + "/routes/network"
    return LaunchDescription([
        DeclareLaunchArgument(
            "route_path", default_value=network +
            "/route_network_segmented_stop_edited_vforward.csv"),
        DeclareLaunchArgument("route_metadata_path", default_value=""),
        DeclareLaunchArgument(
            "display_route_path", default_value=network +
            "/route_network_segmented_all_branches_display_aligned_vforward.csv"),
        DeclareLaunchArgument("display_metadata_path", default_value=""),
        DeclareLaunchArgument("spawn_branch", default_value="A"),
        DeclareLaunchArgument("simulation_speedup", default_value="1.0"),
        DeclareLaunchArgument("controller_hz", default_value="100.0"),
        DeclareLaunchArgument("vehicle_publish_hz", default_value="100.0"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        OpaqueFunction(function=_prepare),
    ])
