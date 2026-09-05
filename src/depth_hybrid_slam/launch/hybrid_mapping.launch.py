"""Protected RTAB-Map mapping fed exclusively by cuVSLAM odometry."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments, rtabmap_include
from depth_hybrid_slam.mapping_session import (
    as_bool, create_mapping_session, current_mapping_graph,
    load_mapping_profile, mapping_graph_conflicts, rtabmap_argument_string,
    write_session_metadata,
)
from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _start(context):
    if not as_bool(LaunchConfiguration("mapping_mode").perform(context)):
        raise RuntimeError("hybrid_mapping requires mapping_mode=true")
    if as_bool(LaunchConfiguration("localization_mode").perform(context)):
        raise RuntimeError("mapping and localization modes cannot run together")
    if as_bool(LaunchConfiguration("enable_control").perform(context)):
        raise RuntimeError("mapping launch forbids enable_control=true")
    names, odometry_publishers = current_mapping_graph()
    conflicts = mapping_graph_conflicts(names, odometry_publishers)
    if conflicts:
        raise RuntimeError("mapping graph rejected: " + ", ".join(conflicts))
    session = create_mapping_session(
        map_path=LaunchConfiguration("map_path").perform(context),
        session_id=LaunchConfiguration("session_id").perform(context),
        auto_session_name=LaunchConfiguration("auto_session_name").perform(context))
    path = session.map_path
    config_dir = Path(get_package_share_directory("depth_hybrid_slam")) / "config"
    high_density = as_bool(LaunchConfiguration("high_density").perform(context))
    profile = load_mapping_profile(config_dir / "high_density_mapping.yaml",
                                   high_density)
    report = LaunchConfiguration("quality_report").perform(context)
    if not report:
        report = str(session.report_path)
    report_path = Path(report).expanduser().resolve()
    if report_path.exists() or report_path.with_suffix(
            report_path.suffix + ".tmp").exists():
        raise RuntimeError(f"quality report target already exists: {report_path}")
    session_values = session.metadata()
    session_values["report_path"] = str(report_path)
    write_session_metadata(session.metadata_path, session_values)
    session_info = Node(
        package="depth_hybrid_slam", executable="mapping_session_info",
        name="mapping_session_info", output="screen",
        parameters=[dict(session_values, metadata_path=str(session.metadata_path))],
    )
    localization = Node(package="depth_hybrid_slam", executable="localization_fusion",
                        name="localization_fusion", output="screen")
    quality = Node(
        package="depth_hybrid_slam", executable="mapping_quality_monitor",
        name="mapping_quality_monitor", output="screen",
        parameters=[{
            "map_path": str(path),
            "quality_config": str(config_dir / "mapping_quality.yaml"),
            "report_path": report,
        }],
    )
    nodes = [
        rtabmap_include(False, rtabmap_args=rtabmap_argument_string(profile),
                        database_path=str(path)),
        localization, session_info,
    ]
    if as_bool(LaunchConfiguration("start_monitor").perform(context)):
        nodes.append(quality)
    if as_bool(LaunchConfiguration("record_bag").perform(context)):
        nodes.append(TimerAction(period=5.0, actions=[ExecuteProcess(
            cmd=["/home/qor/depth_ws/scripts/record_competition_bag.sh",
                 "--case", session.session_id,
                 "--session-id", session.session_id,
                 "--session-path", str(session.bag_path),
                 "--purpose", "mapping",
                 "--profile", "camera_slam_with_aligned_depth",
                 "--map", str(path)],
            output="screen")]))
    return nodes


def generate_launch_description():
    return LaunchDescription(common_arguments({
        "mapping_mode": "true", "localization_mode": "false",
    })+[OpaqueFunction(function=_start)])
