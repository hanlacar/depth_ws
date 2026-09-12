"""Read-only v10 map plus Nav2-first / segmented-CSV fallback follower."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = "/home/qor/depth_ws"
    editor_share = Path(get_package_share_directory("vslam_nav2_route_editor"))
    own_share = Path(get_package_share_directory("nav2_csv_route_follower"))
    # Scope the included editor arguments so its intentionally disabled RViz,
    # ODOM and follower do not overwrite this launch's identically named
    # public arguments.
    base = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(
            editor_share / "launch/v10_route_editor.launch.py")),
        launch_arguments={
            "database_path": root + "/maps/merged_competition_level_aligned_v10/rtabmap.db",
            "ply_path": root + "/analysis/level_aligned_v10/final/merged_competition_level_aligned_v10.ply",
            "map_yaml": root + "/maps/nav2_v10/v10_nav2.yaml",
            "route_path": root + "/routes/v10/START_A.csv",
            "route_name": "START_A",
            "start_index": LaunchConfiguration("start_index"),
            "synthetic_odom": "false",
            "start_test_odom": "false",
            "start_route_follower": "false",
            "start_rviz": "false",
        }.items())])
    common_parameters = [{
        "csv_path": LaunchConfiguration("csv_path"),
        "yaml_path": LaunchConfiguration("yaml_path"),
        "start_index": LaunchConfiguration("start_index"),
        "lookahead_m": LaunchConfiguration("lookahead_m"),
        "csv_to_map_x": LaunchConfiguration("csv_to_map_x"),
        "csv_to_map_y": LaunchConfiguration("csv_to_map_y"),
        "csv_to_map_yaw_deg": LaunchConfiguration("csv_to_map_yaw_deg"),
        "nav_to_csv_position_m": LaunchConfiguration("nav_to_csv_position_m"),
        "csv_to_nav_position_m": LaunchConfiguration("csv_to_nav_position_m"),
        "nav_to_csv_heading_deg": LaunchConfiguration("nav_to_csv_heading_deg"),
        "csv_to_nav_heading_deg": LaunchConfiguration("csv_to_nav_heading_deg"),
        "nav_to_csv_samples": LaunchConfiguration("nav_to_csv_samples"),
        "csv_to_nav_samples": LaunchConfiguration("csv_to_nav_samples"),
        "stop_hold_seconds": LaunchConfiguration("stop_hold_seconds"),
        "stop_trigger_distance_m": LaunchConfiguration(
            "stop_trigger_distance_m"),
        "stop_approach_distance_m": LaunchConfiguration(
            "stop_approach_distance_m"),
        "stop_merge_distance_m": LaunchConfiguration("stop_merge_distance_m"),
    }]
    return LaunchDescription([
        DeclareLaunchArgument("start_index", default_value="43"),
        DeclareLaunchArgument(
            "synthetic_route_start_index",
            default_value=LaunchConfiguration("start_index")),
        DeclareLaunchArgument("synthetic_odom", default_value="true"),
        DeclareLaunchArgument("synthetic_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("synthetic_speed_mps", default_value="0.25"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument(
            "csv_path", default_value="/home/qor/depth_ws/routes/network/route_network_segmented.csv"),
        DeclareLaunchArgument(
            "yaml_path", default_value="/home/qor/depth_ws/routes/network/route_network_segmented.yaml"),
        DeclareLaunchArgument("lookahead_m", default_value="1.0"),
        DeclareLaunchArgument("csv_to_map_x", default_value="-1.84276105"),
        DeclareLaunchArgument("csv_to_map_y", default_value="-1.48548570"),
        DeclareLaunchArgument("csv_to_map_yaw_deg", default_value="-153.98206441"),
        DeclareLaunchArgument("nav_to_csv_position_m", default_value="1.5"),
        DeclareLaunchArgument("csv_to_nav_position_m", default_value="0.75"),
        DeclareLaunchArgument("nav_to_csv_heading_deg", default_value="20.0"),
        DeclareLaunchArgument("csv_to_nav_heading_deg", default_value="10.0"),
        DeclareLaunchArgument("nav_to_csv_samples", default_value="10"),
        DeclareLaunchArgument("csv_to_nav_samples", default_value="20"),
        DeclareLaunchArgument("stop_hold_seconds", default_value="1.0"),
        DeclareLaunchArgument("stop_trigger_distance_m", default_value="0.30"),
        DeclareLaunchArgument("stop_approach_distance_m", default_value="1.0"),
        DeclareLaunchArgument("stop_merge_distance_m", default_value="0.75"),
        base,
        Node(package="vslam_nav2_route_editor", executable="map_odom_alignment",
             name="route_test_map_odom_alignment", output="screen",
             condition=UnlessCondition(LaunchConfiguration("synthetic_odom")),
             parameters=[{"route_path": root + "/routes/v10/START_A.csv",
                          "start_index": LaunchConfiguration("start_index")}]),
        Node(package="nav2_csv_route_follower", executable="controlled_synthetic_odom",
             name="synthetic_route_odom", output="screen",
             condition=IfCondition(LaunchConfiguration("synthetic_odom")),
             parameters=[{
                 "start_index": LaunchConfiguration(
                     "synthetic_route_start_index"),
                 "publish_rate_hz": LaunchConfiguration("synthetic_rate_hz"),
                 "travel_speed_mps": LaunchConfiguration("synthetic_speed_mps"),
                 "path_topic": "/route_compare/csv_path",
                 "log_path": "/tmp/nav2_csv_route_synthetic.csv",
             }]),
        Node(package="vslam_nav2_route_editor", executable="t870_test_odom",
             name="t870_route_test_odom", output="screen",
             condition=UnlessCondition(LaunchConfiguration("synthetic_odom"))),
        Node(package="nav2_csv_route_follower", executable="hybrid_route_follower",
             name="nav2_csv_route_follower", output="screen",
             parameters=common_parameters),
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(own_share / "config/hybrid_route.rviz")],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
    ])
