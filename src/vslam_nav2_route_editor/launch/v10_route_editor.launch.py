"""Read-only v10 RTAB-Map, Nav2 map_server, route publisher, and RViz."""
from pathlib import Path
import csv
import shutil
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def pgm_dimensions(path: Path) -> tuple[int, int]:
    """Read a P5/P2 PGM header without importing OpenCV into the launch process."""
    tokens = []
    with path.open("rb") as stream:
        while len(tokens) < 4:
            line = stream.readline()
            if not line:
                break
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
    if len(tokens) < 4 or tokens[0] not in (b"P5", b"P2"):
        raise RuntimeError(f"invalid PGM map image: {path}")
    width, height = int(tokens[1]), int(tokens[2])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid PGM dimensions in {path}: {width}x{height}")
    return width, height


def cleanup(_context, temporary_directory):
    shutil.rmtree(temporary_directory, ignore_errors=True)
    return []


def boolean_argument(context, name):
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false, got {value!r}")


def prepare(context):
    source_db = Path(LaunchConfiguration("database_path").perform(context)).expanduser().resolve()
    ply = Path(LaunchConfiguration("ply_path").perform(context)).expanduser().resolve()
    map_yaml = Path(LaunchConfiguration("map_yaml").perform(context)).expanduser().resolve()
    route = Path(LaunchConfiguration("route_path").perform(context)).expanduser().resolve()
    try:
        start_index = int(LaunchConfiguration("start_index").perform(context))
    except ValueError as error:
        raise RuntimeError("start_index must be an integer") from error
    if start_index < 0:
        raise RuntimeError("start_index must be non-negative")
    synthetic_odom = boolean_argument(context, "synthetic_odom")
    start_test_odom = boolean_argument(context, "start_test_odom")
    # The map inputs are required. The route is intentionally optional so a
    # brand-new route can be authored in RViz and persisted only on request.
    for path in (source_db, ply, map_yaml):
        if not path.is_file():
            raise RuntimeError(f"required input does not exist: {path}")
    if route.is_file():
        with route.open(newline="", encoding="utf-8") as stream:
            route_point_count = sum(1 for _ in csv.DictReader(stream))
        if start_index >= route_point_count:
            raise RuntimeError(
                f"start_index {start_index} is outside route {route.name}; "
                f"valid range is 0..{route_point_count - 1}")
    elif start_index != 0:
        raise RuntimeError(
            "start_index must be 0 when starting without an existing route")
    map_config = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    map_image = Path(map_config["image"])
    if not map_image.is_absolute():
        map_image = map_yaml.parent / map_image
    if not map_image.is_file():
        raise RuntimeError(f"required map image does not exist: {map_image}")
    width, height = pgm_dimensions(map_image)
    resolution = float(map_config["resolution"])
    origin_x, origin_y = map(float, map_config["origin"][:2])
    map_max_x = origin_x + width * resolution
    map_max_y = origin_y + height * resolution
    rviz_config = Path(get_package_share_directory(
        "vslam_nav2_route_editor")) / "config/route_editor.rviz"
    # RTAB-Map receives a disposable copy, making shutdown-time DB writes unable
    # to alter the v10 source database.
    temporary_directory = tempfile.mkdtemp(prefix="v10_route_view.", dir="/tmp")
    database = Path(temporary_directory) / "rtabmap.db"
    shutil.copy2(source_db, database)
    rtabmap_launch = Path(get_package_share_directory("rtabmap_launch")) / "launch/rtabmap.launch.py"
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(rtabmap_launch)),
        launch_arguments={
            "localization": "true", "database_path": str(database),
            "frame_id": "base_link", "map_frame_id": "map", "odom_frame_id": "",
            "visual_odometry": "false", "icp_odometry": "false",
            "publish_tf_odom": "false", "publish_tf_map": "false",
            "depth": "false", "subscribe_rgb": "false", "subscribe_rgbd": "false",
            "subscribe_stereo": "false", "rtabmap_viz": "false", "rviz": "false",
            "args": ("--Mem/IncrementalMemory false --Mem/InitWMWithAllNodes true "
                     "--Mem/LocalizationDataSaved false"),
        }.items())
    odom_nodes = []
    if synthetic_odom:
        odom_nodes.append(Node(
            package="vslam_nav2_route_editor", executable="synthetic_route_odom",
            name="synthetic_route_odom", output="screen",
            parameters=[{
                "start_index": start_index,
                "publish_rate_hz": float(LaunchConfiguration(
                    "synthetic_rate_hz").perform(context)),
                "travel_speed_mps": float(LaunchConfiguration(
                    "synthetic_speed_mps").perform(context)),
                "stop_hold_s": float(LaunchConfiguration(
                    "synthetic_stop_hold_s").perform(context)),
                "log_path": LaunchConfiguration(
                    "synthetic_log_path").perform(context),
            }]))
    elif start_test_odom:
        odom_nodes.append(Node(
            package="vslam_nav2_route_editor", executable="t870_test_odom",
            name="t870_route_test_odom", output="screen"))
    return [
        rtabmap,
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen", parameters=[{"yaml_filename": str(map_yaml)}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_map", output="screen",
             parameters=[{"autostart": True, "node_names": ["map_server"]}]),
        Node(package="vslam_nav2_route_editor", executable="ply_publisher",
             name="v10_ply_publisher", output="screen",
             parameters=[{"ply_path": str(ply), "topic": "/vslam_map/cloud"}]),
        Node(package="vslam_nav2_route_editor", executable="route_publisher",
             name="route_editor", output="screen",
             parameters=[{"route_path": str(route),
                          "route_name": LaunchConfiguration("route_name"),
                          "source_db": str(source_db),
                          "allow_clicked_point_append": True,
                          "click_plane_enabled": True,
                          "click_plane_min_x": origin_x,
                          "click_plane_max_x": map_max_x,
                          "click_plane_min_y": origin_y,
                          "click_plane_max_y": map_max_y,
                          "click_plane_z": 0.0}]),
        *odom_nodes,
        Node(package="vslam_nav2_route_editor", executable="map_odom_alignment",
             name="route_test_map_odom_alignment", output="screen",
             parameters=[{"route_path": str(route),
                          "start_index": start_index}],
             condition=IfCondition(LaunchConfiguration("start_route_follower"))),
        Node(package="vslam_nav2_route_editor", executable="route_follower",
             name="route_follower", output="screen",
             parameters=[{"start_index": start_index}],
             condition=IfCondition(LaunchConfiguration("start_route_follower"))),
        TimerAction(period=3.0, actions=[Node(
            package="vslam_nav2_route_editor", executable="offline_map_requester",
            name="v10_offline_map_requester", output="screen")]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="route_preview_frame", output="screen",
             arguments=["--x", "0", "--y", "0", "--z", "0", "--yaw", "0",
                        "--pitch", "0", "--roll", "0", "--frame-id", "map",
                        "--child-frame-id", "route_preview"]),
        # Keep the standard RViz interface while preloading only the map, cloud,
        # route and click-surface displays needed for route authoring.
        Node(package="rviz2", executable="rviz2", output="screen",
             arguments=["-d", str(rviz_config)],
             condition=IfCondition(LaunchConfiguration("start_rviz"))),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(
            function=lambda ctx: cleanup(ctx, temporary_directory))])),
    ]


def generate_launch_description():
    root = "/home/qor/depth_ws"
    return LaunchDescription([
        DeclareLaunchArgument("database_path", default_value=root + "/maps/merged_competition_level_aligned_v10/rtabmap.db"),
        DeclareLaunchArgument("ply_path", default_value=root + "/analysis/level_aligned_v10/final/merged_competition_level_aligned_v10.ply"),
        DeclareLaunchArgument("map_yaml", default_value=root + "/maps/nav2_v10/v10_nav2.yaml"),
        DeclareLaunchArgument("route_path", default_value=root + "/routes/v10/START_A.csv"),
        DeclareLaunchArgument("route_name", default_value="START_A"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("synthetic_odom", default_value="false"),
        DeclareLaunchArgument("synthetic_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("synthetic_speed_mps", default_value="1.0"),
        DeclareLaunchArgument("synthetic_stop_hold_s", default_value="0.5"),
        DeclareLaunchArgument(
            "synthetic_log_path",
            default_value="/tmp/vslam_nav2_route_editor_synthetic.csv"),
        DeclareLaunchArgument("start_test_odom", default_value="true"),
        DeclareLaunchArgument("start_route_follower", default_value="true"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        OpaqueFunction(function=prepare),
    ])
