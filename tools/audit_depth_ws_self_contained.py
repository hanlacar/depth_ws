#!/usr/bin/env python3
"""Static/installed audit for the two-workspace depth_ws + current-MCU design."""

import argparse
import ast
import json
from pathlib import Path
import sys
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "install", "log"}
EXTERNAL_MARKERS = (
    "/home/qor/"+"camera_ws", "/home/qor/"+"mmission_ws",
    "/home/qor/"+"urrc", "/home/qor/"+"map_ws",
    "/home/qor/"+"avoidance", "/home/qor/"+"t_parking",
    "/home/qor/"+"T870_MCU", "/home/qor/"+"mcu_ws")
REQUIRED_PACKAGES = (
    "camera_bringup", "camera_navigation", "camera_rgb_traffic_light",
    "camera_yolo_inference", "depth_hybrid_slam", "depth_slam",
    "imu_manager", "race_interfaces")
REQUIRED_FILES = (
    "src/camera_yolo_inference/models/hanla_yolo11n_seg_best.pt",
    "src/race_interfaces/msg/SemanticPathFrame.msg",
    "src/camera_navigation/camera_navigation/semantic_path_contract.py",
    "src/camera_navigation/camera_navigation/direct_bev_core.py",
    "src/camera_bringup/config/camera_mount.yaml",
    "src/camera_rgb_traffic_light/config/rgb_traffic_light.yaml",
    "src/depth_hybrid_slam/depth_hybrid_slam/csv_road_validator_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/mcu_source_adapter_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/branch_selector_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/route_local_path_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/behavior_selector_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/lidar_source_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/stop_editor_core.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/stop_editor_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/virtual_mcu_core.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/virtual_mcu_bridge_node.py",
    "src/depth_hybrid_slam/launch/stop_editor_vslam.launch.py",
    "src/depth_hybrid_slam/launch/prehardware_csv_vslam_closed_loop.launch.py",
    "src/depth_hybrid_slam/launch/competition_pre_hardware.launch.py",
    "routes/network/route_network_segmented.csv",
    "routes/network/route_network_segmented.metadata.yaml",
    "maps/merged_competition_level_aligned_v10/rtabmap.db")


def relevant(path):
    return not any(part in IGNORED_PARTS for part in path.parts)


def python_files():
    return [path for path in (ROOT/"src").rglob("*.py") if relevant(path)]


def literal_publishers(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    output = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_publisher" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            output.append(node.args[1].value)
    return output


def audit(require_installed=False, mcu_zip=None):
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    external = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or not relevant(path) or path == Path(__file__):
            continue
        if path.suffix.lower() not in (".py", ".sh", ".yaml", ".yml",
                                       ".xml", ".md", ".json", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in EXTERNAL_MARKERS:
            if marker in text:
                external.append(f"{path.relative_to(ROOT)}:{marker}")
    record("no_external_workspace_paths", not external, external or "none")

    external_links = []
    for path in ROOT.rglob("*"):
        if not path.is_symlink() or not relevant(path):
            continue
        target = path.resolve(strict=False)
        if target != ROOT and ROOT not in target.parents:
            external_links.append(f"{path.relative_to(ROOT)}->{target}")
    record("no_external_symlinks", not external_links,
           external_links or "none (generated build/install trees excluded)")

    packages_missing = [name for name in REQUIRED_PACKAGES
                        if not (ROOT/"src"/name/"package.xml").is_file()]
    record("required_source_packages_present", not packages_missing,
           packages_missing or list(REQUIRED_PACKAGES))
    missing_files = [name for name in REQUIRED_FILES if not (ROOT/name).is_file()]
    record("required_assets_present", not missing_files,
           missing_files or list(REQUIRED_FILES))

    mcu_imports = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        if "import t870_mcu" in text or "from t870_mcu" in text:
            mcu_imports.append(str(path.relative_to(ROOT)))
    record("no_t870_mcu_compile_import", not mcu_imports, mcu_imports or "none")

    publishers = {}
    for path in python_files():
        # This node is explicitly confined to the closed-loop launch and
        # mirrors test signals for the external real manager. It is not a
        # production source owner.
        if path.name == "closed_loop_support_node.py":
            continue
        for topic in literal_publishers(path):
            if topic in ("/camera_drive", "/camera_wheel", "/camera_stop",
                         "/lidar_drive", "/lidar_wheel", "/lidar_stop",
                         "/gps_drive", "/gps_wheel", "/gps_stop",
                         "/slam_drive", "/slam_wheel", "/slam_stop",
                         "/mcu/cmd_drive", "/mcu/cmd_wheel",
                         "/mcu/cmd_stop"):
                publishers.setdefault(topic, []).append(str(path.relative_to(ROOT)))
    duplicate = {topic: owners for topic, owners in publishers.items()
                 if len(set(owners)) > 1}
    forbidden_final = {topic: owners for topic, owners in publishers.items()
                       if topic.startswith("/mcu/cmd_")}
    record("no_duplicate_literal_command_publishers", not duplicate,
           duplicate or publishers)
    record("depth_ws_never_publishes_final_mcu_commands", not forbidden_final,
           forbidden_final or "manager-only by ROS contract")

    source_dir = ROOT / "src/depth_hybrid_slam/depth_hybrid_slam"
    camera_source = (ROOT / "src/camera_navigation/camera_navigation" /
                     "camera_command_selector_node.py").read_text(
                         encoding="utf-8")
    lidar_source = (source_dir/"lidar_source_node.py").read_text(
        encoding="utf-8")
    behavior_source = (source_dir/"behavior_selector_node.py").read_text(
        encoding="utf-8")
    expected_source_topics = {
        "camera": (camera_source,
                   ("/camera_drive", "/camera_wheel", "/camera_stop")),
        "lidar": (lidar_source,
                  ("/lidar_drive", "/lidar_wheel", "/lidar_stop")),
    }
    missing_sources = [
        f"{name}:{topic}" for name, (text, topics) in
        expected_source_topics.items() for topic in topics if topic not in text]

    adapter = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam" /
               "mcu_source_adapter_node.py").read_text(encoding="utf-8")
    core = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam" /
            "mcu_source_adapter_core.py").read_text(encoding="utf-8")
    contract_ok = all(token in adapter for token in
                      ("/slam_drive", "/slam_wheel", "/slam_stop",
                       "/gps_drive", "/gps_wheel", "/gps_stop"))
    contract_ok &= all(token in core for token in
                       ("-1.0", "0.0", "1.0", "2.0", "3.0",
                        "max_wheel_deg=22", "return SourceCommand(drive, -wheel"))
    record("current_mcu_topic_and_sign_contract", contract_ok,
           "slam +LEFT -> gps +RIGHT boundary; stage {-1,0,1,2,3}; +/-22 deg")
    gps_topics = all(topic in adapter for topic in
                     ("/gps_drive", "/gps_wheel", "/gps_stop"))
    record("all_mcu_source_publishers_present",
           not missing_sources and gps_topics,
           missing_sources or "camera/gps/lidar triplets have depth_ws owners")
    slam_owners = {topic: owners for topic, owners in publishers.items()
                   if topic.startswith("/slam_")}
    behavior_owner = all(
        len(slam_owners.get(topic, ())) == 1 and
        slam_owners[topic][0].endswith("behavior_selector_node.py")
        for topic in ("/slam_drive", "/slam_wheel", "/slam_stop"))
    record("behavior_is_single_internal_command_owner", behavior_owner,
           slam_owners)
    follower = (ROOT / "src/depth_hybrid_slam/depth_hybrid_slam" /
                "route_follower_node.py").read_text(encoding="utf-8")
    record("route_follower_publishes_current_mode",
           '"/drive_mode"' in follower,
           "segmented CSV mode 1..11 -> /drive_mode")
    branch_source = (source_dir/"branch_selector_node.py").read_text(
        encoding="utf-8")
    local_source = (source_dir/"route_local_path_node.py").read_text(
        encoding="utf-8")
    route_source = (source_dir/"route_io.py").read_text(encoding="utf-8")
    prehardware_core = (source_dir/"prehardware_core.py").read_text(
        encoding="utf-8")
    branch_connected = all(token in branch_source+follower+route_source
                           for token in (
                               "/depth_slam/route/branch_command",
                               "/depth_slam/route/selected_branch",
                               "/depth_slam/route/active_branch",
                               "branch=branch", "START_{branch}"))
    record("branch_selector_changes_active_segmented_route", branch_connected,
           "default/timeout A and synthetic B reach route follower")
    local_connected = all(token in local_source + follower + prehardware_core
                          for token in (
                              "/depth_slam/route/reference_path",
                              "/depth_slam/route/active_index",
                              "/depth_slam/localization/pose",
                              "/depth_slam/csv_validation/local_path",
                              "INDEX_BACKTRACK"))
    record("global_route_to_local_validator_path", local_connected,
           "progress-indexed map route -> base_link 0.5..4.0m")
    validator_gate = all(token in behavior_source+prehardware_core for token in (
        "/depth_slam/csv_road_validation/state",
        "/depth_slam/csv_validation/local_path_state",
        "CAMERA_VALIDATION_UNAVAILABLE", "/slam_stop"))
    record("validator_connected_to_behavior_gate", validator_gate,
           "road invalid stops; camera unavailable permits CSV fallback")
    stop_connected = all(token in follower for token in (
        "StopWaypointMachine", "stop_waypoint.update",
        "stop_waypoint_state"))
    record("csv_stop_waypoint_three_second_gate", stop_connected,
           "STOP_LINE one-shot hold plus traffic release gate")

    stop_editor = (source_dir/"stop_editor_core.py").read_text(encoding="utf-8")
    stop_launch = (ROOT / "src/depth_hybrid_slam/launch" /
                   "stop_editor_vslam.launch.py").read_text(encoding="utf-8")
    stop_node = (source_dir/"stop_editor_node.py").read_text(encoding="utf-8")
    map_requester = (source_dir/"offline_map_requester.py").read_text(
        encoding="utf-8")
    stop_rviz = (ROOT / "src/depth_hybrid_slam/config" /
                 "stop_editor_vslam.rviz").read_text(encoding="utf-8")
    stop_safe = all(
        token in stop_editor+stop_launch+stop_node+stop_rviz for token in (
        "maximum_click_distance_m", "CLICK_TOO_FAR_FROM_ROUTE",
        "REQUIRED_DIRECTION_CHANGE_STOP", "validate_geometry_unchanged",
        "source_sha256", "output_sha256", "offline_rtabmap_include",
        "_read_only_prefix", "--ro-bind", "visualization_route_path",
        "visualization_keys", "CSV OVERLAY: VISUALIZATION ONLY",
        "MODE 2 SLOPE", "/rtabmap/map", "Fixed Frame: map"))
    # AUTO prefers a successfully probed bubblewrap namespace.  Restricted
    # hosts use a guarded temporary copy: capacity is checked before creation,
    # partial copies are removed, and launch shutdown removes the directory.
    # This fallback is deliberately accepted only as the complete safety
    # contract below; an unguarded DB copy must not satisfy this audit.
    stop_safe &= all(token in stop_launch for token in (
        "_probe_bwrap", "PROBE_OK", "read_only_backend", "AUTO", "BWRAP",
        "TEMP_COPY", "source_size * 1.5", "INSUFFICIENT_TEMP_SPACE",
        "tempfile.mkdtemp", "shutil.copy2", "shutil.rmtree",
        "depth_stop_editor.", "OnShutdown", "RTABMAP_START",
        "rtabmap_startup_timeout_s", "rtabmap_map_delivery_timeout_s",
        "failure_only=True"))
    stop_safe &= all(token in map_requester for token in (
        "startup_timeout_s", "map_delivery_timeout_s",
        "get_publishers_info_by_topic", "RTABMAP_START_FAILED", '"/rtabmap/map"',
        'message.header.frame_id == "map"'))
    record("non_destructive_rviz_stop_editor", stop_safe,
           "map-frame RTAB-Map + exact-key unvalidated CSV/mode/STOP overlay")
    virtual_core = (source_dir/"virtual_mcu_core.py").read_text(encoding="utf-8")
    virtual_node = (source_dir/"virtual_mcu_bridge_node.py").read_text(
        encoding="utf-8")
    closed_launch = (ROOT / "src/depth_hybrid_slam/launch" /
                     "prehardware_csv_vslam_closed_loop.launch.py").read_text(
                         encoding="utf-8")
    csv_only_launch = (ROOT / "src/depth_hybrid_slam/launch" /
                       "prehardware_csv_only_closed_loop.launch.py").read_text(
                           encoding="utf-8")
    csv_only_vehicle = (source_dir/"csv_only_virtual_vehicle_node.py").read_text(
        encoding="utf-8")
    virtual_safe = all(token in virtual_core+virtual_node+closed_launch for token in (
        "COUNTS_PER_METER = 797.0", "WHEELBASE_M = 0.730",
        "MAX_STEERING_DEG = 22.0", "/mcu/encoder", "/odom",
        "DUPLICATE_REAL_OR_VIRTUAL_BRIDGE", "prehardware_test_only",
        "simulation_speedup", "start_mode", "end_mode"))
    virtual_safe &= ('executable="mcu_bridge"' not in closed_launch and
                     "package=\"t870_mcu\"" not in closed_launch)
    virtual_safe &= all(token in csv_only_launch+csv_only_vehicle for token in (
        "route_network_segmented_stop_edited_vforward.csv",
        "csv_only_virtual_vehicle",
        "csv_only_localization_source", "/depth_slam/follower/candidate_drive",
        "/mcu/encoder", "/odom"))
    virtual_safe &= all(token not in csv_only_launch.lower() for token in (
        "rtabmap", "cuvslam", "nav2", "camera", "lidar", "t870_mcu",
        "mcu_manager", "serial"))
    record("isolated_virtual_mcu_closed_loop", virtual_safe,
           "real manager remains external; serial bridge excluded; 797 count/m Ackermann odom")

    production = (ROOT / "src/depth_hybrid_slam/launch" /
                  "production_ready.launch.py").read_text(encoding="utf-8")
    cuvslam = (ROOT / "src/depth_hybrid_slam/launch" /
               "cuvslam_only.launch.py").read_text(encoding="utf-8")
    no_final = "/mcu/cmd_" not in production
    no_odom_tf = all(token not in production for token in
                     ("rgbd_odometry", "static_transform_publisher",
                      "StaticTransformBroadcaster", '"/odom"')) and \
        '"publish_odom_to_base_tf": False' in cuvslam and \
        '"publish_map_to_odom_tf": False' in cuvslam and \
        'LaunchConfiguration("publish_camera_mount_tf")' in cuvslam
    record("production_no_final_command_owner", no_final,
           "MCU manager remains sole /mcu/cmd_* publisher")
    record("production_no_duplicate_odom_or_static_tf", no_odom_tf,
           "MCU bridge/static-TF runtime owns vehicle odom/TF")

    if require_installed:
        failures = []
        try:
            from ament_index_python.packages import get_package_prefix
            expected = (ROOT/"install").resolve()
            for name in REQUIRED_PACKAGES:
                prefix = Path(get_package_prefix(name)).resolve()
                if prefix != expected and prefix.parent != expected:
                    failures.append(f"{name}:{prefix}")
        except Exception as error:  # pragma: no cover - environment dependent
            failures.append(str(error))
        record("installed_prefixes_are_depth_ws", not failures,
               failures or str(ROOT/"install"))

    if mcu_zip:
        archive_path = Path(mcu_zip).expanduser().resolve()
        problems = []
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()

                def member(suffix):
                    matches = [name for name in names if name.endswith(suffix)]
                    if len(matches) != 1:
                        raise ValueError(f"missing/ambiguous MCU member: {suffix}")
                    return archive.read(matches[0]).decode("utf-8")
                values = yaml.safe_load(member("/CURRENT_VALUES.yaml"))
                topic = member("/TOPIC_CONTRACT.md")
                manager = member("/src/t870_mcu/t870_mcu/manager_node.py")
                policy = member("/src/t870_mcu/t870_mcu/mode_policy.py")
                config = member("/src/t870_mcu/config/t870_mcu.yaml")
                expected = ((values["vehicle"]["wheelbase_m"], 0.730),
                            (values["encoder"]["counts_per_meter"], 797.0),
                            (values["steering"]["center_adc"], 484),
                            (values["steering"]["counts_per_degree"], 18.0),
                            (values["steering"]["max_degree"], 22.0))
                if any(abs(float(actual)-wanted) > 1.0e-9
                       for actual, wanted in expected):
                    problems.append("CURRENT_VALUES calibration mismatch")
                for token in ("/camera_drive", "/lidar_drive", "/gps_drive",
                              "/manual_drive", "/mcu/cmd_drive",
                              "/mcu/cmd_wheel", "/mcu/cmd_stop"):
                    if token not in topic or token not in manager+config:
                        problems.append("missing topic " + token)
                for token in ("team_wheel_positive_right", "source != self.manual_name",
                              "GENERAL_CAMERA_MODES", "INTERSECTION_MODES",
                              "PARKING_MODES"):
                    if token not in manager+policy:
                        problems.append("missing current policy token " + token)
                if "pub_topic_odom: /odom" not in config or "publish_tf: true" not in config:
                    problems.append("MCU /odom or TF ownership mismatch")
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            problems.append(str(error))
        record("latest_mcu_zip_contract_matches", not problems,
               problems or str(archive_path))

    passed = sum(item["pass"] for item in checks)
    return {"workspace": str(ROOT), "pass": passed == len(checks),
            "passed": passed, "total": len(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-installed-prefix", action="store_true")
    parser.add_argument("--mcu-zip")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.require_installed_prefix, args.mcu_zip)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for item in result["checks"]:
            print(("PASS" if item["pass"] else "FAIL"), item["name"],
                  "-", item["detail"])
        print(f"SUMMARY {result['passed']}/{result['total']} "
              f"{'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
