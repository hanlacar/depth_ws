#!/usr/bin/env python3
"""Static audit for the production sensor, route, and command contracts."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "camera_bringup", "camera_navigation", "camera_rgb_traffic_light",
    "camera_yolo_inference", "depth_hybrid_slam", "imu_manager",
    "race_interfaces", "rplidar_ros")
REQUIRED = (
    "src/camera_bringup/launch/d456_production.launch.py",
    "src/depth_hybrid_slam/launch/dual_rplidar.launch.py",
    "src/depth_hybrid_slam/launch/competition_csv_camera_lidar.launch.py",
    "src/depth_hybrid_slam/launch/prehardware_csv_camera_lidar.launch.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/test_odom_publisher.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/odom_localization_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/command_arbiter_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/lidar_perception_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/maneuver_manager_node.py",
    "src/depth_hybrid_slam/depth_hybrid_slam/signal_exit_node.py",
    "src/depth_hybrid_slam/config/rplidar_dual.yaml",
    "routes/network/route_network_segmented_stop_edited_vforward.csv",
    "routes/network/route_network_segmented_stop_edited_vforward.metadata.yaml",
    "tools/ros_network_env.sh",
)
FORBIDDEN_RUNTIME_NAMES = {
    "synthetic_scan_node.py", "virtual_mcu_bridge_node.py",
    "csv_only_virtual_vehicle_node.py", "closed_loop_support_node.py",
    "lidar_source_node.py", "mcu_source_adapter_node.py",
    "csv_only_localization_source_node.py", "video_publisher.py",
}


def _python_sources():
    return [path for path in (ROOT/"src").rglob("*.py")
            if not {"test", "__pycache__"}.intersection(path.parts)]


def _literal_publishers(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    output = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == "create_publisher" and len(node.args) >= 2 and
                isinstance(node.args[1], ast.Constant) and
                isinstance(node.args[1].value, str)):
            output.append(node.args[1].value)
    return output


def audit(require_installed=False, _mcu_zip=None):
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    missing_packages = [name for name in PACKAGES
                        if not (ROOT/"src"/name/"package.xml").is_file()]
    record("production_packages_present", not missing_packages,
           missing_packages or list(PACKAGES))
    missing = [item for item in REQUIRED if not (ROOT/item).is_file()]
    record("required_production_assets", not missing, missing or "present")

    runtime_names = {path.name for path in _python_sources()}
    forbidden = sorted(runtime_names & FORBIDDEN_RUNTIME_NAMES)
    record("no_fake_sensor_or_vehicle_runtime", not forbidden,
           forbidden or "TEST ONLY ODOM is the sole test source")

    gazebo = []
    for root in (ROOT/"src", ROOT/"tools"):
        for path in root.rglob("*"):
            if (not path.is_file() or "test" in path.parts or
                    "rplidar_ros" in path.parts or "__pycache__" in path.parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {".py", ".xml", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(token in text for token in
                   ("gazebo_ros", "gz_sim", "ignition", "spawn_entity")):
                gazebo.append(str(path.relative_to(ROOT)))
    record("no_gazebo_runtime_dependency", not gazebo, gazebo or "none")

    owners = {}
    for path in _python_sources():
        for topic in _literal_publishers(path):
            owners.setdefault(topic, []).append(str(path.relative_to(ROOT)))
    final = {topic: sorted(set(owners.get(topic, [])))
             for topic in ("/cmd_drive", "/cmd_wheel")}
    command_ok = all(len(value) == 1 and
                     value[0].endswith("command_arbiter_node.py")
                     for value in final.values())
    record("single_final_command_owner", command_ok, final)
    aliases = {topic: value for topic, value in owners.items()
               if topic in {"/camera_drive", "/camera_wheel", "/lidar_drive",
                            "/lidar_wheel", "/gps_drive", "/gps_wheel",
                            "/slam_drive", "/slam_wheel"}}
    record("no_legacy_command_publishers", not aliases, aliases or "none")

    lidar = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
             "lidar_perception_node.py").read_text(encoding="utf-8")
    driver = (ROOT/"src/depth_hybrid_slam/launch"/
              "dual_rplidar.launch.py").read_text(encoding="utf-8")
    lidar_ok = (all(topic in lidar for topic in
                    ('"/front/scan"', '"/rear/scan"')) and
                all(topic not in lidar for topic in
                    ('"/scan"', '"/scan_front"', '"/scan_rear"')) and
                'remappings=[("scan", "/front/scan")]' in driver and
                'remappings=[("scan", "/rear/scan")]' in driver)
    record("canonical_lidar_contract", lidar_ok,
           "native scan remaps to /front/scan and /rear/scan")

    hil = (ROOT/"src/depth_hybrid_slam/launch"/
           "prehardware_csv_camera_lidar.launch.py").read_text(encoding="utf-8")
    odom = (ROOT/"src/depth_hybrid_slam/depth_hybrid_slam"/
            "test_odom_publisher.py").read_text(encoding="utf-8")
    hil_ok = ("d456_production.launch.py" in hil and
              "dual_rplidar.launch.py" in hil and
              '"start_mode", default_value="4"' in hil and
              '"end_mode", default_value="5"' in hil and
              '"test_only_acknowledged", False' in odom and
              'Odometry, "/odom"' in odom and
              '"/mcu/encoder"' in odom and '"/mcu/steer_a0"' in odom)
    record("real_sensor_hil_and_isolated_test_odom", hil_ok,
           "real D456/A2M12 plus measured encoder/steering ODOM")

    route = ROOT/"routes/network/route_network_segmented_stop_edited_vforward.csv"
    metadata = yaml.safe_load(route.with_suffix(".metadata.yaml").read_text())
    digest = hashlib.sha256(route.read_bytes()).hexdigest()
    record("canonical_route_hash", metadata.get("route_sha256") == digest,
           digest)

    production = (ROOT/"src/depth_hybrid_slam/launch"/
                  "competition_csv_camera_lidar.launch.py").read_text()
    nodes = ("route_follower", "odom_localization", "csv_road_validator",
             "mission_manager", "signal_exit", "lidar_perception",
             "lidar_rejoin_validator", "maneuver_manager", "command_arbiter",
             "safety_monitor")
    record("production_graph_complete", all(name in production for name in nodes),
           list(nodes))

    if require_installed:
        failures = []
        try:
            from ament_index_python.packages import get_package_prefix
            expected = (ROOT/"install").resolve()
            for name in PACKAGES:
                prefix = Path(get_package_prefix(name)).resolve()
                if prefix != expected and expected not in prefix.parents:
                    failures.append(f"{name}:{prefix}")
        except Exception as error:  # pragma: no cover
            failures.append(str(error))
        record("installed_prefixes_are_depth_ws", not failures,
               failures or str(expected))

    passed = sum(item["pass"] for item in checks)
    return {"workspace": str(ROOT), "pass": passed == len(checks),
            "passed": passed, "total": len(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-installed-prefix", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.require_installed_prefix)
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
