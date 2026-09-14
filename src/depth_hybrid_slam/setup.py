from glob import glob
import os

from setuptools import find_packages, setup


package_name = "depth_hybrid_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["README.md"]),
        (os.path.join("share", package_name, "docs"), glob("docs/*.md")),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml") + glob("config/*.xml") + glob("config/*.rviz")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "PyYAML"],
    # Modern setuptools dropped tests_require.  This PEP 508 extra is also
    # recognized by colcon as the package's test dependency.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="qor",
    maintainer_email="qor@example.com",
    description="GPU cuVSLAM plus RTAB-Map localization, route, mission, and safety stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cuvslam_bridge = depth_hybrid_slam.cuvslam_bridge_node:main",
            "localization_fusion = depth_hybrid_slam.localization_node:main",
            "odom_localization = depth_hybrid_slam.odom_localization_node:main",
            "test_odom_publisher = depth_hybrid_slam.test_odom_publisher:main",
            "route_recorder = depth_hybrid_slam.route_recorder_node:main",
            "map_route_recorder = depth_hybrid_slam.map_route_recorder_node:main",
            "map_route_keyboard = depth_hybrid_slam.map_route_keyboard:main",
            "map_route_validate = depth_hybrid_slam.map_route_validate:main",
            "route_follower = depth_hybrid_slam.route_follower_node:main",
            "csv_road_validator = depth_hybrid_slam.csv_road_validator_node:main",
            "branch_selector = depth_hybrid_slam.branch_selector_node:main",
            "start_validation = depth_hybrid_slam.start_validation_node:main",
            "rtabmap_vslam_gate = depth_hybrid_slam.rtabmap_vslam_gate_node:main",
            "route_local_path = depth_hybrid_slam.route_local_path_node:main",
            "lidar_perception = depth_hybrid_slam.lidar_perception_node:main",
            "lidar_rejoin_validator = depth_hybrid_slam.lidar_rejoin_validator_node:main",
            "maneuver_manager = depth_hybrid_slam.maneuver_manager_node:main",
            "command_arbiter = depth_hybrid_slam.command_arbiter_node:main",
            "csv_only_closed_loop_probe = depth_hybrid_slam.csv_only_closed_loop_probe:main",
            "csv_only_branch_selector = depth_hybrid_slam.csv_only_branch_selector_node:main",
            "csv_only_network_visualizer = depth_hybrid_slam.csv_only_network_visualizer_node:main",
            "mission_manager = depth_hybrid_slam.mission_node:main",
            "signal_exit = depth_hybrid_slam.signal_exit_node:main",
            "safety_monitor = depth_hybrid_slam.safety_node:main",
            "case_manager = depth_hybrid_slam.case_cli:main",
            "performance_probe = depth_hybrid_slam.performance_probe:main",
            "bag_preflight = depth_hybrid_slam.bag_preflight:main",
            "bag_manifest = depth_hybrid_slam.bag_manifest:main",
            "bag_time_audit = depth_hybrid_slam.bag_time_audit:main",
            "bag_replay_probe = depth_hybrid_slam.bag_replay_probe:main",
            "mapping_quality_monitor = depth_hybrid_slam.mapping_quality_node:main",
            "mapping_session_finalize = depth_hybrid_slam.mapping_session:main",
            "mapping_session_info = depth_hybrid_slam.mapping_session_info_node:main",
            "perception_vslam_validator = depth_hybrid_slam.perception_vslam_node:main",
            "perception_vslam_overlay = depth_hybrid_slam.perception_overlay_node:main",
            "traffic_light_live_overlay = depth_hybrid_slam.traffic_light_overlay_node:main",
            "route_path_publisher = depth_hybrid_slam.route_path_publisher:main",
            "offline_map_requester = depth_hybrid_slam.offline_map_requester:main",
        ]
    },
)
