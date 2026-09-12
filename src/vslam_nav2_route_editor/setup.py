from glob import glob
import os
from setuptools import find_packages, setup

package_name = "vslam_nav2_route_editor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools", "PyYAML"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="qor",
    maintainer_email="qor@example.com",
    description="Offline-safe fixed route editor and Nav2 map viewer.",
    license="MIT",
    entry_points={"console_scripts": [
        "route_editor = vslam_nav2_route_editor.route_editor_cli:main",
        "route_publisher = vslam_nav2_route_editor.route_publisher_node:main",
        "ply_map_export = vslam_nav2_route_editor.ply_map_export:main",
        "ply_publisher = vslam_nav2_route_editor.ply_publisher_node:main",
        "offline_map_requester = vslam_nav2_route_editor.offline_map_requester:main",
        "t870_test_odom = vslam_nav2_route_editor.t870_test_odom_node:main",
        "synthetic_route_odom = vslam_nav2_route_editor.synthetic_route_odom_node:main",
        "map_odom_alignment = vslam_nav2_route_editor.map_odom_alignment_node:main",
        "route_follower = vslam_nav2_route_editor.route_test_follower_node:main",
        # Compatibility executable name; it publishes only the new /route_*
        # contract and never recreates the removed /route_test/* topics.
        "route_test_follower = vslam_nav2_route_editor.route_test_follower_node:main",
    ]},
)
