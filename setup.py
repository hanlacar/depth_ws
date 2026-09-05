from glob import glob
import os

from setuptools import find_packages, setup

package_name = "depth_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="qor",
    maintainer_email="qor@example.com",
    description="D456 640x480@60 RTAB-Map mapping stack (fast VO + slow map graph).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fps_monitor = depth_slam.fps_monitor_node:main",
        ],
    },
)
