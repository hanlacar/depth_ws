from glob import glob
import os
from setuptools import find_packages, setup

package_name = "nav2_csv_route_follower"

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
    description="Nav2-first fixed-route follower with segmented CSV fallback.",
    license="MIT",
    entry_points={"console_scripts": [
        "hybrid_route_follower = nav2_csv_route_follower.hybrid_node:main",
        "controlled_synthetic_odom = nav2_csv_route_follower.controlled_synthetic_odom:main",
        "inspect_route_network = nav2_csv_route_follower.inspect_route:main",
    ]},
)
