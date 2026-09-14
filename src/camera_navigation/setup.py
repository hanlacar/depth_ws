from glob import glob
from setuptools import find_packages, setup

package_name = "camera_navigation"
setup(
    name=package_name, version="0.1.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/tools", glob("tools/*.py")),
        ("share/" + package_name + "/tools", glob("tools/*.sh")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="ww", maintainer_email="ww@todo.todo",
    description="D456 semantic perception and advisory-only mission signals",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "camera_mission_perception_node=camera_navigation.camera_mission_perception_node:main",
        "traffic_light_fusion_node=camera_navigation.traffic_light_fusion_node:main",
    ]},
)
