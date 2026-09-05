from glob import glob
import os
from setuptools import find_packages, setup

name = "camera_rgb_traffic_light"
setup(
    name=name, version="0.1.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/"+name]),
        ("share/"+name, ["package.xml"]),
        (os.path.join("share", name, "config"), glob("config/*.yaml")),
        (os.path.join("share", name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"], extras_require={"test": ["pytest"]},
    zip_safe=True, maintainer="qor", maintainer_email="qor@example.com",
    description="YOLO-independent CPU color traffic-light detector",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "rgb_traffic_light_node=camera_rgb_traffic_light.rgb_traffic_light_node:main",
    ]})
