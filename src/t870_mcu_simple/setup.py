from glob import glob
import os

from setuptools import setup


package_name = "t870_mcu_simple"

setup(
    name=package_name,
    version="1.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/"+package_name]),
        ("share/"+package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="URRC",
    maintainer_email="urrc@example.com",
    description="T870 serial bridge with measured encoder/steering odometry",
    license="MIT",
    entry_points={"console_scripts": [
        "bridge = t870_mcu_simple.bridge_node:main",
    ]},
)
