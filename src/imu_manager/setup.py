from setuptools import find_packages, setup

package_name = "imu_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", ["config/imu_manager.yaml"]),
        ("share/" + package_name + "/launch", [
            "launch/imu_manager.launch.py",
            "launch/imu_calibration.launch.py",
        ]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="ww",
    maintainer_email="ww@todo.todo",
    description="Filter D456 raw gyro and accel ROS topics.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "imu_manager_node = imu_manager.imu_manager_node:main",
        "imu_calibration_node = imu_manager.imu_calibration_node:main",
    ]},
)
