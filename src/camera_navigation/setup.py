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
    description="Original-image-coordinate camera path generation",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "camera_path_controller_node=camera_navigation.camera_path_controller_node:main",
        "camera_pixel_controller_node=camera_navigation.camera_pixel_controller_node:main",
        "camera_image_path_node=camera_navigation.camera_image_path_node:main",
        "adaptive_non_bev_node=camera_navigation.adaptive_non_bev_node:main",
        "adaptive_pixel_controller_node=camera_navigation.adaptive_pixel_controller_node:main",
        "direct_bev_planner_node=camera_navigation.direct_bev_planner_node:main",
        "direct_bev_controller_node=camera_navigation.direct_bev_controller_node:main",
        "bev_wheel_selector_node=camera_navigation.bev_wheel_selector_node:main",
        "direct_bev_drive_node=camera_navigation.direct_bev_drive_node:main",
        "camera_metric_path_node=camera_navigation.camera_metric_path_node:main",
        "camera_reference_path_adapter_node=camera_navigation.camera_reference_path_adapter_node:main",
        "debug_mosaic_node=camera_navigation.debug_mosaic_node:main",
        "camera_mission_perception_node=camera_navigation.camera_mission_perception_node:main",
        "camera_mission_decision_node=camera_navigation.camera_mission_decision_node:main",
        "traffic_light_fusion_node=camera_navigation.traffic_light_fusion_node:main",
        "camera_command_selector_node=camera_navigation.camera_command_selector_node:main",
    ]},
)
