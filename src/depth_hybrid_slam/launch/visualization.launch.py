"""Optional lightweight views; never part of the default production graph."""

from ament_index_python.packages import get_package_share_directory
from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = (get_package_share_directory("depth_hybrid_slam") +
              "/config/classroom_localization.rviz")
    rviz = Node(package="rviz2", executable="rviz2", output="screen",
                arguments=["-d", config],
                condition=IfCondition(LaunchConfiguration("start_rviz")))
    image = Node(package="rqt_image_view", executable="rqt_image_view",
                 name="raw_image_view", output="screen",
                 arguments=["/camera/camera/color/image_raw"],
                 condition=IfCondition(LaunchConfiguration("start_image_view")))
    return LaunchDescription(common_arguments({"start_rviz": "true"})+[rviz, image])
