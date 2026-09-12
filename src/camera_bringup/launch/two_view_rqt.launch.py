from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="rqt_image_view", executable="rqt_image_view",
             name="rqt_perception_view",
             arguments=["/camera/perception_overlay_image"]),
        Node(package="rqt_image_view", executable="rqt_image_view",
             name="rqt_path_view", arguments=["/camera/path_overlay_image"])])
