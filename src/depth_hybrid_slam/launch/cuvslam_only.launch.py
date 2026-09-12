"""D456 stereo-VI cuVSLAM graph using an externally managed camera driver."""

from depth_hybrid_slam.launch_common import common_arguments
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    visual_slam = ComposableNode(
        name="visual_slam_node",
        package="isaac_ros_visual_slam",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        parameters=[{
            "enable_image_denoising": False,
            "rectified_images": True,
            "tracking_mode": 1,
            "gyro_noise_density": 0.000244,
            "gyro_random_walk": 0.000019393,
            "accel_noise_density": 0.001862,
            "accel_random_walk": 0.003,
            "calibration_frequency": 400.0,
            "image_jitter_threshold_ms": 19.0,
            "base_frame": "base_link",
            # Production vehicle odometry/TF is owned by the latest MCU
            # bridge. RTAB-Map owns map->odom. cuVSLAM still publishes its
            # tracking Odometry topic, but must not duplicate either TF edge.
            "publish_odom_to_base_tf": False,
            "publish_map_to_odom_tf": False,
            "imu_frame": "camera_gyro_optical_frame",
            "enable_slam_visualization": False,
            "enable_landmarks_view": False,
            "enable_observations_view": False,
            "camera_optical_frames": [
                "camera_infra1_optical_frame",
                "camera_infra2_optical_frame",
            ],
        }],
        remappings=[
            ("visual_slam/image_0", "/camera/camera/infra1/image_rect_raw"),
            ("visual_slam/camera_info_0", "/camera/camera/infra1/camera_info"),
            ("visual_slam/image_1", "/camera/camera/infra2/image_rect_raw"),
            ("visual_slam/camera_info_1", "/camera/camera/infra2/camera_info"),
            ("visual_slam/imu", "/camera/camera/imu"),
        ],
    )
    container = ComposableNodeContainer(
        name="depth_cuvslam_container", namespace="",
        package="rclcpp_components", executable="component_container_mt",
        composable_node_descriptions=[visual_slam], output="screen",
    )
    mount = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="base_to_d456_mount",
        condition=IfCondition(LaunchConfiguration("publish_camera_mount_tf")),
        # Latest MCU CURRENT_VALUES: base_link is at axle height and camera_link
        # is 0.835 m above it. The ground-plane projection's 0.970 m optical
        # height is 0.835 + the 0.135 m base_link-to-ground offset.
        arguments=["--x", "0.015", "--y", "0", "--z", "0.835",
                   "--roll", "0", "--pitch", "0.0872665",
                   "--yaw", "0", "--frame-id", "base_link",
                   "--child-frame-id", "camera_link"],
    )
    bridge = Node(package="depth_hybrid_slam", executable="cuvslam_bridge",
                  name="cuvslam_bridge", output="screen")
    return LaunchDescription(common_arguments()+[mount, container, bridge])
