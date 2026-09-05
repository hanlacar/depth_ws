"""Safety and wiring contracts for the non-spatial live traffic-light view."""

from pathlib import Path

from depth_hybrid_slam.traffic_light_live_guard import live_graph_conflicts
from depth_hybrid_slam.traffic_light_overlay_node import image_to_bgr
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_live_graph_accepts_exactly_one_camera_and_cuvslam():
    assert live_graph_conflicts(1, 1, {}) == []


@pytest.mark.parametrize(
    "camera,cuvslam,forbidden,expected",
    [
        (0, 1, {}, "D456_RGB_PUBLISHER_COUNT:0"),
        (2, 1, {}, "D456_RGB_PUBLISHER_COUNT:2"),
        (1, 0, {}, "CUVSLAM_ODOMETRY_PUBLISHER_COUNT:0"),
        (1, 2, {}, "CUVSLAM_ODOMETRY_PUBLISHER_COUNT:2"),
        (1, 1, {"/camera_drive": 1},
         "FORBIDDEN_CONTROL_PUBLISHER:/camera_drive:1"),
        (1, 1, {"/mcu/cmd_drive": 1},
         "FORBIDDEN_CONTROL_PUBLISHER:/mcu/cmd_drive:1"),
    ],
)
def test_live_graph_fails_closed(camera, cuvslam, forbidden, expected):
    assert expected in live_graph_conflicts(camera, cuvslam, forbidden)


class ImageMessage:
    height = 1
    width = 2
    step = 8
    encoding = "rgb8"
    data = bytes([255, 0, 0, 0, 255, 0, 99, 99])


def test_rgb_image_conversion_handles_row_padding():
    image = image_to_bgr(ImageMessage())
    np.testing.assert_array_equal(
        image, np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8))


def test_launch_is_external_sensor_only_and_has_no_control_or_mapping():
    source = (ROOT / "launch" / "traffic_light_live.launch.py").read_text()
    for required in (
        "camera_yolo_inference", "camera_rgb_traffic_light",
        "traffic_light_fusion_node", "traffic_light_live_overlay",
        "/camera/camera/color/image_raw",
        "/depth_slam/traffic_light/debug_overlay",
    ):
        assert required in source
    for prohibited in (
        "realsense2_camera", "rtabmap", "perception_vslam_validator",
        "camera_mission_perception", "gazebo", "route_follower",
    ):
        assert prohibited not in source.lower()


def test_overlay_is_non_spatial_and_has_no_command_publishers():
    source = (
        ROOT / "depth_hybrid_slam" / "traffic_light_overlay_node.py"
    ).read_text()
    for prohibited in (
        "tf2", "mapdata", "mapgraph", "/camera_drive", "/camera_wheel",
        "/slam_drive", "/slam_wheel", "/mcu/cmd_",
    ):
        assert prohibited not in source.lower()
