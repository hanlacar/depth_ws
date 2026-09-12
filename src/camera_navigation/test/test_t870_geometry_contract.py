import math
from pathlib import Path

import numpy as np
import pytest

from camera_navigation.direct_bev_controller import (
    BevControllerConfig, DirectBevController)
from camera_navigation.ground_plane_calibration import rotation_matrix_rpy
from camera_navigation.stop_line_control import StopLineConfig
from camera_navigation.stop_line_memory import StopLineMemoryConfig


ROOT = Path(__file__).parents[2]


def standard_rep103_pitch_matrix(pitch_deg):
    pitch = math.radians(pitch_deg)
    return np.array([[math.cos(pitch), 0.0, math.sin(pitch)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(pitch), 0.0, math.cos(pitch)]])


def test_projection_negative_five_and_t870_positive_five_both_look_down():
    projection = rotation_matrix_rpy(0.0, -5.0, 0.0)
    t870_tf = standard_rep103_pitch_matrix(5.0)
    center_axis = np.array([1.0, 0.0, 0.0])
    assert (projection @ center_axis)[2] < 0.0
    assert (t870_tf @ center_axis)[2] < 0.0
    assert np.allclose(projection, t870_tf, atol=1.0e-9)


def test_commissioned_mount_and_realsense_internal_tf_contract():
    mount = (ROOT/'depth_hybrid_slam/launch/cuvslam_only.launch.py').read_text()
    assert '"--x", "0.015"' in mount
    assert '"--y", "0"' in mount
    assert '"--z", "0.835"' in mount
    assert '"--pitch", "0.0872665"' in mount
    d456 = (ROOT/'depth_hybrid_slam/config/d456_60hz.yaml').read_text()
    assert 'publish_tf: true' in d456


def test_all_production_control_wheelbases_are_073():
    assert BevControllerConfig().wheelbase_m == 0.73
    paths = [
        ROOT/'camera_navigation/config/bev_controller.yaml',
        ROOT/'camera_navigation/config/bev_path.yaml',
        ROOT/'camera_navigation/config/camera_path_controller.yaml',
    ]
    for path in paths:
        assert 'wheelbase_m: 0.73' in path.read_text(), str(path)


def test_stop_line_offsets_use_t870_center_base_geometry():
    # camera x=0.015, nose x=0.500 and front axle x=0.365.
    assert StopLineConfig().camera_to_front_bumper_m == pytest.approx(0.485)
    memory = StopLineMemoryConfig()
    assert memory.camera_to_bumper_m == pytest.approx(0.485)
    assert memory.front_axle_offset_m == pytest.approx(0.350)


def test_t870_minimum_turning_radius_uses_commissioned_geometry():
    radius = 0.73/math.tan(math.radians(22.0))
    assert radius == pytest.approx(1.8068134, abs=0.001)


def test_camera_production_does_not_publish_vehicle_mount_or_odom_tf():
    launch = (ROOT.parent/'scripts/run_d456_host.sh').read_text()
    assert 'static_transform_publisher' not in launch
    assert 'robot_state_publisher' not in launch
    assert 'Odometry' not in launch
    mission = (ROOT/'camera_navigation/camera_navigation/'
               'camera_mission_perception_node.py').read_text()
    assert '"front_axle_frame"' not in mission
    assert 'target = str(self.p("base_frame"))' in mission


def test_reference_adapter_keeps_exact_source_stamp_and_no_latest_fallback():
    source = (ROOT/'camera_navigation/camera_navigation/'
              'camera_reference_path_adapter_node.py').read_text()
    assert 'Time.from_msg(message.header.stamp)' in source
    assert 'lookup_transform(' in source
    assert 'Time()' not in source
