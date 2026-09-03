import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_all_launch_files_parse_as_python():
    for path in ROOT.glob('src/**/launch/*.launch.py'):
        ast.parse(path.read_text(), filename=str(path))


def test_yaml_files_load():
    for path in ROOT.glob('src/**/*.yaml'):
        assert yaml.safe_load(path.read_text()) is not None


def test_required_launch_arguments_and_topic_remaps_exist():
    full = (ROOT / 'src/depth_bringup/launch/visual_slam_full.launch.py').read_text()
    mapping = (ROOT / 'src/depth_bringup/launch/visual_slam_mapping.launch.py').read_text()
    camera = (ROOT / 'src/depth_bringup/launch/d456_60fps.launch.py').read_text()
    for arg in ('use_mcu_odom', 'publish_mount_tf', 'delete_db', 'database_path'):
        assert arg in full
    for topic in ('rgb_topic', 'depth_topic', 'camera_info_topic', 'imu_topic', 'odom_topic'):
        assert topic in mapping
    for arg in ('serial_no', 'color_profile', 'depth_profile', 'gyro_fps', 'accel_fps'):
        assert arg in camera


def test_tf_publishers_are_mutually_exclusive_by_odom_mode():
    mapping = (ROOT / 'src/depth_bringup/launch/visual_slam_mapping.launch.py').read_text()
    assert 'UnlessCondition(use_mcu)' in mapping
    assert 'publish_tf: true' in (ROOT / 'src/depth_bringup/config/rtabmap_mapping.yaml').read_text()
    assert "'publish_tf': False" in mapping


def test_generated_data_is_ignored():
    ignored = (ROOT / '.gitignore').read_text()
    for entry in ('/build/', '/install/', '/log/', '*.db', '*.mcap'):
        assert entry in ignored
