import importlib.util
from pathlib import Path
import shlex

from depth_hybrid_slam.localization_node import apply_route_policy
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_localization_launch():
    path = ROOT/"launch"/"hybrid_localization.launch.py"
    spec = importlib.util.spec_from_file_location("hybrid_localization_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_only_prefix_protects_database_directory(tmp_path, monkeypatch):
    module = load_localization_launch()
    database = tmp_path/"classroom"/"rtabmap.db"
    database.parent.mkdir()
    database.touch()
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/bwrap")
    parts = shlex.split(module._read_only_prefix(database))
    assert parts[:5] == ["/usr/bin/bwrap", "--bind", "/", "/", "--ro-bind"]
    assert parts[5:7] == [str(database.parent), str(database.parent)]
    assert parts[-2:] == ["--die-with-parent", "--"]


def test_map_only_localization_is_not_forced_to_route_mismatch():
    assert apply_route_policy("RELOCALIZED", True, False, False) == "RELOCALIZED"
    assert apply_route_policy("NOT_LOCALIZED", True, False, False) == "NOT_LOCALIZED"


def test_localization_reports_not_localized_before_first_match():
    source = (ROOT/"depth_hybrid_slam"/"localization_node.py").read_text()
    not_localized = source.index('if localization_mode and not self.relocalized_confirmed:')
    pending = source.index('elif self.core.pending is not None:')
    assert not_localized < pending
    assert 'state = "NOT_LOCALIZED"' in source


def test_route_backed_localization_still_fails_closed():
    assert apply_route_policy("RELOCALIZED", True, True, False) == "MAP_MISMATCH"


def test_classroom_rviz_contract():
    config = yaml.safe_load((ROOT/"config"/"classroom_localization.rviz").read_text())
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "map"
    displays = manager["Displays"]
    topics = {d.get("Topic", {}).get("Value") for d in displays}
    assert "/rtabmap/mapData" in topics
    assert "/rtabmap/mapGraph" in topics
    assert "/rtabmap/cloud_map" in topics
    assert "/depth_slam/localization/odometry" in topics
    assert "/visual_slam/vis/localizer_loop_closure_cloud" not in topics
    map_cloud = next(d for d in displays if d.get("Name") == "Saved Map Cloud")
    assert map_cloud["Download namespace"] == "/rtabmap/rtabmap"
    published = next(d for d in displays if d.get("Name") == "Published Full Cloud")
    assert published["Enabled"] is False


def test_localization_uses_tf_to_avoid_zero_odom_covariance():
    source = (ROOT/"depth_hybrid_slam"/"launch_common.py").read_text()
    assert '"odom_frame_id": "odom" if localization else ""' in source
    assert '"odom_tf_linear_variance": "0.001"' in source
    assert '"odom_tf_angular_variance": "0.01"' in source
