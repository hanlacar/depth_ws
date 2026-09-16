import os
from pathlib import Path
import re
import subprocess

from depth_hybrid_slam.workspace_paths import workspace_root


ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "setup_depth.sh"


def _source_setup(overrides=None):
    env = os.environ.copy()
    for name in (
            "DEPTH_WS_ROOT", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
            "ROS_AUTOMATIC_DISCOVERY_RANGE", "CYCLONEDDS_URI",
            "DEPTH_WS_PRESERVE_CYCLONEDDS_URI"):
        env.pop(name, None)
    env.update(overrides or {})
    command = r'''
source "$1" >/dev/null
printf '%s\n' "$DEPTH_WS_ROOT" "$ROS_DOMAIN_ID" "$RMW_IMPLEMENTATION" \
  "$ROS_AUTOMATIC_DISCOVERY_RANGE" "${CYCLONEDDS_URI-UNSET}"
'''
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(SETUP)], env=env,
        check=True, capture_output=True, text=True)
    return result.stdout.splitlines()


def test_setup_depth_safe_defaults_and_stale_uri_cleanup():
    values = _source_setup({"CYCLONEDDS_URI": "file:///tmp/stale.xml"})
    assert values == [
        str(ROOT), "12", "rmw_cyclonedds_cpp", "LOCALHOST", "UNSET"]


def test_setup_depth_preserves_explicit_user_values():
    values = _source_setup({
        "ROS_DOMAIN_ID": "37",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "OFF",
    })
    assert values[1:] == ["37", "rmw_fastrtps_cpp", "OFF", "UNSET"]


def test_custom_cyclonedds_uri_requires_explicit_opt_in():
    values = _source_setup({
        "CYCLONEDDS_URI": "file:///tmp/advanced.xml",
        "DEPTH_WS_PRESERVE_CYCLONEDDS_URI": "1",
    })
    assert values[-1] == "file:///tmp/advanced.xml"


def test_workspace_root_accepts_arbitrary_clone_locations(monkeypatch):
    for path in (
            "/home/qor/depth_ws", "/home/ww/depth_ws",
            "/home/test/depth_ws", "/opt/project/depth_ws"):
        monkeypatch.setenv("DEPTH_WS_ROOT", path)
        assert workspace_root() == Path(path)


def test_legacy_network_script_is_portable_setup_alias_only():
    text = (ROOT / "tools/ros_network_env.sh").read_text(encoding="utf-8")
    assert "../setup_depth.sh" in text
    for forbidden in (
            "NetworkInterface", "STATIC_PEERS", "ACTIVE_IFACE",
            "ACTIVE_IPV4", "mktemp", "ip -4 route", "SUBNET"):
        assert forbidden not in text


def test_launch_files_do_not_manage_dds_or_bind_home_directory():
    forbidden_network = (
        "CYCLONEDDS_URI", "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "ROS_LOCALHOST_ONLY", "NetworkInterface", "SetEnvironmentVariable",
    )
    home_path = re.compile(r"/home/[A-Za-z0-9_-]+/")
    launch_roots = (
        ROOT / "src/depth_hybrid_slam/launch",
        ROOT / "src/camera_yolo_inference/launch",
    )
    for launch_root in launch_roots:
        for path in launch_root.glob("*.launch.py"):
            text = path.read_text(encoding="utf-8")
            assert not home_path.search(text), path
            for value in forbidden_network:
                assert value not in text, (path, value)


def test_setup_does_not_discover_or_generate_network_configuration():
    text = SETUP.read_text(encoding="utf-8")
    for forbidden in (
            "NetworkInterface", "<Peer", "STATIC_PEERS", "ACTIVE_IFACE",
            "ACTIVE_IPV4", "mktemp", "ip -4 route"):
        assert forbidden not in text
    assert "export ROS_DOMAIN_ID=12" in text
    assert "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in text
    assert "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST" in text
