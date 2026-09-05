from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_production_mask_rate_remains_unlimited():
    config = (ROOT / "camera_yolo_inference" / "config" /
              "yolo_inference.yaml").read_text(encoding="utf-8")
    assert "mask_image_publish_hz: 0.0" in config


def test_validation_launch_is_explicit_a6_none_and_five_hz():
    launch = (ROOT / "camera_navigation" / "launch" /
              "direct_bev_video_rqt_validation.launch.py").read_text(
                  encoding="utf-8")
    assert '"planner_variant", default_value="hybrid_a6"' in launch
    assert '"line_track_mode", default_value="none"' in launch
    assert '"debug_image_publish_hz", default_value="5.0"' in launch
    assert '"mask_image_publish_hz", default_value="5.0"' in launch
    assert '"overlay_publish_hz", default_value="5.0"' in launch


def test_general_direct_bev_validation_defaults_remain_safe():
    launch = (ROOT / "camera_navigation" / "launch" /
              "direct_bev_video_validation.launch.py").read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("active_planner", default_value="none")' in launch
    assert 'DeclareLaunchArgument("planner_variant", default_value="production")' in launch
    assert 'DeclareLaunchArgument("line_track_mode", default_value="none")' in launch
