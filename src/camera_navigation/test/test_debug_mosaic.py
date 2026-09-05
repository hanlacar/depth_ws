import numpy as np

from camera_navigation.debug_mosaic_node import FrameSlot, PANELS, status_text


def test_validation_mosaic_has_required_nine_panels():
    assert [topic for _, topic in PANELS] == [
        "/camera/image_raw",
        "/perception/detections_image",
        "/camera/perception_overlay_image",
        "/perception/masks/road",
        "/perception/refined/road",
        "/perception/masks/white_line",
        "/perception/masks/yellow_line",
        "/camera/bev/overlay_image",
        "/camera/bev/camera_overlay",
    ]


def test_panel_status_distinguishes_no_data_empty_live_and_stale():
    slot = FrameSlot()
    assert status_text(slot, 10.0, 1.0) == "NO DATA"
    slot.image = np.zeros((4, 4), np.uint8)
    slot.encoding = "mono8"
    slot.wall_time = 10.0
    assert status_text(slot, 10.1, 1.0) == "EMPTY MASK"
    slot.image[0, 0] = 255
    assert status_text(slot, 10.1, 1.0) == "LIVE"
    assert status_text(slot, 11.1, 1.0) == "STALE"


def test_launch_keeps_mosaic_opt_in_and_production_defaults_unchanged():
    from pathlib import Path
    launch = (Path(__file__).parents[1] / "launch" /
              "direct_bev_video_rqt_validation.launch.py").read_text()
    assert '"start_debug_mosaic", default_value="false"' in launch
    production = (Path(__file__).parents[1] / "launch" /
                  "camera_bev_standalone.launch.py").read_text()
    assert 'DeclareLaunchArgument("planner_variant", default_value="production")' in production
