from pathlib import Path

from depth_hybrid_slam.bag_common import percentile, resolve_profile, validate_id
from depth_hybrid_slam.bag_time_audit import (connected, gap_statistics,
                                              nearest_sync, series_statistics)


ROOT = Path(__file__).resolve().parents[1]


def test_profile_is_explicit_and_mapdata_is_opt_in():
    import yaml
    config = yaml.safe_load((ROOT / "config" / "competition_bag_topics.yaml").read_text())
    requested, expected, required = resolve_profile(config, "camera_slam")
    assert "/camera/camera/color/image_raw" in required
    assert "/depth_slam/cuvslam/odometry" in required
    assert "/rtabmap/mapData" not in requested
    assert "/camera/camera/aligned_depth_to_color/image_raw" not in requested
    aligned, _, _ = resolve_profile(config, "camera_slam_with_aligned_depth")
    assert "/camera/camera/aligned_depth_to_color/image_raw" in aligned
    assert expected["/tf_static"] == "tf2_msgs/msg/TFMessage"
    assert all(not group["enabled"] and group["topics"] == {}
               for group in config["future_groups"].values())


def test_identifier_cannot_escape_bag_root():
    assert validate_id("case_1") == "case_1"
    for unsafe in ("../case_1", "/tmp/x", "", "a b"):
        try:
            validate_id(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe id: {unsafe}")


def test_stamp_and_frequency_statistics():
    source = [1_000_000_000, 1_010_000_000, 1_020_000_000]
    received = [value + 5_000_000 for value in source]
    result = series_statistics(source, received)
    assert result["average_hz"] == 100.0
    assert result["duplicate_source_stamps"] == 0
    assert result["backwards_source_stamps"] == 0
    assert result["latency_ms"]["p95"] == 5.0


def test_sync_gap_percentile_and_tf_chain():
    sync = nearest_sync([0, 10_000_000, 20_000_000],
                        [1_000_000, 11_000_000, 21_000_000], 2.0)
    assert sync["mean_ms"] == 1.0 and sync["over_tolerance_percent"] == 0.0
    gaps = gap_statistics([0, 2_500_000, 5_000_000])
    assert gaps["mean_gap_ms"] == 2.5
    assert percentile([1, 2, 3], 95) == 2.9
    assert connected({("map", "odom"), ("odom", "base_link"),
                      ("base_link", "camera_link")},
                     ["map", "odom", "base_link", "camera_link"])


def test_safety_configuration_forbids_control():
    import yaml
    config = yaml.safe_load((ROOT / "config" / "bag_recording.yaml").read_text())
    assert config["storage_id"] == "mcap"
    assert config["storage_preset_profile"] == "fastwrite"
    assert config["compression_mode"] == "file"
    assert config["compression_format"] == "zstd"
    assert config["max_bag_size_bytes"] == 4 * 1024**3
    assert config["max_bag_duration_seconds"] == 900
    assert config["safety"] == {
        "enable_control": False,
        "dry_run": True,
        "publish_drive_commands": False,
    }


def test_scripts_never_record_all_or_use_sim_time():
    record = (ROOT.parents[1] / "scripts" / "record_competition_bag.sh").read_text()
    play = (ROOT.parents[1] / "scripts" / "play_competition_bag.sh").read_text()
    assert "ros2 bag record -a" not in record
    assert "--use-sim-time" not in record
    assert "--clock" not in play
    assert "--max-bag-size 4294967296" in record
    assert "--max-bag-duration 900" in record
    assert "--session-path" in record
    assert '[[ "$session_path" == "$bag_root/"* ]]' in record


def test_tracking_loss_is_not_mislabeled_as_pose_reset():
    source = (ROOT / "depth_hybrid_slam" / "cuvslam_bridge_node.py").read_text()
    on_status = source[source.index("    def on_status"):source.index("    def on_odom")]
    assert "self.tracking_loss_count += 1" in on_status
    assert "self.reset_count += 1" not in on_status
