import math
from pathlib import Path

from depth_hybrid_slam.perception_integration_guard import (
    integration_graph_conflicts, REQUIRED_TOPICS,
)
from depth_hybrid_slam.perception_vslam_core import (
    robust_bbox_projection, select_traffic_detection,
    traffic_validation_reasons, transform_point,
)
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def valid_fusion():
    return {
        "stamp": 10_000_000_000,
        "fused_state": "R", "fused_confidence": 0.91,
        "sources_agree": True, "positions_match": True,
        "single_source_used": False,
    }


def test_existing_yolo_schema_detection_is_selected_by_state():
    document = {"detections": [
        {"class_name": "G_light", "confidence": 0.95,
         "xyxy": [10, 20, 30, 50]},
        {"class_name": "R_light", "confidence": 0.80,
         "xyxy": [40, 20, 60, 50]},
    ]}
    selected = select_traffic_detection(document, "R")
    assert selected["class_name"] == "R_light"
    assert selected["bbox"] == (40.0, 20.0, 60.0, 50.0)


def test_robust_bbox_depth_projects_optical_xyz():
    depth = np.full((12, 12), 2000, dtype=np.uint16)
    projection = robust_bbox_projection(
        depth, "16UC1", [100.0, 0.0, 6.0, 0.0, 100.0, 6.0, 0, 0, 1],
        [2, 2, 10, 10], minimum_pixels=4)
    assert projection.valid and projection.reason == "OK"
    assert projection.median_depth_m == 2.0
    assert projection.optical_xyz_m == (0.0, 0.0, 2.0)


def test_depth_projection_rejects_sparse_and_high_spread_data():
    sparse = np.zeros((10, 10), dtype=np.uint16)
    result = robust_bbox_projection(
        sparse, "16UC1", [100, 0, 5, 0, 100, 5, 0, 0, 1],
        [0, 0, 10, 10], minimum_pixels=4)
    assert not result.valid and result.reason == "DEPTH_PIXELS_INSUFFICIENT"
    spread = np.tile(np.array([1000, 4000], dtype=np.uint16), (10, 5))
    result = robust_bbox_projection(
        spread, "16UC1", [100, 0, 5, 0, 100, 5, 0, 0, 1],
        [0, 0, 10, 10], minimum_pixels=4, maximum_mad_m=0.5)
    assert not result.valid and result.reason == "DEPTH_SPREAD_TOO_HIGH"


def test_transform_point_matches_translation_and_yaw():
    half = math.sqrt(0.5)
    result = transform_point((1, 0, 0), (2, 3, 4), (0, 0, half, half))
    assert np.allclose(result, (2, 4, 4))


def test_final_traffic_result_requires_2d_pair_and_timestamped_vslam():
    reasons = traffic_validation_reasons(
        valid_fusion(), 10.0, 10.02, 1, True, "READY", "MISSING",
        True, True)
    assert reasons == []
    fusion = valid_fusion()
    fusion["positions_match"] = None
    reasons = traffic_validation_reasons(
        fusion, 10.0, 10.2, 2, False, "INITIALIZING", "MISSING",
        False, False)
    assert "TWO_DIMENSIONAL_BBOX_NOT_MATCHED" in reasons
    assert "OBSERVATION_TIMESTAMP_MISMATCH" in reasons
    assert "CAMERA_PUBLISHER_COUNT:2" in reasons
    assert "CUVSLAM_TRACKING_INVALID" in reasons
    assert "VSLAM_NOT_READY" in reasons
    assert "MAP_TRANSFORM_INVALID" in reasons


def test_integration_guard_requires_one_camera_and_rejects_all_mcu_cmd_prefixes():
    required = {topic: 1 for topic in REQUIRED_TOPICS}
    assert integration_graph_conflicts(1, required, {}) == []
    problems = integration_graph_conflicts(
        2, required, {"/mcu/cmd_any_future_name": 1})
    assert "D456_RGB_PUBLISHER_COUNT:2" in problems
    assert any("/mcu/cmd_any_future_name" in item for item in problems)


def test_launch_is_sensor_external_and_advisory_only():
    launch = (ROOT / "launch" / "perception_vslam_view.launch.py").read_text()
    for forbidden in (
            "realsense2_camera", "camera_path_controller", "mission_decision",
            "command_selector", "gazebo", '"/camera_drive"', '"/slam_drive"'):
        assert forbidden not in launch
    assert "/camera/camera/color/image_raw" in launch
    assert "perception_vslam_validator" in launch
    assert "perception_vslam_overlay" in launch
    assert "ensure_safe_integration_graph" in launch


def test_t870_mount_and_rviz_topics_are_exact():
    launch = (ROOT / "launch" / "cuvslam_only.launch.py").read_text()
    assert '"--x", "0.015"' in launch
    assert '"--z", "0.835"' in launch
    assert '"--pitch", "0.0872665"' in launch
    rviz = (ROOT / "config" / "perception_vslam.rviz").read_text()
    for topic in ("/rtabmap/mapData", "/rtabmap/mapGraph", "/rtabmap/mapPath",
                  "/depth_slam/perception/markers",
                  "/depth_slam/perception/vehicle_path"):
        assert topic in rviz


def test_validation_node_has_no_database_or_control_output_contract():
    source = (ROOT / "depth_hybrid_slam" /
              "perception_vslam_node.py").read_text()
    assert "sqlite" not in source.lower()
    assert "rtabmap.db" not in source
    for forbidden in ("/camera_drive", "/camera_wheel", "/slam_drive",
                      "/slam_wheel", "/mcu/cmd_"):
        assert forbidden not in source
    assert "if result[\"valid\"]" in source
    assert "Time(nanoseconds=" in source
