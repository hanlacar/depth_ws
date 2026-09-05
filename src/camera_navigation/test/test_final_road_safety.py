import numpy as np

from camera_navigation.image_path_planner import (
    INVALID,
    ImagePathPlanner,
    PlannerConfig,
)


def corridor(left=220, right=420):
    road = np.zeros((480, 640), dtype=np.uint8)
    road[120:476, left:right+1] = 1
    return road


def test_final_projection_recovers_points_with_perspective_margin():
    planner = ImagePathPlanner()
    points = np.asarray([[205.0, 450.0], [225.0, 350.0], [219.0, 200.0]])
    projected, safe, details = planner._project_final_path_to_road(
        points, corridor(), 475)
    assert safe
    assert details["final_road_recovered"]
    assert details["final_road_offroad_points_before"] == 2
    for x, y in projected:
        margin = planner._final_safety_margin_px(y, 475)
        assert 220.0+margin <= x <= 420.0-margin


def test_final_projection_rejects_missing_current_road_row():
    planner = ImagePathPlanner()
    road = corridor()
    road[345:356] = 0
    points = np.asarray([[320.0, 450.0], [320.0, 350.0], [320.0, 250.0]])
    projected, safe, details = planner._project_final_path_to_road(
        points, road, 475)
    assert not safe
    assert not len(projected)
    assert details["final_road_unrecoverable"]


def test_complete_mask_loss_does_not_publish_stale_temporal_path():
    planner = ImagePathPlanner()
    road = corridor()
    empty = np.zeros_like(road)
    first = planner.plan(road, empty, empty, timestamp_sec=0.0)
    assert first.valid
    lost = planner.plan(empty, empty, empty, timestamp_sec=0.05)
    assert lost.state == INVALID
    assert not lost.valid
    assert len(lost.points) == 0
    assert lost.diagnostics["vehicle_containment_ok"]
    assert lost.diagnostics["final_road_unrecoverable"]


def test_source_release_and_confirmation_are_both_required():
    planner = ImagePathPlanner(PlannerConfig(
        source_confirm_frames=2, source_release_frames=3))
    assert planner._update_source_hysteresis("ROAD", 320.0)[0] == "ROAD"
    confirmed, transitioned = planner._update_source_hysteresis("LANE", 321.0)
    assert confirmed == "ROAD" and not transitioned
    confirmed, transitioned = planner._update_source_hysteresis("LANE", 322.0)
    assert confirmed == "ROAD" and not transitioned
    confirmed, transitioned = planner._update_source_hysteresis("LANE", 323.0)
    assert confirmed == "LANE" and transitioned


def test_empty_invalid_path_is_not_reported_as_offroad_path():
    planner = ImagePathPlanner()
    empty = np.zeros((480, 640), dtype=np.uint8)
    result = planner.plan(empty, empty, empty)
    assert result.state == INVALID
    assert len(result.points) == 0
    assert result.diagnostics["vehicle_containment_ok"]
