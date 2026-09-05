import csv
import json

from depth_hybrid_slam.route_finalizer import realign_route
from depth_hybrid_slam.route_recorder_core import RouteRecorder


def test_loop_closure_realigns_old_points_and_preserves_raw(tmp_path):
    raw = tmp_path/"route.csv"
    recorder = RouteRecorder(raw, min_distance_m=.01)
    for index, x in enumerate((0.0, 0.5, 1.0)):
        recorder.append({
            "timestamp": 1.0+index, "x": x, "y": 0.0, "z": 0.0,
            "yaw": 0.0, "frame_id": "map", "tracking_valid": True,
            "localization_state": "TRACKING", "localization_confidence": .9,
            "reset_count": 0, "nearest_rtabmap_node_id": index+1,
            "node_relative_x_m": 0.0, "node_relative_y_m": 0.0,
            "node_relative_yaw_rad": 0.0, "valid": True,
        })
    recorder.finalize()
    before = raw.read_bytes()
    graph = tmp_path/"rtabmap_graph.json"
    graph.write_text(json.dumps({"session_id": "s1", "poses": {
        "1": {"x": 0.0, "y": 1.0, "yaw": 0.0},
        "2": {"x": 0.5, "y": 1.0, "yaw": 0.0},
        "3": {"x": 1.0, "y": 1.0, "yaw": 0.0}}}), encoding="utf-8")
    final = tmp_path/"route_final.csv"
    result = realign_route(raw, graph, final, spacing_m=.05)
    assert raw.read_bytes() == before
    with open(final, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["map_y_m"] == "1.0"
    assert rows[-1]["map_y_m"] == "1.0"
    assert result["final_point_count"] == 21
    assert result["graph_session_id"] == "s1"
