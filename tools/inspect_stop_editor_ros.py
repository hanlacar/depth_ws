#!/usr/bin/env python3
"""One-shot live ROS inspection for the full-network STOP editor."""

import csv
import json
from pathlib import Path
import time

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.srv import PublishMap
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "routes/network/route_network_segmented_all_branches_display_aligned.csv"


class Inspector(Node):
    def __init__(self):
        super().__init__("stop_editor_live_inspector")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.values = {}
        self.create_subscription(
            String, "/depth_slam/stop_editor/status",
            lambda msg: self.values.__setitem__("status", json.loads(msg.data)), latched)
        self.create_subscription(
            String, "/depth_slam/stop_editor/selected",
            lambda msg: self.values.__setitem__("selected", json.loads(msg.data)), latched)
        self.create_subscription(
            MarkerArray, "/depth_slam/stop_editor/network",
            lambda msg: self.values.__setitem__("network", msg), latched)
        self.create_subscription(
            MarkerArray, "/depth_slam/stop_editor/mode_markers",
            lambda msg: self.values.__setitem__("modes", msg), latched)
        self.create_subscription(
            MarkerArray, "/depth_slam/stop_editor/markers",
            lambda msg: self.values.__setitem__("stops", msg), latched)
        self.create_subscription(
            PathMessage, "/depth_slam/stop_editor/csv_overlay",
            lambda msg: self.values.__setitem__("csv", msg), latched)
        self.create_subscription(
            OccupancyGrid, "/rtabmap/map",
            lambda msg: self.values.__setitem__("map", msg), latched)
        self.case_pub = self.create_publisher(
            String, "/depth_slam/stop_editor/case", 10)
        self.mode_pub = self.create_publisher(
            String, "/depth_slam/stop_editor/mode", 10)
        self.action_pub = self.create_publisher(
            String, "/depth_slam/stop_editor/action", 10)
        self.click_pub = self.create_publisher(PointStamped, "/clicked_point", 10)
        self.map_client = self.create_client(
            PublishMap, "/rtabmap/rtabmap/publish_map")

    def wait_for(self, predicate, timeout=30.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return
        raise RuntimeError("live STOP editor inspection timed out")

    def publish_text(self, publisher, value):
        publisher.publish(String(data=value))
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

    def click(self, x, y):
        message = PointStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.point.x, message.point.y = x, y
        self.click_pub.publish(message)


def marker_frames(message):
    return {marker.header.frame_id for marker in message.markers
            if marker.action != marker.DELETEALL}


def main():
    rclpy.init()
    node = Inspector()
    try:
        node.wait_for(lambda: all(name in node.values for name in (
            "status", "network", "modes", "stops", "csv")))
        initial = node.values["status"]
        assert initial["selected_case"] == "ALL"
        assert initial["source_row_count"] == 7463
        assert initial["editable_route_rows"] == 7463
        assert initial["visualization_waypoint_count"] == 7463
        assert initial["route_case_count"] == 16
        assert node.values["csv"].header.frame_id == "map"
        assert len(node.values["csv"].poses) == 7463
        assert marker_frames(node.values["network"]) == {"map"}
        assert marker_frames(node.values["modes"]) == {"map"}
        assert marker_frames(node.values["stops"]) == {"map"}
        labels = {marker.text for marker in node.values["network"].markers}
        for text in ("START_A", "START_B", "T_A", "T_B", "V_A", "V_B",
                     "END A: END_AA", "END B: END_AB"):
            assert text in labels
        mode_labels = {marker.text for marker in node.values["modes"].markers}
        assert "MODE 2 SLOPE" in mode_labels

        node.publish_text(node.case_pub, "BBBB")
        node.wait_for(lambda: node.values["status"]["selected_case"] == "BBBB")
        node.wait_for(lambda: any(marker.ns == "selected_case"
                                  for marker in node.values["network"].markers))

        with OVERLAY.open(newline="", encoding="utf-8") as stream:
            target = next(row for row in csv.DictReader(stream)
                          if row["segment_id"] == "V_B" and
                          int(row["point_index"]) == 10)
        x, y = float(target["x_m"]), float(target["y_m"])
        node.publish_text(node.mode_pub, "ADD")
        node.click(x, y)
        node.wait_for(lambda: node.values.get("selected", {}).get("segment") == "V_B")
        node.wait_for(lambda: node.values["status"]["state"] == "STOP_ADDED")
        node.publish_text(node.mode_pub, "REMOVE")
        node.click(x, y)
        node.wait_for(lambda: node.values["status"]["state"] == "STOP_REMOVED")
        node.publish_text(node.action_pub, "UNDO")
        node.wait_for(lambda: node.values["status"]["state"] == "UNDO_USER_REMOVE")

        if node.map_client.wait_for_service(timeout_sec=5.0):
            request = PublishMap.Request()
            request.global_map, request.optimized, request.graph_only = True, True, False
            future = node.map_client.call_async(request)
            node.wait_for(lambda: future.done(), timeout=30.0)
            node.wait_for(lambda: "map" in node.values, timeout=30.0)
        assert node.values["map"].header.frame_id == "map"
        assert node.values["map"].info.width > 0 and node.values["map"].info.height > 0

        print(json.dumps({
            "rtabmap_map_frame": node.values["map"].header.frame_id,
            "rtabmap_map_size": [node.values["map"].info.width,
                                  node.values["map"].info.height],
            "csv_frame": node.values["csv"].header.frame_id,
            "csv_poses": len(node.values["csv"].poses),
            "network_frame": sorted(marker_frames(node.values["network"])),
            "stop_frame": sorted(marker_frames(node.values["stops"])),
            "mode_2_visible": "MODE 2 SLOPE" in mode_labels,
            "branch_labels_visible": True,
            "case_highlight": "BBBB",
            "stop_add": "PASS", "stop_remove": "PASS", "undo": "PASS",
        }, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
