"""Pose-selected START and independent T/V/END decisions for CSV-only runs."""

import json
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .csv_only_branching import (
    IndependentRouteCaseSelector, StartBranchClassifier,
    csv_only_network_segments, load_csv_only_route_case,
    parking_branch_geometry)
from .models import Pose2D
from .ros_helpers import safe_shutdown, yaw_from_quaternion


class CsvOnlyBranchSelectorNode(Node):
    def __init__(self):
        super().__init__("depth_csv_only_branch_selector")
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        self.declare_parameter("prehardware_test_only", True)
        self.declare_parameter("start_path_length_m", 12.0)
        self.declare_parameter("start_heading_weight_m", 1.0)
        self.declare_parameter("start_ambiguity_margin", 0.35)
        self.declare_parameter("start_maximum_lateral_m", 2.0)
        self.declare_parameter("parking_decision_window_s", 3.0)
        self.declare_parameter("parking_post_commit_hold_s", 3.0)
        if not bool(self.get_parameter("prehardware_test_only").value):
            raise ValueError("CSV-only branch selector is test-only")
        route_path = str(self.get_parameter("route_path").value)
        metadata_path = str(self.get_parameter("route_metadata_path").value)
        segments = csv_only_network_segments(route_path, metadata_path)
        self.segments = segments
        self.geometry = parking_branch_geometry(segments)
        self.classifier = StartBranchClassifier(
            segments["START_A"], segments["START_B"],
            float(self.get_parameter("start_path_length_m").value),
            float(self.get_parameter("start_heading_weight_m").value),
            float(self.get_parameter("start_ambiguity_margin").value),
            float(self.get_parameter("start_maximum_lateral_m").value))
        self.selector = IndependentRouteCaseSelector(
            self.geometry,
            float(self.get_parameter("parking_decision_window_s").value),
            float(self.get_parameter("parking_post_commit_hold_s").value))
        self.last_result = None
        self.active_segment = ""
        self.active_index = None
        self.stop_key = ""
        self.stop_state = "IDLE"
        self.route_cache = {}
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self.on_pose, 10)
        self.create_subscription(
            String, "/depth_slam/route/active_segment", self.on_segment, 10)
        self.create_subscription(
            Int32, "/depth_slam/route/active_index",
            lambda message: setattr(self, "active_index", int(message.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/stop_waypoint_key",
            lambda message: setattr(self, "stop_key", str(message.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/stop_waypoint_state",
            lambda message: setattr(self, "stop_state", str(message.data)), 10)
        self.create_subscription(
            Float32, "/depth_slam/follower/candidate_drive", self.on_drive, 10)
        self.create_subscription(
            String, "/depth_slam/route/branch_command", self.on_command, 10)
        self.pub_case = self.create_publisher(
            String, "/depth_slam/route/selected_case", 10)
        self.pub_stop = self.create_publisher(
            Bool, "/depth_slam/csv_only/branch_stop", 10)
        self.pub_state = self.create_publisher(
            String, "/depth_slam/route/case_state", 10)
        self.create_timer(0.05, self.tick)

    def on_pose(self, message):
        if self.selector.choices["START"] is not None:
            return
        position = message.pose.pose.position
        pose = Pose2D(
            position.x, position.y,
            yaw_from_quaternion(message.pose.pose.orientation), 0.0)
        self.last_result = self.classifier.classify(pose)
        self.selector.set_start(self.last_result)

    def on_segment(self, message):
        self.active_segment = str(message.data)
        self.selector.enter_segment(self.active_segment, time.monotonic())

    def on_command(self, message):
        if not self.selector.request(message.data, time.monotonic()):
            event = self.selector.last_event
            self.get_logger().warn(
                f"{event or 'INVALID_BRANCH_COMMAND_IGNORED'}: {message.data}")

    def on_drive(self, message):
        if float(message.data) >= 0.0:
            return
        for choice in ("T", "V"):
            if self.active_segment in (f"{choice}_A", f"{choice}_B"):
                self.selector.note_reverse_started(choice, time.monotonic())

    def active_point(self):
        if self.active_index is None:
            return None
        case = self.selector.route_case
        if case not in self.route_cache:
            self.route_cache[case] = load_csv_only_route_case(
                str(self.get_parameter("route_path").value),
                str(self.get_parameter("route_metadata_path").value), case)
        route = self.route_cache[case]
        if not 0 <= self.active_index < len(route):
            return None
        point = route[self.active_index]
        if self.active_segment and point.segment_id != self.active_segment:
            return None
        return point

    def tick(self):
        now = time.monotonic()
        point = self.active_point()
        if point is not None:
            self.selector.observe_point(
                point.segment_id, point.point_index, point.direction,
                self.stop_key, self.stop_state, now)
        self.selector.advance(now)
        if self.selector.choices["START"] is not None:
            self.pub_case.publish(String(data=self.selector.route_case))
        self.pub_stop.publish(Bool(data=self.selector.stop))
        score = self.last_result
        data = {
            "state": self.selector.state,
            "route_case": self.selector.route_case,
            "selected": self.selector.status(),
            "requests": dict(sorted(self.selector.requests.items())),
            "parking_lifecycle": self.selector.lifecycle_status(),
            "timestamps": self.selector.timestamp_status(),
            "last_event": self.selector.last_event,
            "geometry_verdict": {
                choice: self.geometry[choice]["verdict"]
                for choice in ("T", "V")},
            "stop": self.selector.stop,
            "start_score_a": None if score is None else score.score_a,
            "start_score_b": None if score is None else score.score_b,
            "start_lateral_a": None if score is None else score.lateral_a,
            "start_lateral_b": None if score is None else score.lateral_b,
            "start_heading_a": None if score is None else score.heading_a,
            "start_heading_b": None if score is None else score.heading_b,
        }
        self.pub_state.publish(String(data=json.dumps(
            data, sort_keys=True, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = CsvOnlyBranchSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            safe_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
