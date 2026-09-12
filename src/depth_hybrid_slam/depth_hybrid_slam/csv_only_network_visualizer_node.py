"""Render every real A/B segment separately from the selected active route."""

from bisect import bisect_right
import json
import math
from pathlib import Path

from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .csv_only_branching import (
    NETWORK_BRANCH_PAIRS, csv_only_network_segments, network_pair_geometry,
    validate_display_correspondence)
from .ros_helpers import safe_shutdown
from .stop_editor_network import CHOICE_SEGMENTS, segment_branch


class CsvOnlyNetworkVisualizerNode(Node):
    def __init__(self):
        super().__init__("depth_csv_only_network_visualizer")
        self.declare_parameter("route_path", "")
        self.declare_parameter("route_metadata_path", "")
        self.declare_parameter("display_route_path", "")
        self.declare_parameter("display_metadata_path", "")
        self.declare_parameter("prehardware_test_only", True)
        if not bool(self.get_parameter("prehardware_test_only").value):
            raise ValueError("CSV-only network visualizer is test-only")
        route_path = str(self.get_parameter("route_path").value)
        metadata_path = str(self.get_parameter("route_metadata_path").value)
        display_path = str(self.get_parameter("display_route_path").value)
        display_metadata = str(
            self.get_parameter("display_metadata_path").value)
        if not display_metadata and display_path:
            display_metadata = str(Path(display_path).with_suffix(".metadata.yaml"))

        # These coordinates intentionally match the follower/virtual vehicle.
        # The piecewise VSLAM overlay is audited below but is not substituted
        # into this CSV-only loop, which would detach the network from /odom.
        self.segments = csv_only_network_segments(route_path, metadata_path)
        self.rendered_geometry = network_pair_geometry(self.segments)
        self.display_geometry = None
        self.correspondence = None
        if display_path:
            self.correspondence = validate_display_correspondence(
                route_path, display_path)
            if not self.correspondence["valid"]:
                raise ValueError(
                    "display-aligned CSV does not exactly correspond by "
                    "(segment_id, point_index)")
            display_segments = csv_only_network_segments(
                display_path, display_metadata)
            self.display_geometry = network_pair_geometry(display_segments)

        points = [point for values in self.segments.values() for point in values]
        self.status_position = (
            min(point.x for point in points)+28.0,
            max(point.y for point in points)-10.0)
        self.status = {
            "selected": {"START": "pending", "T": "pending",
                         "V": "pending", "END": "pending"},
            "parking_lifecycle": {"T": "INACTIVE", "V": "INACTIVE"},
            "route_case": "AAAA", "state": "WAITING_FOR_START_POSE",
            "last_event": "",
        }
        self.create_subscription(
            String, "/depth_slam/route/case_state", self.on_state, 10)
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            MarkerArray, "/depth_slam/csv_only/full_network", qos)
        self.debug_publisher = self.create_publisher(
            String, "/depth_slam/csv_only/network_debug", qos)
        self.debug_data = self.make_debug_data()
        self.dirty = True
        self.debug_pending = True
        self.create_timer(0.2, self.publish_if_changed)

    def make_debug_data(self):
        output = {
            "frame_id": "map",
            "render_coordinate_source": "route_path metadata transform",
            "rendered_pairs": self.rendered_geometry,
            "display_aligned_pairs": self.display_geometry,
            "display_correspondence": self.correspondence,
        }
        for choice, name_a, name_b in NETWORK_BRANCH_PAIRS:
            values = self.rendered_geometry[choice]
            output[f"{name_a}_points"] = values[name_a]["points"]
            output[f"{name_b}_points"] = values[name_b]["points"]
            output[f"{choice}_max_separation_m"] = values[
                "maximum_separation_m"]
        return output

    def on_state(self, message):
        try:
            incoming = json.loads(message.data)
            visible = {
                "selected": incoming.get("selected", {}),
                "parking_lifecycle": incoming.get("parking_lifecycle", {}),
                "route_case": incoming.get("route_case", "AAAA"),
                "state": incoming.get("state", ""),
                "last_event": incoming.get("last_event", ""),
            }
            current = {key: self.status.get(key) for key in visible}
            if visible != current:
                self.status = incoming
                self.dirty = True
        except (TypeError, ValueError):
            self.get_logger().error("invalid CSV-only case state JSON")

    def publish_if_changed(self):
        if self.dirty:
            self.publish()
            self.dirty = False
        if self.debug_pending:
            self.debug_publisher.publish(String(data=json.dumps(
                self.debug_data, sort_keys=True, separators=(",", ":"))))
            self.debug_pending = False

    @staticmethod
    def choice_for_segment(name):
        for choice, branches in CHOICE_SEGMENTS.items():
            if name in branches.values():
                return choice
        return None

    def style(self, name):
        branch = segment_branch(name)
        if branch not in ("A", "B"):
            return 0.050, 0.60
        choice = self.choice_for_segment(name)
        selected = self.status.get("selected", {}).get(choice, "pending")
        if selected == "pending":
            return 0.120, 1.00
        if selected == branch:
            return 0.145, 1.00
        return 0.095, 0.62

    @staticmethod
    def color(marker, branch, alpha):
        if branch == "A":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.70, 1.0
        elif branch == "B":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.10, 0.70
        else:
            marker.color.r = marker.color.g = marker.color.b = 0.72
        marker.color.a = alpha

    @staticmethod
    def point_at(points, cumulative, distance):
        index = min(len(points)-2, max(
            0, bisect_right(cumulative, distance)-1))
        first, second = points[index], points[index+1]
        length = cumulative[index+1]-cumulative[index]
        ratio = 0.0 if length <= 1.0e-12 else \
            (distance-cumulative[index])/length
        return Point(
            x=first.x+ratio*(second.x-first.x),
            y=first.y+ratio*(second.y-first.y), z=0.10)

    @classmethod
    def dashed_points(cls, points, dash_m=0.70, gap_m=0.42):
        """Return real LINE_LIST dash pairs along the unmodified geometry."""
        cumulative = [0.0]
        for first, second in zip(points, points[1:]):
            cumulative.append(cumulative[-1]+math.hypot(
                second.x-first.x, second.y-first.y))
        total = cumulative[-1]
        output = []
        start = 0.0
        while start < total:
            end = min(total, start+dash_m)
            output.extend((cls.point_at(points, cumulative, start),
                           cls.point_at(points, cumulative, end)))
            start += dash_m+gap_m
        return output

    @staticmethod
    def label_offset(points, branch):
        first, second = points[0], points[1]
        dx, dy = second.x-first.x, second.y-first.y
        length = max(1.0e-9, math.hypot(dx, dy))
        side = 1.0 if branch == "A" else -1.0
        vertical = -0.70 if branch == "A" else 0.70
        return (side*(-dy/length)*1.30,
                side*(dx/length)*1.30+vertical)

    def publish(self):
        now = self.get_clock().now().to_msg()
        output = MarkerArray()
        marker_id = 0
        for name, points in sorted(self.segments.items()):
            branch = segment_branch(name)
            width, alpha = self.style(name)
            line = Marker()
            line.header.frame_id, line.header.stamp = "map", now
            # One namespace per physical segment makes topic inspection
            # unambiguous and prevents unrelated geometry from being joined.
            line.ns, line.id = name, marker_id
            line.type, line.action = (
                Marker.LINE_LIST if branch == "B" else Marker.LINE_STRIP,
                Marker.ADD)
            line.pose.orientation.w = 1.0
            line.scale.x = width
            self.color(line, branch, alpha)
            if branch == "B":
                line.points = self.dashed_points(points)
            else:
                line.points = [Point(x=p.x, y=p.y, z=0.10) for p in points]
            output.markers.append(line)
            marker_id += 1

            if branch in ("A", "B"):
                offset_x, offset_y = self.label_offset(points, branch)
                label = Marker()
                label.header = line.header
                label.ns, label.id = f"label_{name}", 1000+marker_id
                label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
                label.pose.position.x = points[0].x+offset_x
                label.pose.position.y = points[0].y+offset_y
                label.pose.position.z, label.pose.orientation.w = 0.85, 1.0
                label.scale.z = 1.35
                self.color(label, branch, max(0.82, alpha))
                label.text = name
                output.markers.append(label)

        stop_id = 0
        for points in self.segments.values():
            for point in points:
                if point.event != "STOP_LINE":
                    continue
                stop = Marker()
                stop.header.frame_id, stop.header.stamp = "map", now
                stop.ns, stop.id = "STOP_MARKERS", 2000+stop_id
                stop.type, stop.action = Marker.CYLINDER, Marker.ADD
                stop.pose.position.x, stop.pose.position.y = point.x, point.y
                stop.pose.position.z, stop.pose.orientation.w = 0.22, 1.0
                stop.scale.x = stop.scale.y = 0.52
                stop.scale.z = 0.36
                stop.color.r, stop.color.a = 1.0, 0.95
                output.markers.append(stop)
                stop_id += 1

        selected = self.status.get("selected", {})
        status = Marker()
        status.header.frame_id, status.header.stamp = "map", now
        status.ns, status.id = "ROUTE_CASE_STATUS", 3000
        status.type, status.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        status.pose.position.x, status.pose.position.y = self.status_position
        status.pose.position.z, status.pose.orientation.w = 1.5, 1.0
        status.scale.z = 1.10
        status.color.r, status.color.g = 1.0, 0.90
        status.color.b, status.color.a = 0.10, 1.0
        lifecycle = self.status.get("parking_lifecycle", {})

        def parking_text(choice):
            phase = lifecycle.get(choice, "INACTIVE")
            branch = selected.get(choice, "pending")
            if phase == "B_REQUESTED":
                return f"{choice}: B REQUESTED"
            if phase.startswith("COMMITTED_"):
                return f"{choice}: COMMITTED {phase[-1]}"
            if phase == "REVERSE_STARTED":
                return f"{choice}: REVERSE STARTED (COMMITTED {branch})"
            if phase == "COMPLETE":
                return f"{choice}: COMPLETE (COMMITTED {branch})"
            return f"{choice}: {phase}"

        status.text = (
            f"START SELECTED: {selected.get('START', 'pending')}\n"
            f"{parking_text('T')}\n{parking_text('V')}\n"
            f"END SELECTED: {selected.get('END', 'pending')}")
        status.text += (f"\nCASE: {self.status.get('route_case', 'AAAA')}"
                        f"\n{self.status.get('state', '')}")
        if self.status.get("last_event") == "LATE_BRANCH_COMMAND_IGNORED":
            status.text += "\nLATE_BRANCH_COMMAND_IGNORED"
        output.markers.append(status)
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = CsvOnlyNetworkVisualizerNode()
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
