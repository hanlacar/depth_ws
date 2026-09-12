"""RViz Publish Point adapter and visualization for STOP_LINE editing."""

import json
import math

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .ros_helpers import quaternion_from_yaw, safe_shutdown
from .stop_editor_core import STOP_EVENT, StopRouteEditor
from .stop_editor_network import segment_branch, segment_label


MODE_LABELS = {
    1: "MODE 1",
    2: "MODE 2 SLOPE",
    3: "MODE 3",
    4: "MODE 4 INTERSECTION",
    5: "MODE 5 S-CURVE",
    6: "MODE 6 INTERSECTION",
    7: "MODE 7 T-PARK",
    8: "MODE 8",
    9: "MODE 9 ACCEL",
    10: "MODE 10 PARALLEL PARK",
    11: "MODE 11 EXIT",
}

MODE_COLORS = {
    1: (0.00, 0.85, 1.00),
    2: (1.00, 1.00, 0.00),
    3: (0.10, 1.00, 0.20),
    4: (1.00, 0.45, 0.00),
    5: (0.65, 0.20, 1.00),
    6: (1.00, 0.25, 0.25),
    7: (1.00, 0.00, 0.75),
    8: (0.10, 0.35, 1.00),
    9: (0.55, 1.00, 0.00),
    10: (0.00, 1.00, 0.65),
    11: (1.00, 1.00, 1.00),
}


class StopEditorNode(Node):
    def __init__(self):
        super().__init__("depth_stop_editor")
        defaults = {
            "route_path": "",
            "route_metadata_path": "",
            "visualization_route_path": "",
            "visualization_metadata_path": "",
            "output_path": (
                "/home/qor/depth_ws/routes/network/"
                "route_network_segmented_stop_edited.csv"
            ),
            "output_metadata_path": (
                "/home/qor/depth_ws/routes/network/"
                "route_network_segmented_stop_edited.metadata.yaml"),
            "maximum_click_distance_m": 0.50,
            "initial_mode": "ADD",
            "initial_case": "ALL",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.editor = StopRouteEditor(
            self.p("route_path"), self.p("route_metadata_path"),
            self.p("maximum_click_distance_m"),
            self.p("visualization_route_path"),
            self.p("visualization_metadata_path"))
        self.mode = str(self.p("initial_mode")).strip().upper()
        if self.mode not in ("ADD", "REMOVE"):
            raise ValueError("initial_mode must be ADD or REMOVE")
        if not self.editor.set_case(self.p("initial_case")):
            raise ValueError("initial_case must be ALL or one of AAAA..BBBB")
        self.last_state = "READY"
        self.last_click = None
        self.saved_paths = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_route_a = self.create_publisher(
            Path, "/depth_slam/stop_editor/route_a", qos)
        self.pub_route_b = self.create_publisher(
            Path, "/depth_slam/stop_editor/route_b", qos)
        self.pub_csv_overlay = self.create_publisher(
            Path, "/depth_slam/stop_editor/csv_overlay", qos)
        self.pub_network = self.create_publisher(
            MarkerArray, "/depth_slam/stop_editor/network", qos)
        self.pub_mode_markers = self.create_publisher(
            MarkerArray, "/depth_slam/stop_editor/mode_markers", qos)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/depth_slam/stop_editor/markers", qos)
        self.pub_status = self.create_publisher(
            String, "/depth_slam/stop_editor/status", qos)
        self.pub_selected = self.create_publisher(
            String, "/depth_slam/stop_editor/selected", qos)
        self.pub_count = self.create_publisher(
            Int32, "/depth_slam/stop_editor/stop_count", qos)
        self.create_subscription(PointStamped, "/clicked_point", self.clicked, 10)
        self.create_subscription(
            String, "/depth_slam/stop_editor/mode", self.set_mode, 10)
        self.create_subscription(
            String, "/depth_slam/stop_editor/action", self.action, 10)
        self.create_subscription(
            String, "/depth_slam/stop_editor/case", self.set_case, 10)
        root = "/depth_slam/stop_editor"
        self.create_service(Trigger, root+"/undo", self.undo_service)
        self.create_service(Trigger, root+"/save", self.save_service)
        self.create_service(Trigger, root+"/list", self.list_service)
        self.create_service(Trigger, root+"/status_request", self.status_service)
        self.create_timer(1.0, self.publish_all)
        self.publish_all()

    def p(self, name):
        return self.get_parameter(name).value

    def set_case(self, message):
        value = str(message.data).strip().upper()
        if self.editor.set_case(value):
            self.last_state = "CASE_" + value
        else:
            self.last_state = "CASE_REJECTED"
        self.publish_all()

    def set_mode(self, message):
        value = str(message.data).strip().upper()
        if value not in ("ADD", "REMOVE"):
            self.last_state = "MODE_REJECTED"
        else:
            self.mode = value
            self.last_state = "MODE_"+value
        self.publish_all()

    def action(self, message):
        value = str(message.data).strip().upper()
        if value == "UNDO":
            self._undo()
        elif value == "SAVE":
            self._save()
        elif value in ("LIST", "STATUS"):
            self.last_state = value
        else:
            self.last_state = "ACTION_REJECTED"
        self.publish_all()

    def clicked(self, message):
        frame = str(message.header.frame_id).lstrip("/")
        if frame != "map":
            self.last_state = "CLICK_FRAME_REJECTED"
            self.publish_all()
            return
        self.last_click = (float(message.point.x), float(message.point.y))
        result = (self.editor.add(*self.last_click) if self.mode == "ADD" else
                  self.editor.remove(*self.last_click))
        self.last_state = result.state
        self.publish_all()

    def _undo(self):
        result = self.editor.undo()
        self.last_state = result.state
        return result

    def _save(self):
        try:
            self.saved_paths = self.editor.save(
                self.p("output_path"), self.p("output_metadata_path"))
            self.last_state = "SAVED_GEOMETRY_UNCHANGED"
            return True, str(self.saved_paths[0])
        except (OSError, TypeError, ValueError) as error:
            self.last_state = "SAVE_REJECTED:"+str(error)
            return False, str(error)

    def undo_service(self, _request, response):
        result = self._undo()
        response.success = result.accepted
        response.message = result.state
        self.publish_all()
        return response

    def save_service(self, _request, response):
        response.success, response.message = self._save()
        self.publish_all()
        return response

    def list_service(self, _request, response):
        response.success = True
        response.message = json.dumps(self.stop_list(), ensure_ascii=False)
        self.last_state = "LIST"
        self.publish_all()
        return response

    def status_service(self, _request, response):
        response.success = True
        response.message = json.dumps(self.status(), ensure_ascii=False)
        self.last_state = "STATUS"
        self.publish_all()
        return response

    def _path_rows(self, rows, z):
        message = Path()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        points = [self.editor.map_xy[self.editor.key(row)] for row in rows]
        for index, (x, y) in enumerate(points):
            other = points[index+1] if index+1 < len(points) else points[max(0, index-1)]
            yaw = math.atan2(other[1]-y, other[0]-x)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.position.z = z
            pose.pose.orientation = quaternion_from_yaw(yaw)
            message.poses.append(pose)
        return message

    def _path(self, branch):
        return self._path_rows(
            self.editor.route_rows(branch), 0.18 if branch == "A" else 0.16)

    def _csv_overlay_path(self):
        return self._path_rows(self.editor.visualization_rows, 0.12)

    @staticmethod
    def _color(marker, red, green, blue, alpha=1.0):
        marker.color.r, marker.color.g = float(red), float(green)
        marker.color.b, marker.color.a = float(blue), float(alpha)

    @staticmethod
    def _chunks(rows):
        chunks, current = [], []
        previous = None
        for row in rows:
            identity = (str(row["segment_id"]), int(row["mode"]))
            contiguous = (previous is not None and identity == previous[:2] and
                          int(row["point_index"]) == previous[2] + 1)
            if current and not contiguous:
                chunks.append(current)
                current = []
            current.append(row)
            previous = (*identity, int(row["point_index"]))
        if current:
            chunks.append(current)
        return chunks

    def _route_marker(self, rows, marker_id, namespace, alpha, width, z):
        branch = segment_branch(rows[0]["segment_id"])
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = namespace, marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        self._color(marker, *MODE_COLORS[int(rows[0]["mode"])], alpha)
        points = [Point(x=self.editor.map_xy[self.editor.key(row)][0],
                        y=self.editor.map_xy[self.editor.key(row)][1], z=z)
                  for row in rows]
        if branch == "B":
            marker.type = Marker.LINE_LIST
            for index in range(len(points) - 1):
                if (index // 4) % 2 == 0:
                    marker.points.extend((points[index], points[index + 1]))
        else:
            marker.type = Marker.LINE_STRIP
            marker.points = points
        return marker

    def _network_markers(self):
        """Draw every unique segment once; optionally emphasize one case."""
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        marker_id = 1
        selected = self.editor.selected_case
        for chunk in self._chunks(self.editor.visualization_rows):
            branch = segment_branch(chunk[0]["segment_id"])
            if selected == "ALL":
                alpha = {"A": 1.0, "B": 0.95, "COMMON": 0.82,
                         "REFERENCE": 0.16}[branch]
                width = {"A": 0.13, "B": 0.17, "COMMON": 0.10,
                         "REFERENCE": 0.04}[branch]
            else:
                alpha, width = (0.08, 0.05) if branch != "REFERENCE" else (0.03, 0.03)
            array.markers.append(self._route_marker(
                chunk, marker_id, "full_network", alpha, width, 0.20))
            marker_id += 1

        if selected != "ALL":
            for chunk in self._chunks(self.editor.case_rows(selected)):
                array.markers.append(self._route_marker(
                    chunk, marker_id, "selected_case", 1.0, 0.28, 0.31))
                marker_id += 1

        now = self.get_clock().now().to_msg()
        for name, rows in self.editor.segments.items():
            row = rows[len(rows) // 2]
            x, y = self.editor.map_xy[self.editor.key(row)]
            label = Marker()
            label.header.frame_id, label.header.stamp = "map", now
            label.ns, label.id = "segment_labels", marker_id
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z, label.pose.orientation.w = 0.92, 1.0
            label.scale.z = 0.30 if segment_branch(name) != "REFERENCE" else 0.22
            branch = segment_branch(name)
            color = {"A": (0.25, 1.0, 1.0), "B": (1.0, 0.55, 1.0),
                     "COMMON": (0.9, 0.9, 0.9),
                     "REFERENCE": (0.55, 0.55, 0.55)}[branch]
            self._color(label, *color, 0.95 if branch != "REFERENCE" else 0.55)
            label.text = segment_label(name)
            array.markers.append(label)
            marker_id += 1
        return array

    def _mode_markers(self):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        now = self.get_clock().now().to_msg()
        marker_id = 1
        for mode in range(1, 12):
            rows = [row for row in self.editor.visualization_rows
                    if int(row["mode"]) == mode and
                    segment_branch(row["segment_id"]) != "REFERENCE"]
            if not rows:
                continue
            label_row = rows[len(rows)//2]
            x, y = self.editor.map_xy[self.editor.key(label_row)]
            label = Marker()
            label.header.frame_id, label.header.stamp = "map", now
            label.ns, label.id = "mode_labels", marker_id
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z, label.pose.orientation.w = 1.25, 1.0
            label.scale.z = 0.55 if mode == 2 else 0.38
            self._color(label, *MODE_COLORS[mode], 1.0)
            label.text = MODE_LABELS[mode]
            array.markers.append(label)
            marker_id += 1
        return array

    def _markers(self):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        now = self.get_clock().now().to_msg()
        marker_id = 1
        for row in self.editor.visualization_rows:
            if str(row["event"]).strip().upper() != STOP_EVENT:
                continue
            key = self.editor.key(row)
            x, y = self.editor.map_xy[key]
            required = key in self.editor.required_transitions
            recently_added = key == self.editor.last_added_key
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "map", now
            marker.ns, marker.id = "stop_points", marker_id
            marker.type = Marker.CUBE if required else Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = x, y
            marker.pose.position.z, marker.pose.orientation.w = 0.35, 1.0
            size = 0.48 if recently_added else 0.34
            marker.scale.x = marker.scale.y = marker.scale.z = size
            if required:
                self._color(marker, 1.0, 0.10, 0.75)
            else:
                self._color(marker, 1.0, 0.0, 0.0)
            array.markers.append(marker)
            marker_id += 1
            label = Marker()
            label.header = marker.header
            label.ns, label.id = "stop_labels", marker_id
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z, label.pose.orientation.w = 0.75, 1.0
            label.scale.z = 0.22
            self._color(label, 1.0, 0.8 if required else 1.0,
                        0.2 if required else 1.0)
            reason = "DIRECTION-CHANGE STOP" if required else "STOP"
            label.text = (f"{reason}\nM{row['mode']}\n"
                          f"segment={row['segment_id']}\n"
                          f"idx={row['point_index']}\n"
                          f"branch={segment_branch(row['segment_id'])}")
            array.markers.append(label)
            marker_id += 1
        if self.last_click is not None:
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "map", now
            marker.ns, marker.id = "last_click", marker_id
            marker.type, marker.action = Marker.CYLINDER, Marker.ADD
            marker.pose.position.x, marker.pose.position.y = self.last_click
            marker.pose.position.z, marker.pose.orientation.w = 0.08, 1.0
            marker.scale.x = marker.scale.y = 0.22
            marker.scale.z = 0.15
            self._color(marker, 0.0, 1.0, 1.0)
            array.markers.append(marker)
            marker_id += 1
        selected = self.editor.selected()
        if selected:
            key = (selected["segment"], selected["point_index"])
            x, y = self.editor.map_xy[key]
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "map", now
            marker.ns, marker.id = "selected", marker_id
            marker.type, marker.action = Marker.CUBE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y = x, y
            marker.pose.position.z, marker.pose.orientation.w = 0.18, 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.55
            self._color(marker, 0.0, 0.85, 1.0, 0.65)
            array.markers.append(marker)
            marker_id += 1
        anchor_key = self.editor.key(self.editor.visualization_rows[0])
        anchor = self.editor.map_xy[anchor_key]
        banner = Marker()
        banner.header.frame_id, banner.header.stamp = "map", now
        banner.ns, banner.id = "editor_status", marker_id
        banner.type, banner.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        banner.pose.position.x, banner.pose.position.y = anchor
        banner.pose.position.z, banner.pose.orientation.w = 2.0, 1.0
        banner.scale.z = 0.42
        self._color(banner, 1.0, 0.25 if not self.editor.alignment_validated else 1.0,
                    0.1 if not self.editor.alignment_validated else 1.0)
        alignment = "VALIDATED" if self.editor.alignment_validated else "UNVALIDATED"
        banner.text = ("CSV OVERLAY: VISUALIZATION ONLY\n"
                       f"ALIGNMENT: {alignment}\n"
                       f"STOP EDIT MODE: {self.mode}  CASE: {self.editor.selected_case}\n"
                       f"VISIBLE STOPS: {self.editor.visible_stop_count}  "
                       f"{self.last_state}")
        array.markers.append(banner)
        return array

    def stop_list(self):
        output = []
        for row in self.editor.visualization_rows:
            if str(row["event"]).strip().upper() != STOP_EVENT:
                continue
            key = self.editor.key(row)
            x, y = self.editor.map_xy[key]
            output.append({
                "mode": int(row["mode"]), "segment": row["segment_id"],
                "point_index": int(row["point_index"]), "x_m": x, "y_m": y,
                "branch": segment_branch(row["segment_id"]),
                "required_direction_change": key in self.editor.required_transitions,
            })
        return output

    def status(self):
        added, removed = self.editor.changes()
        return {
            "state": self.last_state, "edit_mode": self.mode,
            "selected_case": self.editor.selected_case,
            "case_display_default": "ALL",
            "route_case_count": len(self.editor.route_cases),
            "route_cases": self.editor.case_report(),
            "source_row_count": len(self.editor.rows),
            "editable_route_rows": len(self.editor.rows),
            "unique_segment_count": len(self.editor.segments),
            "route_network_waypoint_count": len(self.editor.route_network_keys),
            "reference_waypoint_count": len(self.editor.reference_keys),
            "alignment": ("VALIDATED" if self.editor.alignment_validated else
                          "UNVALIDATED"),
            "prehardware_test_only": not self.editor.alignment_validated,
            "csv_overlay": "VISUALIZATION_ONLY",
            "visualization_alignment": "UNVALIDATED",
            "visualization_frame": "map",
            "visualization_route_path": str(
                self.editor.visualization_route_path or ""),
            "visualization_waypoint_count": len(
                self.editor.visualization_rows),
            "maximum_click_distance_m": self.editor.maximum_click_distance_m,
            "stop_before_count": self.editor.source_stop_count,
            "stop_count": self.editor.stop_count,
            "visible_stop_count": self.editor.visible_stop_count,
            "added_count": len(added), "removed_count": len(removed),
            "required_direction_stop_count": len(self.editor.required_transitions),
            "geometry_unchanged": self.editor.validate_geometry_unchanged(),
            "output_path": str(self.p("output_path")),
        }

    def publish_all(self):
        self.pub_csv_overlay.publish(self._csv_overlay_path())
        self.pub_network.publish(self._network_markers())
        self.pub_route_a.publish(self._path("A"))
        self.pub_route_b.publish(self._path("B"))
        self.pub_mode_markers.publish(self._mode_markers())
        self.pub_markers.publish(self._markers())
        self.pub_status.publish(String(data=json.dumps(
            self.status(), sort_keys=True, separators=(",", ":"))))
        self.pub_selected.publish(String(data=json.dumps(
            self.editor.selected() or {}, sort_keys=True, separators=(",", ":"))))
        self.pub_count.publish(Int32(data=self.editor.stop_count))


def main(args=None):
    rclpy.init(args=args)
    node = StopEditorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
