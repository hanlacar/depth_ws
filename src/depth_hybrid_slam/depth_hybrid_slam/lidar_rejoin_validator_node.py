"""ROS wiring for bounded same-segment CSV rejoin validation."""

import json
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String

from .csv_only_branching import load_csv_only_route_case
from .lidar_mission_core import bounded_rejoin, route_rejoin_candidates
from .route_io import load_segmented_route


class LidarRejoinValidatorNode(Node):
    def __init__(self):
        super().__init__("depth_lidar_rejoin_validator")
        parameters = (
            ("route_path", ""), ("route_metadata_path", ""),
            ("publish_hz", 20.0), ("pose_timeout_s", 0.5),
            ("forward_window", 120), ("max_distance_m", 0.25),
            ("max_heading_deg", 10.0),
        )
        for name, default in parameters:
            self.declare_parameter(name, default)
        self.route_path = str(self.get_parameter("route_path").value)
        self.metadata_path = str(
            self.get_parameter("route_metadata_path").value)
        if not self.route_path:
            raise ValueError("route_path is required")
        self.routes = {}
        self.branch = "A"
        self.case = ""
        self.index = None
        self.planned_rejoin_index = -1
        self.segment = ""
        self.pose = None
        self.pose_at = None
        self.road_state = "UNKNOWN"
        self.create_subscription(
            PoseWithCovarianceStamped, "/depth_slam/localization/pose",
            self._pose, 10)
        self.create_subscription(
            Int32, "/depth_slam/route/active_index",
            lambda m: setattr(self, "index", int(m.data)), 10)
        self.create_subscription(
            Int32, "/depth_slam/lidar/planned_rejoin_index",
            lambda m: setattr(
                self, "planned_rejoin_index", int(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/active_segment",
            lambda m: setattr(self, "segment", str(m.data)), 10)
        self.create_subscription(
            String, "/depth_slam/route/selected_branch",
            self._branch, 10)
        self.create_subscription(
            String, "/depth_slam/route/selected_case", self._case, 10)
        self.create_subscription(
            String, "/depth_slam/camera/csv_validation", self._camera, 10)
        self.valid_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/csv_rejoin_valid", 10)
        self.diag_pub = self.create_publisher(
            String, "/depth_slam/lidar/csv_rejoin", 10)
        self.index_pub = self.create_publisher(
            Int32, "/depth_slam/lidar/csv_rejoin_index", 10)
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    def _pose(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0*(orientation.w*orientation.z+orientation.x*orientation.y),
            1.0-2.0*(orientation.y**2+orientation.z**2))
        self.pose = (float(position.x), float(position.y), yaw)
        self.pose_at = time.monotonic()

    def _branch(self, message):
        value = str(message.data).strip().upper()
        if value in ("A", "B"):
            self.branch = value

    def _case(self, message):
        value = str(message.data).strip().upper()
        if len(value) == 4 and all(item in "AB" for item in value):
            self.case = value

    def _camera(self, message):
        try:
            self.road_state = str(json.loads(message.data).get(
                "state", "UNKNOWN"))
        except (TypeError, json.JSONDecodeError):
            self.road_state = "UNKNOWN"

    def _route(self):
        key = self.case or self.branch
        if key not in self.routes:
            if len(key) == 4:
                route = load_csv_only_route_case(
                    self.route_path, self.metadata_path, key)
            else:
                route = load_segmented_route(
                    self.route_path, self.metadata_path, branch=key).points
            self.routes[key] = tuple(route)
        return self.routes[key]

    def _tick(self):
        now = time.monotonic()
        fresh = (self.pose is not None and self.pose_at is not None and
                 now-self.pose_at <= float(
                     self.get_parameter("pose_timeout_s").value))
        selected = None
        validation_basis = "ACTIVE_FORWARD_WINDOW"
        if fresh:
            route = self._route()
            if 0 <= self.planned_rejoin_index < len(route):
                # Mode 5 fixes an exact same-CSV endpoint when its path is
                # generated. The suspended CSV follower's live cursor may
                # advance past this endpoint while the offset path is active;
                # validate the planned endpoint rather than excluding it as
                # historical progress.
                target = route[self.planned_rejoin_index]
                road_valid = self.road_state != "OUTSIDE_ROAD"
                candidates = route_rejoin_candidates(
                    route, target.segment_id,
                    self.planned_rejoin_index, self.pose, 0,
                    road_valid=road_valid)
                selected = bounded_rejoin(
                    candidates, target.segment_id,
                    self.planned_rejoin_index, int(target.direction), 0,
                    float(self.get_parameter("max_distance_m").value),
                    float(self.get_parameter("max_heading_deg").value))
                validation_basis = "PLANNED_EXACT_ENDPOINT"
            elif (self.index is not None and self.segment and
                  0 <= self.index < len(route)):
                current = route[self.index]
                direction = int(current.direction)
                road_valid = self.road_state != "OUTSIDE_ROAD"
                window = int(self.get_parameter("forward_window").value)
                candidates = route_rejoin_candidates(
                    route, self.segment, self.index, self.pose, window,
                    road_valid=road_valid)
                selected = bounded_rejoin(
                    candidates, self.segment, self.index, direction, window,
                    float(self.get_parameter("max_distance_m").value),
                    float(self.get_parameter("max_heading_deg").value))
        valid = selected is not None
        self.valid_pub.publish(Bool(data=valid))
        if selected is not None:
            # Keep the last positively validated handoff index latched at the
            # subscriber. Publishing -1 on the falling edge would race the
            # maneuver owner's final candidate_valid=False notification.
            self.index_pub.publish(Int32(data=int(selected.index)))
        self.diag_pub.publish(String(data=json.dumps({
            "valid": valid, "fresh": fresh, "segment": self.segment,
            "current_index": self.index,
            "planned_rejoin_index": self.planned_rejoin_index,
            "validation_basis": validation_basis,
            "selected_index": None if selected is None else selected.index,
            "required_steering_deg": (
                None if selected is None else selected.required_steering_deg),
            "road_state": self.road_state,
        }, separators=(",", ":"))))


def main(args=None):
    rclpy.init(args=args)
    node = LidarRejoinValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
