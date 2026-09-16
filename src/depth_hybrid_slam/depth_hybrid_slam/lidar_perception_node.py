"""Canonical steering-aware front/rear LiDAR perception; no motion command."""

from dataclasses import replace
import json
import math
import time

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .lidar_roi_core import (
    assess_curved_roi, assess_mode5_broad, cluster_points,
    clusters_in_centerline_corridor, DYNAMIC, DynamicClusterTracker, mode_gates,
    mode5_avoidance_requested, speed_bump_suppressed)
from .lidar_scan_core import optional_rear_hard_stop, ScanSafety
from .lidar_mission_core import (
    Mode9EmergencyLatch, Mode9Safety, SteeringSlowdownLatch)


def scan_points(message):
    output = []
    angle = float(message.angle_min)
    for raw in message.ranges:
        distance = float(raw)
        if (math.isfinite(distance) and message.range_min <= distance <=
                message.range_max):
            output.append((distance*math.cos(angle), distance*math.sin(angle)))
        angle += float(message.angle_increment)
    return tuple(output)


def _point(x, y, z=0.04):
    value = Point()
    value.x, value.y, value.z = float(x), float(y), float(z)
    return value


class LidarPerceptionNode(Node):
    def __init__(self):
        super().__init__("depth_lidar_perception")
        parameters = (
            ("scan_timeout_s", 0.5), ("publish_hz", 20.0),
            ("steering_timeout_s", 0.5), ("odom_timeout_s", 0.5),
            ("wheelbase_m", 0.73),
            ("emergency_distance_m", 0.50), ("clear_distance_m", 0.60),
            ("mode9_clear_distance_m", 1.5),
            ("mode9_clear_duration_s", 1.0),
            ("mode9_stop_distance_m", 1.0),
            ("mode9_slow_distance_m", 1.5),
            ("mode9_reaction_time_s", 0.35),
            ("mode9_deceleration_mps2", 1.5),
            ("mode9_braking_margin_m", 0.20),
            ("mode9_ttc_stop_s", 0.75),
            ("mode9_ttc_slow_s", 1.5),
            ("steering_slowdown_threshold_deg", 10.0),
            ("steering_slowdown_enter_s", 1.0),
            ("steering_slowdown_exit_s", 1.0),
            ("corridor_half_width_m", 0.30),
            ("minimum_cluster_points", 2), ("cluster_gap_m", 0.16),
            ("clear_confirm_scans", 3), ("parking_free_distance_m", 1.0),
            ("parking_use_front_lidar", True),
            ("dynamic_velocity_threshold_mps", 0.25),
            ("dynamic_confirmation_count", 3),
            ("dynamic_timeout_s", 0.75),
            ("dynamic_association_distance_m", 0.50),
            ("mode5_range_m", 1.5), ("mode5_fov_deg", 80.0),
            ("csv_path_timeout_s", 0.5),
            ("mode5_lateral_m", 1.0), ("vehicle_width_m", 0.80),
            ("obstacle_margin_m", 0.15), ("speed_bump_zones", [""]),
        )
        for name, default in parameters:
            self.declare_parameter(name, default)

        def p(name):
            return self.get_parameter(name).value

        self.rear_safety = ScanSafety(
            p("emergency_distance_m"), p("clear_distance_m"), 1.5,
            p("corridor_half_width_m"), p("minimum_cluster_points"),
            p("clear_confirm_scans"))
        self.tracker = DynamicClusterTracker(
            p("dynamic_velocity_threshold_mps"),
            p("dynamic_confirmation_count"), p("dynamic_timeout_s"),
            p("dynamic_association_distance_m"))
        self.mode = -1
        self.active_index = 0
        self.front = self.front_local = self.rear = ()
        self.front_frame = "front_laser"
        self.front_sensor_pose = None
        self.front_at = self.rear_at = None
        self.front_sequence = self.processed_sequence = 0
        self.clusters = self.local_clusters = ()
        self.odom = None
        self.odom_speed = 0.0
        self.odom_at = None
        self.actual_steering = 0.0
        self.actual_steering_at = None
        self.local_path = ()
        self.local_path_frame = "odom"
        self.csv_path = ()
        self.csv_path_at = None
        self.camera_road = self.camera_fresh = False
        self.object_evidence_valid, self.object_detected = False, True
        self.last_event_state = None
        self.front_hard_latched = False
        self.front_clear_count = 0
        self.mode9_emergency = Mode9EmergencyLatch(
            p("mode9_clear_distance_m"), p("mode9_clear_duration_s"))
        self.mode9_safety = Mode9Safety(
            p("mode9_stop_distance_m"), p("mode9_slow_distance_m"),
            p("mode9_reaction_time_s"), p("mode9_deceleration_mps2"),
            p("mode9_braking_margin_m"), p("mode9_ttc_stop_s"),
            p("mode9_ttc_slow_s"))
        self.steering_slowdown = SteeringSlowdownLatch(
            p("steering_slowdown_threshold_deg"),
            p("steering_slowdown_enter_s"),
            p("steering_slowdown_exit_s"))
        self.speed_bump_zones = self._zones(p("speed_bump_zones"))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, "/front/scan", self._front, 10)
        self.create_subscription(LaserScan, "/rear/scan", self._rear, 10)
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(Int32, "/depth_slam/route/active_index",
                                 lambda m: setattr(self, "active_index", int(m.data)), 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(Float32, "/mcu/steer_deg", self._actual_steering, 10)
        self.create_subscription(Path, "/depth_slam/lidar/local_path",
                                 self._local_path, 10)
        self.create_subscription(
            Path, "/depth_slam/csv_validation/local_path",
            self._csv_path, 10)
        self.create_subscription(String, "/depth_slam/camera/csv_validation",
                                 self._camera_validation, 10)
        self.create_subscription(String, "/camera/mission/diagnostics",
                                 self._camera_objects, 10)
        self.hard_pub = self.create_publisher(Bool, "/depth_slam/lidar/hard_emergency", 10)
        self.front_fresh_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/front_scan_fresh", 10)
        self.rear_hard_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/rear_hard_emergency", 10)
        self.rear_available_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/rear_scan_available", 10)
        self.slow_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/slowdown_required", 10)
        self.distance_slow_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/distance_slowdown_required", 10)
        self.steering_slow_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/steering_slowdown_required", 10)
        self.avoid_pub = self.create_publisher(
            Bool, "/depth_slam/lidar/avoidance_required", 10)
        self.slot_pub = self.create_publisher(String, "/depth_slam/lidar/parking_slots", 10)
        self.event_pub = self.create_publisher(String, "/depth_slam/lidar/safety_event", 10)
        self.diag_pub = self.create_publisher(String, "/depth_slam/lidar/perception", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/depth_slam/lidar/roi_markers", 10)
        hz = float(p("publish_hz"))
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self._tick)

    @staticmethod
    def _zones(values):
        output = []
        for raw in values:
            try:
                output.append(tuple(int(value) for value in str(raw).split(":")))
            except (TypeError, ValueError):
                continue
        return tuple(value for value in output if len(value) == 3)

    def _front(self, message):
        local_points = scan_points(message)
        transformed = self._base_points(local_points, message.header.frame_id)
        if transformed is None:
            self.front = self.front_local = ()
            self.front_sensor_pose, self.front_at = None, None
        else:
            self.front, self.front_sensor_pose = transformed
            self.front_local = local_points
            self.front_frame = str(message.header.frame_id).lstrip("/") or \
                "front_laser"
            self.front_at = time.monotonic()
        self.front_sequence += 1

    def _rear(self, message):
        self.rear, self.rear_at = scan_points(message), time.monotonic()

    def _base_points(self, points, frame_id):
        frame = str(frame_id).lstrip("/")
        if not frame or frame == "base_link":
            return points, (0.0, 0.0, 0.0)
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", frame, Time())
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(2.0*(rotation.w*rotation.z +
                              rotation.x*rotation.y),
                         1.0-2.0*(rotation.y**2+rotation.z**2))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        transformed = tuple((translation.x+cosine*x-sine*y,
                             translation.y+sine*x+cosine*y)
                            for x, y in points)
        return transformed, (translation.x, translation.y, yaw)

    def _mode(self, message):
        previous = self.mode
        try:
            self.mode = int(str(message.data).strip())
        except ValueError:
            self.mode = -1
        if previous == 9 and self.mode != 9:
            self.mode9_emergency.reset()

    def _odom(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(2.0*(orientation.w*orientation.z +
                              orientation.x*orientation.y),
                         1.0-2.0*(orientation.y**2+orientation.z**2))
        self.odom = (float(position.x), float(position.y), yaw)
        self.odom_speed = abs(float(message.twist.twist.linear.x))
        self.odom_at = time.monotonic()

    def _actual_steering(self, message):
        self.actual_steering, self.actual_steering_at = float(message.data), time.monotonic()

    def _local_path(self, message):
        self.local_path = tuple((p.pose.position.x, p.pose.position.y)
                                for p in message.poses)
        self.local_path_frame = str(message.header.frame_id).lstrip("/") or \
            "odom"

    def _csv_path(self, message):
        frame = str(message.header.frame_id).lstrip("/")
        if frame != "base_link":
            self.csv_path = ()
            self.csv_path_at = None
            return
        self.csv_path = tuple((p.pose.position.x, p.pose.position.y)
                              for p in message.poses)
        self.csv_path_at = time.monotonic()

    def _camera_validation(self, message):
        try:
            value = json.loads(message.data)
            self.camera_fresh = bool(value.get("fresh", False))
            self.camera_road = str(value.get("state", "")).startswith("VALID")
        except (TypeError, ValueError, json.JSONDecodeError):
            self.camera_fresh = self.camera_road = False

    def _camera_objects(self, message):
        try:
            value = json.loads(message.data)
            self.object_evidence_valid = bool(value.get("object_evidence_valid", False))
            self.object_detected = bool(value.get("object_detected", True))
        except (TypeError, ValueError, json.JSONDecodeError):
            self.object_evidence_valid, self.object_detected = False, True

    def _steering(self, now):
        timeout = float(self.get_parameter("steering_timeout_s").value)
        if self.actual_steering_at is not None and now-self.actual_steering_at <= timeout:
            return self.actual_steering, "/mcu/steer_deg"
        return 0.0, "STALE"

    def _markers(self, assessment, broad, clusters):
        values = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def line(marker_id, namespace, points, width, color, frame_id=None):
            marker = Marker()
            marker.header.frame_id = frame_id or self.front_frame
            marker.header.stamp = stamp
            marker.ns, marker.id, marker.type = namespace, marker_id, Marker.LINE_STRIP
            marker.action, marker.pose.orientation.w = Marker.ADD, 1.0
            marker.scale.x = width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.points = [_point(x, y) for x, y in points]
            values.markers.append(marker)

        line(0, "centerline", [(x, y) for _, x, y in assessment.centerline],
             0.025, (0.1, 0.8, 1.0, 1.0))
        for marker_id, first, last, color in (
                (1, 0.0, 0.5, (1.0, 0.0, 0.0, 0.28)),
                (2, 0.5, 1.0, (1.0, 0.7, 0.0, 0.24)),
                (3, 1.0, 1.5, (0.2, 0.5, 1.0, 0.20))):
            line(marker_id, "zones", [(x, y) for s, x, y in assessment.centerline
                                      if first <= s <= last], 0.60, color)
        half_width = float(self.get_parameter("corridor_half_width_m").value)
        left_edge, right_edge = [], []
        samples = assessment.centerline
        for index, (_, x, y) in enumerate(samples):
            before = samples[max(0, index-1)]
            after = samples[min(len(samples)-1, index+1)]
            dx, dy = after[1]-before[1], after[2]-before[2]
            norm = max(1.0e-9, math.hypot(dx, dy))
            nx, ny = -dy/norm, dx/norm
            left_edge.append((x+half_width*nx, y+half_width*ny))
            right_edge.append((x-half_width*nx, y-half_width*ny))
        line(7, "corridor_left_0p30m", left_edge, 0.018,
             (0.1, 0.9, 1.0, 0.95))
        line(8, "corridor_right_0p30m", right_edge, 0.018,
             (0.1, 0.9, 1.0, 0.95))
        for marker_id, boundary in (
                (9, 0.0), (10, 0.5), (11, 1.0), (13, 1.5)):
            index = min(range(len(samples)),
                        key=lambda item: abs(samples[item][0]-boundary))
            line(marker_id, f"boundary_{boundary:.1f}m",
                 (left_edge[index], right_edge[index]), 0.035,
                 (1.0, 1.0, 1.0, 0.95))
        forward = Marker()
        forward.header.frame_id, forward.header.stamp = self.front_frame, stamp
        forward.ns, forward.id, forward.type = (
            "vehicle_forward_plus_x", 12, Marker.ARROW)
        forward.action, forward.pose.orientation.w = Marker.ADD, 1.0
        forward.scale.x, forward.scale.y, forward.scale.z = 0.04, 0.10, 0.10
        forward.color.r, forward.color.g = 0.2, 1.0
        forward.color.b, forward.color.a = 0.2, 1.0
        forward.points = [_point(0.0, 0.0, 0.08), _point(0.45, 0.0, 0.08)]
        values.markers.append(forward)
        # IDs 4/5 were the obsolete broad-FOV outline and green base_link CSV
        # sweep.  Explicit DELETE actions also clear them from an RViz session
        # that was already running before this node restarted.
        for marker_id, namespace, frame_id in (
                (4, "mode5_broad", self.front_frame),
                (5, "csv_sweep", "base_link")):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = frame_id, stamp
            marker.ns, marker.id, marker.action = (
                namespace, marker_id, Marker.DELETE)
            values.markers.append(marker)
        line(6, "local_path", self.local_path, 0.04,
             (1.0, 1.0, 1.0, 1.0), self.local_path_frame)
        # Classification colors are disjoint. Anything intersecting the
        # active steering ROI overrides its motion color and is shown red.
        roi = clusters_in_centerline_corridor(
            clusters, assessment.centerline,
            half_width_m=self.get_parameter("corridor_half_width_m").value,
            length_m=self.get_parameter("mode5_range_m").value)
        roi_ids = {id(cluster) for cluster in roi}
        curb_track_ids = {
            cluster.track_id for cluster in broad.curbs
            if cluster.track_id >= 0}
        ordinary = tuple(
            cluster for cluster in clusters
            if id(cluster) not in roi_ids and
            cluster.track_id not in curb_track_ids)
        marker_groups = (
                (20, "static_obstacles_blue", tuple(
                    c for c in ordinary if c.motion == "STATIC"),
                 (0.05, 0.25, 1.0, 1.0), self.front_frame),
                (21, "dynamic_obstacles_green", tuple(
                    c for c in ordinary if c.motion == DYNAMIC),
                 (0.05, 1.0, 0.15, 1.0), self.front_frame),
                (23, "roi_obstacles_red", roi,
                 (1.0, 0.05, 0.05, 1.0), self.front_frame),
                (24, "unknown_obstacles_yellow", tuple(
                    c for c in ordinary if c.motion not in ("STATIC", DYNAMIC)),
                 (1.0, 0.75, 0.05, 1.0), self.front_frame),
                (22, "curbs", broad.curbs,
                 (0.2, 1.0, 0.9, 1.0), "base_link"))
        for namespace, marker_id in (("obstacles", 20), ("dynamic", 21)):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = self.front_frame, stamp
            marker.ns, marker.id, marker.action = (
                namespace, marker_id, Marker.DELETE)
            values.markers.append(marker)
        for marker_id, namespace, group, color, frame_id in marker_groups:
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = frame_id, stamp
            marker.ns, marker.id, marker.type = namespace, marker_id, Marker.POINTS
            marker.action, marker.pose.orientation.w = Marker.ADD, 1.0
            marker.scale.x = marker.scale.y = 0.07
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.points = [_point(*point) for cluster in group
                             for point in cluster.points]
            values.markers.append(marker)
        return values

    def _tick(self):
        now = time.monotonic()
        timeout = float(self.get_parameter("scan_timeout_s").value)
        front_fresh = self.front_at is not None and now-self.front_at <= timeout
        rear_fresh = self.rear_at is not None and now-self.rear_at <= timeout
        odom_fresh = (self.odom_at is not None and
                      now-self.odom_at <= float(self.get_parameter("odom_timeout_s").value))
        new_scan = self.front_sequence != self.processed_sequence
        if new_scan:
            raw_clusters = cluster_points(
                self.front, self.get_parameter("cluster_gap_m").value,
                self.get_parameter("minimum_cluster_points").value)
            self.clusters = self.tracker.update(
                raw_clusters, now, self.odom if odom_fresh else None)
            local_clusters = cluster_points(
                self.front_local, self.get_parameter("cluster_gap_m").value,
                self.get_parameter("minimum_cluster_points").value)
            self.local_clusters = tuple(
                replace(local, track_id=tracked.track_id,
                        motion=tracked.motion, speed_mps=tracked.speed_mps)
                for local, tracked in zip(local_clusters, self.clusters))
            self.processed_sequence = self.front_sequence
        steering, steering_source = self._steering(now)
        front_active, rear_active = mode_gates(self.mode)
        filtered, local_filtered = [], []
        for cluster, local in zip(self.clusters, self.local_clusters):
            suppressed = speed_bump_suppressed(
                self.mode, self.active_index, self.speed_bump_zones,
                camera_fresh=self.camera_fresh,
                drivable_road=self.camera_road,
                object_evidence_valid=self.object_evidence_valid,
                object_detected=self.object_detected,
                motion=cluster.motion)
            if not suppressed:
                filtered.append(cluster)
                local_filtered.append(local)
        filtered, local_filtered = tuple(filtered), tuple(local_filtered)
        assessment = assess_curved_roi(
            local_filtered, steering, front_active=front_active,
            fresh=front_fresh,
            half_width_m=self.get_parameter("corridor_half_width_m").value,
            wheelbase_m=self.get_parameter("wheelbase_m").value,
            length_m=1.5)
        mode9_safety = self.mode9_safety.evaluate(
            assessment.nearest_m, self.odom_speed,
            fresh=front_fresh and odom_fresh)
        distance_slowdown_evidence = (
            mode9_safety.slowdown if self.mode == 9 else assessment.slowdown)
        steering_slowdown = self.steering_slowdown.update(
            steering if steering_source != "STALE" else None,
            now)
        slowdown = distance_slowdown_evidence or steering_slowdown
        if self.mode == 9:
            hard_stop = self.mode9_emergency.update(
                mode9_safety.stop, assessment.nearest_m,
                front_fresh, new_scan, now)
            # A stale scan is always fail-safe even before an obstacle has
            # established the persistent emergency latch.
            hard_stop = hard_stop or not front_fresh
            self.front_hard_latched, self.front_clear_count = False, 0
        elif not front_active:
            self.front_hard_latched, self.front_clear_count = False, 0
        elif assessment.hard_stop:
            self.front_hard_latched, self.front_clear_count = True, 0
        elif self.front_hard_latched:
            clear = (front_fresh and
                     (not math.isfinite(assessment.nearest_m) or
                      assessment.nearest_m >= float(
                          self.get_parameter("clear_distance_m").value)))
            if new_scan:
                self.front_clear_count = self.front_clear_count+1 if clear else 0
            if self.front_clear_count >= int(self.get_parameter(
                    "clear_confirm_scans").value):
                self.front_hard_latched, self.front_clear_count = False, 0
        if self.mode != 9:
            hard_stop = self.front_hard_latched
        mode5_local = clusters_in_centerline_corridor(
            local_filtered, assessment.centerline,
            half_width_m=self.get_parameter("corridor_half_width_m").value,
            length_m=self.get_parameter("mode5_range_m").value)
        mode5_ids = {id(cluster) for cluster in mode5_local}
        csv_path_fresh = (
            self.csv_path_at is not None and
            now-self.csv_path_at <= float(self.get_parameter(
                "csv_path_timeout_s").value))
        broad = assess_mode5_broad(
            filtered, self.csv_path if csv_path_fresh else (),
            range_m=self.get_parameter("mode5_range_m").value,
            fov_deg=self.get_parameter("mode5_fov_deg").value,
            lateral_m=self.get_parameter("mode5_lateral_m").value,
            clearance_m=(self.get_parameter("vehicle_width_m").value/2.0 +
                         self.get_parameter("obstacle_margin_m").value),
            sensor_pose=(self.front_sensor_pose or (0.0, 0.0, 0.0)))
        mode5_obstacles = broad.collision_obstacles
        collision_ids = {id(cluster) for cluster in mode5_obstacles}
        collision_local = tuple(
            local for base, local in zip(filtered, local_filtered)
            if id(base) in collision_ids)
        nearest_collision_lidar_m = min(
            (math.hypot(x, y) for cluster in collision_local
             for x, y in cluster.points), default=math.inf)
        # The 1.5 m ROI is a sensing/visualization boundary. Mode 5 stops only
        # when the fresh CSV local path's swept vehicle footprint intersects
        # an obstacle inside that front_laser-origin boundary.
        avoidance = mode5_avoidance_requested(
            self.mode, front_fresh, csv_path_fresh, broad.path_blocked)
        # The rear LiDAR is optional production equipment. A missing/stale
        # rear stream must not become a synthetic obstacle and immobilize
        # Modes 7/10. Fresh rear data keeps normal obstacle protection.
        rear_result = (self.rear_safety.assess(self.rear, True)
                       if rear_active and rear_fresh else None)
        rear_hard = optional_rear_hard_stop(
            rear_active, rear_fresh,
            bool(rear_result and rear_result.hard_obstacle))
        parking_use_front = bool(self.get_parameter(
            "parking_use_front_lidar").value)
        parking_scan = self.front_local if parking_use_front else self.rear
        parking_fresh = front_fresh if parking_use_front else rear_fresh
        left_min = min((math.hypot(x, y) for x, y in parking_scan if y >= 0.0),
                       default=math.inf)
        right_min = min((math.hypot(x, y) for x, y in parking_scan if y < 0.0),
                        default=math.inf)
        parking_free = float(self.get_parameter("parking_free_distance_m").value)
        parking_available = bool(
            parking_fresh and (parking_use_front or rear_active))
        slots = {"a_free": parking_available and left_min > parking_free,
                 "b_free": parking_available and right_min > parking_free,
                 "fresh": parking_available,
                 "source": "FRONT" if parking_use_front else "REAR"}
        self.hard_pub.publish(Bool(data=hard_stop))
        self.front_fresh_pub.publish(Bool(data=front_fresh))
        self.rear_hard_pub.publish(Bool(data=rear_hard))
        self.rear_available_pub.publish(Bool(
            data=bool(rear_active and rear_fresh)))
        self.slow_pub.publish(Bool(data=slowdown))
        self.distance_slow_pub.publish(Bool(
            data=distance_slowdown_evidence))
        self.steering_slow_pub.publish(Bool(data=steering_slowdown))
        self.avoid_pub.publish(Bool(data=avoidance))
        self.slot_pub.publish(String(data=json.dumps(slots, separators=(",", ":"))))
        event_state = (self.mode, hard_stop, slowdown,
                       avoidance, slots["a_free"], slots["b_free"])
        if event_state != self.last_event_state:
            event = {"mode": self.mode, "hard_stop": hard_stop,
                     "emergency_stop_latched": (
                         self.mode == 9 and self.mode9_emergency.latched),
                     "slowdown": slowdown,
                     "distance_m": None if not math.isfinite(assessment.nearest_m)
                     else assessment.nearest_m,
                     "path_blocked": avoidance,
                     "avoidance_required": avoidance,
                     "slot_a": slots["a_free"], "slot_b": slots["b_free"]}
            payload = json.dumps(event, separators=(",", ":"))
            self.event_pub.publish(String(data=payload))
            self.get_logger().info("[LIDAR ROI] "+payload)
            self.last_event_state = event_state
        obstacle_y = (sum(c.centroid[1] for c in mode5_obstacles) /
                      len(mode5_obstacles) if mode5_obstacles else 0.0)
        diagnostic = {
            "mode": self.mode, "steering_deg": steering,
            "steering_source": steering_source,
            "roi_frame": self.front_frame,
            "roi_origin_m": [0.0, 0.0],
            "curve_radius_m": None if not math.isfinite(assessment.radius_m)
            else assessment.radius_m,
            "zone1_count": assessment.zone1_count,
            "zone2_count": assessment.zone2_count,
            "zone3_dynamic_count": assessment.zone3_dynamic_count,
            "front_active": front_active, "rear_active": rear_active,
            "fresh": front_fresh, "fixed_frame_valid": odom_fresh,
            "nearest_m": None if not math.isfinite(assessment.nearest_m)
            else assessment.nearest_m,
            "hard_obstacle": hard_stop,
            "emergency_stop_latched": (
                self.mode == 9 and self.mode9_emergency.latched),
            "emergency_clear_elapsed_s": (
                0.0 if self.mode != 9 or
                self.mode9_emergency.clear_since is None else
                max(0.0, now-self.mode9_emergency.clear_since)),
            "slowdown_required": slowdown,
            "distance_slowdown_required": distance_slowdown_evidence,
            "steering_slowdown_required": steering_slowdown,
            "slowdown_policy": "DISTANCE_1M_OR_MEASURED_STEERING_HYSTERESIS",
            "mode9_safety_state": mode9_safety.state,
            "mode9_stopping_distance_m": (
                None if not math.isfinite(mode9_safety.stopping_distance_m)
                else mode9_safety.stopping_distance_m),
            "mode9_ttc_s": (None if not math.isfinite(mode9_safety.ttc_s)
                            else mode9_safety.ttc_s),
            "odom_speed_mps": self.odom_speed,
            "distance_slowdown_evidence": distance_slowdown_evidence,
            "steering_slowdown_threshold_deg": self.steering_slowdown.threshold_deg,
            "steering_above_elapsed_s": (
                0.0 if self.steering_slowdown.above_since is None else
                max(0.0, now-self.steering_slowdown.above_since)),
            "steering_below_elapsed_s": (
                0.0 if self.steering_slowdown.below_since is None else
                max(0.0, now-self.steering_slowdown.below_since)),
            "rear_hard_obstacle": rear_hard,
            "rear_scan_available": bool(rear_active and rear_fresh),
            "rear_lidar_policy": "OPTIONAL_VERIFY_IF_PRESENT",
            "path_blocked": avoidance,
            "avoidance_required": avoidance,
            "csv_path_fresh": csv_path_fresh,
            "csv_collision_count": len(mode5_obstacles),
            "avoidance_nearest_lidar_m": (
                None if not math.isfinite(nearest_collision_lidar_m) else
                nearest_collision_lidar_m),
            "obstacle_y_m": obstacle_y,
            "motion_classification": {
                "static": sum(c.motion == "STATIC" for c in local_filtered),
                "dynamic": sum(c.motion == DYNAMIC for c in local_filtered),
                "unknown": sum(c.motion not in ("STATIC", DYNAMIC)
                               for c in local_filtered),
                "roi_red": len(mode5_local),
            },
            "obstacle_tracks": [
                {"track_id": c.track_id, "motion": c.motion,
                 "speed_mps": round(c.speed_mps, 3),
                 "centroid": [round(c.centroid[0], 3),
                              round(c.centroid[1], 3)],
                 "in_roi": id(c) in mode5_ids}
                for c in local_filtered],
            "obstacles": [[round(p[0], 3), round(p[1], 3)]
                          for c in local_filtered for p in c.points][:128],
            "curbs": [[round(p[0], 3), round(p[1], 3)]
                      for c in broad.curbs for p in c.points][:64],
            "speed_bump_suppression_configured": bool(self.speed_bump_zones),
        }
        self.diag_pub.publish(String(data=json.dumps(diagnostic, separators=(",", ":"))))
        self.marker_pub.publish(self._markers(
            assessment, broad, local_filtered))


def main(args=None):
    rclpy.init(args=args)
    node = LidarPerceptionNode()
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
