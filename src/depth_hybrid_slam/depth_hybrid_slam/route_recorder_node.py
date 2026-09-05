"""Explicitly controlled map-frame vehicle reference route recorder."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMessage
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import Info, MapGraph
from std_msgs.msg import Bool, Float32, String, UInt32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from .geometry import wrap_angle
from .ros_helpers import safe_shutdown, stamp_seconds, yaw_from_quaternion
from .route_recorder_core import RouteRecorder


def _atomic_yaml(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, sort_keys=True, allow_unicode=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RouteRecorderNode(Node):
    def __init__(self):
        super().__init__("depth_route_recorder")
        defaults = (
            ("route_path", ""), ("route_metadata_path", ""),
            ("map_path", ""), ("session_id", ""),
            ("high_density", False),
            ("route_min_distance_m", 0.05),
            ("route_min_angle_rad", 0.03),
            ("pose_timeout_s", 0.5),
            ("graph_snapshot_path", ""),
        )
        for name, default in defaults:
            self.declare_parameter(name, default)
        raw_route = str(self.get_parameter("route_path").value)
        self.route_path = (Path(raw_route).expanduser().resolve()
                           if raw_route else None)
        metadata = str(self.get_parameter("route_metadata_path").value)
        self.metadata_path = (Path(metadata).expanduser().resolve() if metadata else
                              (self.route_path.parent / "route_metadata.yaml"
                               if self.route_path else None))
        raw_map = str(self.get_parameter("map_path").value)
        self.map_path = Path(raw_map).expanduser().resolve() if raw_map else None
        self.session_id = (str(self.get_parameter("session_id").value) or
                           (self.route_path.parent.name if self.route_path else ""))
        self.recorder = None
        self.active = False
        self.mapping_ready = False
        self.mapping_state = "INITIALIZING"
        self.tracking = False
        self.reset_count = None
        self.reset_start = None
        self.localization_state = "INITIALIZING"
        self.localization_confidence = 0.0
        self.nearest_node_id = -1
        self.graph_poses = {}
        graph_path = str(self.get_parameter("graph_snapshot_path").value)
        self.graph_path = (Path(graph_path).expanduser().resolve() if graph_path else
                           (self.route_path.parent/"rtabmap_graph.json"
                            if self.route_path else None))
        self.recording_allowed = False
        self.recording_hold_reason = "WAITING_FOR_QUALITY_MONITOR"
        self.last_pose = None
        self.last_pose_receipt = None
        self.invalid_session = False
        self.tracking_true = 0
        self.tracking_total = 0
        self.path_message = PathMessage()
        self.path_message.header.frame_id = "map"
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.pub_status = self.create_publisher(String, "/depth_slam/route/status", 10)
        self.pub_path = self.create_publisher(PathMessage, "/depth_slam/route/path", 10)
        self.pub_sample_valid = self.create_publisher(
            Bool, "/depth_slam/route/sample_valid", 10)
        self.create_subscription(PoseWithCovarianceStamped,
                                 "/depth_slam/localization/pose", self.on_pose, 10)
        self.create_subscription(Bool, "/depth_slam/mapping/ready",
                                 lambda message: setattr(self, "mapping_ready", message.data), 10)
        self.create_subscription(String, "/depth_slam/mapping/state",
                                 lambda message: setattr(self, "mapping_state", message.data), 10)
        self.create_subscription(Bool, "/depth_slam/cuvslam/tracking_valid",
                                 lambda message: setattr(self, "tracking", message.data), 10)
        self.create_subscription(UInt32, "/depth_slam/cuvslam/reset_count", self.on_reset, 10)
        self.create_subscription(String, "/depth_slam/localization/state",
                                 lambda message: setattr(self, "localization_state", message.data), 10)
        self.create_subscription(Float32, "/depth_slam/localization/confidence",
                                 lambda message: setattr(self, "localization_confidence", message.data), 10)
        self.create_subscription(Info, "/rtabmap/info",
                                 lambda message: setattr(self, "nearest_node_id", message.ref_id), 10)
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self.on_graph, 10)
        self.create_subscription(Bool, "/depth_slam/mapping/recording_allowed",
                                 lambda message: setattr(
                                     self, "recording_allowed", bool(message.data)), 10)
        self.create_subscription(String, "/depth_slam/mapping/recording_hold_reason",
                                 lambda message: setattr(
                                     self, "recording_hold_reason", str(message.data)), 10)
        session_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/depth_slam/mapping/session_id", self.on_session_id, session_qos)
        self.create_subscription(
            String, "/depth_slam/mapping/map_path", self.on_map_path, session_qos)
        self.create_subscription(
            String, "/depth_slam/mapping/route_path", self.on_route_path, session_qos)
        self.create_service(Trigger, "/depth_slam/route/start", self.start)
        self.create_service(Trigger, "/depth_slam/route/stop", self.stop)
        self.create_service(Trigger, "/depth_slam/route/status", self.status)
        self.create_timer(0.1, self.tick)

    def on_reset(self, message):
        self.reset_count = int(message.data)
        if self.active and self.reset_start is not None and self.reset_count > self.reset_start:
            self.invalid_session = True

    def on_graph(self, message):
        self.graph_poses = {
            int(node_id): {
                "x": float(pose.position.x), "y": float(pose.position.y),
                "z": float(pose.position.z),
                "yaw": float(yaw_from_quaternion(pose.orientation)),
            }
            for node_id, pose in zip(message.poses_id, message.poses)
        }

    def on_session_id(self, message):
        if not self.session_id:
            self.session_id = str(message.data)

    def on_map_path(self, message):
        if self.map_path is None:
            self.map_path = Path(message.data).expanduser().resolve()

    def on_route_path(self, message):
        if self.route_path is None:
            self.route_path = Path(message.data).expanduser().resolve()
            if self.metadata_path is None:
                self.metadata_path = self.route_path.parent / "route_metadata.yaml"
            if self.graph_path is None:
                self.graph_path = self.route_path.parent / "rtabmap_graph.json"

    def pose_is_fresh(self):
        return (self.last_pose is not None and self.last_pose_receipt is not None and
                time.monotonic() - self.last_pose_receipt <=
                float(self.get_parameter("pose_timeout_s").value))

    def start(self, _request, response):
        problems = []
        if self.active:
            problems.append("ALREADY_RECORDING")
        if not self.mapping_ready:
            problems.append("MAPPING_NOT_READY:" + self.mapping_state)
        if not self.tracking:
            problems.append("TRACKING_INVALID")
        if not self.recording_allowed:
            problems.append("QUALITY_RECORDING_HOLD:" + self.recording_hold_reason)
        if self.reset_count is None:
            problems.append("RESET_COUNT_MISSING")
        if not self.pose_is_fresh() or self.last_pose.header.frame_id != "map":
            problems.append("VALID_MAP_POSE_MISSING")
        if self.route_path is None or self.map_path is None or not self.session_id:
            problems.append("SESSION_PATHS_MISSING")
        elif self.route_path.exists() or self.route_path.with_suffix(
                self.route_path.suffix + ".partial").exists():
            problems.append("ROUTE_PATH_EXISTS")
        if self.metadata_path is not None and self.metadata_path.exists():
            problems.append("ROUTE_METADATA_EXISTS")
        if (self.graph_path is not None and
                (self.graph_path.exists() or self.graph_path.with_suffix(
                    self.graph_path.suffix+".tmp").exists())):
            problems.append("GRAPH_SNAPSHOT_EXISTS")
        if problems:
            response.success = False
            response.message = ",".join(problems)
            return response
        self.recorder = RouteRecorder(
            self.route_path,
            min_distance_m=float(self.get_parameter("route_min_distance_m").value),
            min_angle_rad=float(self.get_parameter("route_min_angle_rad").value))
        self.reset_start = self.reset_count
        self.active = True
        self.invalid_session = False
        response.success = True
        response.message = f"recording map-frame route to {self.route_path}"
        return response

    def stop(self, _request, response):
        if not self.active:
            response.success = False
            response.message = "NOT_RECORDING"
            return response
        summary = self._finalize(True)
        response.success = True
        response.message = (f"points={summary['point_count']} "
                            f"distance_m={summary['total_distance_m']:.3f} "
                            f"sha256={summary['route_csv_sha256']}")
        return response

    def status(self, _request, response):
        response.success = not self.invalid_session
        response.message = (
            f"active={self.active} ready={self.mapping_ready} tracking={self.tracking} "
            f"reset={self.reset_count} state={self.mapping_state} "
            f"session_id={self.session_id or 'WAITING'} "
            f"route={self.route_path or 'WAITING'} "
            f"points={self.recorder.index if self.recorder else 0} "
            f"invalid_session={self.invalid_session}")
        return response

    def on_pose(self, message):
        self.last_pose = message
        self.last_pose_receipt = time.monotonic()
        if self.active:
            self.record_pose(message, "localization_pose")

    def record_pose(self, message, pose_source):
        position = message.pose.pose.position
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        self.tracking_total += 1
        self.tracking_true += int(self.tracking)
        valid = (message.header.frame_id == "map" and self.tracking and
                 not self.invalid_session and self.mapping_ready and
                 self.recording_allowed)
        relative_x = relative_y = relative_yaw = ""
        anchor = self.graph_poses.get(int(self.nearest_node_id))
        if anchor is not None:
            dx, dy = position.x-anchor["x"], position.y-anchor["y"]
            cosine, sine = math.cos(anchor["yaw"]), math.sin(anchor["yaw"])
            relative_x = cosine*dx+sine*dy
            relative_y = -sine*dx+cosine*dy
            relative_yaw = wrap_angle(yaw-anchor["yaw"])
        else:
            valid = False
        try:
            appended = self.recorder.append({
                "timestamp": stamp_seconds(message.header.stamp),
                "map_x_m": position.x, "map_y_m": position.y,
                "map_z_m": position.z, "yaw_rad": yaw,
                "direction": "", "localization_state": self.localization_state,
                "localization_confidence": self.localization_confidence,
                "tracking_valid": self.tracking,
                "reset_count": self.reset_count,
                "nearest_rtabmap_node_id": self.nearest_node_id,
                "node_relative_x_m": relative_x,
                "node_relative_y_m": relative_y,
                "node_relative_yaw_rad": relative_yaw,
                "pose_source": pose_source, "valid": valid,
                "frame_id": message.header.frame_id,
            })
        except ValueError as error:
            self.invalid_session = True
            self.pub_status.publish(String(data="INVALID_SESSION:" + str(error)))
            return
        self.pub_sample_valid.publish(Bool(data=valid))
        if appended:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose = message.pose.pose
            self.path_message.poses.append(pose)

    def tick(self):
        if self.active and not self.pose_is_fresh():
            try:
                transform = self.buffer.lookup_transform(
                    "map", "base_link", rclpy.time.Time())
            except TransformException:
                transform = None
            if transform is not None:
                message = PoseWithCovarianceStamped()
                message.header = transform.header
                message.header.frame_id = "map"
                message.pose.pose.position.x = transform.transform.translation.x
                message.pose.pose.position.y = transform.transform.translation.y
                message.pose.pose.position.z = transform.transform.translation.z
                message.pose.pose.orientation = transform.transform.rotation
                self.last_pose = message
                self.last_pose_receipt = time.monotonic()
                self.record_pose(message, "tf_map_to_base_link")
        self.path_message.header.stamp = self.get_clock().now().to_msg()
        self.pub_path.publish(self.path_message)
        self.pub_status.publish(String(data=(
            "RECORDING" if self.active else
            ("INVALID_SESSION" if self.invalid_session else "IDLE"))))

    def _finalize(self, normal_stop):
        summary = self.recorder.finalize()
        if self.graph_path is None or not self.graph_poses:
            self.invalid_session = True
        else:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.graph_path.with_suffix(self.graph_path.suffix+".tmp")
            with open(temporary, "x", encoding="utf-8") as stream:
                json.dump({"session_id": self.session_id,
                           "poses": self.graph_poses}, stream,
                          indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.graph_path)
        self.active = False
        reset_increase = ((self.reset_count or 0) - (self.reset_start or 0))
        tracking_percent = (100.0 * self.tracking_true / self.tracking_total
                            if self.tracking_total else 0.0)
        metadata = {
            "session_id": self.session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "finalized": False,
            "normal_route_stop": normal_stop,
            "map_db_path": str(self.map_path),
            "map_db_sha256": "PENDING_MAPPING_SHUTDOWN",
            "route_path": str(self.route_path),
            "raw_route_path": str(self.route_path),
            "graph_snapshot_path": str(self.graph_path) if self.graph_path else "",
            "route_csv_sha256": summary["route_csv_sha256"],
            "frame_id": "map",
            "pose_source": "localization_pose_then_tf_fallback",
            "d456_serial": "338122302896",
            "sensor_profiles": "RGB/Depth/IR 640x480@60; IMU 400Hz",
            "rtabmap_settings": ({
                "Rtabmap/DetectionRate": "10",
                "RGBD/LinearUpdate": "0.03",
                "RGBD/AngularUpdate": "0.02",
                "Vis/MinInliers": "15",
            } if bool(self.get_parameter("high_density").value) else {
                "Rtabmap/DetectionRate": "5",
                "RGBD/LinearUpdate": "0.05",
                "RGBD/AngularUpdate": "0.03",
                "Vis/MinInliers": "15",
            }),
            "route_min_distance_m": float(
                self.get_parameter("route_min_distance_m").value),
            "route_min_angle_rad": float(
                self.get_parameter("route_min_angle_rad").value),
            "point_count": summary["point_count"],
            "total_distance_m": summary["total_distance_m"],
            "tracking_true_percent": tracking_percent,
            "reset_increase": reset_increase,
            "quality_verdict": ("INVALID_SESSION" if self.invalid_session or reset_increase
                                else "PENDING_MAPPING_FINALIZATION"),
        }
        _atomic_yaml(self.metadata_path, metadata)
        return summary

    def destroy_node(self):
        if self.active:
            self._finalize(False)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RouteRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
