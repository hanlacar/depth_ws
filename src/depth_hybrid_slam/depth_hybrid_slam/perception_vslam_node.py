"""Cross-check existing 2D perception against timestamped VSLAM transforms."""

from collections import deque
import json
import math

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .perception_vslam_core import (
    json_stamp_seconds, robust_bbox_projection, select_traffic_detection,
    traffic_validation_reasons, transform_point, vslam_ready,
)
from .ros_helpers import safe_shutdown


def image_array(message):
    encoding = message.encoding.upper()
    if encoding in ("16UC1", "MONO16"):
        dtype = np.dtype("<u2")
    elif encoding == "32FC1":
        dtype = np.dtype("<f4")
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    row_items = message.step//dtype.itemsize
    raw = np.frombuffer(message.data, dtype=dtype)
    return raw[:message.height*row_items].reshape(
        message.height, row_items)[:, :message.width].copy()


def stamp_seconds(stamp):
    return float(stamp.sec)+float(stamp.nanosec)*1.0e-9


class PerceptionVslamNode(Node):
    def __init__(self):
        super().__init__("perception_vslam_validator")
        defaults = {
            "color_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "detections_topic": "/perception/detections_json",
            "fusion_diagnostics_topic": "/camera/traffic_light_fused/diagnostics",
            "mission_diagnostics_topic": "/camera/mission/diagnostics",
            "map_frame": "map", "base_frame": "base_link",
            "front_axle_x_m": 0.365,
            "maximum_stamp_delta_sec": 0.08,
            "maximum_input_age_sec": 0.50,
            "tf_timeout_sec": 0.05,
            "minimum_traffic_confidence": 0.60,
            "minimum_depth_pixels": 12,
            "minimum_depth_m": 0.20,
            "maximum_depth_m": 20.0,
            "maximum_depth_mad_m": 0.75,
            "marker_lifetime_sec": 0.60,
            "vehicle_path_min_distance_m": 0.03,
            "vehicle_path_max_poses": 4000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=15.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.depth_cache = deque(maxlen=12)
        self.camera_info = None
        self.detection = None
        self.tracking_valid = False
        self.mapping_state = "MISSING"
        self.localization_state = "MISSING"
        self.traffic = {"valid": False, "reason": "NO_INPUT"}
        self.stop_line = {"valid": False, "reason": "NO_INPUT"}
        self.last_final_traffic_stamp = None
        self.last_final_stop_stamp = None
        self.vehicle_path = Path()
        self.vehicle_path.header.frame_id = self.p("map_frame")

        self.marker_pub = self.create_publisher(
            MarkerArray, "/depth_slam/perception/markers", 10)
        self.path_pub = self.create_publisher(
            Path, "/depth_slam/perception/vehicle_path", 10)
        self.status_pub = self.create_publisher(
            String, "/depth_slam/perception/cross_validation_status", 10)
        self.traffic_valid_pub = self.create_publisher(
            Bool, "/depth_slam/perception/traffic_light/validated", 10)
        self.traffic_result_pub = self.create_publisher(
            String, "/depth_slam/perception/traffic_light/result", 10)
        self.stop_valid_pub = self.create_publisher(
            Bool, "/depth_slam/perception/stop_line/validated", 10)
        self.stop_result_pub = self.create_publisher(
            String, "/depth_slam/perception/stop_line/result", 10)

        self.create_subscription(Image, self.p("depth_topic"), self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.p("camera_info_topic"),
                                 self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(String, self.p("detections_topic"),
                                 self.on_detections, 10)
        self.create_subscription(String, self.p("fusion_diagnostics_topic"),
                                 self.on_fusion, 10)
        self.create_subscription(String, self.p("mission_diagnostics_topic"),
                                 self.on_mission, 10)
        self.create_subscription(
            Bool, "/depth_slam/cuvslam/tracking_valid",
            lambda message: setattr(self, "tracking_valid", bool(message.data)), 10)
        self.create_subscription(
            String, "/depth_slam/mapping/state",
            lambda message: setattr(self, "mapping_state", message.data), 10)
        self.create_subscription(
            String, "/depth_slam/localization/state",
            lambda message: setattr(self, "localization_state", message.data), 10)
        self.create_timer(0.1, self.publish_visuals)

    def p(self, name):
        return self.get_parameter(name).value

    def on_depth(self, message):
        try:
            self.depth_cache.append(
                (stamp_seconds(message.header.stamp), message, image_array(message)))
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)

    def on_camera_info(self, message):
        self.camera_info = message

    def on_detections(self, message):
        try:
            document = json.loads(message.data)
            stamp = json_stamp_seconds(document)
        except (KeyError, TypeError, ValueError):
            return
        self.detection = (stamp, document)

    def _camera_publishers(self):
        return len(self.get_publishers_info_by_topic(str(self.p("color_topic"))))

    def _transform(self, target_frame, source_frame, stamp):
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame,
            Time(nanoseconds=int(round(float(stamp)*1.0e9))),
            timeout=Duration(seconds=float(self.p("tf_timeout_sec"))))
        value = transform.transform
        return value, (
            value.translation.x, value.translation.y, value.translation.z), (
            value.rotation.x, value.rotation.y,
            value.rotation.z, value.rotation.w)

    def on_fusion(self, message):
        try:
            fusion = json.loads(message.data)
            fusion_stamp = json_stamp_seconds(fusion)
        except (KeyError, TypeError, ValueError):
            self.traffic = {"valid": False, "reason": "FUSION_JSON_INVALID"}
            self._publish_status()
            return
        result = {
            "kind": "traffic_light", "valid": False,
            "stamp": fusion_stamp, "state": fusion.get("fused_state", "UNKNOWN"),
            "confidence": float(fusion.get("fused_confidence", 0.0) or 0.0),
            "reason": "DETECTION_MISSING", "map_point": None,
        }
        detection_stamp = math.nan
        depth_stamp = math.nan
        projection_valid = False
        transform_valid = False
        detail_reasons = []
        selected = None
        if self.detection is not None:
            detection_stamp, document = self.detection
            selected = select_traffic_detection(
                document, fusion.get("fused_state", ""))
        if selected is None:
            detail_reasons.append("TRAFFIC_BBOX_MISSING")
        if self.camera_info is None:
            detail_reasons.append("CAMERA_INFO_MISSING")
        depth_item = None
        if self.depth_cache and math.isfinite(detection_stamp):
            depth_item = min(self.depth_cache,
                             key=lambda item: abs(item[0]-detection_stamp))
        if depth_item is None:
            detail_reasons.append("DEPTH_MISSING")
        if selected is not None and self.camera_info is not None and depth_item:
            depth_stamp, depth_message, depth = depth_item
            projection = robust_bbox_projection(
                depth, depth_message.encoding, self.camera_info.k,
                selected["bbox"], int(self.p("minimum_depth_pixels")),
                float(self.p("minimum_depth_m")),
                float(self.p("maximum_depth_m")),
                float(self.p("maximum_depth_mad_m")))
            projection_valid = projection.valid
            result.update({
                "bbox": list(selected["bbox"]),
                "yolo_confidence": selected["confidence"],
                "depth_m": (projection.median_depth_m
                            if math.isfinite(projection.median_depth_m) else None),
                "depth_pixels": projection.valid_pixels,
                "depth_mad_m": (projection.depth_mad_m
                                if math.isfinite(projection.depth_mad_m) else None),
            })
            if not projection.valid:
                detail_reasons.append(projection.reason)
            else:
                source_frame = (depth_message.header.frame_id or
                                self.camera_info.header.frame_id)
                try:
                    _transform, translation, quaternion = self._transform(
                        self.p("map_frame"), source_frame, depth_stamp)
                    result["map_point"] = list(transform_point(
                        projection.optical_xyz_m, translation, quaternion))
                    transform_valid = True
                except (TransformException, ValueError):
                    detail_reasons.append("MAP_CAMERA_TF_MISSING")
        reasons = traffic_validation_reasons(
            fusion, detection_stamp, depth_stamp, self._camera_publishers(),
            self.tracking_valid, self.mapping_state, self.localization_state,
            projection_valid, transform_valid,
            float(self.p("maximum_stamp_delta_sec")),
            float(self.p("minimum_traffic_confidence")))
        reasons.extend(item for item in detail_reasons if item not in reasons)
        result["valid"] = not reasons
        result["reason"] = "VALIDATED" if not reasons else reasons[0]
        result["reasons"] = reasons
        result["same_observation"] = not any(
            value in reasons for value in (
                "OBSERVATION_TIMESTAMP_MISMATCH",
                "OBSERVATION_TIMESTAMP_INVALID"))
        self.traffic = result
        self.traffic_valid_pub.publish(Bool(data=result["valid"]))
        if result["valid"] and fusion_stamp != self.last_final_traffic_stamp:
            self.last_final_traffic_stamp = fusion_stamp
            self.traffic_result_pub.publish(String(data=json.dumps(
                result, separators=(",", ":"), sort_keys=True,
                allow_nan=False)))
        self._publish_status()

    def on_mission(self, message):
        try:
            mission = json.loads(message.data)
            stamp = float(mission["input_timestamp"])
            distance = float(mission["stop_line_distance_m"])
        except (KeyError, TypeError, ValueError):
            self.stop_line = {"valid": False, "reason": "STOP_INPUT_INVALID"}
            self._publish_status()
            return
        valid_2d = bool(mission.get("stop_line_detected") and
                        mission.get("stop_line_distance_valid") and
                        mission.get("stop_line_tf_valid") and
                        math.isfinite(stamp) and math.isfinite(distance) and
                        distance > 0.0)
        reasons = []
        if self._camera_publishers() != 1:
            reasons.append(f"CAMERA_PUBLISHER_COUNT:{self._camera_publishers()}")
        if not valid_2d:
            reasons.append(str(mission.get(
                "stop_line_failure_reason", "STOP_LINE_UNCONFIRMED")))
        if not self.tracking_valid:
            reasons.append("CUVSLAM_TRACKING_INVALID")
        if not vslam_ready(self.mapping_state, self.localization_state):
            reasons.append("VSLAM_NOT_READY")
        result = {
            "kind": "stop_line", "valid": False, "stamp": stamp,
            "front_axle_distance_m": distance,
            "confidence": min(1.0, max(0.0, float(
                mission.get("stop_line_depth_valid_pixels", 0))/60.0)),
            "reason": "STOP_LINE_UNCONFIRMED", "map_point": None,
            "line_endpoints": None, "map_orientation": None,
        }
        if valid_2d:
            base_x = float(self.p("front_axle_x_m"))+distance
            try:
                transform, translation, quaternion = self._transform(
                    self.p("map_frame"), self.p("base_frame"), stamp)
                center = transform_point((base_x, 0.0, 0.0),
                                         translation, quaternion)
                left = transform_point((base_x, 1.0, 0.0),
                                       translation, quaternion)
                right = transform_point((base_x, -1.0, 0.0),
                                        translation, quaternion)
                result["map_point"] = list(center)
                result["line_endpoints"] = [list(left), list(right)]
                result["map_orientation"] = [
                    transform.rotation.x, transform.rotation.y,
                    transform.rotation.z, transform.rotation.w]
            except (TransformException, ValueError):
                reasons.append("MAP_BASE_TF_MISSING")
        result["valid"] = not reasons
        result["reason"] = "VALIDATED" if not reasons else reasons[0]
        result["reasons"] = reasons
        self.stop_line = result
        self.stop_valid_pub.publish(Bool(data=result["valid"]))
        if result["valid"] and stamp != self.last_final_stop_stamp:
            self.last_final_stop_stamp = stamp
            self.stop_result_pub.publish(String(data=json.dumps(
                result, separators=(",", ":"), sort_keys=True,
                allow_nan=False)))
        self._publish_status()

    def _publish_status(self):
        status = {
            "camera_publishers": self._camera_publishers(),
            "cuvslam_tracking_valid": self.tracking_valid,
            "mapping_state": self.mapping_state,
            "localization_state": self.localization_state,
            "traffic_light": self.traffic,
            "stop_line": self.stop_line,
        }
        self.status_pub.publish(String(data=json.dumps(
            status, separators=(",", ":"), sort_keys=True, allow_nan=False)))

    def _marker(self, marker_id, marker_type, namespace):
        marker = Marker()
        marker.header.frame_id = self.p("map_frame")
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        duration = max(0.05, float(self.p("marker_lifetime_sec")))
        marker.lifetime.sec = int(duration)
        marker.lifetime.nanosec = int((duration-int(duration))*1.0e9)
        return marker

    @staticmethod
    def _set_color(marker, valid, blue=False):
        if valid and blue:
            marker.color.r, marker.color.g, marker.color.b = 0.1, 0.5, 1.0
        elif valid:
            marker.color.r, marker.color.g, marker.color.b = 0.1, 1.0, 0.2
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.75, 0.0
        marker.color.a = 0.95

    def _point(self, values):
        return Point(x=float(values[0]), y=float(values[1]), z=float(values[2]))

    def _object_markers(self):
        markers = []
        traffic = self.traffic
        if traffic.get("map_point"):
            shape = self._marker(10, Marker.SPHERE, "traffic_light_observation")
            shape.pose.position = self._point(traffic["map_point"])
            shape.scale.x = shape.scale.y = shape.scale.z = 0.28
            self._set_color(shape, traffic.get("valid", False))
            markers.append(shape)
            text = self._marker(11, Marker.TEXT_VIEW_FACING,
                                "traffic_light_observation")
            text.pose.position = self._point(traffic["map_point"])
            text.pose.position.z += 0.35
            text.scale.z = 0.22
            self._set_color(text, traffic.get("valid", False))
            # Deliberately omit current R/G state from map markers. State is
            # live-only in the 2D overlay/result topic and never stored in DB.
            text.text = ("TRAFFIC LIGHT VALID" if traffic.get("valid") else
                         "TRAFFIC LIGHT UNCONFIRMED")
            markers.append(text)
        stop = self.stop_line
        if stop.get("line_endpoints"):
            line = self._marker(20, Marker.LINE_LIST, "stop_line_observation")
            line.scale.x = 0.08
            line.points = [self._point(value) for value in stop["line_endpoints"]]
            self._set_color(line, stop.get("valid", False), blue=True)
            markers.append(line)
            arrow = self._marker(21, Marker.ARROW, "stop_line_observation")
            arrow.pose.position = self._point(stop["map_point"])
            orientation = stop["map_orientation"]
            arrow.pose.orientation.x, arrow.pose.orientation.y = orientation[:2]
            arrow.pose.orientation.z, arrow.pose.orientation.w = orientation[2:]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.6, 0.10, 0.10
            self._set_color(arrow, stop.get("valid", False), blue=True)
            markers.append(arrow)
        return markers

    def _vehicle_marker_and_path(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.p("map_frame"), self.p("base_frame"), Time(),
                timeout=Duration(seconds=float(self.p("tf_timeout_sec"))))
        except TransformException:
            return None
        marker = self._marker(1, Marker.ARROW, "vehicle_current_pose")
        marker.pose.position.x = transform.transform.translation.x
        marker.pose.position.y = transform.transform.translation.y
        marker.pose.position.z = transform.transform.translation.z
        marker.pose.orientation = transform.transform.rotation
        marker.scale.x, marker.scale.y, marker.scale.z = 0.65, 0.22, 0.22
        marker.color.r, marker.color.g = 0.2, 0.8
        marker.color.b, marker.color.a = 1.0, 1.0
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position = marker.pose.position
        pose.pose.orientation = marker.pose.orientation
        append = not self.vehicle_path.poses
        if self.vehicle_path.poses:
            previous = self.vehicle_path.poses[-1].pose.position
            append = math.hypot(pose.pose.position.x-previous.x,
                                pose.pose.position.y-previous.y) >= float(
                                    self.p("vehicle_path_min_distance_m"))
        if append:
            self.vehicle_path.poses.append(pose)
            maximum = int(self.p("vehicle_path_max_poses"))
            if len(self.vehicle_path.poses) > maximum:
                self.vehicle_path.poses = self.vehicle_path.poses[-maximum:]
        self.vehicle_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.vehicle_path)
        return marker

    def publish_visuals(self):
        if not rclpy.ok():
            return
        markers = self._object_markers()
        vehicle = self._vehicle_marker_and_path()
        if vehicle is not None:
            markers.append(vehicle)
        self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionVslamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
