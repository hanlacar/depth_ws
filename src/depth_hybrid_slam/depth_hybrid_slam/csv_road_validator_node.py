"""ROS adapter for calibrated, validation-only CSV-to-road containment."""

import json
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Path
from race_interfaces.msg import SemanticPathFrame
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String

from camera_navigation.direct_bev_projection import (
    CameraModel, build_ground_remap, project_mask_to_bev)
from camera_navigation.ground_plane_calibration import (
    load_camera_mount_config, rotation_matrix_rpy)
from camera_navigation.semantic_path_contract import decode_binary_rle

from .csv_road_validator_core import (
    CAMERA_UNAVAILABLE, INVALID_GEOMETRY, PATH_UNAVAILABLE, ValidatorConfig,
    classify_input_state, render_bev_overlay, unavailable_result,
    validate_metric_bev)


def _yaw(quaternion):
    norm = math.sqrt(quaternion.x**2+quaternion.y**2+
                     quaternion.z**2+quaternion.w**2)
    if norm < 1.0e-6:
        return None
    return math.atan2(2.0*(quaternion.w*quaternion.z+
                           quaternion.x*quaternion.y),
                      1.0-2.0*(quaternion.y**2+quaternion.z**2))


class CsvRoadValidatorNode(Node):
    """Consumes semantic frames and local candidate paths; never commands."""

    def __init__(self):
        super().__init__("csv_road_validator")
        defaults = {
            "semantic_topic": "/perception/semantic_path_frame",
            "camera_info_topic": "/camera/camera_info",
            "local_path_topic": "/depth_slam/csv_validation/local_path",
            "path_frame_id": "base_link",
            "semantic_timeout_s": 0.50,
            "path_timeout_s": 0.50,
            "camera_info_timeout_s": 5.0,
            "publish_hz": 10.0,
            "forward_min_m": 0.5,
            "forward_max_m": 4.0,
            "y_min_m": -3.0,
            "y_max_m": 3.0,
            "resolution_m": 0.04,
            "vehicle_width_m": 0.78,
            "minimum_center_inside_ratio": 0.90,
            "minimum_vehicle_corridor_inside_ratio": 0.75,
            "minimum_visible_path_ratio": 0.70,
            "minimum_road_confidence": 0.70,
            "camera_mount.configured": False,
            "camera_mount.position_x_m": 0.0,
            "camera_mount.position_y_m": 0.0,
            "camera_mount.height_z_m": 0.0,
            "camera_mount.reference_roll_deg": 0.0,
            "camera_mount.reference_pitch_deg": 0.0,
            "camera_mount.reference_yaw_deg": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        self.config = ValidatorConfig(
            forward_min_m=float(p("forward_min_m")),
            forward_max_m=float(p("forward_max_m")),
            y_min_m=float(p("y_min_m")), y_max_m=float(p("y_max_m")),
            resolution_m=float(p("resolution_m")),
            vehicle_width_m=float(p("vehicle_width_m")),
            minimum_center_inside_ratio=float(
                p("minimum_center_inside_ratio")),
            minimum_vehicle_corridor_inside_ratio=float(
                p("minimum_vehicle_corridor_inside_ratio")),
            minimum_visible_path_ratio=float(p("minimum_visible_path_ratio")),
            minimum_road_confidence=float(p("minimum_road_confidence")))
        self.config.validate()
        self.mount = load_camera_mount_config({
            "configured": p("camera_mount.configured"),
            "position_x_m": p("camera_mount.position_x_m"),
            "position_y_m": p("camera_mount.position_y_m"),
            "height_z_m": p("camera_mount.height_z_m"),
            "reference_roll_deg": p("camera_mount.reference_roll_deg"),
            "reference_pitch_deg": p("camera_mount.reference_pitch_deg"),
            "reference_yaw_deg": p("camera_mount.reference_yaw_deg")})
        self.semantic_timeout = float(p("semantic_timeout_s"))
        self.path_timeout = float(p("path_timeout_s"))
        self.camera_timeout = float(p("camera_info_timeout_s"))
        self.expected_path_frame = str(p("path_frame_id")).lstrip("/")
        self.camera = None
        self.camera_seen = False
        self.camera_frame = ""
        self.camera_receipt = None
        self.map_x = self.map_y = self.visibility = None
        self.road = self.lane = None
        self.semantic_receipt = None
        self.semantic_seen = False
        self.semantic_error = "semantic frame not received"
        self.path = np.empty((0, 3), dtype=float)
        self.path_receipt = None
        self.path_error = "path not received"

        semantic_topic = str(p("semantic_topic"))
        camera_topic = str(p("camera_info_topic"))
        path_topic = str(p("local_path_topic"))
        self.create_subscription(SemanticPathFrame, semantic_topic,
                                 self._on_semantic, 10)
        self.create_subscription(CameraInfo, camera_topic,
                                 self._on_camera_info, 10)
        self.create_subscription(Path, path_topic, self._on_path, 10)
        prefix = "/depth_slam/csv_road_validation"
        self.pub_valid = self.create_publisher(Bool, prefix+"/valid", 10)
        self.pub_state = self.create_publisher(String, prefix+"/state", 10)
        self.pub_center = self.create_publisher(
            Float32, prefix+"/center_inside_ratio", 10)
        self.pub_corridor = self.create_publisher(
            Float32, prefix+"/corridor_inside_ratio", 10)
        self.pub_visible = self.create_publisher(
            Float32, prefix+"/path_visible_ratio", 10)
        self.pub_road_confidence = self.create_publisher(
            Float32, prefix+"/road_confidence", 10)
        self.pub_lane_visible = self.create_publisher(
            Bool, prefix+"/lane_visible", 10)
        self.pub_lane_consistent = self.create_publisher(
            Bool, prefix+"/lane_consistent", 10)
        self.pub_diagnostics = self.create_publisher(
            String, prefix+"/diagnostics", 10)
        self.pub_overlay = self.create_publisher(
            Image, prefix+"/overlay", 10)
        frequency = float(p("publish_hz"))
        if frequency <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/frequency, self._tick)

    def _on_camera_info(self, message):
        self.camera_seen = True
        try:
            matrix = np.asarray(message.k, dtype=float).reshape(3, 3)
            if (message.width <= 0 or message.height <= 0 or
                    matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or
                    not np.all(np.isfinite(matrix))):
                raise ValueError("invalid CameraInfo intrinsics")
            self.camera = CameraModel(
                int(message.width), int(message.height), matrix,
                np.asarray(message.d, dtype=float), message.distortion_model)
            self.camera_frame = str(message.header.frame_id).lstrip("/")
            rotation = rotation_matrix_rpy(
                self.mount.reference_roll_deg,
                self.mount.reference_pitch_deg,
                self.mount.reference_yaw_deg)
            position = np.asarray((self.mount.position_x_m,
                                   self.mount.position_y_m,
                                   self.mount.height_z_m), dtype=float)
            if not self.mount.is_usable():
                raise ValueError("camera mount is not commissioned")
            self.map_x, self.map_y = build_ground_remap(
                self.config, self.camera, rotation, position)
            self.visibility = ((self.map_x >= 0.0) &
                               (self.map_y >= 0.0)).astype(np.uint8)
            if not np.any(self.visibility):
                raise ValueError("camera ground projection is empty")
            self.camera_receipt = time.monotonic()
        except (TypeError, ValueError) as error:
            self.camera = None
            self.map_x = self.map_y = self.visibility = None
            self.semantic_error = str(error)

    @staticmethod
    def _decode_lane(message, height, width):
        lane = decode_binary_rle(message.lane_rle, height, width)
        if np.any(lane):
            return lane
        for values in (message.white_line_rle, message.yellow_line_rle,
                       message.unknown_line_rle):
            lane = np.maximum(lane, decode_binary_rle(values, height, width))
        return lane

    def _on_semantic(self, message):
        self.semantic_seen = True
        try:
            if self.camera is None or self.map_x is None:
                raise ValueError("calibrated camera geometry unavailable")
            height, width = int(message.image_height), int(message.image_width)
            if (width != self.camera.width or height != self.camera.height or
                    width <= 0 or height <= 0):
                raise ValueError("semantic dimensions do not match CameraInfo")
            frame = str(message.header.frame_id).lstrip("/")
            if self.camera_frame and frame and frame != self.camera_frame:
                raise ValueError("semantic frame does not match CameraInfo frame")
            road_image = decode_binary_rle(message.road_rle, height, width)
            lane_image = self._decode_lane(message, height, width)
            self.road = project_mask_to_bev(road_image, self.map_x, self.map_y)
            self.lane = project_mask_to_bev(lane_image, self.map_x, self.map_y)
            self.semantic_receipt = time.monotonic()
            self.semantic_error = ""
        except (TypeError, ValueError) as error:
            self.road = self.lane = None
            self.semantic_receipt = None
            self.semantic_error = str(error)

    def _on_path(self, message):
        frame = str(message.header.frame_id).lstrip("/")
        if frame != self.expected_path_frame:
            self.path = np.empty((0, 3), dtype=float)
            self.path_receipt = None
            self.path_error = (f"path frame {frame!r} != "
                               f"{self.expected_path_frame!r}")
            return
        points, yaws = [], []
        for stamped in message.poses:
            position = stamped.pose.position
            points.append((float(position.x), float(position.y)))
            yaws.append(_yaw(stamped.pose.orientation))
        if not points or not np.all(np.isfinite(points)):
            self.path = np.empty((0, 3), dtype=float)
            self.path_receipt = None
            self.path_error = "path is empty or non-finite"
            return
        points = np.asarray(points, dtype=float)
        if any(value is None for value in yaws):
            if len(points) > 1:
                delta = np.gradient(points, axis=0)
                yaws = np.arctan2(delta[:, 1], delta[:, 0])
            else:
                yaws = np.zeros(1)
        self.path = np.column_stack((points, np.asarray(yaws, dtype=float)))
        self.path_receipt = time.monotonic()
        self.path_error = ""

    def _result(self):
        now = time.monotonic()
        if self.camera_seen and self.camera is None:
            return unavailable_result(INVALID_GEOMETRY, self.semantic_error)
        if self.semantic_seen and self.semantic_receipt is None:
            return unavailable_result(INVALID_GEOMETRY, self.semantic_error)
        state = classify_input_state(
            self.mount.is_usable(),
            self.camera is not None and self.camera_receipt is not None,
            self.semantic_receipt is not None,
            self.path_receipt is not None,
            float("inf") if self.camera_receipt is None else now-self.camera_receipt,
            float("inf") if self.semantic_receipt is None else now-self.semantic_receipt,
            float("inf") if self.path_receipt is None else now-self.path_receipt,
            self.camera_timeout, self.semantic_timeout, self.path_timeout)
        if state is not None:
            reason = self.path_error if state.state == PATH_UNAVAILABLE else \
                self.semantic_error if state.state == CAMERA_UNAVAILABLE else state.reason
            return unavailable_result(state.state, reason)
        return validate_metric_bev(self.road, self.lane, self.visibility,
                                   self.path, self.config)

    def _publish_overlay(self, result):
        if self.road is None or self.lane is None or self.visibility is None:
            return
        image = render_bev_overlay(self.road, self.lane, self.visibility,
                                   self.path, result, self.config)
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.expected_path_frame
        message.height, message.width = image.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = int(message.width*3)
        message.data = image.tobytes()
        self.pub_overlay.publish(message)

    def _tick(self):
        result = self._result()
        self.pub_valid.publish(Bool(data=result.valid))
        self.pub_state.publish(String(data=result.state))
        self.pub_center.publish(Float32(data=float(result.center_inside_ratio)))
        self.pub_corridor.publish(Float32(
            data=float(result.vehicle_corridor_inside_ratio)))
        self.pub_visible.publish(Float32(data=float(result.visible_path_ratio)))
        self.pub_road_confidence.publish(Float32(data=float(result.road_confidence)))
        self.pub_lane_visible.publish(Bool(data=result.lane_visible))
        self.pub_lane_consistent.publish(Bool(data=result.lane_consistent))
        diagnostics = result.diagnostics()
        diagnostics.update({
            "validation_only": True,
            "path_frame": self.expected_path_frame,
            "projection": "calibrated_ground_plane_bev",
            "camera_mount_configured": self.mount.is_usable()})
        self.pub_diagnostics.publish(String(
            data=json.dumps(diagnostics, sort_keys=True)))
        self._publish_overlay(result)


def main(args=None):
    rclpy.init(args=args)
    node = CsvRoadValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
