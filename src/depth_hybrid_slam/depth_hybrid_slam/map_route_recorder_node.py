"""Service-controlled recorder for VSLAM localization poses in ``map``."""

import math
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMessage
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from .map_route_recorder_core import (
    GOOD_STATES, MapPoseSample, MapRouteRecordingSession,
    approve_rviz_review, resample_by_distance,
)
from .ros_helpers import quaternion_from_yaw, safe_shutdown, yaw_from_quaternion


def map_pose_sample(message, localization_state, localization_confidence):
    """Parse the actual PoseWithCovarianceStamped contract without TF fallback."""
    stamp = message.header.stamp
    position = message.pose.pose.position
    return MapPoseSample(
        int(stamp.sec)*1_000_000_000+int(stamp.nanosec),
        float(position.x), float(position.y), float(position.z),
        float(yaw_from_quaternion(message.pose.pose.orientation)),
        str(message.header.frame_id), str(localization_state).strip().upper(),
        float(localization_confidence), tuple(message.pose.covariance))


class MapRouteRecorderNode(Node):
    """Record and visualize only; this node creates no command publisher."""

    def __init__(self):
        super().__init__("map_route_recorder")
        defaults = (
            ("output_directory", "/home/qor/depth_ws/routes/recorded_map"),
            ("map_path", "/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db"),
            ("gps_reference_path", "/home/qor/depth_ws/routes/network/route_network_segmented.csv"),
            ("pose_topic", "/depth_slam/localization/pose"),
            ("state_topic", "/depth_slam/localization/state"),
            ("confidence_topic", "/depth_slam/localization/confidence"),
            ("pose_timeout_s", 0.50),
            ("resample_spacing_m", 0.10),
            ("stationary_duplicate_m", 0.01),
            ("recovery_jump_m", 0.75),
            ("wheelbase_m", 0.73),
            ("max_steering_deg", 22.0),
            ("use_drive_command_direction", False),
            ("drive_command_topic", "/slam_drive"),
        )
        for name, default in defaults:
            self.declare_parameter(name, default)
        self.localization_state = "INITIALIZING"
        self.localization_confidence = 0.0
        self.last_pose = None
        self.last_pose_receipt = None
        self.session = None
        self.last_summary = None
        self.raw_path = PathMessage()
        self.raw_path.header.frame_id = "map"
        self.resampled_path = PathMessage()
        self.resampled_path.header.frame_id = "map"
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_raw_path = self.create_publisher(
            PathMessage, "/depth_slam/recorded_route/raw_path", qos)
        self.pub_resampled_path = self.create_publisher(
            PathMessage, "/depth_slam/recorded_route/resampled_path", qos)
        self.pub_status = self.create_publisher(
            String, "/depth_slam/recorded_route/status", qos)
        self.create_subscription(
            PoseWithCovarianceStamped, str(self.get_parameter("pose_topic").value),
            self.on_pose, 50)
        self.create_subscription(
            String, str(self.get_parameter("state_topic").value),
            self.on_state, 10)
        self.create_subscription(
            Float32, str(self.get_parameter("confidence_topic").value),
            self.on_confidence, 10)
        # This optional input is disabled by default.  It observes the verified
        # existing Float32 command sign and never republishes it.
        if bool(self.get_parameter("use_drive_command_direction").value):
            self.create_subscription(
                Float32, str(self.get_parameter("drive_command_topic").value),
                self.on_drive_command, 10)
        root = "/depth_slam/map_route_record"
        self.create_service(Trigger, root+"/start", self.start)
        self.create_service(Trigger, root+"/pause", self.pause)
        self.create_service(Trigger, root+"/resume", self.resume)
        self.create_service(Trigger, root+"/finish", self.finish)
        self.create_service(Trigger, root+"/status", self.status)
        self.create_service(Trigger, root+"/set_forward", self.set_forward)
        self.create_service(Trigger, root+"/set_reverse", self.set_reverse)
        self.create_service(Trigger, root+"/approve_rviz", self.approve_rviz)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(
            "Map route recorder ready (manual direction; no drive publishers)")

    def on_state(self, message):
        self.localization_state = str(message.data).strip().upper()

    def on_confidence(self, message):
        self.localization_confidence = float(message.data)

    def on_drive_command(self, message):
        if self.session is not None and self.session.state in ("RECORDING", "PAUSED"):
            if message.data > 0.0:
                self.session.set_direction(1)
            elif message.data < 0.0:
                self.session.set_direction(-1)

    def pose_is_fresh(self):
        return (self.last_pose is not None and self.last_pose_receipt is not None and
                time.monotonic()-self.last_pose_receipt <=
                float(self.get_parameter("pose_timeout_s").value))

    def _sample(self, message):
        return map_pose_sample(
            message, self.localization_state, self.localization_confidence)

    @staticmethod
    def _pose_stamped(message):
        pose = PoseStamped()
        pose.header = message.header
        pose.pose = message.pose.pose
        return pose

    def on_pose(self, message):
        self.last_pose = message
        self.last_pose_receipt = time.monotonic()
        if self.session is None or self.session.state not in ("RECORDING", "PAUSED"):
            return
        sample = self._sample(message)
        eligible, _break = self.session.append(sample)
        if message.header.frame_id == "map" and all(math.isfinite(value) for value in
                (sample.x, sample.y, sample.z, sample.yaw)):
            self.raw_path.poses.append(self._pose_stamped(message))
        if eligible:
            self._refresh_resampled_path()

    def _refresh_resampled_path(self):
        points = resample_by_distance(
            self.session.samples, self.session.spacing_m,
            self.session.duplicate_distance_m)
        message = PathMessage()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        for point in points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp.sec = point.stamp_ns//1_000_000_000
            pose.header.stamp.nanosec = point.stamp_ns % 1_000_000_000
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation = quaternion_from_yaw(point.yaw)
            message.poses.append(pose)
        self.resampled_path = message

    def start(self, _request, response):
        problems = []
        if self.session is not None and self.session.state in ("RECORDING", "PAUSED"):
            problems.append("ALREADY_RECORDING")
        if self.localization_state not in GOOD_STATES:
            problems.append("LOCALIZATION_STATE_NOT_READY:"+self.localization_state)
        if not self.pose_is_fresh():
            problems.append("POSE_STALE_OR_MISSING")
        elif self.last_pose.header.frame_id != "map":
            problems.append("POSE_FRAME_NOT_MAP:"+self.last_pose.header.frame_id)
        if problems:
            response.success = False
            response.message = ",".join(problems)
            return response
        try:
            self.session = MapRouteRecordingSession(
                str(self.get_parameter("output_directory").value),
                str(self.get_parameter("map_path").value),
                spacing_m=float(self.get_parameter("resample_spacing_m").value),
                duplicate_distance_m=float(self.get_parameter("stationary_duplicate_m").value),
                recovery_jump_m=float(self.get_parameter("recovery_jump_m").value),
                wheelbase_m=float(self.get_parameter("wheelbase_m").value),
                max_steering_deg=float(self.get_parameter("max_steering_deg").value),
                gps_reference_path=str(self.get_parameter("gps_reference_path").value))
            self.raw_path = PathMessage(); self.raw_path.header.frame_id = "map"
            self.resampled_path = PathMessage(); self.resampled_path.header.frame_id = "map"
            self.session.append(self._sample(self.last_pose))
        except Exception as error:
            response.success = False
            response.message = f"START_FAILED:{error}"
            return response
        response.success = True
        response.message = f"RECORDING raw={self.session.raw_path}"
        return response

    def pause(self, _request, response):
        return self._transition(response, "pause", "PAUSED")

    def resume(self, _request, response):
        if self.localization_state not in GOOD_STATES or not self.pose_is_fresh():
            response.success = False
            response.message = "RESUME_GATE_FAILED"
            return response
        return self._transition(response, "resume", "RECORDING")

    def _transition(self, response, method, label):
        try:
            if self.session is None:
                raise RuntimeError("NO_SESSION")
            getattr(self.session, method)()
            response.success = True
            response.message = label
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def finish(self, _request, response):
        try:
            if self.session is None:
                raise RuntimeError("NO_SESSION")
            self.last_summary = self.session.finish()
            self._refresh_resampled_path()
            # Trigger success means the files were saved.  Quality remains an
            # explicit field and can still be false.
            response.success = True
            response.message = (
                f"automated_pass={self.last_summary['automated_pass']} "
                f"route={self.last_summary['route_path']} "
                f"metadata={self.last_summary['metadata_path']} "
                "approval=PENDING_RVIZ_REVIEW")
        except Exception as error:
            response.success = False
            response.message = f"FINISH_FAILED:{error}"
        return response

    def set_forward(self, _request, response):
        return self._set_direction(response, 1)

    def set_reverse(self, _request, response):
        return self._set_direction(response, -1)

    def _set_direction(self, response, value):
        try:
            if self.session is None or self.session.state not in ("RECORDING", "PAUSED"):
                raise RuntimeError("NO_ACTIVE_SESSION")
            self.session.set_direction(value)
            response.success = True
            response.message = "FORWARD" if value > 0 else "REVERSE"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def approve_rviz(self, _request, response):
        try:
            if self.session is None or self.session.state != "FINISHED":
                raise RuntimeError("FINISHED_SESSION_REQUIRED")
            approve_rviz_review(self.session.metadata_path)
            response.success = True
            response.message = f"APPROVED:{self.session.metadata_path}"
        except Exception as error:
            response.success = False
            response.message = f"APPROVAL_FAILED:{error}"
        return response

    def status(self, _request, response):
        state = self.session.state if self.session is not None else "IDLE"
        direction = self.session.direction if self.session is not None else 1
        response.success = True
        response.message = (
            f"state={state} localization={self.localization_state} "
            f"pose_fresh={self.pose_is_fresh()} direction={'F' if direction > 0 else 'R'} "
            f"raw={self.session.raw_count if self.session else 0} "
            f"eligible={len(self.session.samples) if self.session else 0}")
        return response

    def tick(self):
        if (self.session is not None and self.session.state == "RECORDING" and
                not self.pose_is_fresh()):
            self.session.notify_pose_timeout()
        now = self.get_clock().now().to_msg()
        self.raw_path.header.stamp = now
        self.resampled_path.header.stamp = now
        self.pub_raw_path.publish(self.raw_path)
        self.pub_resampled_path.publish(self.resampled_path)
        state = self.session.state if self.session is not None else "IDLE"
        direction = self.session.direction if self.session is not None else 1
        self.pub_status.publish(String(data=(
            f"{state};localization={self.localization_state};"
            f"direction={'F' if direction > 0 else 'R'};"
            f"pose_fresh={self.pose_is_fresh()}")))

    def destroy_node(self):
        if self.session is not None:
            self.session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MapRouteRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
