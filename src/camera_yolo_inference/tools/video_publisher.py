#!/usr/bin/env python3
"""Standalone mp4 -> /camera/image_raw + /camera/camera_info test publisher.

Feeds a recorded video file into the camera_yolo_inference pipeline so it
can be validated without a live camera. Not a package entry point: run
directly, e.g.

    python3 tools/video_publisher.py --ros-args \\
        -p video_path:=/home/qor/urrc_hanla/20260827_070334_from_0530.mp4 \\
        -p fps:=30.0 -p loop:=true -p frame_id:=camera_link

QoS and topic names intentionally mirror camera_yolo_inference_node's
input defaults (input_reliability=best_effort, input_depth=1,
/camera/image_raw, /camera/camera_info) so both nodes talk without remaps.

Uses camera_yolo_inference.ros_image.bgr8_to_image instead of cv_bridge:
the inference node avoids cv_bridge on purpose because its binary build
can crash against a NumPy 2.x environment (see ros_image.py's module
docstring), so this test tool follows the same convention rather than
reintroducing that risk.
"""
import csv
import json
from pathlib import Path
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

from camera_yolo_inference.ros_image import bgr8_to_image

WIDTH, HEIGHT = 640, 480


def _dummy_camera_info(frame_id):
    """640x480 plumb_bob intrinsics satisfying image_contract's checks.
    Values are not calibrated to the real camera -- they only need to be
    finite and well-formed; nothing downstream uses them for geometry in
    this segmentation pipeline."""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width, info.height = WIDTH, HEIGHT
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    fx = fy = 500.0
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.declare_parameter("video_path", "")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("frame_id", "camera_link")
        # Opt-in, off by default (empty path): logs (source video frame
        # index, published stamp) per published frame to a local CSV so a
        # captured ROS message's header.stamp can be matched back to the
        # exact cv2.VideoCapture frame index for offline re-inference on
        # the identical frame. No effect on /camera/image_raw or any other
        # published topic/contract when left unset.
        self.declare_parameter("frame_index_log_path", "")
        # Optional deterministic diagnostic mode: seek to these source indices
        # instead of decoding sequentially. Production/default behavior is unchanged.
        self.declare_parameter("frame_indices_path", "")
        self.declare_parameter("diagnostics_topic", "/video_test/publisher_diagnostics")
        self.declare_parameter("diagnostics_hz", 1.0)

        video_path = Path(str(self.get_parameter("video_path").value)).expanduser()
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        frame_index_log_path = str(
            self.get_parameter("frame_index_log_path").value).strip()
        self.frame_index_log = None
        indices_path = str(self.get_parameter("frame_indices_path").value).strip()
        self.frame_indices = None
        self.frame_indices_cursor = 0
        self.timer_callbacks = 0
        self.published_frames = 0
        self.read_failures = 0
        self.loop_count = 0
        self.current_frame_index = -1
        self.current_stamp_ns = 0
        self._started_wall = time.monotonic()
        if indices_path:
            self.frame_indices = [int(value) for value in json.loads(
                Path(indices_path).expanduser().read_text())]
        if frame_index_log_path:
            self.frame_index_log = open(  # noqa: SIM115 -- closed in main()'s finally
                Path(frame_index_log_path).expanduser(), "w", newline="")
            self._frame_index_writer = csv.writer(self.frame_index_log)
            self._frame_index_writer.writerow(
                ["source_frame_index", "stamp_sec", "stamp_nanosec"])
        if not video_path.is_file():
            raise FileNotFoundError(f"video_path does not exist: {video_path}")
        if self.fps <= 0.0:
            raise ValueError("fps must be > 0")

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_logger().info(
            f"publishing {video_path} at {self.fps:.1f} fps "
            f"(loop={self.loop}, frame_id={self.frame_id}, "
            f"{frame_count} source frames, resized to {WIDTH}x{HEIGHT})")

        # Matches camera_yolo_inference_node's default input QoS
        # (input_reliability=best_effort, input_depth=1) exactly.
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", qos)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", qos)
        self.diagnostics_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10)
        self.camera_info = _dummy_camera_info(self.frame_id)

        diagnostics_hz = float(self.get_parameter("diagnostics_hz").value)
        if diagnostics_hz <= 0.0:
            raise ValueError("diagnostics_hz must be > 0")
        self.frame_timer = self.create_timer(1.0 / self.fps, self._publish_frame)
        self.diagnostics_timer = self.create_timer(
            1.0 / diagnostics_hz, self._publish_diagnostics)

    def _publish_frame(self):
        self.timer_callbacks += 1
        if self.frame_indices:
            requested = self.frame_indices[self.frame_indices_cursor]
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
        ok, frame = self.capture.read()
        if not ok:
            self.read_failures += 1
            if self.loop:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
                if ok:
                    self.loop_count += 1
                    self.get_logger().info(
                        f"EOF reached; loop restarted at frame 0 "
                        f"(loop_count={self.loop_count}, "
                        f"published_frames={self.published_frames})")
            if not ok:
                self.get_logger().warn(
                    "video read failed; no frame published "
                    f"(loop={self.loop}, failures={self.read_failures})")
                return
        # POS_FRAMES already advanced past the frame read() just returned,
        # so subtract 1 to log the index of *this* frame -- robust across
        # the loop-reset above too, since it reads the true decoder
        # position rather than a manually kept counter.
        source_frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if self.frame_indices:
            self.frame_indices_cursor = (self.frame_indices_cursor + 1) % len(self.frame_indices)
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # max_image_age_sec=0.4 in camera_yolo_inference_node means this
        # stamp must be the current ROS time, not the video's own PTS.
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self.frame_id)
        self.image_pub.publish(bgr8_to_image(frame, header))
        self.camera_info.header.stamp = stamp
        self.info_pub.publish(self.camera_info)
        self.published_frames += 1
        self.current_frame_index = source_frame_index
        self.current_stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if self.frame_index_log is not None:
            self._frame_index_writer.writerow(
                [source_frame_index, stamp.sec, stamp.nanosec])
            self.frame_index_log.flush()

    def _publish_diagnostics(self):
        """Publish low-rate proof that both time and image content advance."""
        position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        # Read no extra source frame here: hash the frame at the current decoder
        # position by seeking would perturb sequential playback.  The companion
        # frame_hash_diagnostic.py hashes received ROS images.  This publisher
        # reports the decoder/timer side of the same contract.
        document = {
            "video_open": bool(self.capture.isOpened()),
            "video_path": str(self.get_parameter("video_path").value),
            "configured_fps": self.fps,
            "loop": self.loop,
            "source_frame_count": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "source_fps": float(self.capture.get(cv2.CAP_PROP_FPS)),
            "decoder_next_frame_index": position,
            "current_frame_index": self.current_frame_index,
            "current_stamp_ns": self.current_stamp_ns,
            "timer_callbacks": self.timer_callbacks,
            "published_frames": self.published_frames,
            "read_failures": self.read_failures,
            "loop_count": self.loop_count,
            "observed_publish_fps": self.published_frames / max(
                1.0e-9, time.monotonic() - self._started_wall),
        }
        self.diagnostics_pub.publish(String(data=json.dumps(
            document, separators=(",", ":"))))


def main():
    rclpy.init()
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.capture.release()
        if node.frame_index_log is not None:
            node.frame_index_log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
