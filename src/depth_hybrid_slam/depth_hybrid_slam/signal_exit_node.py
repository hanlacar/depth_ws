"""ROS adapter for the self-contained Mode 11 three-lamp detector."""

import json
import time

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .signal_exit_core import (
    DetectorConfig, HSVRange, NormalizedROI, ObservationState,
    SelectedRoute, ExitSignalTrack, SignalExitDetector, SignalState,
    SignalVoteWindow)


class SignalExitNode(Node):
    """Track the upper-image lamps and publish only explicit stable A/B."""

    def __init__(self):
        super().__init__("depth_signal_exit")
        defaults = {
            "image_topic": "/camera/image_raw",
            "exit_roi": [0.05, 0.00, 0.95, 0.40],
            "red_1_low": [0, 60, 55], "red_1_high": [12, 255, 255],
            "red_2_low": [168, 60, 55], "red_2_high": [179, 255, 255],
            "green_core_low": [35, 60, 55],
            "green_core_high": [105, 255, 255],
            "green_extended_low": [35, 60, 55],
            "green_extended_high": [105, 255, 255],
            "minimum_signal_area_px": 40.0,
            "dark_pixel_threshold": 80,
            "dark_ratio_threshold": 0.08,
            "exit_slot_centers": [0.25, 0.50, 0.75],
            "exit_slot_max_distance": 0.18,
            "exit_slot_conflict_margin": 0.12,
            "exit_row_y_tolerance_lamp_heights": 1.25,
            "exit_row_minimum_x_gap": 0.05,
            "exit_row_maximum_gap_ratio": 2.25,
            "exit_row_minimum_size_ratio": 0.35,
            "exit_green_weight": 3.0,
            "exit_red_pair_weight": 1.0,
            "exit_min_green_observations": 2,
            "exit_min_red_pair_observations": 2,
            "minimum_valid_frames": 60, "decision_ratio": 0.75,
            "exit_decision_time_sec": 5.0,
            "track_confirmations": 3, "track_missing_hold_s": 0.25,
            "track_minimum_confidence": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def value(name):
            return self.get_parameter(name).value

        roi = NormalizedROI(*(float(item) for item in value("exit_roi")))
        config = DetectorConfig(
            (HSVRange(tuple(value("red_1_low")), tuple(value("red_1_high"))),
             HSVRange(tuple(value("red_2_low")), tuple(value("red_2_high")))),
            HSVRange(tuple(value("green_core_low")),
                     tuple(value("green_core_high"))),
            HSVRange(tuple(value("green_extended_low")),
                     tuple(value("green_extended_high"))),
            minimum_contour_area=float(value("minimum_signal_area_px")),
            panel_dark_value=int(value("dark_pixel_threshold")),
            dark_ratio_threshold=float(value("dark_ratio_threshold")),
            slot_centers=tuple(float(item) for item in value("exit_slot_centers")),
            slot_max_distance=float(value("exit_slot_max_distance")),
            slot_conflict_margin=float(value("exit_slot_conflict_margin")),
            row_y_tolerance_lamp_heights=float(value(
                "exit_row_y_tolerance_lamp_heights")),
            row_minimum_x_gap=float(value("exit_row_minimum_x_gap")),
            row_maximum_gap_ratio=float(value(
                "exit_row_maximum_gap_ratio")),
            row_minimum_size_ratio=float(value(
                "exit_row_minimum_size_ratio")))
        self.detector = SignalExitDetector(config, roi)
        self.window = SignalVoteWindow(
            value("exit_decision_time_sec"), value("minimum_valid_frames"),
            value("decision_ratio"), value("exit_green_weight"),
            value("exit_red_pair_weight"),
            value("exit_min_green_observations"),
            value("exit_min_red_pair_observations"))
        self.track = ExitSignalTrack(
            value("track_confirmations"), value("track_missing_hold_s"),
            value("track_minimum_confidence"))
        self.latest_track = {}
        self.bridge = CvBridge()
        self.mode = -1
        self.last_window_state = ObservationState.IDLE
        self.last_debug_log = 0.0
        self.create_subscription(String, "/drive_mode", self._mode, 10)
        self.create_subscription(
            Image, str(value("image_topic")), self._image,
            qos_profile_sensor_data)
        self.signal_pub = self.create_publisher(
            String, "/camera/exit_branch_signal", 10)
        self.state_pub = self.create_publisher(
            String, "/depth_slam/mode11/signal_state", 10)
        self.event_pub = self.create_publisher(
            String, "/depth_slam/mode11/signal_event", 10)
        self.create_timer(0.05, self._tick)

    def _mode(self, message):
        try:
            mode = int(str(message.data).strip())
        except ValueError:
            mode = -1
        if mode == 11 and self.mode != 11:
            self.window.reset()
            self.window.start(time.monotonic())
            self.track.reset()
        elif mode != 11 and self.mode == 11:
            self.window.reset()
        self.mode = mode

    @staticmethod
    def _raw_route(state):
        if state == SignalState.GREEN:
            return SelectedRoute.A.value
        if state == SignalState.RED:
            return SelectedRoute.B.value
        return SelectedRoute.UNKNOWN.value

    def _image(self, message):
        if self.mode != 11:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            detection = self.detector.detect(frame)
        except Exception as error:
            self.get_logger().error(f"Mode11 image processing failed: {error}")
            return
        now = time.monotonic()
        self.latest_track = self.track.update(detection, now)
        snapshot = self.window.observe_detection(detection, now)
        # Preserve the legacy raw A/B topic while the weighted final event is
        # the authoritative branch decision after the five-second window.
        self.signal_pub.publish(String(data=self._raw_route(detection.state)))
        if now-self.last_debug_log >= 1.0:
            self.last_debug_log = now
            counts = snapshot.position_counts or ((0, 0),)*3
            self.get_logger().info(
                "[EXIT_SIGNAL] elapsed=%.1fs L(R=%d,G=%d) C(R=%d,G=%d) "
                "R(R=%d,G=%d) A=%.1f B=%.1f reason=%s" % (
                    snapshot.elapsed, counts[0][0], counts[0][1],
                    counts[1][0], counts[1][1], counts[2][0], counts[2][1],
                    snapshot.a_score, snapshot.b_score,
                    detection.reason))
        self._publish(snapshot)

    def _tick(self):
        if self.mode == 11:
            self._publish(self.window.evaluate(time.monotonic()))

    def _publish(self, snapshot):
        self.state_pub.publish(String(data=snapshot.state.value))
        if snapshot.state == self.last_window_state:
            return
        source = ("CAMERA" if snapshot.state == ObservationState.LATCHED
                  else "DEFAULT" if snapshot.state == ObservationState.DEFAULTED
                  else "")
        self.event_pub.publish(String(data=json.dumps({
            "state": snapshot.state.value, "route": snapshot.route.value,
            "source": source, "confidence": snapshot.confidence,
            "green": snapshot.green, "red": snapshot.red,
            "unknown": snapshot.unknown, "elapsed_s": snapshot.elapsed,
            "a_score": snapshot.a_score, "b_score": snapshot.b_score,
            "reason": snapshot.reason,
            "position_counts": snapshot.position_counts,
            "bbox": self.latest_track.get("bbox", ()),
            "track_confidence": self.latest_track.get("confidence", 0.0),
            "last_seen": self.latest_track.get("last_seen"),
            "missing_duration": self.latest_track.get(
                "missing_duration", 0.0),
            "track_continuity": self.latest_track.get(
                "track_continuity", 0),
        }, separators=(",", ":"))))
        if snapshot.state in (ObservationState.LATCHED,
                              ObservationState.DEFAULTED):
            self.get_logger().info(
                f"[EXIT_SIGNAL] FINAL={snapshot.route.value} "
                f"reason={snapshot.reason}")
        self.last_window_state = snapshot.state


def main(args=None):
    rclpy.init(args=args)
    node = SignalExitNode()
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
