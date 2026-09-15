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
    SelectedRoute, SignalExitDetector, SignalState, SignalVoteWindow)


class SignalExitNode(Node):
    """Publish A only for a stable left-to-right G/R/R observation."""

    def __init__(self):
        super().__init__("depth_signal_exit")
        defaults = {
            "image_topic": "/camera/image_raw",
            "exit_roi": [0.05, 0.00, 0.95, 0.40],
            "red_1_low": [0, 90, 90], "red_1_high": [12, 255, 255],
            "red_2_low": [168, 90, 90], "red_2_high": [179, 255, 255],
            "green_core_low": [40, 90, 90],
            "green_core_high": [85, 255, 255],
            "green_extended_low": [86, 60, 50],
            "green_extended_high": [110, 255, 125],
            "minimum_valid_frames": 60, "decision_ratio": 0.75,
            "observation_duration_s": 5.0,
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
                     tuple(value("green_extended_high"))))
        self.detector = SignalExitDetector(config, roi)
        self.window = SignalVoteWindow(
            value("observation_duration_s"), value("minimum_valid_frames"),
            value("decision_ratio"))
        self.bridge = CvBridge()
        self.mode = -1
        self.last_window_state = ObservationState.IDLE
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
        snapshot = self.window.observe(detection.state, now)
        # Raw A/B evidence feeds the existing five-second commit gate.  The
        # gate, not this advisory camera node, owns DEFAULT and branch commit.
        self.signal_pub.publish(String(data=self._raw_route(detection.state)))
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
        }, separators=(",", ":"))))
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
