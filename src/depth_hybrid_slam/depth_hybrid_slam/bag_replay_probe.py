"""Hardware-free replay subscriber and TF reception probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformListener


REQUIRED = {
    "/camera/camera/color/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/infra1/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/infra2/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/imu": "sensor_msgs/msg/Imu",
    "/depth_slam/cuvslam/odometry": "nav_msgs/msg/Odometry",
    "/depth_slam/cuvslam/tracking_valid": "std_msgs/msg/Bool",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}
running = True


def stop(_signum, _frame):
    global running
    running = False


class ReplayProbe(Node):
    def __init__(self):
        super().__init__("competition_bag_replay_probe")
        self.counts = {topic: 0 for topic in REQUIRED}
        self.tracking = []
        self.tf_chain_seen = False
        sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        static_qos = QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._bag_subscriptions = []
        for topic, type_name in REQUIRED.items():
            cls = get_message(type_name)

            def callback(message, name=topic):
                self.counts[name] += 1
                if name == "/depth_slam/cuvslam/tracking_valid":
                    self.tracking.append(bool(message.data))

            qos = static_qos if topic == "/tf_static" else sensor_qos
            self._bag_subscriptions.append(
                self.create_subscription(cls, topic, callback, qos))
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self, spin_thread=False)
        self.create_timer(0.2, self.check_tf)

    def check_tf(self):
        self.tf_chain_seen |= self.buffer.can_transform(
            "map", "camera_link", Time(), timeout=Duration(seconds=0.01))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = ReplayProbe()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    result = {
        "counts": node.counts,
        "all_required_topics_received": all(value > 0 for value in node.counts.values()),
        "tf_map_to_camera_link_seen": node.tf_chain_seen,
        "tracking_valid_percent": (
            100.0 * sum(node.tracking) / len(node.tracking)
            if node.tracking else 0.0),
        "use_sim_time": False,
    }
    result["pass"] = result["all_required_topics_received"] and result["tf_map_to_camera_link_seen"]
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
