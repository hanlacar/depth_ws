"""Publish the single mapping session identity selected and persisted at launch."""

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .ros_helpers import safe_shutdown


class MappingSessionInfoNode(Node):
    TOPICS = {
        "session_id": "/depth_slam/mapping/session_id",
        "map_path": "/depth_slam/mapping/map_path",
        "route_path": "/depth_slam/mapping/route_path",
        "final_route_path": "/depth_slam/mapping/final_route_path",
        "graph_path": "/depth_slam/mapping/graph_path",
    }

    def __init__(self):
        super().__init__("mapping_session_info")
        defaults = (
            ("session_id", ""), ("map_path", ""), ("route_path", ""),
            ("final_route_path", ""), ("graph_path", ""),
            ("bag_path", ""), ("report_path", ""), ("metadata_path", ""),
            ("created_at_local", ""), ("created_at_utc", ""),
            ("timezone", ""), ("utc_offset", ""),
            ("automatic_name", False),
        )
        for name, default in defaults:
            self.declare_parameter(name, default)
        self.values = {name: self.get_parameter(name).value for name, _ in defaults}
        required = ("session_id", "map_path", "route_path", "final_route_path",
                    "graph_path", "bag_path",
                    "report_path", "metadata_path", "created_at_local",
                    "created_at_utc", "timezone", "utc_offset")
        missing = [name for name in required if not str(self.values[name])]
        if missing:
            raise ValueError("missing mapping session parameters: " + ", ".join(missing))
        if not Path(str(self.values["metadata_path"])).expanduser().is_file():
            raise FileNotFoundError(self.values["metadata_path"])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.session_publishers = {
            name: self.create_publisher(String, topic, qos)
            for name, topic in self.TOPICS.items()
        }
        self._log_session()
        self.publish()
        self.create_timer(5.0, self.publish)

    def _log_session(self):
        lines = [
            "\n=== NEW MAPPING SESSION ===",
            f"session_id: {self.values['session_id']}",
            f"map_db: {self.values['map_path']}",
            f"route: {self.values['route_path']}",
            f"final_route: {self.values['final_route_path']}",
            f"bag: {self.values['bag_path']}",
            f"report: {self.values['report_path']}",
            f"timezone: {self.values['timezone']}",
            f"utc_offset: {self.values['utc_offset']}",
            "===========================",
        ]
        self.get_logger().info("\n".join(lines))

    def publish(self):
        for name, publisher in self.session_publishers.items():
            publisher.publish(String(data=str(self.values[name])))


def main(args=None):
    rclpy.init(args=args)
    node = MappingSessionInfoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
