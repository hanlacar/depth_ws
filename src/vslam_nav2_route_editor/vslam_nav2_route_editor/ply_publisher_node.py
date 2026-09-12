"""Publish the saved v10 PLY in map coordinates with transient-local QoS."""
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

from .ply_map_export import load_ply


class PlyPublisher(Node):
    def __init__(self):
        super().__init__("v10_ply_publisher")
        self.declare_parameter("ply_path", "")
        self.declare_parameter("topic", "/vslam_map/cloud")
        path = Path(str(self.get_parameter("ply_path").value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        source = load_ply(path)
        dtype = np.dtype({"names": ["x", "y", "z", "rgb"],
                          "formats": ["<f4", "<f4", "<f4", "<u4"],
                          "offsets": [0, 4, 8, 12], "itemsize": 16})
        self.points = np.empty(len(source), dtype=dtype)
        self.points["x"] = source["x"]; self.points["y"] = source["y"]
        self.points["z"] = source["z"]
        self.points["rgb"] = ((source["r"].astype(np.uint32) << 16) |
                              (source["g"].astype(np.uint32) << 8) |
                              source["b"].astype(np.uint32))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            PointCloud2, str(self.get_parameter("topic").value), qos)
        self.sent = False
        self.create_timer(1.0, self.publish)

    def publish(self):
        message = PointCloud2()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.height = 1; message.width = len(self.points)
        message.is_bigendian = False; message.is_dense = False
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        message.point_step = 16; message.row_step = 16 * len(self.points)
        message.data = self.points.tobytes()
        self.publisher.publish(message)
        if not self.sent:
            self.get_logger().info(f"published {len(self.points)} PLY points in frame map")
            self.sent = True


def main(args=None):
    rclpy.init(args=args); node = PlyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
