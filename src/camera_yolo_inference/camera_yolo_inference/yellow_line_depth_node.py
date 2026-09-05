#!/usr/bin/env python3
"""Measure nearest yellow-line depth and report which side of the image it occupies."""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int8

from .depth_utils import depth_array


class YellowLineDepthNode(Node):
    def __init__(self):
        super().__init__("yellow_line_depth_node")
        self.declare_parameter("max_pair_age_sec", 0.35)
        self.declare_parameter("minimum_valid_pixels", 20)
        self.mask = None; self.mask_time = -1e9
        self.distance_pub = self.create_publisher(Float32, "/perception/yellow_line_distance_m", 10)
        self.valid_pub = self.create_publisher(Bool, "/perception/yellow_line_distance_valid", 10)
        self.side_pub = self.create_publisher(Int8, "/perception/yellow_line_side", 10)
        self.create_subscription(Image, "/camera/yellow_line_mask", self.on_mask, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, qos_profile_sensor_data)

    def on_mask(self, msg):
        try:
            row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            self.mask = row[:, :msg.width] > 0
            self.mask_time = time.monotonic()
        except ValueError:
            self.mask = None

    def on_depth(self, msg):
        valid = False
        try:
            if self.mask is None or time.monotonic()-self.mask_time > float(self.get_parameter("max_pair_age_sec").value):
                raise ValueError("stale mask")
            depth = depth_array(msg)
            if depth.shape != self.mask.shape:
                raise ValueError("shape mismatch")
            rows, cols = np.nonzero(self.mask & np.isfinite(depth) & (depth > 0.05))
            if len(rows) < int(self.get_parameter("minimum_valid_pixels").value):
                raise ValueError("insufficient pixels")
            values = depth[rows, cols]
            distance = float(np.percentile(values, 10.0))
            close = values <= np.percentile(values, 20.0)
            side = -1 if float(np.median(cols[close])) < msg.width/2.0 else 1
            valid = math.isfinite(distance)
            if valid:
                self.distance_pub.publish(Float32(data=distance))
                self.side_pub.publish(Int8(data=side))
        except (ValueError, TypeError):
            pass
        self.valid_pub.publish(Bool(data=valid))


def main(args=None):
    rclpy.init(args=args); node = YellowLineDepthNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
