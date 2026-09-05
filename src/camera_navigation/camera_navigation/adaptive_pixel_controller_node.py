"""ROS entry point retaining the stock controller I/O and safety features."""

import rclpy

from .adaptive_pixel_controller import AdaptivePixelController, DynamicLookaheadConfig
from .camera_pixel_controller_node import CameraPixelController


class AdaptiveCameraPixelController(CameraPixelController):
    def __init__(self):
        super().__init__()
        defaults = {name: field.default for name, field in
                    DynamicLookaheadConfig.__dataclass_fields__.items()}
        for name, value in defaults.items():
            self.declare_parameter("dynamic_lookahead_"+name, value)
        dynamic = DynamicLookaheadConfig(**{
            name: self.get_parameter("dynamic_lookahead_"+name).value
            for name in defaults})
        self.controller = AdaptivePixelController(self.controller.config, dynamic)
        self.get_logger().info("dynamic non-BEV look-ahead enabled")


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveCameraPixelController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
