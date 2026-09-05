"""ROS entry point selecting the adaptive non-BEV planner."""

import cv2
import numpy as np
import rclpy

from .adaptive_non_bev_planner import AdaptiveNonBevConfig, AdaptiveNonBevPlanner
from .camera_image_path_node import CameraImagePathNode


class AdaptiveNonBevNode(CameraImagePathNode):
    def __init__(self):
        super().__init__()
        defaults = {name: field.default for name, field in
                    AdaptiveNonBevConfig.__dataclass_fields__.items()}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        adaptive = AdaptiveNonBevConfig(**{
            name: self.get_parameter(name).value for name in defaults})
        self.planner = AdaptiveNonBevPlanner(self.planner.config, adaptive)
        self.get_logger().info(
            "adaptive non-BEV planner enabled (multi-band/DT/robust-fit/time-hold)")

    @staticmethod
    def _draw_path_layers(image, result):
        CameraImagePathNode._draw_path_layers(image, result)
        diagnostics = result.diagnostics or {}
        for x, y in diagnostics.get("robust_fit_rejected_points", []):
            cv2.drawMarker(image, (int(round(x)), int(round(y))), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 8, 1)
        ratio = diagnostics.get("selected_lookahead_y_ratio")
        if ratio is not None and len(result.points):
            points = np.asarray(result.points, dtype=float)
            order = np.argsort(points[:, 1])[::-1]
            index = min(len(points)-1, max(0, int(round(float(ratio)*(len(points)-1)))))
            x, y = points[order[index]]
            cv2.circle(image, (int(round(x)), int(round(y))), 7, (255, 255, 0), 2)
            cv2.putText(image, f"LA {float(ratio):.2f}",
                        (int(round(x))+8, int(round(y))-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveNonBevNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
