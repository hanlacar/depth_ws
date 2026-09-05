"""Request one full saved-map publication without depending on ros2cli daemon."""

import rclpy
from rclpy.node import Node
from rtabmap_msgs.srv import PublishMap

from .ros_helpers import safe_shutdown


class OfflineMapRequester(Node):
    def __init__(self):
        super().__init__("offline_map_requester")
        # RTAB-Map 0.22.1 creates this service relative to its namespaced node.
        self.service_name = "/rtabmap/rtabmap/publish_map"
        self.client = self.create_client(PublishMap, self.service_name)
        self.future = None
        self.complete = False
        self.timer = self.create_timer(0.5, self.request)

    def request(self):
        if self.complete or self.future is not None:
            return
        if not self.client.service_is_ready():
            self.get_logger().info(
                f"waiting for {self.service_name}", throttle_duration_sec=5.0)
            return
        request = PublishMap.Request()
        request.global_map = True
        request.optimized = True
        request.graph_only = False
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.finished)

    def finished(self, future):
        try:
            future.result()
        except Exception as error:  # service failure must stay visible
            self.get_logger().error(f"saved map publication failed: {error}")
            self.future = None
            return
        self.complete = True
        self.get_logger().info("full saved map publication requested")


def main(args=None):
    rclpy.init(args=args)
    node = OfflineMapRequester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
