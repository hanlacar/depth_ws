"""Request one full saved-map publication without depending on ros2cli daemon."""

import time

from nav_msgs.msg import OccupancyGrid
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
        self.declare_parameter("startup_timeout_s", 15.0)
        self.declare_parameter("map_delivery_timeout_s", 900.0)
        self.startup_timeout = float(
            self.get_parameter("startup_timeout_s").value)
        self.map_delivery_timeout = float(
            self.get_parameter("map_delivery_timeout_s").value)
        if self.startup_timeout <= 0.0 or self.map_delivery_timeout <= 0.0:
            raise ValueError("startup and map delivery timeouts must be positive")
        self.started_at = time.monotonic()
        self.future = None
        self.complete = False
        self.service_complete = False
        self.map_received = False
        self.failure_message = ""
        self.map_frame = ""
        self.create_subscription(OccupancyGrid, "/rtabmap/map", self.map, 10)
        self.timer = self.create_timer(0.25, self.request)

    def map(self, message):
        self.map_frame = message.header.frame_id
        if message.header.frame_id == "map" and message.info.width > 0:
            self.map_received = True
            self._finish_if_ready()

    def _finish_if_ready(self):
        if self.service_complete and self.map_received and not self.complete:
            self.complete = True
            self.get_logger().info(
                "RTAB-Map startup verified: service response and /rtabmap/map")

    def request(self):
        if self.complete:
            return
        elapsed = time.monotonic() - self.started_at
        service_ready = self.client.service_is_ready()
        map_topic_ready = bool(self.get_publishers_info_by_topic("/rtabmap/map"))
        if elapsed >= self.startup_timeout and not (
                service_ready and map_topic_ready):
            missing = []
            if not service_ready:
                missing.append(self.service_name)
            if not map_topic_ready:
                missing.append("/rtabmap/map")
            self.fail("RTABMAP_START_FAILED missing=" + ",".join(missing))
            return
        if elapsed >= self.map_delivery_timeout:
            detail = (f"frame={self.map_frame}" if self.map_frame else
                      "no valid map payload")
            self.fail("RTABMAP_START_FAILED " + detail)
            return
        if self.future is not None:
            return
        if not service_ready or not map_topic_ready:
            self.get_logger().info(
                f"waiting for {self.service_name} and /rtabmap/map publisher",
                throttle_duration_sec=5.0)
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
            self.fail(f"RTABMAP_START_FAILED service_error={error}")
            return
        self.service_complete = True
        self.get_logger().info("full saved map publication requested")
        self._finish_if_ready()

    def fail(self, message):
        if self.failure_message:
            return
        self.failure_message = message
        self.get_logger().fatal(message)
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OfflineMapRequester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        failure = node.failure_message
        node.destroy_node()
        if rclpy.ok():
            safe_shutdown()
    if failure:
        raise RuntimeError(failure)
