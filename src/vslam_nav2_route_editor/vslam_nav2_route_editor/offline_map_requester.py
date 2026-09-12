"""Request one optimized full-map publication from the offline RTAB-Map node."""
import rclpy
from rclpy.node import Node
from rtabmap_msgs.srv import PublishMap


class Requester(Node):
    def __init__(self):
        super().__init__("v10_offline_map_requester")
        self.client = self.create_client(PublishMap, "/rtabmap/rtabmap/publish_map")
        self.future = None
        self.complete = False
        self.create_timer(0.5, self.request)

    def request(self):
        if self.complete or self.future is not None or not self.client.service_is_ready():
            return
        request = PublishMap.Request()
        request.global_map = True; request.optimized = True; request.graph_only = False
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.finished)

    def finished(self, future):
        try:
            future.result(); self.complete = True
            self.get_logger().info("published optimized v10 RTAB-Map graph in frame map")
        except Exception as error:
            self.get_logger().error(f"publish_map failed: {error}"); self.future = None


def main(args=None):
    rclpy.init(args=args); node = Requester()
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
