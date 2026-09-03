"""Console-friendly observer for the aggregate depth SLAM diagnostics topic."""

from diagnostic_msgs.msg import DiagnosticArray
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class SlamDiagnostics(Node):
    def __init__(self):
        super().__init__('depth_slam_diagnostics_observer')
        self.create_subscription(DiagnosticArray, '/depth_slam/diagnostics',
                                 self._callback, 10)

    def _callback(self, msg):
        if msg.status:
            status = msg.status[0]
            values = ' '.join(f'{v.key}={v.value}' for v in status.values)
            self.get_logger().info(f'{status.message}: {values}')


def main(args=None):
    rclpy.init(args=args)
    node = SlamDiagnostics()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
