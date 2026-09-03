"""Low-overhead RGB-D/SLAM monitor with a latest-frame-only output path."""

import math
import time

import cv2
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Bool, Float32, String, UInt64

from .metrics import LatencyTracker, StampRateTracker, TrackingWatchdog


def stamp_seconds(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def quaternion_to_rpy(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return roll, pitch, math.atan2(siny, cosy)


class FpsMonitor(Node):
    def __init__(self):
        super().__init__('depth_slam_monitor')
        defaults = {
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'imu_topic': '/camera/camera/imu',
            'visual_odom_topic': '/visual_odom',
            'map_topic': '/rtabmap/mapData',
            'expected_rgb_fps': 60.0,
            'expected_depth_fps': 30.0,
            'tracking_timeout_sec': 1.0,
            'overlay_enabled': True,
            'diagnostic_period_sec': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def p(name):
            return self.get_parameter(name).value
        self._rgb = StampRateTracker(expected_fps=p('expected_rgb_fps'))
        self._depth = StampRateTracker(expected_fps=p('expected_depth_fps'))
        self._imu = StampRateTracker()
        self._vo = StampRateTracker()
        self._map = StampRateTracker()
        self._output = StampRateTracker(expected_fps=p('expected_rgb_fps'))
        self._latency = LatencyTracker()
        self._watchdog = TrackingWatchdog(p('tracking_timeout_sec'))
        self._bridge = CvBridge()
        self._pose = (0.0,) * 6
        self._overlay = bool(p('overlay_enabled'))
        self._last_rgb_stamp = None

        self._pubs = {}
        for name in ('rgb_fps', 'depth_fps', 'imu_fps', 'visual_odom_fps',
                     'map_update_fps', 'output_fps', 'latency_ms'):
            self._pubs[name] = self.create_publisher(Float32, '/depth_slam/' + name, 10)
        self._drop_pub = self.create_publisher(UInt64, '/depth_slam/dropped_frames', 10)
        self._tracking_pub = self.create_publisher(Bool, '/depth_slam/tracking_valid', 10)
        self._status_pub = self.create_publisher(String, '/depth_slam/status', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/depth_slam/diagnostics', 10)
        self._image_pub = self.create_publisher(
            Image, '/depth_slam/output_image', qos_profile_sensor_data)

        self.create_subscription(Image, p('rgb_topic'), self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic'), self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Imu, p('imu_topic'), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, p('visual_odom_topic'), self._on_odom,
                                 qos_profile_sensor_data)
        try:
            from rtabmap_msgs.msg import MapData
            self.create_subscription(
                MapData, p('map_topic'), self._on_map, qos_profile_sensor_data)
        except ImportError:
            self.get_logger().warning('rtabmap_msgs unavailable; map update FPS is disabled')
        self.create_timer(float(p('diagnostic_period_sec')), self._publish_diagnostics)

    def _on_rgb(self, msg):
        stamp = stamp_seconds(msg)
        if not self._rgb.add(stamp):
            return
        self._last_rgb_stamp = stamp
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        latency_ms = (now_ros - stamp) * 1000.0
        # Ignore invalid latency when device and ROS clocks do not share an epoch.
        if 0.0 <= latency_ms < 60000.0:
            self._latency.add(latency_ms)
        if self._image_pub.get_subscription_count() == 0:
            return
        try:
            if self._overlay:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                frame = frame.copy()
                self._draw_overlay(frame)
                out = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                out.header = msg.header
            else:
                out = msg
            self._image_pub.publish(out)
            self._output.add(stamp)
        except Exception as exc:  # keep diagnostics alive on malformed frames
            self.get_logger().error(f'output image conversion failed: {exc}')

    def _draw_overlay(self, image):
        x, y, z, roll, pitch, yaw = self._pose
        valid = self._watchdog.valid(time.monotonic())
        lines = [
            f'RGB {self._rgb.fps:.1f}  Depth {self._depth.fps:.1f}  VO {self._vo.fps:.1f} FPS',
            f'pose {x:+.2f} {y:+.2f} {z:+.2f} m',
            (f'RPY {math.degrees(roll):+.1f} {math.degrees(pitch):+.1f} '
             f'{math.degrees(yaw):+.1f} deg'),
            (f'tracking {"OK" if valid else "LOST"}  '
             f'drop {self._rgb.dropped + self._depth.dropped}'),
            f'latency avg {self._latency.average:.1f} ms p95 {self._latency.p95:.1f} ms',
        ]
        for row, line in enumerate(lines):
            cv2.putText(image, line, (8, 22 + row * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0) if valid else (0, 165, 255), 1, cv2.LINE_AA)

    def _on_depth(self, msg):
        self._depth.add(stamp_seconds(msg))

    def _on_imu(self, msg):
        self._imu.add(stamp_seconds(msg))

    def _on_odom(self, msg):
        if not self._vo.add(stamp_seconds(msg)):
            return
        pose = msg.pose.pose
        self._pose = (pose.position.x, pose.position.y, pose.position.z,
                      *quaternion_to_rpy(pose.orientation))
        self._watchdog.mark(time.monotonic())

    def _on_map(self, msg):
        stamp = stamp_seconds(msg)
        if stamp == 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self._map.add(stamp)

    def _publish_diagnostics(self):
        values = {
            'rgb_fps': self._rgb.fps,
            'depth_fps': self._depth.fps,
            'imu_fps': self._imu.fps,
            'visual_odom_fps': self._vo.fps,
            'map_update_fps': self._map.fps,
            'output_fps': self._output.fps,
            'latency_ms': self._latency.average,
        }
        for name, value in values.items():
            self._pubs[name].publish(Float32(data=float(value)))
        dropped = self._rgb.dropped + self._depth.dropped
        valid = self._watchdog.valid(time.monotonic())
        self._drop_pub.publish(UInt64(data=dropped))
        self._tracking_pub.publish(Bool(data=valid))
        status_text = ('TRACKING' if valid else
                       ('WAITING_FOR_ODOMETRY' if self._rgb.fps else
                        'WAITING_FOR_RGB'))
        self._status_pub.publish(String(data=status_text))
        level = DiagnosticStatus.OK if valid else DiagnosticStatus.WARN
        status = DiagnosticStatus(level=level, name='depth_slam', message=status_text,
                                  hardware_id='d456')
        status.values = [KeyValue(key=k, value=f'{v:.3f}') for k, v in values.items()]
        status.values.extend([
            KeyValue(key='latency_p95_ms', value=f'{self._latency.p95:.3f}'),
            KeyValue(key='dropped_frames', value=str(dropped)),
            KeyValue(key='timestamp_regressions', value=str(
                self._rgb.regressions + self._depth.regressions)),
        ])
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self._diag_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = FpsMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
