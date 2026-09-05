"""Standardize cuVSLAM odometry without changing its timestamp or covariance."""

from collections import deque
import math
import subprocess
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32, String, UInt32

from .ros_helpers import safe_shutdown, stamp_seconds, status


class CuvslamBridge(Node):
    def __init__(self):
        super().__init__("cuvslam_bridge")
        self.declare_parameter("input_odometry", "/visual_slam/tracking/odometry")
        self.declare_parameter("timeout_s", 0.15)
        self.declare_parameter("status_timeout_s", 0.20)
        self.declare_parameter("jump_translation_m", 0.75)
        self.declare_parameter("jump_yaw_deg", 30.0)
        self.declare_parameter("expected_parent_frame", "odom")
        self.declare_parameter("expected_child_frame", "base_link")
        self.timeout = float(self.get_parameter("timeout_s").value)
        self.status_timeout = float(self.get_parameter("status_timeout_s").value)
        self.last_receipt = None
        self.last_stamp = None
        self.last_position = None
        self.status_received = None
        self.vo_success = False
        self.previous_vo_success = None
        self.tracking_loss_count = 0
        self.status_latency_ms = None
        self.reset_count = 0
        self.intervals = deque(maxlen=180)
        self.latencies = deque(maxlen=180)
        self.pub_odom = self.create_publisher(Odometry, "/depth_slam/cuvslam/odometry", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/depth_slam/cuvslam/pose", 10)
        self.pub_status = self.create_publisher(String, "/depth_slam/cuvslam/status", 10)
        self.pub_valid = self.create_publisher(Bool, "/depth_slam/cuvslam/tracking_valid", 10)
        self.pub_conf = self.create_publisher(Float32, "/depth_slam/cuvslam/confidence", 10)
        self.pub_reset = self.create_publisher(UInt32, "/depth_slam/cuvslam/reset_count", 10)
        self.pub_latency = self.create_publisher(Float32, "/depth_slam/cuvslam/latency_ms", 10)
        self.pub_fps = self.create_publisher(Float32, "/depth_slam/cuvslam/fps", 10)
        self.pub_gpu = self.create_publisher(Float32, "/depth_slam/cuvslam/gpu_utilization", 10)
        self.pub_vram = self.create_publisher(Float32, "/depth_slam/cuvslam/gpu_memory_mb", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/depth_slam/cuvslam/diagnostics", 10)
        self.create_subscription(Odometry, str(self.get_parameter("input_odometry").value),
                                 self.on_odom, qos_profile_sensor_data)
        try:
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            self.create_subscription(VisualSlamStatus, "/visual_slam/status",
                                     self.on_status, qos_profile_sensor_data)
        except ImportError:
            self.get_logger().warning(
                "VisualSlamStatus unavailable; freshness-only tracking fallback")
        self.create_timer(1.0/30.0, self.tick)
        self.create_timer(1.0, self.update_gpu)

    def on_status(self, message):
        success = int(message.vo_state) == 1
        if self.previous_vo_success is True and not success:
            # A transient tracking loss is not a pose reset. Keep it visible in
            # tracking_valid/diagnostics while reserving reset_count for source
            # timestamp regressions and discontinuous pose jumps.
            self.tracking_loss_count += 1
        self.previous_vo_success = success
        self.vo_success = success
        self.status_received = time.monotonic()
        self.status_latency_ms = float(message.node_callback_execution_time)*1000.0

    def on_odom(self, message):
        now_wall = time.monotonic()
        stamp = stamp_seconds(message.header.stamp)
        if self.last_stamp is not None:
            delta = stamp-self.last_stamp
            if delta > 0.0:
                self.intervals.append(delta)
            elif delta < 0.0:
                self.reset_count += 1
        position = message.pose.pose.position
        if self.last_position is not None:
            jump = math.sqrt((position.x-self.last_position[0])**2 +
                             (position.y-self.last_position[1])**2 +
                             (position.z-self.last_position[2])**2)
            if jump > float(self.get_parameter("jump_translation_m").value):
                self.reset_count += 1
        self.last_position = (position.x, position.y, position.z)
        self.last_stamp = stamp
        self.last_receipt = now_wall
        ros_now = self.get_clock().now().nanoseconds*1.0e-9
        latency = max(0.0, (ros_now-stamp)*1000.0)
        self.latencies.append(latency)
        # Preserve the complete original odometry contract verbatim.
        self.pub_odom.publish(message)
        pose = PoseStamped(header=message.header, pose=message.pose.pose)
        self.pub_pose.publish(pose)

    def tick(self):
        age = float("inf") if self.last_receipt is None else time.monotonic()-self.last_receipt
        status_fresh = (self.status_received is not None and
                        time.monotonic()-self.status_received <= self.status_timeout)
        valid = (age <= self.timeout and self.last_stamp is not None and
                 (self.vo_success if self.status_received is not None else True) and
                 (status_fresh if self.status_received is not None else True))
        fps = (len(self.intervals)/sum(self.intervals)) if self.intervals and sum(self.intervals) > 0 else 0.0
        latency = (self.status_latency_ms if self.status_latency_ms is not None else
                   (sum(self.latencies)/len(self.latencies) if self.latencies else float("nan")))
        state = "TRACKING" if valid else ("INITIALIZING" if self.last_stamp is None else "STALE")
        # Confidence is deliberately fail-closed: data freshness only, not a fabricated
        # cuVSLAM likelihood. Downstream localization must also use RTAB-Map evidence.
        confidence = 1.0 if valid else 0.0
        self.pub_status.publish(String(data=state))
        self.pub_valid.publish(Bool(data=valid))
        self.pub_conf.publish(Float32(data=confidence))
        self.pub_reset.publish(UInt32(data=self.reset_count))
        self.pub_latency.publish(Float32(data=float(latency)))
        self.pub_fps.publish(Float32(data=float(fps)))
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        diag.status = [status("depth_slam/cuvslam", DiagnosticStatus.OK if valid else DiagnosticStatus.ERROR,
                              state, (("fps", fps), ("latency_ms", latency),
                                      ("source_stamp", self.last_stamp),
                                      ("reset_count", self.reset_count),
                                      ("tracking_loss_count", self.tracking_loss_count)))]
        self.pub_diag.publish(diag)

    def update_gpu(self):
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"], check=True, capture_output=True,
                text=True, timeout=0.5).stdout.splitlines()[0]
            utilization, memory = (float(item.strip()) for item in output.split(","))
            self.pub_gpu.publish(Float32(data=utilization))
            self.pub_vram.publish(Float32(data=memory))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self.pub_gpu.publish(Float32(data=float("nan")))
            self.pub_vram.publish(Float32(data=float("nan")))


def main(args=None):
    rclpy.init(args=args)
    node = CuvslamBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()
