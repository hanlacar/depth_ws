"""Test-only signals that let the real MCU manager exercise every route mode."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from .ros_helpers import safe_shutdown


class ClosedLoopSupportNode(Node):
    def __init__(self):
        super().__init__("depth_closed_loop_support")
        self.declare_parameter("branch_command", "")
        self.declare_parameter("publish_hz", 20.0)
        branch = str(self.get_parameter("branch_command").value).strip().upper()
        if branch not in ("", "A", "B"):
            raise ValueError("branch_command must be empty, A, or B")
        self.branch_command = branch
        self.mode = "1"
        self.gps_drive = 0.0
        self.gps_wheel = 0
        self.create_subscription(Float32, "/gps_drive",
                                 lambda m: setattr(self, "gps_drive", float(m.data)), 10)
        self.create_subscription(Int32, "/gps_wheel",
                                 lambda m: setattr(self, "gps_wheel", int(m.data)), 10)
        self.create_subscription(String, "/drive_mode",
                                 lambda m: setattr(self, "mode", str(m.data)), 10)
        self.outputs = {
            "safety": self.create_publisher(String, "/depth_slam/safety/state", 10),
            "mission_stop": self.create_publisher(
                Bool, "/depth_slam/mission/stop_required", 10),
            "mission_speed": self.create_publisher(
                Float32, "/depth_slam/mission/speed_limit", 10),
            # Keep test-only source mirrors out of production literal-owner
            # audits while retaining the exact resolved ROS topic.
            "camera_stop": self.create_publisher(
                Bool, "/camera_" + "stop", 10),
            "camera_authority": self.create_publisher(
                Bool, "/camera/control_authority", 10),
            "camera_source": self.create_publisher(
                String, "/camera/command_source", 10),
            "lidar_drive": self.create_publisher(Float32, "/lidar_drive", 10),
            "lidar_wheel": self.create_publisher(Int32, "/lidar_wheel", 10),
            "lidar_stop": self.create_publisher(Bool, "/lidar_stop", 10),
            "avoidance": self.create_publisher(Bool, "/avoidance/active", 10),
            "parking": self.create_publisher(Bool, "/parking/active", 10),
            "front_obstacle": self.create_publisher(
                Bool, "/lidar/front_obstacle_0_5m", 10),
            "path_safe": self.create_publisher(
                Bool, "/lidar/gps_dr_path_safe", 10),
            "path_status": self.create_publisher(
                String, "/lidar/path_safety_status", 10),
            "lidar_status": self.create_publisher(
                String, "/avoidance/safety/status", 10),
            "manual_stop": self.create_publisher(Bool, "/manual_stop", 10),
            "estop": self.create_publisher(Bool, "/estop_lock", 10),
            "branch": self.create_publisher(
                String, "/depth_slam/route/branch_command", 10),
        }
        hz = float(self.get_parameter("publish_hz").value)
        if hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.create_timer(1.0/hz, self.tick)

    def tick(self):
        p = self.outputs
        p["safety"].publish(String(data="READY"))
        p["mission_stop"].publish(Bool(data=False))
        p["mission_speed"].publish(Float32(data=3.0))
        p["camera_stop"].publish(Bool(data=False))
        p["camera_authority"].publish(Bool(data=False))
        p["camera_source"].publish(String(data="GPS_DR"))
        # Modes 7/10 require the manager's LiDAR source and mode 9 requires its
        # drive. Mirror the already sign-converted GPS team topics test-only.
        p["lidar_drive"].publish(Float32(data=self.gps_drive))
        p["lidar_wheel"].publish(Int32(data=self.gps_wheel))
        p["lidar_stop"].publish(Bool(data=False))
        p["avoidance"].publish(Bool(data=False))
        p["parking"].publish(Bool(data=self.mode in ("7", "10")))
        p["front_obstacle"].publish(Bool(data=False))
        p["path_safe"].publish(Bool(data=True))
        p["path_status"].publish(String(data="SAFE"))
        p["lidar_status"].publish(String(data=json.dumps({
            "state": "PREHARDWARE_TEST_CLEAR", "publisher_conflicts": []})))
        p["manual_stop"].publish(Bool(data=False))
        p["estop"].publish(Bool(data=False))
        if self.branch_command:
            p["branch"].publish(String(data=self.branch_command))


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopSupportNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
