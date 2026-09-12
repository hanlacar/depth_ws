"""Small terminal client for the map-route recorder services."""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from .ros_helpers import safe_shutdown


class RecorderKeyboard(Node):
    def __init__(self):
        super().__init__("map_route_recorder_keyboard")
        root = "/depth_slam/map_route_record"
        self.clients = {name: self.create_client(Trigger, root+"/"+service)
                        for name, service in (
                            ("s", "start"), ("pause", "pause"),
                            ("resume", "resume"), ("f", "set_forward"),
                            ("r", "set_reverse"), ("q", "finish"),
                            ("?", "status"))}
        self.paused = False

    def call(self, key):
        name = key
        if key == "p":
            name = "resume" if self.paused else "pause"
        client = self.clients.get(name)
        if client is None:
            return False, "unknown key"
        if not client.wait_for_service(timeout_sec=1.0):
            return False, f"service unavailable for {key}"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        response = future.result()
        if response is None:
            return False, f"service timeout for {key}"
        if key == "p" and response.success:
            self.paused = not self.paused
        if key == "s" and response.success:
            self.paused = False
        return response.success, response.message


def main(args=None):
    if not sys.stdin.isatty():
        raise RuntimeError("map_route_keyboard requires an interactive terminal")
    rclpy.init(args=args)
    node = RecorderKeyboard()
    original = termios.tcgetattr(sys.stdin)
    print("keys: s=start, p=pause/resume, F=forward, R=reverse, ?=status, q=finish/save")
    try:
        tty.setcbreak(sys.stdin.fileno())
        done = False
        while rclpy.ok() and not done:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            key = sys.stdin.read(1)
            normalized = key.lower()
            if normalized not in ("s", "p", "f", "r", "q", "?"):
                continue
            success, message = node.call(normalized)
            print(f"\n{'OK' if success else 'REJECTED'}: {message}")
            done = normalized == "q" and success
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
