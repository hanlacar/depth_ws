"""Interactive terminal helper for the RViz STOP editor."""

import select
import sys
import termios
import tty
from itertools import product

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .ros_helpers import safe_shutdown


class StopEditorKeyboard(Node):
    def __init__(self):
        super().__init__("stop_editor_keyboard")
        root = "/depth_slam/stop_editor"
        self.mode_pub = self.create_publisher(String, root+"/mode", 10)
        self.case_pub = self.create_publisher(String, root+"/case", 10)
        self.service_clients = {
            key: self.create_client(Trigger, root+"/"+service)
            for key, service in (("u", "undo"), ("s", "save"),
                                 ("l", "list"), ("?", "status_request"))}

    def select_case(self, value):
        cases = ["".join(items) for items in product("AB", repeat=4)]
        text = str(value).strip().upper()
        if text.isdigit():
            number = int(text)
            text = "ALL" if number == 0 else (
                cases[number - 1] if 1 <= number <= len(cases) else "")
        if text != "ALL" and text not in cases:
            return False, "case must be ALL/0, AAAA..BBBB, or 1..16"
        self.case_pub.publish(String(data=text))
        rclpy.spin_once(self, timeout_sec=0.15)
        return True, "case=" + text

    def command(self, key):
        if key in ("a", "r"):
            mode = "ADD" if key == "a" else "REMOVE"
            self.mode_pub.publish(String(data=mode))
            rclpy.spin_once(self, timeout_sec=0.15)
            return True, "mode="+mode
        client = self.service_clients.get(key)
        if client is None:
            return False, "unknown key"
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "STOP editor service unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        response = future.result()
        if response is None:
            return False, "service timeout"
        return response.success, response.message


def main(args=None):
    if not sys.stdin.isatty():
        raise RuntimeError("stop_editor_keyboard requires an interactive terminal")
    rclpy.init(args=args)
    node = StopEditorKeyboard()
    original = termios.tcgetattr(sys.stdin)
    print("keys: a=ADD, r=REMOVE, u=undo, s=save, l=list, c=case, ?=status, q=quit")
    try:
        tty.setcbreak(sys.stdin.fileno())
        done = False
        while rclpy.ok() and not done:
            rclpy.spin_once(node, timeout_sec=0.05)
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                done = True
                continue
            if key == "c":
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
                print("\ncase (ALL/0, AAAA..BBBB, or 1..16): ", end="", flush=True)
                value = sys.stdin.readline()
                success, message = node.select_case(value)
                tty.setcbreak(sys.stdin.fileno())
                print(f"{'OK' if success else 'REJECTED'}: {message}")
                continue
            if key not in ("a", "r", "u", "s", "l", "?"):
                continue
            success, message = node.command(key)
            print(f"\n{'OK' if success else 'REJECTED'}: {message}")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        node.destroy_node()
        safe_shutdown()


if __name__ == "__main__":
    main()
