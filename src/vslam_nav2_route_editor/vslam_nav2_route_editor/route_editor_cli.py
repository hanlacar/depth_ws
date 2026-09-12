"""Command-line waypoint editor for route CSV files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

from .route_model import (
    ROUTE_NAMES, RoutePoint, VehiclePolicy, apply_vehicle_policy,
    insert_required_stops, load_route, save_route, smooth_route, straight_points,
    summary_json,
)

DEFAULT_DB = Path("/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db")


def call_live_editor(command: str, name: str) -> None:
    """Call the running editor after verifying its active route name."""
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node(f"route_editor_cli_{command}")
    state: dict[str, str] = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def metadata_callback(message: String) -> None:
        state.update(json.loads(message.data))

    node.create_subscription(String, "/nav2_route/metadata", metadata_callback, qos)
    try:
        deadline = time.monotonic() + 5.0
        while "route_name" not in state and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "route_name" not in state:
            raise RuntimeError("running route editor metadata was not received")
        if state["route_name"] != name:
            raise RuntimeError(
                f"running route is {state['route_name']}, not requested {name}")

        client = node.create_client(Trigger, f"/nav2_route/{command}")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"/nav2_route/{command} service is not available")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        response = future.result()
        if response is None:
            raise RuntimeError(f"/nav2_route/{command} did not respond")
        if not response.success:
            raise RuntimeError(response.message)
        print(response.message)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _metadata_qos():
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def read_live_state(name: str, timeout_sec: float = 5.0) -> dict:
    """Read the running editor's transient-local route state."""
    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("route_editor_cli_show")
    state: dict = {}

    def metadata_callback(message: String) -> None:
        state.update(json.loads(message.data))

    node.create_subscription(String, "/nav2_route/metadata", metadata_callback,
                             _metadata_qos())
    try:
        deadline = time.monotonic() + timeout_sec
        while "route_name" not in state and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "route_name" not in state:
            raise RuntimeError("running route editor metadata was not received")
        if state["route_name"] != name:
            raise RuntimeError(
                f"running route is {state['route_name']}, not requested {name}")
        return state
    finally:
        node.destroy_node()
        rclpy.shutdown()


def send_live_command(command: str, name: str, **payload) -> dict:
    """Send one acknowledged, in-memory route edit to the running node."""
    import rclpy
    from std_msgs.msg import String

    request_id = uuid.uuid4().hex
    rclpy.init()
    node = rclpy.create_node(f"route_editor_cli_{command.replace('-', '_')}")
    state: dict = {}

    def metadata_callback(message: String) -> None:
        state.update(json.loads(message.data))

    node.create_subscription(String, "/nav2_route/metadata", metadata_callback,
                             _metadata_qos())
    publisher = node.create_publisher(String, "/nav2_route/command", 10)
    try:
        deadline = time.monotonic() + 5.0
        while ("route_name" not in state or publisher.get_subscription_count() < 1) \
                and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "route_name" not in state:
            raise RuntimeError("running route editor metadata was not received")
        if state["route_name"] != name:
            raise RuntimeError(
                f"running route is {state['route_name']}, not requested {name}")
        if publisher.get_subscription_count() < 1:
            raise RuntimeError("/nav2_route/command has no running subscriber")

        request = {"request_id": request_id, "command": command,
                   "route_name": name, **payload}
        message = String(); message.data = json.dumps(request, separators=(",", ":"))
        next_publish = 0.0
        deadline = time.monotonic() + 5.0
        acknowledgement = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                publisher.publish(message)
                next_publish = now + 0.5
            rclpy.spin_once(node, timeout_sec=0.1)
            candidate = state.get("last_command")
            if candidate and candidate.get("request_id") == request_id:
                acknowledgement = candidate
                break
        if acknowledgement is None:
            raise RuntimeError(f"{command} acknowledgement timed out")
        if not acknowledgement.get("success"):
            raise RuntimeError(str(acknowledgement.get("message", "command failed")))
        print(acknowledgement["message"])
        return acknowledgement
    finally:
        node.destroy_node()
        rclpy.shutdown()


def show_live_route(name: str) -> None:
    state = read_live_state(name)
    invalid = set(state["validation"]["invalid_steering_indices"])
    for point in state["points"]:
        suffix = " INVALID" if point["index"] in invalid else ""
        print(
            f"index {point['index']:<3} {point['direction']:<4} "
            f"speed={point['speed']:.2f} "
            f"steering={point['required_steering_deg']:+.2f} "
            f"yaw={point['yaw_deg']:+.2f} segment={point['segment_id']} "
            f"event={point.get('event', 'NONE')}"
            f"{suffix}")
    validation = state["validation"]
    print("validation " + ("PASS" if validation["valid"] else "FAIL"))
    if validation["direction_changes_without_stop"]:
        print("missing STOP before indices: " + ",".join(
            map(str, validation["direction_changes_without_stop"])))


def show_selected(name: str) -> None:
    selected = read_live_state(name).get("selected")
    if selected is None:
        print("no waypoint selected")
        return
    print(
        f"index={selected['index']} x={selected['x']:.6f} "
        f"y={selected['y']:.6f} direction={selected['direction']} "
        f"event={selected['event']} distance={selected['distance_m']:.3f}m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def route_name(args, path: Path) -> str:
    name = getattr(args, "name", None) or path.stem
    if name not in ROUTE_NAMES:
        raise ValueError(f"route name must be one of: {', '.join(ROUTE_NAMES)}")
    return name


def persist(args, path: Path, points: list[RoutePoint]) -> None:
    points = apply_vehicle_policy(points, update_forward_yaw=False)
    metadata = save_route(path, points, route_name(args, path), VehiclePolicy(),
                          str(DEFAULT_DB), sha256(DEFAULT_DB))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("save", "persist the running in-memory route"),
        ("new", "start an empty running route without changing files"),
        ("clear", "clear the running route without changing files"),
        ("undo", "remove the final in-memory waypoint"),
    ):
        live = sub.add_parser(command, help=help_text)
        live.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    set_direction = sub.add_parser(
        "set-direction", help="set an inclusive waypoint range to F or R")
    set_direction.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    set_direction.add_argument("--start", type=int, required=True)
    set_direction.add_argument("--end", type=int, required=True)
    set_direction.add_argument("--direction", choices=("F", "R"), required=True)
    set_stop = sub.add_parser("set-stop", help="set one waypoint to STOP")
    set_stop.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    set_stop.add_argument("--index", type=int, required=True)
    show = sub.add_parser("show", help="show the running route state")
    show.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    for command, help_text in (
        ("mark-stop", "mark selected waypoint event STOP"),
        ("mark-accel", "mark selected waypoint event ACCEL"),
        ("clear-event", "clear selected waypoint event"),
        ("selected", "show selected waypoint"),
    ):
        live = sub.add_parser(command, help=help_text)
        live.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    select = sub.add_parser("select", help="select nearest waypoint to map x/y")
    select.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    select.add_argument("--x", type=float, required=True)
    select.add_argument("--y", type=float, required=True)
    indices = sub.add_parser("show-indices", help="show/hide every RViz index label")
    indices.add_argument("--name", default="START_A", choices=ROUTE_NAMES)
    indices.add_argument("--enabled", choices=("true", "false"), required=True)
    create = sub.add_parser("create")
    create.add_argument("path", type=Path); create.add_argument("--name", required=True, choices=ROUTE_NAMES)
    line = sub.add_parser("line")
    line.add_argument("path", type=Path); line.add_argument("x0", type=float); line.add_argument("y0", type=float)
    line.add_argument("x1", type=float); line.add_argument("y1", type=float)
    line.add_argument("--name", required=True, choices=ROUTE_NAMES); line.add_argument("--spacing", type=float, default=.20)
    line.add_argument("--direction", choices=("F", "R"), default="F")
    line.add_argument("--vehicle-yaw-deg", type=float); line.add_argument("--append", action="store_true")
    add = sub.add_parser("add")
    add.add_argument("path", type=Path); add.add_argument("x", type=float); add.add_argument("y", type=float)
    add.add_argument("yaw_deg", type=float); add.add_argument("direction", choices=("F", "R"))
    add.add_argument("--event", choices=("NONE", "STOP", "ACCEL"), default="NONE")
    add.add_argument("--at", type=int); add.add_argument("--name", choices=ROUTE_NAMES)
    delete = sub.add_parser("delete")
    delete.add_argument("path", type=Path); delete.add_argument("indices", nargs="+", type=int); delete.add_argument("--name", choices=ROUTE_NAMES)
    move = sub.add_parser("move")
    move.add_argument("path", type=Path); move.add_argument("index", type=int); move.add_argument("x", type=float); move.add_argument("y", type=float)
    move.add_argument("--yaw-deg", type=float); move.add_argument("--name", choices=ROUTE_NAMES)
    many = sub.add_parser("move-many")
    many.add_argument("path", type=Path); many.add_argument("indices", nargs="+", type=int)
    many.add_argument("--dx", type=float, required=True); many.add_argument("--dy", type=float, required=True); many.add_argument("--name", choices=ROUTE_NAMES)
    reorder = sub.add_parser("reorder")
    reorder.add_argument("path", type=Path); reorder.add_argument("old_index", type=int); reorder.add_argument("new_index", type=int); reorder.add_argument("--name", choices=ROUTE_NAMES)
    direction = sub.add_parser("direction")
    direction.add_argument("path", type=Path); direction.add_argument("start", type=int); direction.add_argument("end", type=int)
    direction.add_argument("value", choices=("F", "R", "STOP")); direction.add_argument("--name", choices=ROUTE_NAMES)
    direction.add_argument("--insert-stops", action="store_true")
    smooth = sub.add_parser("smooth")
    smooth.add_argument("source", type=Path); smooth.add_argument("output", type=Path)
    smooth.add_argument("--iterations", type=int, default=2); smooth.add_argument("--name", required=True, choices=ROUTE_NAMES)
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    return ap


def _main() -> None:
    args = parser().parse_args()
    if args.command in {"save", "new", "clear", "undo"}:
        service = "delete_last" if args.command == "undo" else args.command
        call_live_editor(service, args.name)
    elif args.command == "set-direction":
        send_live_command("set-direction", args.name, start=args.start,
                          end=args.end, direction=args.direction)
    elif args.command == "set-stop":
        send_live_command("set-stop", args.name, index=args.index)
    elif args.command == "show":
        show_live_route(args.name)
    elif args.command in {"mark-stop", "mark-accel", "clear-event"}:
        send_live_command(args.command, args.name)
    elif args.command == "selected":
        show_selected(args.name)
    elif args.command == "select":
        send_live_command("select-nearest", args.name, x=args.x, y=args.y)
    elif args.command == "show-indices":
        send_live_command("show-indices", args.name,
                          enabled=args.enabled == "true")
    elif args.command in {"mark-stop", "mark-accel", "clear-event"}:
        send_live_command(args.command, args.name)
    elif args.command == "selected":
        show_selected(args.name)
    elif args.command == "select":
        send_live_command("select-nearest", args.name, x=args.x, y=args.y)
    elif args.command == "show-indices":
        send_live_command("show-indices", args.name,
                          enabled=args.enabled == "true")
    elif args.command == "create":
        persist(args, args.path, [])
    elif args.command == "line":
        points = load_route(args.path) if args.append and args.path.exists() else []
        segment = max((point.segment_id for point in points), default=0) + 1
        generated = straight_points(args.x0, args.y0, args.x1, args.y1,
                                    args.spacing, args.direction, segment,
                                    args.vehicle_yaw_deg)
        if points and points[-1].direction != generated[0].direction:
            anchor = points[-1]
            points.append(RoutePoint(0, segment, anchor.x, anchor.y,
                                     anchor.yaw_deg, anchor.direction,
                                     0.0, 0.0, "STOP"))
            segment += 1
            for point in generated:
                point.segment_id = segment
        points.extend(generated)
        persist(args, args.path, points)
    elif args.command == "validate":
        points = load_route(args.path)
        print(summary_json(points))
        raise SystemExit(0 if json.loads(summary_json(points))["valid"] else 2)
    elif args.command == "smooth":
        points = smooth_route(load_route(args.source), args.iterations)
        persist(args, args.output, points)
    else:
        points = load_route(args.path)
        if args.command == "add":
            point = RoutePoint(0, max((p.segment_id for p in points), default=1),
                               args.x, args.y, args.yaw_deg, args.direction,
                               event=args.event)
            points.insert(len(points) if args.at is None else args.at, point)
        elif args.command == "delete":
            remove = set(args.indices); points = [p for i, p in enumerate(points) if i not in remove]
        elif args.command == "move":
            points[args.index].x = args.x; points[args.index].y = args.y
            if args.yaw_deg is not None: points[args.index].yaw_deg = args.yaw_deg
        elif args.command == "move-many":
            for index in args.indices:
                points[index].x += args.dx; points[index].y += args.dy
        elif args.command == "reorder":
            points.insert(args.new_index, points.pop(args.old_index))
        elif args.command == "direction":
            for index in range(args.start, args.end + 1):
                if args.value == "STOP":
                    points[index].event = "STOP"
                else:
                    points[index].direction = args.value
            if args.insert_stops:
                points = insert_required_stops(points)
        persist(args, args.path, points)


def main() -> None:
    try:
        _main()
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
