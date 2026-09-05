"""Fail-closed graph checks for live D456/cuVSLAM/traffic-light viewing."""

import time


COLOR_TOPIC = "/camera/camera/color/image_raw"
CUVSLAM_TOPIC = "/depth_slam/cuvslam/odometry"
FORBIDDEN_EXACT = (
    "/camera_drive", "/camera_wheel", "/slam_drive", "/slam_wheel",
)


def live_graph_conflicts(camera_publishers, cuvslam_publishers,
                         forbidden_publishers):
    problems = []
    if int(camera_publishers) != 1:
        problems.append(f"D456_RGB_PUBLISHER_COUNT:{int(camera_publishers)}")
    if int(cuvslam_publishers) != 1:
        problems.append(
            f"CUVSLAM_ODOMETRY_PUBLISHER_COUNT:{int(cuvslam_publishers)}")
    for topic, count in sorted(forbidden_publishers.items()):
        if int(count):
            problems.append(f"FORBIDDEN_CONTROL_PUBLISHER:{topic}:{int(count)}")
    return problems


def current_live_graph(discovery_seconds=1.5):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node("traffic_light_live_start_guard", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        deadline = time.monotonic()+float(discovery_seconds)
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)

        def count(topic):
            return len(node.get_publishers_info_by_topic(topic))

        topics = {name for name, _types in node.get_topic_names_and_types()}
        forbidden = set(FORBIDDEN_EXACT)
        forbidden.update(name for name in topics if name.startswith("/mcu/cmd_"))
        return {
            "camera_publishers": count(COLOR_TOPIC),
            "cuvslam_publishers": count(CUVSLAM_TOPIC),
            "forbidden_publishers": {
                topic: count(topic) for topic in sorted(forbidden)},
        }
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def ensure_safe_live_graph(discovery_seconds=1.5):
    graph = current_live_graph(discovery_seconds)
    problems = live_graph_conflicts(
        graph["camera_publishers"], graph["cuvslam_publishers"],
        graph["forbidden_publishers"])
    if problems:
        raise RuntimeError("live traffic-light graph rejected: "+", ".join(problems))
    return graph
