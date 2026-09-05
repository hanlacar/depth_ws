"""Fail-closed graph audit for the camera/VSLAM visualization launch."""

import time


CAMERA_TOPIC = "/camera/camera/color/image_raw"
REQUIRED_TOPICS = (
    "/depth_slam/cuvslam/odometry",
    "/rtabmap/mapGraph",
)
FORBIDDEN_CONTROL_TOPICS = (
    "/camera_drive", "/camera_wheel",
    "/slam_drive", "/slam_wheel",
)


def integration_graph_conflicts(camera_publishers, required_publishers,
                                forbidden_publishers):
    problems = []
    if int(camera_publishers) != 1:
        problems.append(f"D456_RGB_PUBLISHER_COUNT:{int(camera_publishers)}")
    for topic in REQUIRED_TOPICS:
        if int(required_publishers.get(topic, 0)) < 1:
            problems.append("REQUIRED_TOPIC_MISSING:"+topic)
    for topic, value in forbidden_publishers.items():
        count = int(value)
        if count:
            problems.append(f"FORBIDDEN_CONTROL_PUBLISHER:{topic}:{count}")
    return problems


def current_integration_graph(discovery_seconds=1.5):
    """Inspect publishers without creating or changing any external state."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node("perception_vslam_start_guard", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        deadline = time.monotonic()+float(discovery_seconds)
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)

        def count(topic):
            return len(node.get_publishers_info_by_topic(topic))
        topic_names = {name for name, _types in node.get_topic_names_and_types()}
        forbidden = set(FORBIDDEN_CONTROL_TOPICS)
        forbidden.update(name for name in topic_names
                         if name.startswith("/mcu/cmd_"))
        return {
            "camera_publishers": count(CAMERA_TOPIC),
            "required_publishers": {
                topic: count(topic) for topic in REQUIRED_TOPICS},
            "forbidden_publishers": {
                topic: count(topic) for topic in sorted(forbidden)},
        }
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def ensure_safe_integration_graph(discovery_seconds=1.5):
    graph = current_integration_graph(discovery_seconds)
    problems = integration_graph_conflicts(
        graph["camera_publishers"], graph["required_publishers"],
        graph["forbidden_publishers"])
    if problems:
        raise RuntimeError("perception/VSLAM graph rejected: "+", ".join(problems))
    return graph
