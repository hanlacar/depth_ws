import importlib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_node_imports_without_yolo():
    module = importlib.import_module(
        "camera_rgb_traffic_light.rgb_traffic_light_node")
    assert module.RgbTrafficLightNode


def test_shutdown_keeps_context_valid_for_final_unknown():
    source = (ROOT / "camera_rgb_traffic_light" /
              "rgb_traffic_light_node.py").read_text()
    assert "SignalHandlerOptions.NO" in source
    assert source.index("node.destroy_node()") < source.index("rclpy.shutdown()")


def test_no_control_or_existing_traffic_light_publish_contract():
    source = (ROOT / "camera_rgb_traffic_light" /
              "rgb_traffic_light_node.py").read_text()
    forbidden = ("/camera_drive", "/camera_wheel", "/mcu_drive",
                 "/mcu_wheel", "/camera_traffic_light", "ultralytics")
    assert all(value not in source for value in forbidden)


def test_launches_have_no_yolo_planner_controller_or_fake_publisher():
    text = "\n".join(path.read_text() for path in (ROOT / "launch").glob("*.py"))
    for forbidden in ("camera_yolo_inference", "planner", "controller",
                      "fake_", "video_publisher"):
        assert forbidden not in text
    assert "/camera/traffic_light_rgb/overlay_image" in text
    config = (ROOT / "config" / "rgb_traffic_light.yaml").read_text()
    assert "/camera/traffic_light_rgb/aspect" in config


def test_launch_modules_have_valid_syntax():
    for path in (ROOT / "launch").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
