"""ROS mono8 Image conversion without the cv_bridge NumPy ABI dependency."""
import numpy as np


def image_to_mono8(message):
    if message.encoding.lower() != "mono8":
        raise ValueError(f"unsupported mask encoding: {message.encoding}")
    width, height, step = int(message.width), int(message.height), int(message.step)
    if step < width:
        raise ValueError("mask step is shorter than one pixel row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = step * height
    if raw.size < required:
        raise ValueError("mask data is shorter than height * step")
    return np.ascontiguousarray(raw[:required].reshape(height, step)[:, :width])
