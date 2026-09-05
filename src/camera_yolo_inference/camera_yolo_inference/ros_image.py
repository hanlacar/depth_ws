"""Small ROS Image/NumPy adapter that does not depend on cv_bridge.

ROS Jazzy's binary cv_bridge may be built against NumPy 1.x.  Keeping the
adapter in Python avoids an ABI crash when an inference environment supplies
NumPy 2.x.
"""
import numpy as np
from sensor_msgs.msg import Image


def image_to_bgr8(message):
    encoding = message.encoding.lower()
    if encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"unsupported image encoding: {message.encoding}")
    row_bytes = int(message.width) * 3
    if int(message.step) < row_bytes:
        raise ValueError("image step is shorter than one pixel row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = int(message.step) * int(message.height)
    if raw.size < required:
        raise ValueError("image data is shorter than height * step")
    image = raw[:required].reshape(int(message.height), int(message.step))[:, :row_bytes]
    image = image.reshape(int(message.height), int(message.width), 3)
    if encoding == "rgb8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def mono8_to_image(array, header):
    mask = np.ascontiguousarray(array, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mono8 output must be a two-dimensional array")
    message = Image()
    message.header = header
    message.height, message.width = map(int, mask.shape)
    message.encoding = "mono8"
    message.is_bigendian = False
    message.step = int(message.width)
    message.data = mask.tobytes()
    return message


def bgr8_to_image(array, header):
    image = np.ascontiguousarray(array, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("bgr8 output must have shape (height, width, 3)")
    message = Image()
    message.header = header
    message.height, message.width = map(int, image.shape[:2])
    message.encoding = "bgr8"
    message.is_bigendian = False
    message.step = int(message.width) * 3
    message.data = image.tobytes()
    return message
