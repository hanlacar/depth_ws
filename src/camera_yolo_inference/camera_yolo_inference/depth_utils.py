#!/usr/bin/env python3
"""Shared depth-image decoding utility (no mission logic)."""
import numpy as np


def depth_array(msg):
    if msg.encoding in ("16UC1", "mono16"):
        row = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        return row[:, :msg.width].astype(np.float32) * 0.001
    if msg.encoding == "32FC1":
        row = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return row[:, :msg.width]
    raise ValueError("unsupported depth encoding")
