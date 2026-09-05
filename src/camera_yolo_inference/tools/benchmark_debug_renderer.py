#!/usr/bin/env python3
"""Micro-benchmark debug rendering options at the production 640x480 shape."""

import argparse
import time

import cv2
import numpy as np

from camera_yolo_inference.debug_renderer import (
    DebugRenderer, DebugRenderingConfig)


def sample_frame():
    image = np.full((480, 640, 3), (62, 64, 66), np.uint8)
    masks = {name: np.zeros((480, 640), np.uint8) for name in (
        "road", "white_line", "yellow_line", "red_light",
        "green_light", "left_sign", "other_light", "stop_line",
        "speed_20_sign", "c_line", "words")}
    cv2.fillPoly(masks["road"], [np.array(
        [[30, 475], [160, 220], [480, 220], [620, 475]], np.int32)], 255)
    cv2.line(masks["yellow_line"], (40, 430), (230, 235), 255, 8)
    cv2.line(masks["white_line"], (600, 430), (420, 235), 255, 8)
    instances = [
        {"class_id": 0, "confidence": 0.91, "xyxy": [30, 220, 620, 475]},
        {"class_id": 1, "confidence": 0.78, "xyxy": [420, 235, 600, 430]},
        {"class_id": 2, "confidence": 0.82, "xyxy": [40, 235, 230, 430]},
    ]
    names = {0: "road", 1: "W_line", 2: "Y_line"}
    return image, masks, instances, names


def measure(config, masks, iterations):
    image, _, instances, names = sample_frame()
    renderer = DebugRenderer(config)
    for _ in range(20):
        renderer.render(image, instances, names, semantic_masks=masks)
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        renderer.render(image, instances, names, semantic_masks=masks)
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values)
    return float(array.mean()), float(np.percentile(array, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    _, masks, _, _ = sample_frame()
    optional_masks = {name: mask.copy() for name, mask in masks.items()}
    cv2.circle(optional_masks["stop_line"], (320, 300), 30, 255, -1)
    cases = {
        "masks+boxes+labels+confidence": DebugRenderingConfig(
            draw_contours=False, draw_boxes=True, draw_labels=True,
            draw_confidence=True),
        "masks+contours": DebugRenderingConfig(
            draw_contours=True, draw_boxes=False, draw_labels=False,
            draw_confidence=False),
        "masks+boxes": DebugRenderingConfig(
            draw_contours=False, draw_boxes=True, draw_labels=False,
            draw_confidence=False),
        "masks+labels": DebugRenderingConfig(
            draw_contours=False, draw_boxes=False, draw_labels=True,
            draw_confidence=False),
        "masks_only": DebugRenderingConfig(
            draw_contours=False, draw_boxes=False, draw_labels=False,
            draw_confidence=False),
    }
    for name, config in cases.items():
        selected = optional_masks if name == "masks_only+optional" else masks
        mean, p95 = measure(config, selected, args.iterations)
        print(f"{name}: mean={mean:.3f} ms p95={p95:.3f} ms")
    mean, p95 = measure(cases["masks_only"], optional_masks, args.iterations)
    print(f"masks_only+optional: mean={mean:.3f} ms p95={p95:.3f} ms")


if __name__ == "__main__":
    main()
