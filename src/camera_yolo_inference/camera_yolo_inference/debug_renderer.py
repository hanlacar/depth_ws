"""Pure OpenCV rendering for YOLO segmentation diagnostics."""

from dataclasses import dataclass
from types import MappingProxyType
import cv2
import numpy as np
from .detection_schema import model_name
from .mask_postprocessor import restore_masks_to_raw_shape


@dataclass(frozen=True)
class DebugRenderingConfig:
    overlay_alpha: float = 0.35
    draw_contours: bool = True
    draw_boxes: bool = True
    draw_labels: bool = True
    draw_confidence: bool = True
    mask_threshold: float = 0.5

    def validate(self):
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("debug_overlay_alpha must be in [0, 1]")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")


CLASS_COLORS = MappingProxyType({
    "road": (40, 170, 40), "W_line": (255, 255, 255), "Y_line": (0, 220, 255),
    "R_light": (0, 0, 255), "G_light": (0, 255, 0),
    "Left": (255, 255, 0), "etc_light": (255, 0, 255), "stop": (100, 0, 255),
    "traffic20": (255, 100, 0), "C_line": (220, 80, 180), "words": (180, 60, 180),
})
RENDER_PRIORITY = MappingProxyType({
    "road": 0, "W_line": 10, "Y_line": 10, "stop": 20, "C_line": 20,
    "words": 20, "traffic20": 30, "R_light": 40,
    "G_light": 40, "Left": 40, "etc_light": 40,
})
ROLE_TO_CLASS_NAME = MappingProxyType({
    "road": "road", "white_line": "W_line", "yellow_line": "Y_line",
    "red_light": "R_light",
    "green_light": "G_light", "left_arrow": "Left",
    "other_light": "etc_light", "stop": "stop",
    "traffic20": "traffic20", "center_line": "C_line", "words": "words",
})


class DebugRenderer:
    def __init__(self, config):
        config.validate()
        self.config = config
        self._overlay = None
        self._blended = None
        self._active = None

    def _ensure_buffers(self, bgr):
        if self._overlay is None or self._overlay.shape != bgr.shape:
            self._overlay = np.empty_like(bgr)
            self._blended = np.empty_like(bgr)
            self._active = np.empty(bgr.shape[:2], dtype=np.uint8)

    def render(self, bgr, instances, model_names, diagnostics=None,
               semantic_masks=None):
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("debug input must be a BGR image")
        output = bgr.copy()
        self._ensure_buffers(bgr)
        ordered = sorted(instances, key=lambda item: RENDER_PRIORITY.get(
            model_name(model_names, int(item["class_id"])), 25))
        rendered = []
        if semantic_masks is not None:
            render_source = []
            for role, source in semantic_masks.items():
                source = np.asarray(source)
                if not source.any():
                    continue
                # Semantic masks are canonical mono8 (0/255).  Reinterpret
                # their bytes as bool instead of allocating a full 640x480
                # comparison result for every role on every debug frame.
                mask = (source.view(np.bool_) if source.dtype == np.uint8
                        else source.astype(bool, copy=False))
                render_source.append((ROLE_TO_CLASS_NAME.get(role, role), mask))
        else:
            grouped = {}
            for item in ordered:
                name = model_name(model_names, int(item["class_id"]))
                grouped.setdefault(name, []).append(item)
            render_source = ((name, restore_masks_to_raw_shape(
                np.maximum.reduce([np.asarray(item["mask"], dtype=np.float32)
                                   for item in items]), bgr.shape[:2]) >=
                self.config.mask_threshold) for name, items in grouped.items())
        for name, mask in render_source:
            if semantic_masks is None and not np.any(mask):
                continue
            color = CLASS_COLORS.get(name, (200, 200, 50))
            rendered.append((name, color, mask))

        # At most two full-frame blends: low-alpha road, then all other classes.
        for road_pass in (True, False):
            self._active.fill(False)
            for name, color, mask in rendered:
                if (name == "road") != road_pass:
                    continue
                self._overlay[mask] = color
                self._active[mask] = 255
            if not self._active.any():
                continue
            alpha = min(self.config.overlay_alpha, 0.18) if road_pass else self.config.overlay_alpha
            cv2.addWeighted(output, 1.0 - alpha, self._overlay, alpha, 0.0,
                            dst=self._blended)
            cv2.copyTo(self._blended, self._active, output)

        if self.config.draw_contours:
            for name, color, mask in rendered:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                contour_color = (25, 25, 25) if name == "W_line" else color
                cv2.drawContours(output, contours, -1, contour_color, 2)
        for item in ordered:
            name = model_name(model_names, int(item["class_id"]))
            color = CLASS_COLORS.get(name, (200, 200, 50))
            box = item.get("xyxy", ())
            if self.config.draw_boxes and len(box) == 4:
                x1, y1, x2, y2 = (int(round(value)) for value in box)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            if self.config.draw_labels:
                confidence = f" {float(item.get('confidence', 0.0)):.2f}" if self.config.draw_confidence else ""
                label = name + confidence
                if len(box) == 4:
                    origin = (max(0, int(box[0])), max(18, int(box[1]) - 5))
                else:
                    origin = (5, 18)
                cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(output, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)
        if diagnostics:
            text = " | ".join(str(value) for value in diagnostics if value)
            cv2.putText(output, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(output, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (240, 240, 240), 1, cv2.LINE_AA)
        return output

    def render_perception(self, bgr, semantic_masks):
        """Render the compact canonical perception view on the real RGB frame."""
        output = self.render(bgr, [], {}, semantic_masks=semantic_masks)
        presence = {role: bool(np.any(mask)) for role, mask in semantic_masks.items()}
        status = (
            ("W_LINE", presence.get("white_line", False)),
            ("Y_LINE", presence.get("yellow_line", False)),
            ("STOP", presence.get("stop", False)),
        )
        for index, (name, active) in enumerate(status):
            text = f"{name}: {'YES' if active else 'NO'}"
            color = (0, 255, 0) if active else (180, 180, 180)
            origin = (8, 20 + 18 * index)
            cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)

        active_names = [ROLE_TO_CLASS_NAME.get(role, role)
                        for role, active in presence.items() if active]
        x = max(8, output.shape[1] - 132)
        for index, name in enumerate(active_names):
            y = 18 + 17 * index
            color = CLASS_COLORS.get(name, (200, 200, 50))
            cv2.rectangle(output, (x, y - 9), (x + 10, y + 1), color, -1)
            cv2.rectangle(output, (x, y - 9), (x + 10, y + 1), (25, 25, 25), 1)
            cv2.putText(output, name, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(output, name, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1, cv2.LINE_AA)
        return output
