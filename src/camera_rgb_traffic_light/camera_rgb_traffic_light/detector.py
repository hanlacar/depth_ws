"""ROS-independent HSV/Lab/luminance traffic-light detector and tracker."""

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


STATES = ("R", "G", "UNKNOWN")
ASPECTS = ("RED", "RED_X", "YELLOW", "GREEN_CIRCLE", "GREEN_LEFT",
           "GREEN_DOWN", "GREEN_OTHER", "UNKNOWN")
GREEN_SHAPES = (
    "CIRCLE", "LEFT_ARROW", "DOWN_ARROW", "OTHER_GREEN_SHAPE",
    "UNKNOWN_SHAPE")


@dataclass(frozen=True)
class DetectorConfig:
    roi_x_min_ratio: float = 0.05
    roi_x_max_ratio: float = 0.95
    roi_y_min_ratio: float = 0.02
    roi_y_max_ratio: float = 0.55
    minimum_area_ratio: float = 0.00003
    maximum_area_ratio: float = 0.012
    minimum_aspect_ratio: float = 0.25
    maximum_aspect_ratio: float = 4.0
    minimum_solidity: float = 0.55
    minimum_convexity: float = 0.72
    round_minimum_score: float = 0.80
    left_minimum_score: float = 0.65
    down_minimum_score: float = 0.65
    left_direction_margin: float = 0.12
    other_green_minimum_score: float = 0.58
    minimum_confidence: float = 0.55
    conflict_margin: float = 0.12
    red_priority_confidence: float = 0.82
    hsv_minimum_saturation: int = 90
    hsv_minimum_value: int = 130
    red_hue_low_max: int = 12
    red_hue_high_min: int = 168
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    green_hue_min: int = 40
    green_hue_max: int = 95
    red_lab_a_min: int = 145
    yellow_lab_b_min: int = 145
    green_lab_a_max: int = 140
    minimum_brightness_delta: float = 18.0
    minimum_bright_pixel_ratio: float = 0.45
    housing_expand_ratio: float = 0.55
    housing_dark_luma_max: int = 80
    minimum_housing_dark_ratio: float = 0.08
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 3
    reject_roi_boundary_contact: bool = True
    red_x_min_diagonal_score: float = 0.62
    red_x_angle_tolerance_deg: float = 18.0
    red_x_center_tolerance_ratio: float = 0.32
    red_x_min_diagonal_balance: float = 0.55
    red_x_max_circle_score: float = 0.76
    red_x_max_rectangularity: float = 0.72
    red_x_hough_threshold: int = 8
    red_x_min_line_length_ratio: float = 0.45
    red_x_max_line_gap_ratio: float = 0.18

    def validate(self):
        ratios = (self.roi_x_min_ratio, self.roi_x_max_ratio,
                  self.roi_y_min_ratio, self.roi_y_max_ratio)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in ratios):
            raise ValueError("ROI ratios must be finite and in [0, 1]")
        if not (self.roi_x_min_ratio < self.roi_x_max_ratio and
                self.roi_y_min_ratio < self.roi_y_max_ratio):
            raise ValueError("ROI minimums must be below maximums")
        if not 0.0 < self.minimum_area_ratio < self.maximum_area_ratio < 1.0:
            raise ValueError("area ratios are invalid")
        for name in ("minimum_solidity", "minimum_convexity",
                     "round_minimum_score", "left_minimum_score",
                     "down_minimum_score", "other_green_minimum_score",
                     "minimum_confidence", "red_priority_confidence",
                     "minimum_bright_pixel_ratio",
                     "minimum_housing_dark_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("red_x_min_diagonal_score", "red_x_min_diagonal_balance",
                     "red_x_max_circle_score", "red_x_max_rectangularity"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("morphology_open_kernel", "morphology_close_kernel"):
            value = int(getattr(self, name))
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        return self


@dataclass(frozen=True)
class Candidate:
    state: str
    color: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    circularity: float
    solidity: float
    convexity: float
    circle_score: float
    left_score: float
    right_score: float
    brightness_delta: float
    bright_ratio: float
    housing_dark_ratio: float
    hu_moments: Tuple[float, ...] = field(default_factory=tuple)
    raw_shape: str = "UNKNOWN_SHAPE"
    down_score: float = 0.0
    green_shape_score: float = 0.0
    red_x_score: float = 0.0


@dataclass(frozen=True)
class DetectionResult:
    raw_state: str
    confidence: float
    candidates: Tuple[Candidate, ...]
    selected: Optional[Candidate]
    conflict: bool
    rejection_reasons: Dict[str, int]
    roi: Tuple[int, int, int, int]

    @property
    def raw_aspect(self):
        return candidate_aspect(self.selected) if not self.conflict else "UNKNOWN"


def candidate_aspect(candidate):
    if candidate is None:
        return "UNKNOWN"
    if candidate.color == "red":
        return "RED_X" if candidate.raw_shape == "RED_X" else "RED"
    if candidate.color == "yellow":
        return "YELLOW"
    return {"CIRCLE": "GREEN_CIRCLE", "LEFT_ARROW": "GREEN_LEFT",
            "DOWN_ARROW": "GREEN_DOWN",
            "OTHER_GREEN_SHAPE": "GREEN_OTHER"}.get(
                candidate.raw_shape, "UNKNOWN")


def normalize_to_bgr(image: np.ndarray, encoding: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image is empty")
    name = str(encoding).lower()
    if name in ("bgr8", "8uc3") and image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)
    if name == "rgb8" and image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if name == "bgra8" and image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if name == "rgba8" and image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if name in ("mono8", "8uc1") and image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding/shape: {encoding} {image.shape}")


def _bounded(value):
    return float(max(0.0, min(1.0, value)))


class ColorTrafficLightDetector:
    COLORS = {"red": (0, 0, 255), "yellow": (0, 255, 255),
              "green": (0, 255, 0)}

    def __init__(self, config=DetectorConfig()):
        self.config = config.validate()

    def roi_bounds(self, shape):
        if len(shape) < 2 or shape[0] < 2 or shape[1] < 2:
            raise ValueError("invalid image dimensions")
        height, width = int(shape[0]), int(shape[1])
        x0 = int(round(width*self.config.roi_x_min_ratio))
        x1 = int(round(width*self.config.roi_x_max_ratio))
        y0 = int(round(height*self.config.roi_y_min_ratio))
        y1 = int(round(height*self.config.roi_y_max_ratio))
        return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))

    def _color_masks(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        h, s, v = cv2.split(hsv)
        _l, a, b = cv2.split(lab)
        bright = ((s >= self.config.hsv_minimum_saturation) &
                  (v >= self.config.hsv_minimum_value))
        masks = {
            "red": (bright & ((h <= self.config.red_hue_low_max) |
                               (h >= self.config.red_hue_high_min)) &
                    (a >= self.config.red_lab_a_min)),
            "yellow": bright & (h >= self.config.yellow_hue_min) &
                      (h <= self.config.yellow_hue_max) &
                      (b >= self.config.yellow_lab_b_min),
            "green": bright & (h >= self.config.green_hue_min) &
                     (h <= self.config.green_hue_max) &
                     (a <= self.config.green_lab_a_max),
        }
        opened = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.morphology_open_kernel,)*2)
        closed = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.morphology_close_kernel,)*2)
        return {
            name: cv2.morphologyEx(cv2.morphologyEx(
                mask.astype(np.uint8)*255, cv2.MORPH_OPEN, opened),
                cv2.MORPH_CLOSE, closed)
            for name, mask in masks.items()
        }, v

    @staticmethod
    def _shape_features(contour, bbox):
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        hull_perimeter = float(cv2.arcLength(hull, True))
        circularity = 4.0*math.pi*area/max(perimeter*perimeter, 1.0e-9)
        solidity = area/max(hull_area, 1.0e-9)
        convexity = hull_perimeter/max(perimeter, 1.0e-9)
        approx = cv2.approxPolyDP(contour, 0.025*perimeter, True)
        _x, _y, width, height = bbox
        aspect = width/max(height, 1)
        vertex_score = _bounded((len(approx)-4)/4.0)
        aspect_score = math.exp(-1.6*abs(math.log(max(aspect, 1.0e-6))))
        circle_score = _bounded(
            0.50*circularity+0.20*solidity+0.15*aspect_score+
            0.15*vertex_score)
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu = tuple(float(-math.copysign(1.0, value)*math.log10(abs(value)))
                   if abs(value) > 1.0e-30 else 0.0 for value in hu)
        return (area, circularity, solidity, convexity, aspect,
                circle_score, hu)

    @staticmethod
    def _horizontal_arrow_scores(component):
        ys, xs = np.nonzero(component)
        if len(xs) < 5:
            return 0.0, 0.0
        width, height = component.shape[1], component.shape[0]
        aspect = width/max(height, 1)
        if aspect < 1.15:
            return 0.0, 0.0
        spans = np.zeros(width, dtype=float)
        for column in range(width):
            rows = np.flatnonzero(component[:, column])
            spans[column] = 0.0 if len(rows) == 0 else rows[-1]-rows[0]+1
        bins = [float(np.max(chunk, initial=0.0))/max(height, 1)
                for chunk in np.array_split(spans, 5)]
        left_head = 0.55*bins[1]+0.45*bins[0]
        right_head = 0.55*bins[3]+0.45*bins[4]
        left_tail = 0.5*(bins[3]+bins[4])
        right_tail = 0.5*(bins[0]+bins[1])
        peak = float(np.argmax(spans))/max(width-1, 1)
        center_y = (height-1)/2.0
        left_rows = ys[xs <= max(1, int(0.12*width))]
        right_rows = ys[xs >= min(width-2, int(0.88*width))]
        left_tip_center = (1.0 if len(left_rows) == 0 else
                           1.0-abs(float(np.mean(left_rows))-center_y)/max(center_y, 1.0))
        right_tip_center = (1.0 if len(right_rows) == 0 else
                            1.0-abs(float(np.mean(right_rows))-center_y)/max(center_y, 1.0))
        horizontal = _bounded((aspect-1.10)/1.20)
        left = _bounded(0.35*horizontal+0.35*_bounded(
            0.5+(left_head-left_tail))+0.20*_bounded((0.58-peak)/0.35)+
            0.10*left_tip_center)
        right = _bounded(0.35*horizontal+0.35*_bounded(
            0.5+(right_head-right_tail))+0.20*_bounded((peak-0.42)/0.35)+
            0.10*right_tip_center)
        return left, right

    @classmethod
    def _arrow_scores(cls, component):
        left, right = cls._horizontal_arrow_scores(component)
        # A downward arrow becomes a left arrow after a clockwise rotation.
        down, _up = cls._horizontal_arrow_scores(np.rot90(component, k=3))
        return left, right, down

    @staticmethod
    def _green_shape(component, circle_score, left_score, right_score,
                     down_score, rectangularity, config):
        if circle_score >= config.round_minimum_score:
            return "CIRCLE", circle_score
        directional = max(left_score, right_score, down_score)
        if (left_score >= config.left_minimum_score and
                left_score-right_score >= config.left_direction_margin):
            return "LEFT_ARROW", left_score
        if down_score >= config.down_minimum_score:
            return "DOWN_ARROW", down_score
        if right_score >= config.left_minimum_score:
            return "OTHER_GREEN_SHAPE", right_score
        # Preserve uncertain arrow-like lamps without admitting rectangular
        # signs.  Housing, brightness, area, and temporal checks still apply.
        non_rectangular = _bounded((0.90-rectangularity)/0.35)
        score = _bounded(0.55*directional+0.45*non_rectangular)
        if score >= config.other_green_minimum_score:
            return "OTHER_GREEN_SHAPE", score
        return "UNKNOWN_SHAPE", score

    def _brightness_features(self, luma, contour, bbox):
        x, y, width, height = bbox
        contour_mask = np.zeros(luma.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        inside = luma[contour_mask > 0]
        bright_ratio = float(np.mean(
            inside >= self.config.hsv_minimum_value)) if inside.size else 0.0
        pad_x = max(2, int(round(width*self.config.housing_expand_ratio)))
        pad_y = max(2, int(round(height*self.config.housing_expand_ratio)))
        xa, xb = max(0, x-pad_x), min(luma.shape[1], x+width+pad_x)
        ya, yb = max(0, y-pad_y), min(luma.shape[0], y+height+pad_y)
        ring_mask = np.zeros(luma.shape, np.uint8)
        ring_mask[ya:yb, xa:xb] = 255
        ring_mask[contour_mask > 0] = 0
        ring = luma[ring_mask > 0]
        background = float(np.median(ring)) if ring.size else 0.0
        center = float(np.mean(inside)) if inside.size else 0.0
        delta = center-background
        dark_ratio = float(np.mean(
            ring <= self.config.housing_dark_luma_max)) if ring.size else 0.0
        return delta, bright_ratio, dark_ratio

    def _red_x_score(self, component):
        """Score a lit X from two balanced diagonals crossing near its centre."""
        height, width = component.shape
        scale = max(width, height, 1)
        lines = cv2.HoughLinesP(
            component.astype(np.uint8)*255, 1, np.pi/180.0,
            threshold=int(self.config.red_x_hough_threshold),
            minLineLength=max(3, int(scale*self.config.red_x_min_line_length_ratio)),
            maxLineGap=max(1, int(scale*self.config.red_x_max_line_gap_ratio)))
        if lines is None:
            return 0.0
        positive, negative = [], []
        tolerance = self.config.red_x_angle_tolerance_deg
        for x1, y1, x2, y2 in lines[:, 0, :]:
            angle = math.degrees(math.atan2(float(y2-y1), float(x2-x1)))
            angle = ((angle+90.0) % 180.0)-90.0
            length = math.hypot(x2-x1, y2-y1)
            midpoint = ((x1+x2)/2.0, (y1+y2)/2.0)
            if abs(angle-45.0) <= tolerance:
                positive.append((length, midpoint))
            elif abs(angle+45.0) <= tolerance:
                negative.append((length, midpoint))
        if not positive or not negative:
            return 0.0
        pos = max(positive); neg = max(negative)
        balance = min(pos[0], neg[0])/max(pos[0], neg[0], 1.0)
        center = (width/2.0, height/2.0)
        center_error = max(
            math.hypot(pos[1][0]-center[0], pos[1][1]-center[1]),
            math.hypot(neg[1][0]-center[0], neg[1][1]-center[1]))/scale
        center_score = _bounded(1.0-center_error/max(
            self.config.red_x_center_tolerance_ratio, 1.0e-6))
        if balance < self.config.red_x_min_diagonal_balance:
            return 0.0
        return _bounded(0.55*balance+0.45*center_score)

    def detect(self, image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            return DetectionResult(
                "UNKNOWN", 0.0, (), None, False, {"invalid_image": 1},
                (0, 0, 0, 0))
        x0, y0, x1, y1 = self.roi_bounds(image.shape)
        roi = image[y0:y1, x0:x1]
        masks, luma = self._color_masks(roi)
        image_area = float(image.shape[0]*image.shape[1])
        candidates: List[Candidate] = []
        rejected: Dict[str, int] = {}

        def reject(reason):
            rejected[reason] = rejected.get(reason, 0)+1

        for color, mask in masks.items():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bbox = cv2.boundingRect(contour)
                x, y, width, height = bbox
                area, circularity, solidity, convexity, aspect, circle, hu = (
                    self._shape_features(contour, bbox))
                component = mask[y:y+height, x:x+width] > 0
                rectangularity = area/max(float(width*height), 1.0)
                red_x_score = self._red_x_score(component) if color == "red" else 0.0
                red_x_candidate = (
                    red_x_score >= self.config.red_x_min_diagonal_score and
                    circle <= self.config.red_x_max_circle_score and
                    rectangularity <= self.config.red_x_max_rectangularity)
                area_ratio = area/image_area
                if area_ratio < self.config.minimum_area_ratio:
                    reject("area_too_small"); continue
                if area_ratio > self.config.maximum_area_ratio:
                    reject("area_too_large"); continue
                if not (self.config.minimum_aspect_ratio <= aspect <=
                        self.config.maximum_aspect_ratio):
                    reject("aspect_ratio"); continue
                if solidity < self.config.minimum_solidity and not red_x_candidate:
                    reject("low_solidity"); continue
                if convexity < self.config.minimum_convexity and not red_x_candidate:
                    reject("low_convexity"); continue
                if (self.config.reject_roi_boundary_contact and
                        (x <= 0 or y <= 0 or x+width >= roi.shape[1]-1 or
                         y+height >= roi.shape[0]-1)):
                    reject("roi_boundary_contact"); continue
                delta, bright_ratio, dark_ratio = self._brightness_features(
                    luma, contour, bbox)
                if delta < self.config.minimum_brightness_delta:
                    reject("low_relative_brightness"); continue
                if bright_ratio < self.config.minimum_bright_pixel_ratio:
                    reject("low_bright_pixel_ratio"); continue
                if dark_ratio < self.config.minimum_housing_dark_ratio:
                    reject("insufficient_dark_housing"); continue
                left_score, right_score, down_score = self._arrow_scores(component)
                state = "UNKNOWN"
                raw_shape = "UNKNOWN_SHAPE"
                green_shape_score = 0.0
                if color in ("red", "yellow"):
                    if color == "red" and red_x_candidate:
                        raw_shape = "RED_X"
                    elif circle < self.config.round_minimum_score:
                        reject("non_round_stop_lamp"); continue
                    else:
                        raw_shape = "CIRCLE"
                    state = "R"
                else:
                    raw_shape, green_shape_score = self._green_shape(
                        component, circle, left_score, right_score,
                        down_score, rectangularity, self.config)
                    if raw_shape == "UNKNOWN_SHAPE":
                        reject("ambiguous_green_shape"); continue
                    state = "G"
                housing = _bounded((dark_ratio-self.config.minimum_housing_dark_ratio)/0.5)
                shape_confidence = (red_x_score if raw_shape == "RED_X" else
                                    circle if state == "R" else
                                    green_shape_score)
                confidence = _bounded(
                    0.25*_bounded(delta/100.0)+0.15*bright_ratio+
                    0.15*solidity+0.10*convexity+0.10*housing+
                    0.25*shape_confidence)
                if (not math.isfinite(confidence) or
                        confidence < self.config.minimum_confidence):
                    reject("low_confidence"); continue
                candidates.append(Candidate(
                    state, color, (x+x0, y+y0, width, height), confidence,
                    circularity, solidity, convexity, circle, left_score,
                    right_score, delta, bright_ratio, dark_ratio, hu,
                    raw_shape, down_score, green_shape_score, red_x_score))
        return self._resolve(candidates, rejected, (x0, y0, x1, y1))

    def _resolve(self, candidates, rejected, roi):
        by_state = {state: sorted(
            (candidate for candidate in candidates if candidate.state == state),
            key=lambda value: value.confidence, reverse=True)
            for state in ("R", "G")}
        red = by_state["R"][0] if by_state["R"] else None
        go_options = by_state["G"]
        go = max(go_options, key=lambda value: value.confidence,
                 default=None)
        selected, conflict = None, False
        if red is not None and go is not None:
            if self._same_housing(red, go):
                selected = red
            elif abs(red.confidence-go.confidence) <= self.config.conflict_margin:
                conflict = True
            elif (red.confidence >= self.config.red_priority_confidence or
                  red.confidence > go.confidence):
                selected = red
            else:
                selected = go
        elif red is not None:
            selected = red
        elif go is not None:
            selected = go
        return DetectionResult(
            "UNKNOWN" if selected is None or conflict else selected.state,
            0.0 if selected is None or conflict else selected.confidence,
            tuple(candidates), selected if not conflict else None, conflict,
            rejected, roi)

    @staticmethod
    def _same_housing(first, second):
        ax, ay, aw, ah = first.bbox
        bx, by, bw, bh = second.bbox
        distance = math.hypot(
            ax+aw/2.0-bx-bw/2.0, ay+ah/2.0-by-bh/2.0)
        lamp_scale = max(aw, ah, bw, bh, 1)
        return distance <= 2.5*lamp_scale

    def render_overlay(self, image, result, confirmed_state="UNKNOWN",
                       confirmation_count=0, input_fps=0.0,
                       processing_fps=0.0, latency_ms=0.0):
        overlay = image.copy()
        x0, y0, x1, y1 = result.roi
        cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (255, 255, 0), 1)
        for candidate in result.candidates:
            x, y, width, height = candidate.bbox
            color = self.COLORS[candidate.color]
            cv2.rectangle(overlay, (x, y), (x+width, y+height), color, 2)
            text = (f"{candidate.color.upper()}/{candidate.raw_shape} "
                    f"{candidate.confidence:.2f}")
            cv2.putText(overlay, text, (x, max(12, y-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                        cv2.LINE_AA)
        selected = result.selected
        raw_color = selected.color.upper() if selected else "UNKNOWN"
        raw_shape = selected.raw_shape if selected else "UNKNOWN_SHAPE"
        reason = ("CONFLICT" if result.conflict else
                  next(iter(result.rejection_reasons), "NONE"))
        confidence = result.confidence if math.isfinite(result.confidence) else 0.0
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, 76), (0, 0, 0), -1)
        cv2.putText(overlay, f"STATE={confirmed_state} raw={result.raw_state} count={confirmation_count}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"COLOR={raw_color} SHAPE={raw_shape} conf={confidence:.2f}",
                    (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"reject={reason}", (5, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(overlay, f"in={input_fps:.1f} proc={processing_fps:.1f} FPS latency={latency_ms:.1f} ms",
                    (5, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return overlay


@dataclass(frozen=True)
class TemporalConfig:
    confirmation_frames: int = 3
    switch_confirmation_frames: int = 4
    lost_timeout_sec: float = 0.35
    input_timeout_sec: float = 0.50
    track_iou_threshold: float = 0.20
    track_center_distance_ratio: float = 0.08

    def validate(self):
        if self.confirmation_frames < 1 or self.switch_confirmation_frames < 1:
            raise ValueError("confirmation counts must be positive")
        if self.lost_timeout_sec <= 0.0 or self.input_timeout_sec <= 0.0:
            raise ValueError("timeouts must be positive")
        if not 0.0 <= self.track_iou_threshold <= 1.0:
            raise ValueError("track_iou_threshold must be in [0,1]")
        if self.track_center_distance_ratio < 0.0:
            raise ValueError("track_center_distance_ratio must be nonnegative")
        return self


@dataclass(frozen=True)
class TemporalDecision:
    state: str
    confidence: float
    raw_state: str
    confirmation_count: int
    held: bool
    aspect: str = "UNKNOWN"
    raw_aspect: str = "UNKNOWN"


def bbox_iou(first, second):
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    intersection = max(0, x1-x0)*max(0, y1-y0)
    union = aw*ah+bw*bh-intersection
    return intersection/max(union, 1)


class TemporalTrafficLightFilter:
    def __init__(self, config=TemporalConfig()):
        self.config = config.validate()
        self.state = "UNKNOWN"
        self.aspect = "UNKNOWN"
        self.confidence = 0.0
        self.pending_state = "UNKNOWN"
        self.pending_aspect = "UNKNOWN"
        self.pending_count = 0
        self.pending_bbox = None
        self.last_detection_time = None
        self.last_input_time = None

    def _same_track(self, bbox, image_shape):
        if bbox is None or self.pending_bbox is None:
            return bbox is self.pending_bbox
        if bbox_iou(bbox, self.pending_bbox) >= self.config.track_iou_threshold:
            return True
        ax, ay, aw, ah = bbox; bx, by, bw, bh = self.pending_bbox
        distance = math.hypot(ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2)
        diagonal = math.hypot(image_shape[1], image_shape[0])
        return distance/max(diagonal, 1.0) <= self.config.track_center_distance_ratio

    def update(self, result, timestamp, image_shape):
        timestamp = float(timestamp)
        self.last_input_time = timestamp
        raw_aspect = result.raw_aspect
        if result.raw_state not in STATES or raw_aspect not in ASPECTS:
            return self.force_unknown("UNKNOWN")
        if result.raw_state == "UNKNOWN":
            if result.conflict:
                return self.force_unknown(result.raw_state)
            if (self.state != "UNKNOWN" and self.last_detection_time is not None and
                    timestamp-self.last_detection_time <= self.config.lost_timeout_sec):
                return TemporalDecision(self.state, self.confidence,
                                        result.raw_state, 0, True,
                                        self.aspect, raw_aspect)
            return self.force_unknown(result.raw_state)
        bbox = result.selected.bbox if result.selected else None
        if (raw_aspect == self.pending_aspect and
                self._same_track(bbox, image_shape)):
            self.pending_count += 1
        else:
            self.pending_state = result.raw_state
            self.pending_aspect = raw_aspect
            self.pending_count = 1
            self.pending_bbox = bbox
        self.last_detection_time = timestamp
        required = (self.config.confirmation_frames if self.state == "UNKNOWN"
                    else self.config.switch_confirmation_frames)
        if raw_aspect == self.aspect:
            self.confidence = result.confidence
            self.pending_count = max(self.pending_count, required)
        elif self.pending_count >= required:
            self.state = result.raw_state
            self.aspect = raw_aspect
            self.confidence = result.confidence
        return TemporalDecision(self.state, self.confidence,
                                result.raw_state, self.pending_count, False,
                                self.aspect, raw_aspect)

    def tick(self, timestamp):
        timestamp = float(timestamp)
        if (self.last_input_time is None or
                timestamp-self.last_input_time > self.config.input_timeout_sec):
            return self.force_unknown("UNKNOWN")
        if (self.last_detection_time is None or
                timestamp-self.last_detection_time > self.config.lost_timeout_sec):
            return self.force_unknown("UNKNOWN")
        return TemporalDecision(self.state, self.confidence,
                                self.pending_state, self.pending_count, False,
                                self.aspect, self.pending_aspect)

    def force_unknown(self, raw_state="UNKNOWN"):
        self.state = "UNKNOWN"
        self.aspect = "UNKNOWN"
        self.confidence = 0.0
        self.pending_state = "UNKNOWN"
        self.pending_aspect = "UNKNOWN"
        self.pending_count = 0
        self.pending_bbox = None
        return TemporalDecision("UNKNOWN", 0.0, raw_state, 0, False,
                                "UNKNOWN", "UNKNOWN")


def finite_diagnostics(payload):
    def finite(value):
        if isinstance(value, (float, np.floating)):
            return math.isfinite(float(value))
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True
    return finite(payload)


def build_diagnostics(stamp, result, decision, input_age_ms,
                      processing_latency_ms, input_fps, processing_fps):
    selected = result.selected
    counts = {
        "red_candidates": sum(c.color == "red" for c in result.candidates),
        "yellow_candidates": sum(c.color == "yellow" for c in result.candidates),
        "green_candidates": sum(c.color == "green" for c in result.candidates),
        "left_candidates": sum(c.raw_shape == "LEFT_ARROW"
                               for c in result.candidates),
        "down_candidates": sum(c.raw_shape == "DOWN_ARROW"
                               for c in result.candidates),
    }
    payload = {
        "stamp": int(stamp), "state": decision.state,
        "aspect": decision.aspect, "raw_aspect": result.raw_aspect,
        "confidence": float(decision.confidence),
        "input_age_ms": float(input_age_ms),
        "processing_latency_ms": float(processing_latency_ms),
        "candidate_count": len(result.candidates), **counts,
        "raw_color": (selected.color.upper() if selected else "UNKNOWN"),
        "selected_bbox": ([] if selected is None else [
            int(selected.bbox[0]), int(selected.bbox[1]),
            int(selected.bbox[0]+selected.bbox[2]),
            int(selected.bbox[1]+selected.bbox[3])]),
        "raw_shape": (selected.raw_shape if selected else "UNKNOWN_SHAPE"),
        "raw_state": result.raw_state,
        "confirmed_state": decision.state,
        "green_shape_score": (float(selected.green_shape_score)
                              if selected else 0.0),
        "circle_score": float(selected.circle_score) if selected else 0.0,
        "left_arrow_score": float(selected.left_score) if selected else 0.0,
        "down_arrow_score": float(selected.down_score) if selected else 0.0,
        "red_x_score": float(selected.red_x_score) if selected else 0.0,
        "confirmation_count": int(decision.confirmation_count),
        "held": bool(decision.held), "conflict": bool(result.conflict),
        "rejection_reasons": result.rejection_reasons,
        "roi": list(result.roi), "input_fps": float(input_fps),
        "processing_fps": float(processing_fps),
    }
    if not finite_diagnostics(payload):
        raise ValueError("diagnostics contain non-finite values")
    return payload
