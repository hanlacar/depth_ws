"""A-panel signal detector and five-second vote for Mode 11 exit."""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class SignalState(str, Enum):
    RED = "RED"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


class SelectedRoute(str, Enum):
    A = "A"
    B = "B"
    UNKNOWN = "UNKNOWN"


class ObservationState(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    LATCHED = "LATCHED"
    DEFAULTED = "DEFAULTED"


@dataclass(frozen=True)
class NormalizedROI:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not 0.0 <= self.x1 < self.x2 <= 1.0:
            raise ValueError("invalid normalized ROI x bounds")
        if not 0.0 <= self.y1 < self.y2 <= 1.0:
            raise ValueError("invalid normalized ROI y bounds")

    def pixel_bounds(self, frame):
        height, width = frame.shape[:2]
        x1 = max(0, min(width-1, round(self.x1*width)))
        y1 = max(0, min(height-1, round(self.y1*height)))
        x2 = max(x1+1, min(width, round(self.x2*width)))
        y2 = max(y1+1, min(height, round(self.y2*height)))
        return x1, y1, x2, y2


@dataclass(frozen=True)
class HSVRange:
    low: tuple
    high: tuple


@dataclass(frozen=True)
class DetectorConfig:
    red_ranges: tuple
    core_green: HSVRange
    extended_green: HSVRange
    minimum_contour_area: float = 1.0
    maximum_contour_area: float = 1200.0
    extended_maximum_contour_area: float = 700.0
    minimum_aspect_ratio: float = 0.12
    maximum_aspect_ratio: float = 12.0
    minimum_color_pixel_ratio: float = 0.0001
    panel_dark_value: int = 80
    red_dark_ratio: float = 0.95
    core_green_dark_ratio: float = 0.50
    extended_green_dark_ratio: float = 0.55
    local_context_scale: float = 0.60
    local_context_min_padding: int = 8
    morphology_kernel_size: int = 3
    candidate_bounds: tuple = (0.33, 0.40, 0.15, 0.26)


@dataclass(frozen=True)
class SignalDetection:
    state: SignalState
    red_pixel_count: int
    green_pixel_count: int


class SignalExitDetector:
    """Two-stage HSV detector restricted to the leftmost A panel."""

    def __init__(self, config, roi):
        self.config = config
        self.roi = roi

    def _mask(self, hsv, ranges):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for value in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(
                hsv, np.array(value.low, dtype=np.uint8),
                np.array(value.high, dtype=np.uint8)))
        size = int(self.config.morphology_kernel_size)
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _components(self, mask, hsv, offset, frame_size, maximum_area,
                    minimum_dark_ratio):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixels = 0
        frame_width, frame_height = frame_size
        min_x, max_x, min_y, max_y = self.config.candidate_bounds
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width/max(1, height)
            center_x = (offset[0]+x+width/2)/frame_width
            center_y = (offset[1]+y+height/2)/frame_height
            if not self.config.minimum_contour_area <= area <= maximum_area:
                continue
            if not self.config.minimum_aspect_ratio <= aspect <= \
                    self.config.maximum_aspect_ratio:
                continue
            if not min_x <= center_x <= max_x or not min_y <= center_y <= max_y:
                continue
            padding = max(
                self.config.local_context_min_padding,
                round(max(width, height)*self.config.local_context_scale))
            xa, ya = max(0, x-padding), max(0, y-padding)
            xb = min(mask.shape[1], x+width+padding)
            yb = min(mask.shape[0], y+height+padding)
            if float(np.mean(hsv[ya:yb, xa:xb, 2] <
                             self.config.panel_dark_value)) < minimum_dark_ratio:
                continue
            component = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, -1)
            pixels += int(np.count_nonzero((component > 0) & (mask > 0)))
        return pixels

    def detect(self, frame):
        x1, y1, x2, y2 = self.roi.pixel_bounds(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        size = (frame.shape[1], frame.shape[0])
        red = self._components(
            self._mask(hsv, self.config.red_ranges), hsv, (x1, y1), size,
            self.config.maximum_contour_area, self.config.red_dark_ratio)
        core_mask = self._mask(hsv, (self.config.core_green,))
        core = self._components(
            core_mask, hsv, (x1, y1), size,
            self.config.maximum_contour_area,
            self.config.core_green_dark_ratio)
        extended_mask = cv2.bitwise_and(
            self._mask(hsv, (self.config.extended_green,)),
            cv2.bitwise_not(core_mask))
        extended = self._components(
            extended_mask, hsv, (x1, y1), size,
            self.config.extended_maximum_contour_area,
            self.config.extended_green_dark_ratio)
        area = float(crop.shape[0]*crop.shape[1])
        red_valid = red/area >= self.config.minimum_color_pixel_ratio
        green_valid = (core+extended)/area >= \
            self.config.minimum_color_pixel_ratio
        state = (SignalState.UNKNOWN if red_valid == green_valid else
                 SignalState.RED if red_valid else SignalState.GREEN)
        return SignalDetection(state, red, core+extended)


@dataclass(frozen=True)
class VoteSnapshot:
    state: ObservationState
    route: SelectedRoute
    confidence: float
    green: int
    red: int
    unknown: int
    elapsed: float


class SignalVoteWindow:
    """Five-second vote with fail-safe DEFAULT B unless A is explicit."""

    def __init__(self, duration_s=5.0, minimum_valid_frames=60,
                 decision_ratio=0.75):
        self.duration_s = max(5.0, float(duration_s))
        self.minimum_valid_frames = max(1, int(minimum_valid_frames))
        self.decision_ratio = float(decision_ratio)
        if not 0.5 < self.decision_ratio <= 1.0:
            raise ValueError("decision ratio must be in (0.5, 1.0]")
        self.reset()

    def reset(self):
        self.state = ObservationState.IDLE
        self.started_at = None
        self.route = SelectedRoute.UNKNOWN
        self.confidence = 0.0
        self.green = self.red = self.unknown = 0

    def start(self, now):
        if self.state not in (ObservationState.IDLE,
                              ObservationState.DEFAULTED):
            return False
        self.reset()
        self.state = ObservationState.OBSERVING
        self.started_at = float(now)
        return True

    def observe(self, state, now):
        if self.state != ObservationState.OBSERVING:
            return self.snapshot(now)
        value = SignalState(str(getattr(state, "value", state)).upper())
        if value == SignalState.GREEN:
            self.green += 1
        elif value == SignalState.RED:
            self.red += 1
        else:
            self.unknown += 1
        return self.evaluate(now)

    def evaluate(self, now):
        if self.state != ObservationState.OBSERVING:
            return self.snapshot(now)
        if float(now)-self.started_at < self.duration_s:
            return self.snapshot(now)
        valid = self.green+self.red
        green_ratio = self.green/valid if valid else 0.0
        red_ratio = self.red/valid if valid else 0.0
        self.confidence = max(green_ratio, red_ratio)
        if valid >= self.minimum_valid_frames and \
                green_ratio >= self.decision_ratio:
            self.route, self.state = SelectedRoute.A, ObservationState.LATCHED
        elif valid >= self.minimum_valid_frames and \
                red_ratio >= self.decision_ratio:
            self.route, self.state = SelectedRoute.B, ObservationState.LATCHED
        else:
            self.route, self.state = SelectedRoute.B, ObservationState.DEFAULTED
        return self.snapshot(now)

    def snapshot(self, now):
        elapsed = (0.0 if self.started_at is None else
                   min(self.duration_s, max(0.0, float(now)-self.started_at)))
        return VoteSnapshot(
            self.state, self.route, self.confidence,
            self.green, self.red, self.unknown, elapsed)
