"""Three-lamp signal detector and five-second vote for Mode 11 exit."""

from dataclasses import dataclass
from enum import Enum
import itertools

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
    minimum_contour_area: float = 40.0
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
    candidate_bounds: tuple = (0.05, 0.95, 0.02, 0.40)
    dark_ratio_threshold: float = 0.08
    slot_centers: tuple = (0.25, 0.50, 0.75)
    slot_max_distance: float = 0.18
    slot_conflict_margin: float = 0.12
    row_y_tolerance_lamp_heights: float = 1.25
    row_minimum_x_gap: float = 0.05
    row_maximum_gap_ratio: float = 2.25
    row_minimum_size_ratio: float = 0.35


@dataclass(frozen=True)
class SignalDetection:
    state: SignalState
    red_pixel_count: int
    green_pixel_count: int
    lamp_states: tuple = ()
    bbox: tuple = ()
    confidence: float = 0.0
    candidates: tuple = ()
    reason: str = "NO_SIGNAL"


def _signal_state(value):
    text = str(getattr(value, "value", value)).strip().upper()
    return {"G": SignalState.GREEN, "GREEN": SignalState.GREEN,
            "R": SignalState.RED, "RED": SignalState.RED}.get(
                text, SignalState.UNKNOWN)


def _candidate_geometry(item):
    """Return pixel geometry for a Mode-11 color component, if available."""
    if len(item) <= 3 or len(item[3]) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in item[3])
    width, height = x2-x1, y2-y1
    if width <= 0.0 or height <= 0.0:
        return None
    return ((y1+y2)/2.0, width, height, width*height)


def select_aligned_exit_triplet(
        candidates, y_tolerance_lamp_heights=1.25,
        minimum_x_gap=0.05, maximum_gap_ratio=2.25,
        minimum_size_ratio=0.35):
    """Select one horizontal three-lamp row from noisy Mode-11 candidates.

    Candidate x is normalized, while bbox geometry is in pixels.  Requiring
    similar y centers, component sizes and left/right spacing rejects colored
    objects that are not part of the physical exit signal panel.  Color is
    deliberately not part of the score; grouping happens before GRR/RGR is
    classified.
    """
    usable = tuple(item for item in candidates
                   if _candidate_geometry(item) is not None)
    if len(usable) < 3:
        return ()
    best = None
    for values in itertools.combinations(usable, 3):
        ordered = tuple(sorted(values, key=lambda item: float(item[0])))
        first_gap = float(ordered[1][0])-float(ordered[0][0])
        second_gap = float(ordered[2][0])-float(ordered[1][0])
        if first_gap < float(minimum_x_gap) or \
                second_gap < float(minimum_x_gap):
            continue
        gap_ratio = max(first_gap, second_gap) / max(
            min(first_gap, second_gap), 1.0e-9)
        if gap_ratio > float(maximum_gap_ratio):
            continue
        geometry = tuple(_candidate_geometry(item) for item in ordered)
        y_centers = tuple(value[0] for value in geometry)
        heights = tuple(value[2] for value in geometry)
        areas = tuple(value[3] for value in geometry)
        reference_height = float(np.median(heights))
        y_spread = max(y_centers)-min(y_centers)
        if y_spread > float(y_tolerance_lamp_heights)*reference_height:
            continue
        size_ratio = min(areas)/max(max(areas), 1.0e-9)
        if size_ratio < float(minimum_size_ratio):
            continue
        confidence = sum(float(item[4]) if len(item) > 4 else 0.0
                         for item in ordered)
        # Geometry dominates confidence so a bright off-panel object cannot
        # displace a dim but physically aligned exit lamp.
        score = (4.0*y_spread/max(reference_height, 1.0e-9) +
                 2.0*(gap_ratio-1.0) + (1.0-size_ratio) -
                 0.20*confidence)
        if best is None or score < best[0]:
            best = (score, ordered)
    return () if best is None else best[1]


def assign_exit_lamps(candidates, slot_centers=(0.25, 0.50, 0.75),
                      slot_max_distance=0.18, conflict_margin=0.12,
                      row_y_tolerance_lamp_heights=1.25,
                      row_minimum_x_gap=0.05,
                      row_maximum_gap_ratio=2.25,
                      row_minimum_size_ratio=0.35):
    """Assign lamps by full-group order or calibrated partial positions."""
    geometric_candidates = sum(
        _candidate_geometry(item) is not None for item in candidates)
    aligned = select_aligned_exit_triplet(
        candidates, row_y_tolerance_lamp_heights, row_minimum_x_gap,
        row_maximum_gap_ratio, row_minimum_size_ratio)
    if aligned:
        candidates = aligned
    elif geometric_candidates >= 3:
        # Three or more visible colored objects that do not form one physical
        # horizontal panel are noise, not a calibrated partial observation.
        return (SignalState.UNKNOWN,)*3
    normalized = []
    for item in candidates:
        confidence = float(item[4]) if len(item) > 4 else min(
            1.0, float(item[2])/40.0)
        normalized.append((float(item[0]), _signal_state(item[1]),
                           int(item[2]), confidence))
    # A single surviving dark-red candidate must never translate the whole
    # panel: that made a physical CENTER GREEN look like LEFT GREEN. When all
    # three lamps are present, their left-to-right order is authoritative.
    clusters = []
    for item in sorted(normalized, key=lambda value: value[0]):
        if not clusters or item[0]-clusters[-1][-1][0] > 0.06:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    if aligned or len(clusters) == 3:
        slots = clusters
    else:
        expected = tuple(float(value) for value in slot_centers)
        slots = [[], [], []]
        for item in normalized:
            distances = [abs(item[0]-center) for center in expected]
            index = int(np.argmin(distances))
            if distances[index] <= float(slot_max_distance):
                slots[index].append(item)
    output = []
    for values in slots:
        if not values:
            output.append(SignalState.UNKNOWN)
            continue
        by_state = {}
        for item in values:
            previous = by_state.get(item[1])
            if previous is None or item[3] > previous[3]:
                by_state[item[1]] = item
        red, green = by_state.get(SignalState.RED), by_state.get(SignalState.GREEN)
        if red and green and abs(red[3]-green[3]) <= float(conflict_margin):
            output.append(SignalState.UNKNOWN)
        else:
            output.append(max(values, key=lambda item: item[3])[1])
    return tuple(output)


def exit_route_evidence(lamps):
    """Return A/B evidence and a human-readable position-specific reason."""
    if len(lamps) != 3:
        return SignalState.UNKNOWN, "INCOMPLETE"
    left, center, right = tuple(_signal_state(value) for value in lamps)
    if left == SignalState.GREEN:
        return SignalState.GREEN, "LEFT_GREEN"
    if center == SignalState.GREEN:
        return SignalState.RED, "CENTER_GREEN"
    if left == center == right == SignalState.RED:
        return SignalState.UNKNOWN, "AMBIGUOUS_ALL_RED"
    if center == SignalState.RED and right == SignalState.RED:
        return SignalState.GREEN, "CENTER_RED+RIGHT_RED"
    if left == SignalState.RED and right == SignalState.RED:
        return SignalState.RED, "LEFT_RED+RIGHT_RED"
    return SignalState.UNKNOWN, "INSUFFICIENT_POSITIONAL_EVIDENCE"


def classify_exit_triplet(candidates, merge_distance_ratio=0.04):
    """Compatibility entry point backed by position-aware partial evidence."""
    del merge_distance_ratio
    lamps = assign_exit_lamps(candidates)
    return exit_route_evidence(lamps)[0], lamps


class SignalExitDetector:
    """Detect and x-sort the three Mode 11 lamps in one image ROI."""

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
                    minimum_dark_ratio, state):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
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
            dark_ratio = float(np.mean(
                hsv[ya:yb, xa:xb, 2] < self.config.panel_dark_value))
            component = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, -1)
            pixels = int(np.count_nonzero((component > 0) & (mask > 0)))
            housing_bonus = min(
                1.0, dark_ratio/max(self.config.dark_ratio_threshold, 1.0e-6))
            confidence = min(1.0, 0.75*min(1.0, area/80.0)+0.25*housing_bonus)
            candidates.append((center_x, state, pixels,
                               (offset[0]+x, offset[1]+y,
                                offset[0]+x+width, offset[1]+y+height),
                               confidence, dark_ratio, float(area)))
        return tuple(candidates)

    def detect(self, frame):
        x1, y1, x2, y2 = self.roi.pixel_bounds(frame)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        size = (frame.shape[1], frame.shape[0])
        red_candidates = self._components(
            self._mask(hsv, self.config.red_ranges), hsv, (x1, y1), size,
            self.config.maximum_contour_area, self.config.red_dark_ratio,
            SignalState.RED)
        core_mask = self._mask(hsv, (self.config.core_green,))
        core_candidates = self._components(
            core_mask, hsv, (x1, y1), size,
            self.config.maximum_contour_area,
            self.config.core_green_dark_ratio, SignalState.GREEN)
        extended_mask = cv2.bitwise_and(
            self._mask(hsv, (self.config.extended_green,)),
            cv2.bitwise_not(core_mask))
        extended_candidates = self._components(
            extended_mask, hsv, (x1, y1), size,
            self.config.extended_maximum_contour_area,
            self.config.extended_green_dark_ratio, SignalState.GREEN)
        candidates = red_candidates+core_candidates+extended_candidates
        minimum_pixels = (float(crop.shape[0]*crop.shape[1]) *
                          self.config.minimum_color_pixel_ratio)
        candidates = tuple(value for value in candidates
                           if value[2] >= minimum_pixels)
        lamps = assign_exit_lamps(
            candidates, self.config.slot_centers,
            self.config.slot_max_distance, self.config.slot_conflict_margin,
            self.config.row_y_tolerance_lamp_heights,
            self.config.row_minimum_x_gap,
            self.config.row_maximum_gap_ratio,
            self.config.row_minimum_size_ratio)
        aligned = select_aligned_exit_triplet(
            candidates, self.config.row_y_tolerance_lamp_heights,
            self.config.row_minimum_x_gap,
            self.config.row_maximum_gap_ratio,
            self.config.row_minimum_size_ratio)
        if aligned:
            # Diagnostics/tracking must describe the selected physical panel,
            # not every red/green object in the upper camera image.
            candidates = aligned
        elif sum(_candidate_geometry(item) is not None
                 for item in candidates) >= 3:
            candidates = ()
        state, reason = exit_route_evidence(lamps)
        red = sum(value[2] for value in candidates
                  if value[1] == SignalState.RED)
        green = sum(value[2] for value in candidates
                    if value[1] == SignalState.GREEN)
        boxes = [value[3] for value in candidates]
        bbox = (() if not boxes else (
            min(value[0] for value in boxes), min(value[1] for value in boxes),
            max(value[2] for value in boxes), max(value[3] for value in boxes)))
        confidence = min(1.0, (red+green)/max(minimum_pixels*3.0, 1.0))
        return SignalDetection(
            state, red, green, lamps, bbox, confidence, candidates, reason)


@dataclass(frozen=True)
class VoteSnapshot:
    state: ObservationState
    route: SelectedRoute
    confidence: float
    green: int
    red: int
    unknown: int
    elapsed: float
    a_score: float = 0.0
    b_score: float = 0.0
    reason: str = ""
    position_counts: tuple = ()


class SignalVoteWindow:
    """Five-second position vote with direct-green priority and default A."""

    def __init__(self, duration_s=5.0, minimum_valid_frames=60,
                 decision_ratio=0.75, green_weight=3.0,
                 red_pair_weight=1.0, minimum_green_observations=2,
                 minimum_red_pair_observations=2):
        self.duration_s = max(5.0, float(duration_s))
        self.minimum_valid_frames = max(1, int(minimum_valid_frames))
        self.decision_ratio = float(decision_ratio)
        if not 0.5 < self.decision_ratio <= 1.0:
            raise ValueError("decision ratio must be in (0.5, 1.0]")
        self.green_weight = max(1.0, float(green_weight))
        self.red_pair_weight = max(0.0, float(red_pair_weight))
        self.minimum_green_observations = max(
            1, int(minimum_green_observations))
        self.minimum_red_pair_observations = max(
            1, int(minimum_red_pair_observations))
        self.reset()

    def reset(self):
        self.state = ObservationState.IDLE
        self.started_at = None
        self.route = SelectedRoute.UNKNOWN
        self.confidence = 0.0
        self.green = self.red = self.unknown = 0
        self.position_counts = [{SignalState.RED: 0, SignalState.GREEN: 0}
                                for _ in range(3)]
        self.a_red_pair = self.b_red_pair = 0
        self.a_score = self.b_score = 0.0
        self.reason = ""

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

    def observe_detection(self, detection, now):
        """Accumulate visible slots; absent lamps never erase prior evidence."""
        if self.state != ObservationState.OBSERVING:
            return self.snapshot(now)
        lamps = tuple(detection.lamp_states)
        if len(lamps) == 3:
            for index, value in enumerate(lamps):
                value = _signal_state(value)
                if value in (SignalState.RED, SignalState.GREEN):
                    self.position_counts[index][value] += 1
            left, center, right = tuple(_signal_state(value) for value in lamps)
            if center == SignalState.RED and right == SignalState.RED:
                self.a_red_pair += 1
            if left == SignalState.RED and right == SignalState.RED:
                self.b_red_pair += 1
        return self.observe(detection.state, now)

    def evaluate(self, now):
        if self.state != ObservationState.OBSERVING:
            return self.snapshot(now)
        if float(now)-self.started_at < self.duration_s:
            return self.snapshot(now)
        observed_left_green = self.position_counts[0][SignalState.GREEN]
        observed_center_green = self.position_counts[1][SignalState.GREEN]
        left_green = (observed_left_green if observed_left_green >=
                      self.minimum_green_observations else 0)
        center_green = (observed_center_green if observed_center_green >=
                        self.minimum_green_observations else 0)
        a_red_pair = (self.a_red_pair if self.a_red_pair >=
                      self.minimum_red_pair_observations else 0)
        b_red_pair = (self.b_red_pair if self.b_red_pair >=
                      self.minimum_red_pair_observations else 0)
        self.a_score = self.green_weight*left_green + \
            self.red_pair_weight*a_red_pair
        self.b_score = self.green_weight*center_green + \
            self.red_pair_weight*b_red_pair
        positional = self.a_score+self.b_score
        if positional > 0.0:
            self.confidence = max(self.a_score, self.b_score)/positional
            if left_green != center_green:
                choose_a = left_green > center_green
                self.reason = "LEFT_GREEN" if choose_a else "CENTER_GREEN"
            elif self.a_score != self.b_score:
                choose_a = self.a_score > self.b_score
                self.reason = ("CENTER_RED+RIGHT_RED" if choose_a else
                               "LEFT_RED+RIGHT_RED")
            else:
                choose_a = True
                self.reason = "5SEC_TIE_FALLBACK_A"
            self.route = SelectedRoute.A if choose_a else SelectedRoute.B
            self.state = (ObservationState.DEFAULTED if
                          self.reason.endswith("FALLBACK_A") else
                          ObservationState.LATCHED)
        else:
            # Preserve the public legacy vote API used by older callers, but
            # never allow indecision beyond five seconds.
            valid = self.green+self.red
            green_ratio = self.green/valid if valid else 0.0
            red_ratio = self.red/valid if valid else 0.0
            self.confidence = max(green_ratio, red_ratio)
            if valid >= self.minimum_valid_frames and green_ratio >= self.decision_ratio:
                self.route, self.state = SelectedRoute.A, ObservationState.LATCHED
                self.reason = "LEGACY_A"
            elif valid >= self.minimum_valid_frames and red_ratio >= self.decision_ratio:
                self.route, self.state = SelectedRoute.B, ObservationState.LATCHED
                self.reason = "LEGACY_B"
            else:
                self.route, self.state = SelectedRoute.A, ObservationState.DEFAULTED
                self.reason = "5SEC_FALLBACK_A"
        return self.snapshot(now)

    def snapshot(self, now):
        elapsed = (0.0 if self.started_at is None else
                   min(self.duration_s, max(0.0, float(now)-self.started_at)))
        return VoteSnapshot(
            self.state, self.route, self.confidence,
            self.green, self.red, self.unknown, elapsed,
            self.a_score, self.b_score, self.reason,
            tuple((counts[SignalState.RED], counts[SignalState.GREEN])
                  for counts in self.position_counts))


class ExitSignalTrack:
    """Small continuity tracker for brief occlusion of the upper-image lamps."""

    def __init__(self, confirmations=3, missing_hold_s=0.25,
                 minimum_confidence=0.5):
        self.confirmations = max(2, int(confirmations))
        self.missing_hold_s = float(missing_hold_s)
        self.minimum_confidence = float(minimum_confidence)
        self.reset()

    def reset(self):
        self.state = SignalState.UNKNOWN
        self.count = 0
        self.bbox = ()
        self.confidence = 0.0
        self.last_seen = None

    @staticmethod
    def _iou(first, second):
        if len(first) != 4 or len(second) != 4:
            return 0.0
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x2-x1)*max(0.0, y2-y1)
        a = max(0.0, first[2]-first[0])*max(0.0, first[3]-first[1])
        b = max(0.0, second[2]-second[0])*max(0.0, second[3]-second[1])
        return intersection/max(a+b-intersection, 1.0e-9)

    def update(self, detection, now):
        now = float(now)
        observed = (detection.state != SignalState.UNKNOWN and
                    detection.confidence >= self.minimum_confidence and
                    len(detection.bbox) == 4)
        if observed:
            same = (detection.state == self.state and
                    (not self.bbox or self._iou(detection.bbox, self.bbox) > 0.05))
            self.count = self.count+1 if same else 1
            self.state = detection.state
            self.bbox = detection.bbox
            self.confidence = detection.confidence
            self.last_seen = now
        missing = (None if self.last_seen is None else
                   max(0.0, now-self.last_seen))
        valid = (self.count >= self.confirmations and missing is not None and
                 missing <= self.missing_hold_s)
        return {
            "state": self.state if valid else SignalState.UNKNOWN,
            "bbox": self.bbox, "confidence": self.confidence,
            "last_seen": self.last_seen, "missing_duration": missing,
            "track_continuity": self.count, "valid": valid}
