"""Shared, timestamp-driven semantic refinement for both camera planners.

The implementation is intentionally ROS independent.  Raw YOLO masks remain
immutable; this module returns a second set of masks for navigation consumers.
All pixel-heavy work is vectorised OpenCV/NumPy and colour conversion is
limited to instance/road ROIs.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from .mask_postprocessor import restore_masks_to_raw_shape


@dataclass(frozen=True)
class RefinementConfig:
    yellow_hsv_h_min: int = 14
    yellow_hsv_h_max: int = 42
    yellow_hsv_s_min: int = 55
    yellow_lab_b_road_delta_min: float = 8.0
    yellow_min_ratio: float = 0.28
    white_hsv_s_max: int = 85
    white_hsv_v_min: int = 125
    white_lab_l_road_delta_min: float = -8.0
    white_min_ratio: float = 0.35
    unknown_line_score_max: float = 0.32
    class_hysteresis_frames: int = 3
    temporal_history_frames: int = 5
    temporal_iou_min: float = 0.18
    temporal_center_distance_ratio: float = 0.08
    track_timeout_sec: float = 0.30
    stop_angle_tolerance_deg: float = 18.0
    stop_min_road_width_ratio: float = 0.34
    stop_max_thickness_ratio: float = 0.10
    stop_min_confidence: float = 0.55
    stop_confirmation_frames: int = 2
    yellow_line_adjacency_px_ratio: float = 0.025
    yellow_line_adjacency_bonus: float = 0.12
    crosswalk_min_stripes: int = 3
    crosswalk_spacing_cv_max: float = 0.45
    crosswalk_parallel_tolerance_deg: float = 12.0
    crosswalk_confirmation_frames: int = 2
    marking_min_area_ratio: float = 0.00008
    lane_min_vertical_span_ratio: float = 0.16
    lane_max_width_ratio: float = 0.18
    road_restore_max_gap_ratio: float = 0.16
    road_restore_max_width_growth_ratio: float = 0.08
    previous_road_hold_sec: float = 0.20
    road_drop_area_ratio: float = 0.65

    def validate(self):
        if not 0 <= self.yellow_hsv_h_min < self.yellow_hsv_h_max <= 179:
            raise ValueError("yellow HSV hue range is invalid")
        if self.class_hysteresis_frames < 2:
            raise ValueError("class_hysteresis_frames must be at least 2")
        if not 3 <= self.temporal_history_frames <= 10:
            raise ValueError("temporal_history_frames must be in [3, 10]")
        if not 0.0 <= self.temporal_iou_min <= 1.0:
            raise ValueError("temporal_iou_min must be in [0, 1]")
        if self.track_timeout_sec <= 0.0 or self.previous_road_hold_sec < 0.0:
            raise ValueError("temporal timeouts are invalid")
        if self.stop_confirmation_frames < 1 or self.crosswalk_confirmation_frames < 1:
            raise ValueError("candidate confirmation frames must be positive")
        if self.crosswalk_min_stripes < 2:
            raise ValueError("crosswalk_min_stripes must be at least 2")


@dataclass
class _Track:
    track_id: int
    mask: np.ndarray
    center: tuple
    stamp: float
    stable_class: str
    pending_class: str = ""
    pending_count: int = 0
    scores: deque = field(default_factory=deque)


@dataclass
class RefinementResult:
    road: np.ndarray
    white_line: np.ndarray
    yellow_line: np.ndarray
    unknown_line: np.ndarray
    stop_line: np.ndarray
    stop_confidence: float
    crosswalk: np.ndarray
    crosswalk_confidence: float
    words: np.ndarray
    road_marking_unknown: np.ndarray
    restored_markings: np.ndarray
    diagnostics: dict


def _binary(mask, shape=None):
    value = np.asarray(mask)
    if shape is not None and value.shape != shape:
        raise ValueError("semantic mask shape mismatch")
    return (value > 0).astype(np.uint8)


def _bbox(mask, pad=0):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    return (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
            min(width, int(xs.max())+1+pad), min(height, int(ys.max())+1+pad))


def _center(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0.0:
        return (0.0, 0.0)
    return (moments["m10"]/moments["m00"], moments["m01"]/moments["m00"])


def _iou(first, second):
    intersection = np.count_nonzero((first > 0) & (second > 0))
    if not intersection:
        return 0.0
    union = np.count_nonzero((first > 0) | (second > 0))
    return float(intersection/max(1, union))


def _component_features(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(points)
    if side_a < side_b:
        side_a, side_b = side_b, side_a
        angle += 90.0
    angle = ((angle+90.0) % 180.0)-90.0
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    return {"area": int(len(xs)), "center": (float(xs.mean()), float(ys.mean())),
            "length": float(side_a), "thickness": float(max(1.0, side_b)),
            "angle_deg": float(angle), "bbox": (int(x0), int(y0), int(x1), int(y1)),
            "horizontal_error_deg": float(abs(angle)),
            "vertical_span": int(y1-y0), "horizontal_span": int(x1-x0)}


class CommonPerceptionRefiner:
    """Refine raw semantic masks without mutating them."""

    def __init__(self, config=None):
        self.config = config or RefinementConfig()
        self.config.validate()
        self._tracks = []
        self._next_track_id = 1
        self._last_stamp = None
        self._previous_road = None
        self._candidate_history = {"stop": deque(maxlen=8),
                                   "crosswalk": deque(maxlen=8)}

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 1
        self._last_stamp = None
        self._previous_road = None
        for values in self._candidate_history.values():
            values.clear()

    def _instance_mask(self, instance, shape, threshold):
        restored = instance.get("restored_mask")
        probability = (np.asarray(restored, np.float32) if restored is not None
                       else restore_masks_to_raw_shape(instance["mask"], shape))
        return (probability >= threshold).astype(np.uint8)

    def _line_colour_scores(self, bgr, mask, road):
        box = _bbox(mask, pad=4)
        if box is None:
            return 0.0, 0.0, {"pixels": 0}
        x0, y0, x1, y1 = box
        roi_mask = mask[y0:y1, x0:x1] > 0
        roi = bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        pixels = int(np.count_nonzero(roi_mask))
        if not pixels:
            return 0.0, 0.0, {"pixels": 0}
        ring_kernel = np.ones((7, 7), np.uint8)
        ring = cv2.dilate(roi_mask.astype(np.uint8), ring_kernel) > 0
        ring &= ~roi_mask
        ring &= road[y0:y1, x0:x1] > 0
        if np.any(ring):
            road_v = float(np.median(hsv[:, :, 2][ring]))
            road_s = float(np.median(hsv[:, :, 1][ring]))
            road_l = float(np.median(lab[:, :, 0][ring]))
            road_b = float(np.median(lab[:, :, 2][ring]))
        else:
            road_v, road_s, road_l, road_b = 100.0, 45.0, 100.0, 128.0
        hue, sat, val = (hsv[:, :, index][roi_mask] for index in range(3))
        light, lab_b = lab[:, :, 0][roi_mask], lab[:, :, 2][roi_mask]
        yellow_pixels = ((hue >= self.config.yellow_hsv_h_min) &
                         (hue <= self.config.yellow_hsv_h_max) &
                         (sat >= max(self.config.yellow_hsv_s_min, road_s+8.0)) &
                         (lab_b >= road_b+self.config.yellow_lab_b_road_delta_min))
        white_pixels = ((sat <= min(self.config.white_hsv_s_max, road_s+32.0)) &
                        (val >= max(self.config.white_hsv_v_min, road_v-5.0)) &
                        (light >= road_l+self.config.white_lab_l_road_delta_min))
        yellow_ratio = float(np.mean(yellow_pixels))
        white_ratio = float(np.mean(white_pixels))
        yellow_score = 0.65*yellow_ratio+0.35*float(np.mean(
            np.clip((lab_b-road_b)/32.0, 0.0, 1.0)))
        white_score = 0.65*white_ratio+0.35*float(np.mean(
            np.clip((light-road_l+8.0)/64.0, 0.0, 1.0)))
        return yellow_score, white_score, {
            "pixels": pixels, "yellow_ratio": yellow_ratio,
            "white_ratio": white_ratio, "road_v": road_v, "road_l": road_l,
            "yellow_score": yellow_score, "white_score": white_score}

    def _match_track(self, mask, center, stamp):
        height, width = mask.shape
        distance_limit = self.config.temporal_center_distance_ratio*math.hypot(width, height)
        best, best_score = None, -1.0
        for track in self._tracks:
            # One physical track may consume at most one instance in a frame;
            # adjacent white/yellow masks must retain independent statistics.
            if abs(stamp-track.stamp) <= 1.0e-9:
                continue
            if stamp-track.stamp > self.config.track_timeout_sec:
                continue
            overlap = _iou(mask, track.mask)
            distance = math.hypot(center[0]-track.center[0], center[1]-track.center[1])
            if overlap < self.config.temporal_iou_min and distance > distance_limit:
                continue
            score = overlap-max(0.0, distance/distance_limit)*0.1
            if score > best_score:
                best, best_score = track, score
        return best

    def _stable_line_class(self, mask, stamp, raw_class, yellow, white):
        if yellow >= self.config.yellow_min_ratio and yellow > white+0.05:
            observed = "yellow_line"
        elif white >= self.config.white_min_ratio and white > yellow+0.05:
            observed = "white_line"
        elif max(yellow, white) <= self.config.unknown_line_score_max:
            observed = "unknown_line"
        else:
            observed = "unknown_line"
        center = _center(mask)
        track = self._match_track(mask, center, stamp)
        if track is None:
            stable = observed if observed == "unknown_line" else raw_class
            track = _Track(self._next_track_id, mask.copy(), center, stamp, stable,
                           scores=deque(maxlen=self.config.temporal_history_frames))
            self._next_track_id += 1
            self._tracks.append(track)
        track.scores.append((yellow, white))
        mean_yellow = float(np.mean([value[0] for value in track.scores]))
        mean_white = float(np.mean([value[1] for value in track.scores]))
        aggregate = ("yellow_line" if mean_yellow >= self.config.yellow_min_ratio and
                     mean_yellow > mean_white+0.04 else
                     "white_line" if mean_white >= self.config.white_min_ratio and
                     mean_white > mean_yellow+0.04 else "unknown_line")
        if aggregate == track.stable_class:
            track.pending_class, track.pending_count = "", 0
        elif aggregate == "unknown_line":
            track.stable_class = aggregate
            track.pending_class, track.pending_count = "", 0
        else:
            if track.pending_class == aggregate:
                track.pending_count += 1
            else:
                track.pending_class, track.pending_count = aggregate, 1
            if track.pending_count >= self.config.class_hysteresis_frames:
                track.stable_class = aggregate
                track.pending_class, track.pending_count = "", 0
        track.mask, track.center, track.stamp = mask.copy(), center, stamp
        return track.stable_class, track.track_id, aggregate

    def _road_width(self, road, row):
        xs = np.flatnonzero(road[int(np.clip(row, 0, road.shape[0]-1))])
        return 0 if not len(xs) else int(xs[-1]-xs[0]+1)

    def _shape_candidates(self, bgr, road, raw_masks, line_union, yellow):
        shape = road.shape
        marking = np.bitwise_or.reduce((line_union, _binary(raw_masks.get("words", 0), shape),
                                        _binary(raw_masks.get("stop", raw_masks.get("stop_line", 0)), shape),
                                        _binary(raw_masks.get("center_line", raw_masks.get("c_line", 0)), shape)))
        # Recover missed bright/coloured planar paint, but only inside the road ROI.
        road_box = _bbox(road)
        if road_box is not None:
            x0, y0, x1, y1 = road_box
            roi = bgr[y0:y1, x0:x1]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            paint = (((hsv[:, :, 1] < self.config.white_hsv_s_max) &
                      (hsv[:, :, 2] >= self.config.white_hsv_v_min)) |
                     ((hsv[:, :, 0] >= self.config.yellow_hsv_h_min) &
                      (hsv[:, :, 0] <= self.config.yellow_hsv_h_max) &
                      (hsv[:, :, 1] >= self.config.yellow_hsv_s_min)))
            gap = max(3, int(round(self.config.road_restore_max_gap_ratio*shape[0])))
            road_support = cv2.morphologyEx(
                road, cv2.MORPH_CLOSE, np.ones((gap, 3), np.uint8))
            paint &= road_support[y0:y1, x0:x1] > 0
            marking[y0:y1, x0:x1] |= paint.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(marking, 8)
        minimum_area = max(4, int(round(self.config.marking_min_area_ratio*shape[0]*shape[1])))
        horizontal = []
        irregular = np.zeros(shape, np.uint8)
        lane_like = np.zeros(shape, np.uint8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < minimum_area:
                continue
            component = (labels == label).astype(np.uint8)
            features = _component_features(component)
            if features is None:
                continue
            if (features["vertical_span"] >= self.config.lane_min_vertical_span_ratio*shape[0] and
                  features["horizontal_span"] <= self.config.lane_max_width_ratio*shape[1]):
                lane_like |= component
            else:
                irregular |= component
        # Horizontal opening separates a transverse stripe from an attached
        # longitudinal yellow line before measuring angle/width.
        horizontal_kernel = np.ones(
            (1, max(3, int(round(0.06*shape[1])))), np.uint8)
        horizontal_seed = cv2.morphologyEx(
            marking, cv2.MORPH_OPEN, horizontal_kernel)
        h_count, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(
            horizontal_seed, 8)
        for label in range(1, h_count):
            if h_stats[label, cv2.CC_STAT_AREA] < minimum_area:
                continue
            component = (h_labels == label).astype(np.uint8)
            features = _component_features(component)
            if features is None or features["horizontal_error_deg"] > self.config.stop_angle_tolerance_deg:
                continue
            road_width = self._road_width(road, features["center"][1])
            if road_width <= 0:
                # A marking may be the very hole that split this row. Use
                # neighbouring road extents without authorising restoration.
                nearby = [self._road_width(road, row) for row in range(
                    max(0, int(features["center"][1])-5),
                    min(shape[0], int(features["center"][1])+6))]
                road_width = max(nearby, default=0)
            if road_width > 0:
                features["road_width"] = road_width
                features["mask"] = component
                horizontal.append(features)
        irregular &= (horizontal_seed == 0)
        stop = np.zeros(shape, np.uint8)
        stop_score = 0.0
        stop_feature = None
        for feature in horizontal:
            width_ratio = feature["length"]/max(1.0, feature["road_width"])
            thickness_ratio = feature["thickness"]/max(1.0, shape[0])
            if width_ratio < self.config.stop_min_road_width_ratio or thickness_ratio > self.config.stop_max_thickness_ratio:
                continue
            score = min(1.0, 0.40+0.55*width_ratio)
            distance_px = max(1, int(round(self.config.yellow_line_adjacency_px_ratio*shape[1])))
            adjacent = np.any(cv2.dilate(feature["mask"], np.ones((3, 2*distance_px+1), np.uint8)) & yellow)
            if adjacent:
                score = min(1.0, score+self.config.yellow_line_adjacency_bonus)
            if score > stop_score:
                stop, stop_score, stop_feature = feature["mask"], score, feature
        crosswalk = np.zeros(shape, np.uint8)
        crosswalk_score = 0.0
        if len(horizontal) >= self.config.crosswalk_min_stripes:
            ordered = sorted(horizontal, key=lambda item: item["center"][1])
            for start in range(len(ordered)):
                group = [ordered[start]]
                for candidate in ordered[start+1:]:
                    if abs(candidate["angle_deg"]-group[-1]["angle_deg"]) > self.config.crosswalk_parallel_tolerance_deg:
                        continue
                    if candidate["center"][1]-group[-1]["center"][1] > 0.18*shape[0]:
                        break
                    group.append(candidate)
                if len(group) < self.config.crosswalk_min_stripes:
                    continue
                spacing = np.diff([item["center"][1] for item in group])
                spacing_cv = float(np.std(spacing)/max(1.0, np.mean(spacing)))
                if spacing_cv <= self.config.crosswalk_spacing_cv_max:
                    candidate_mask = np.bitwise_or.reduce([item["mask"] for item in group])
                    score = min(1.0, 0.42+0.12*len(group)+0.2*(1.0-spacing_cv))
                    if score > crosswalk_score:
                        crosswalk, crosswalk_score = candidate_mask, score
        # A repeated group is not also a single stop line.
        if np.any(crosswalk) and _iou(stop, crosswalk) > 0.05:
            stop, stop_score, stop_feature = np.zeros(shape, np.uint8), 0.0, None
        diagnostics = {"horizontal_component_count": len(horizontal),
                       "stop_width_ratio": (None if stop_feature is None else
                                             stop_feature["length"]/max(1.0, stop_feature["road_width"])),
                       "lane_like_pixels": int(np.count_nonzero(lane_like))}
        return stop, stop_score, crosswalk, crosswalk_score, irregular, diagnostics

    def _confirm_candidate(self, name, mask, score, stamp, minimum_frames):
        history = self._candidate_history[name]
        history.append((float(stamp), mask.copy(), float(score), _center(mask)))
        while history and stamp-history[0][0] > self.config.track_timeout_sec:
            history.popleft()
        matching = [item for item in history if score > 0.0 and
                    (_iou(mask, item[1]) >= self.config.temporal_iou_min or
                     math.hypot(_center(mask)[0]-item[3][0], _center(mask)[1]-item[3][1]) <=
                     self.config.temporal_center_distance_ratio*math.hypot(*mask.shape))]
        confirmed = len(matching) >= minimum_frames
        accumulated = 0.0 if not matching else float(np.mean([item[2] for item in matching]))
        approaching = (len(matching) < 2 or
                       matching[-1][3][1] >= matching[0][3][1]-0.01*mask.shape[0])
        if confirmed and not approaching:
            accumulated *= 0.8
        return (mask if confirmed else np.zeros_like(mask),
                accumulated if confirmed else score*0.5, len(matching),
                bool(approaching))

    def _restore_road(self, raw_road, candidates, obstacle, stamp):
        height, width = raw_road.shape
        missing = candidates & (raw_road == 0) & (obstacle == 0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, 8)
        restored = np.zeros_like(raw_road)
        max_gap = max(2, int(round(self.config.road_restore_max_gap_ratio*height)))
        accepted, rejected = 0, 0
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            component = (labels == label).astype(np.uint8)
            if h > max_gap and w > max_gap:
                rejected += 1
                continue
            pad = max(2, min(max_gap, max(w, h)//3+1))
            x0, y0, x1, y1 = max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)
            local_road = raw_road[y0:y1, x0:x1]
            above = np.any(raw_road[max(0, y-pad):y, x:x+w])
            below = np.any(raw_road[y+h:min(height, y+h+pad), x:x+w])
            left = np.any(raw_road[y:y+h, max(0, x-pad):x])
            right = np.any(raw_road[y:y+h, x+w:min(width, x+w+pad)])
            connects = (above and below) or (left and right)
            previous_ok = (self._previous_road is not None and
                           np.count_nonzero(component & self._previous_road) >= 0.35*area)
            if not connects and not previous_ok:
                rejected += 1
                continue
            before_width = max((self._road_width(raw_road, row) for row in range(y, min(height, y+h))), default=0)
            proposed = raw_road | component
            after_width = max((self._road_width(proposed, row) for row in range(y, min(height, y+h))), default=0)
            if after_width-before_width > self.config.road_restore_max_width_growth_ratio*width:
                rejected += 1
                continue
            if np.count_nonzero(local_road) == 0 and not previous_ok:
                rejected += 1
                continue
            restored |= component
            accepted += 1
        refined = raw_road | restored
        held_pixels = 0
        if self._previous_road is not None and self._last_stamp is not None:
            age = stamp-self._last_stamp
            previous_area = np.count_nonzero(self._previous_road)
            current_area = np.count_nonzero(refined)
            if (0.0 <= age <= self.config.previous_road_hold_sec and previous_area and
                    current_area < self.config.road_drop_area_ratio*previous_area):
                hold = self._previous_road & (obstacle == 0)
                # Limit hold to a narrow dilation of current/candidate evidence.
                radius = max(1, max_gap//2)
                support = cv2.dilate((refined | candidates).astype(np.uint8),
                                     np.ones((2*radius+1, 3), np.uint8))
                hold &= support
                held_pixels = int(np.count_nonzero(hold & (refined == 0)))
                refined |= hold
        return refined, restored, {"accepted_components": accepted,
                                    "rejected_components": rejected,
                                    "held_previous_road_pixels": held_pixels}

    def refine(self, bgr, instances, raw_masks, role_class_ids, stamp_sec,
               mask_threshold=0.5):
        bgr = np.asarray(bgr, np.uint8)
        shape = bgr.shape[:2]
        stamp = float(stamp_sec)
        if self._last_stamp is not None and stamp <= self._last_stamp:
            raise ValueError("stale_or_duplicate_refinement_frame")
        raw = {name: _binary(mask, shape) for name, mask in raw_masks.items()}
        road = raw.get("road", np.zeros(shape, np.uint8))
        outputs = {name: np.zeros(shape, np.uint8) for name in
                   ("white_line", "yellow_line", "unknown_line")}
        line_diagnostics = []
        white_ids = set(role_class_ids.get("white_line", ()))
        yellow_ids = set(role_class_ids.get("yellow_line", ()))
        line_ids = white_ids | yellow_ids
        used_masks = set()
        for instance in instances:
            class_id = int(instance["class_id"])
            if class_id not in line_ids:
                continue
            mask = self._instance_mask(instance, shape, mask_threshold)
            if not np.any(mask):
                continue
            # Preserve per-instance statistics even when adjacent line masks overlap.
            key = id(instance.get("mask"))
            if key in used_masks:
                continue
            used_masks.add(key)
            yellow_score, white_score, evidence = self._line_colour_scores(bgr, mask, road)
            raw_class = "white_line" if class_id in white_ids else "yellow_line"
            stable, track_id, observed = self._stable_line_class(
                mask, stamp, raw_class, yellow_score, white_score)
            outputs[stable] |= mask
            evidence.update({"track_id": track_id, "raw_class": raw_class,
                             "observed_class": observed, "refined_class": stable,
                             "raw_confidence": float(instance.get("confidence", 0.0))})
            line_diagnostics.append(evidence)
        # Aggregated line pixels absent from instance payload remain unknown.
        raw_line = (raw.get("white_line", np.zeros(shape, np.uint8)) |
                    raw.get("yellow_line", np.zeros(shape, np.uint8)))
        assigned = outputs["white_line"] | outputs["yellow_line"] | outputs["unknown_line"]
        outputs["unknown_line"] |= raw_line & (assigned == 0)
        line_union = assigned | outputs["unknown_line"]
        stop_candidate, stop_score, crosswalk_candidate, cross_score, irregular, shape_diag = self._shape_candidates(
            bgr, road, raw, line_union, outputs["yellow_line"])
        if stop_score < self.config.stop_min_confidence:
            stop_candidate = np.zeros_like(stop_candidate)
        stop, stop_confidence, stop_frames, stop_approaching = self._confirm_candidate(
            "stop", stop_candidate, stop_score, stamp,
            self.config.stop_confirmation_frames)
        crosswalk, cross_confidence, cross_frames, cross_approaching = self._confirm_candidate(
            "crosswalk", crosswalk_candidate, cross_score, stamp,
            self.config.crosswalk_confirmation_frames)
        words = raw.get("words", np.zeros(shape, np.uint8)).copy()
        words &= (stop == 0) & (crosswalk == 0)
        marking_exclusion = (words | irregular | stop_candidate |
                             crosswalk_candidate)
        for name in outputs:
            outputs[name] &= (marking_exclusion == 0)
        unknown_marking = (irregular |
                           (stop_candidate & (stop == 0)) |
                           (crosswalk_candidate & (crosswalk == 0)))
        unknown_marking &= (words == 0) & (stop == 0) & (crosswalk == 0)
        restoration_candidates = words | stop | crosswalk | unknown_marking
        all_role_ids = set().union(*(set(value) for value in role_class_ids.values()))
        obstacle = np.zeros(shape, np.uint8)
        for instance in instances:
            if int(instance["class_id"]) not in all_role_ids:
                obstacle |= self._instance_mask(instance, shape, mask_threshold)
        refined_road, restored, restore_diag = self._restore_road(
            road, restoration_candidates, obstacle, stamp)
        self._tracks = [track for track in self._tracks
                        if stamp-track.stamp <= self.config.track_timeout_sec]
        previous_area = 0 if self._previous_road is None else int(np.count_nonzero(self._previous_road))
        self._previous_road = refined_road.copy()
        self._last_stamp = stamp
        diagnostics = {
            "stamp_sec": stamp, "line_instances": line_diagnostics,
            "raw_pixels": {name: int(np.count_nonzero(mask)) for name, mask in raw.items()},
            "refined_pixels": {"road": int(np.count_nonzero(refined_road)),
                               **{name: int(np.count_nonzero(mask)) for name, mask in outputs.items()},
                               "stop_line": int(np.count_nonzero(stop)),
                               "crosswalk": int(np.count_nonzero(crosswalk)),
                               "words": int(np.count_nonzero(words)),
                               "road_marking_unknown": int(np.count_nonzero(unknown_marking)),
                               "restored_markings": int(np.count_nonzero(restored))},
            "stop_line": {"confidence": stop_confidence,
                          "support_frames": stop_frames,
                          "approaching": stop_approaching},
            "crosswalk": {"confidence": cross_confidence,
                          "support_frames": cross_frames,
                          "approaching": cross_approaching},
            "shape": shape_diag, "road_restore": restore_diag,
            "previous_road_pixels": previous_area,
            "active_track_count": len(self._tracks),
        }
        return RefinementResult(
            refined_road*255, outputs["white_line"]*255,
            outputs["yellow_line"]*255, outputs["unknown_line"]*255,
            stop*255, stop_confidence, crosswalk*255, cross_confidence,
            words*255, unknown_marking*255, restored*255, diagnostics)
