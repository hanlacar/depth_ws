"""ROS-independent, latest-result traffic-light fusion policy."""

from dataclasses import dataclass
import math


STATES = frozenset(("R", "G", "UNKNOWN"))
ASPECTS = frozenset(("RED", "RED_X", "YELLOW", "GREEN_CIRCLE",
                     "GREEN_LEFT", "GREEN_DOWN", "GREEN_OTHER", "UNKNOWN"))
STOP_ASPECTS = frozenset(("RED", "RED_X", "YELLOW"))
GO_ASPECTS = frozenset(("GREEN_CIRCLE", "GREEN_LEFT", "GREEN_DOWN",
                        "GREEN_OTHER"))

# Names are model metadata carried by detections_json, not class indices.
# The current manifest maps these names to semantic roles and therefore keeps
# this safe when model class order changes.
YOLO_CLASS_ASPECT = {
    "R_light": ("R", "UNKNOWN"),  # training merges red and yellow
    "G_light": ("G", "UNKNOWN"),
    "Left": ("G", "GREEN_LEFT"),
    "etc_light": ("G", "GREEN_OTHER"),
}


@dataclass(frozen=True)
class FusionConfig:
    fusion_yolo_timeout_sec: float = 0.50
    fusion_rgb_timeout_sec: float = 0.50
    fusion_max_stamp_delta_sec: float = 0.08
    fusion_min_yolo_confidence: float = 0.50
    fusion_min_rgb_confidence: float = 0.55
    fusion_confirm_frames: int = 3
    fusion_switch_confirm_frames: int = 4
    fusion_conflict_margin: float = 0.15
    fusion_single_source_confirm_frames: int = 4
    fusion_bbox_iou_threshold: float = 0.05
    fusion_bbox_center_distance_px: float = 80.0

    def validate(self):
        numeric = tuple(vars(self).values())
        if not all(math.isfinite(float(value)) and float(value) >= 0.0
                   for value in numeric):
            raise ValueError("fusion parameters must be finite and nonnegative")
        if min(self.fusion_yolo_timeout_sec,
               self.fusion_rgb_timeout_sec) <= 0.0:
            raise ValueError("fusion timeouts must be positive")
        if min(self.fusion_confirm_frames, self.fusion_switch_confirm_frames,
               self.fusion_single_source_confirm_frames) < 1:
            raise ValueError("fusion confirmation counts must be positive")
        if not 0.0 <= self.fusion_bbox_iou_threshold <= 1.0:
            raise ValueError("fusion_bbox_iou_threshold must be in [0,1]")
        return self


@dataclass(frozen=True)
class SourceObservation:
    source: str
    state: str
    aspect: str
    confidence: float
    stamp: float
    received_at: float
    sequence: int
    class_name: str = ""
    bbox: tuple | None = None


@dataclass(frozen=True)
class FusionDecision:
    state: str
    aspect: str
    confidence: float
    reason: str
    diagnostics: dict


def combine_confidence(first, second):
    """Independent-evidence union, bounded to [0,1]."""
    first, second = float(first), float(second)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in (first, second)):
        return 0.0
    return min(1.0, 1.0-(1.0-first)*(1.0-second))


def state_for_aspect(aspect):
    if aspect in STOP_ASPECTS:
        return "R"
    if aspect in GO_ASPECTS:
        return "G"
    return "UNKNOWN"


def _finite_bbox(value):
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (len(values) != 4 or not all(math.isfinite(item) for item in values) or
            values[2] <= values[0] or values[3] <= values[1]):
        return None
    return values


def normalize_yolo_document(document, received_at, sequence):
    """Select conservative traffic evidence from detections_json."""
    try:
        timestamp = document["timestamp"]
        stamp = float(timestamp["sec"])+float(timestamp["nanosec"])*1.0e-9
        detections = document.get("detections", ())
    except (KeyError, TypeError, ValueError):
        return None, "YOLO_DOCUMENT_INVALID"
    if not math.isfinite(stamp) or stamp < 0.0:
        return None, "YOLO_TIMESTAMP_INVALID"
    candidates = []
    for item in detections:
        name = str(item.get("class_name", ""))
        mapping = YOLO_CLASS_ASPECT.get(name)
        try:
            confidence = float(item.get("confidence", math.nan))
        except (TypeError, ValueError):
            continue
        if mapping is None or not math.isfinite(confidence) or not (
                0.0 <= confidence <= 1.0):
            continue
        candidates.append((confidence, name, mapping, _finite_bbox(
            item.get("xyxy"))))
    if not candidates:
        return SourceObservation("YOLO", "UNKNOWN", "UNKNOWN", 0.0, stamp,
                                 float(received_at), int(sequence)), "NO_YOLO_LIGHT"
    families = {item[2][0] for item in candidates}
    if len(families) > 1:
        confidence = max(item[0] for item in candidates)
        names = "+".join(sorted({item[1] for item in candidates}))
        return SourceObservation("YOLO", "UNKNOWN", "UNKNOWN", confidence,
                                 stamp, float(received_at), int(sequence), names), \
            "YOLO_INTERNAL_RG_CONFLICT"
    candidates.sort(reverse=True, key=lambda item: item[0])
    confidence, name, (state, aspect), bbox = candidates[0]
    detailed = {item[2][1] for item in candidates if item[2][1] != "UNKNOWN"}
    if len(detailed) > 1:
        return SourceObservation("YOLO", "UNKNOWN", "UNKNOWN", confidence,
                                 stamp, float(received_at), int(sequence), name,
                                 bbox), "YOLO_INTERNAL_ASPECT_CONFLICT"
    if detailed:
        aspect = next(iter(detailed))
        detailed_item = max((item for item in candidates
                             if item[2][1] == aspect), key=lambda item: item[0])
        confidence, name, bbox = detailed_item[0], detailed_item[1], detailed_item[3]
    return SourceObservation("YOLO", state, aspect, confidence, stamp,
                             float(received_at), int(sequence), name, bbox), "OK"


def normalize_rgb_diagnostics(document, received_at, sequence):
    try:
        state = str(document.get("state", "UNKNOWN")).upper()
        aspect = str(document.get("aspect", "UNKNOWN")).upper()
        confidence = float(document.get("confidence", math.nan))
        stamp = float(document["stamp"])*1.0e-9
    except (KeyError, TypeError, ValueError):
        return None, "RGB_DOCUMENT_INVALID"
    if (state not in STATES or aspect not in ASPECTS or
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0 or
            not math.isfinite(stamp) or stamp < 0.0):
        return None, "RGB_DOCUMENT_INVALID"
    aspect_state = state_for_aspect(aspect)
    if aspect != "UNKNOWN" and aspect_state != state:
        return None, "RGB_STATE_ASPECT_MISMATCH"
    return SourceObservation(
        "RGB", state, aspect, confidence, stamp, float(received_at),
        int(sequence), bbox=_finite_bbox(document.get("selected_bbox"))), "OK"


def _bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = first; bx1, by1, bx2, by2 = second
    area = max(0.0, min(ax2, bx2)-max(ax1, bx1))*max(
        0.0, min(ay2, by2)-max(ay1, by1))
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-area
    return area/max(union, 1.0e-9)


def positions_match(first, second, config):
    if first is None or second is None:
        return None
    if _bbox_iou(first, second) >= config.fusion_bbox_iou_threshold:
        return True
    ac = ((first[0]+first[2])/2.0, (first[1]+first[3])/2.0)
    bc = ((second[0]+second[2])/2.0, (second[1]+second[3])/2.0)
    return math.hypot(ac[0]-bc[0], ac[1]-bc[1]) <= \
        config.fusion_bbox_center_distance_px


class TrafficLightFusion:
    def __init__(self, config=FusionConfig()):
        self.config = config.validate()
        self.yolo = None
        self.rgb = None
        self.last_stamps = {"YOLO": None, "RGB": None}
        self.rejections = {}
        self.confirmed_key = None
        self.confirmed_confidence = 0.0
        self.pending_key = None
        self.pending_count = 0
        self.last_evidence_token = None
        self.last_counted_sequences = {"YOLO": 0, "RGB": 0}

    def _reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0)+1

    def ingest(self, observation):
        if observation is None or observation.source not in self.last_stamps:
            self._reject("INVALID_OBSERVATION")
            return False
        if not all(math.isfinite(value) for value in (
                observation.stamp, observation.received_at,
                observation.confidence)):
            self._reject(f"{observation.source}_NONFINITE")
            return False
        previous = self.last_stamps[observation.source]
        if previous is not None and observation.stamp < previous:
            self._reject(f"{observation.source}_TIMESTAMP_REWIND")
            return False
        self.last_stamps[observation.source] = observation.stamp
        if observation.source == "YOLO":
            self.yolo = observation
        else:
            self.rgb = observation
        return True

    def _source_status(self, observation, now, timeout, minimum):
        age = None if observation is None else max(
            0.0, float(now)-observation.received_at)
        fresh = bool(observation is not None and age <= timeout)
        reliable = bool(fresh and observation.state in ("R", "G") and
                        math.isfinite(observation.confidence) and
                        observation.confidence >= minimum)
        return age, fresh, reliable

    def _raw_candidate(self, now):
        ya, yf, yr = self._source_status(
            self.yolo, now, self.config.fusion_yolo_timeout_sec,
            self.config.fusion_min_yolo_confidence)
        ra, rf, rr = self._source_status(
            self.rgb, now, self.config.fusion_rgb_timeout_sec,
            self.config.fusion_min_rgb_confidence)
        delta = (None if self.yolo is None or self.rgb is None else
                 abs(self.yolo.stamp-self.rgb.stamp))
        paired = bool(yr and rr and delta is not None and
                      delta <= self.config.fusion_max_stamp_delta_sec)
        meta = {"yolo_age": ya, "yolo_fresh": yf, "yolo_reliable": yr,
                "rgb_age": ra, "rgb_fresh": rf, "rgb_reliable": rr,
                "stamp_delta": delta, "paired": paired,
                "agree": False, "conflict": False, "single": False,
                "confidence_gap": (None if self.yolo is None or self.rgb is None
                    else abs(self.yolo.confidence-self.rgb.confidence)),
                "position_match": None}
        if paired:
            yolo, rgb = self.yolo, self.rgb
            if yolo.state != rgb.state:
                meta["conflict"] = True
                return None, "CONFLICT", ("PAIR", yolo.sequence, rgb.sequence), meta
            position = positions_match(yolo.bbox, rgb.bbox, self.config)
            meta["position_match"] = position
            if position is False:
                meta["conflict"] = True
                return None, "CONFLICT", ("PAIR", yolo.sequence, rgb.sequence), meta
            if (yolo.aspect != "UNKNOWN" and rgb.aspect != "UNKNOWN" and
                    yolo.aspect != rgb.aspect):
                meta["conflict"] = True
                return None, "CONFLICT", ("PAIR", yolo.sequence, rgb.sequence), meta
            aspect = rgb.aspect if rgb.aspect != "UNKNOWN" else yolo.aspect
            confidence = combine_confidence(yolo.confidence, rgb.confidence)
            reason = ("RGB_SHAPE_WITH_YOLO_GREEN" if yolo.state == "G" and
                      yolo.aspect == "UNKNOWN" and rgb.aspect != "UNKNOWN"
                      else "AGREED")
            meta["agree"] = True
            return (yolo.state, aspect, confidence, "PAIR"), reason, \
                ("PAIR", yolo.sequence, rgb.sequence), meta

        usable = []
        if yr:
            usable.append(self.yolo)
        if rr:
            usable.append(self.rgb)
        if len(usable) == 2:
            # Different source stamps are never treated as a same-frame clash.
            usable = [max(usable, key=lambda item: item.stamp)]
        if usable:
            source = usable[0]
            if now-source.received_at < self.config.fusion_max_stamp_delta_sec:
                return None, "WAITING_CONFIRMATION", ("WAIT", source.source,
                    source.sequence), meta
            meta["single"] = True
            reason = f"{source.source}_ONLY_CONFIRMED"
            return (source.state, source.aspect, source.confidence,
                    source.source), reason, (source.source, source.sequence), meta
        if not yf and not rf:
            return None, "NO_INPUT" if self.yolo is None and self.rgb is None \
                else "STALE", ("EMPTY",), meta
        return None, "LOW_CONFIDENCE", ("INVALID",), meta

    def evaluate(self, now, route_mode=None):
        now = float(now)
        raw, reason, token, meta = self._raw_candidate(now)
        if raw is None:
            if reason in ("CONFLICT", "STALE", "NO_INPUT", "LOW_CONFIDENCE"):
                self.confirmed_key = None
                self.confirmed_confidence = 0.0
                self.pending_key = None
                self.pending_count = 0
            return self._decision(now, "UNKNOWN", "UNKNOWN", 0.0, reason,
                                  meta, route_mode)
        state, aspect, confidence, basis = raw
        key = (state, aspect, basis)
        if key == self.confirmed_key:
            self.confirmed_confidence = confidence
            return self._decision(now, state, aspect, confidence, reason, meta,
                                  route_mode)
        if basis == "PAIR":
            new_evidence = bool(
                self.yolo.sequence > self.last_counted_sequences["YOLO"] and
                self.rgb.sequence > self.last_counted_sequences["RGB"])
        else:
            source = self.yolo if basis == "YOLO" else self.rgb
            new_evidence = source.sequence > self.last_counted_sequences[basis]
        if new_evidence and token != self.last_evidence_token:
            if key == self.pending_key:
                self.pending_count += 1
            else:
                self.pending_key, self.pending_count = key, 1
            self.last_evidence_token = token
            if basis == "PAIR":
                self.last_counted_sequences["YOLO"] = self.yolo.sequence
                self.last_counted_sequences["RGB"] = self.rgb.sequence
            else:
                self.last_counted_sequences[basis] = source.sequence
        required = (self.config.fusion_single_source_confirm_frames
                    if basis != "PAIR" else self.config.fusion_confirm_frames)
        if self.confirmed_key is not None:
            required = max(required, self.config.fusion_switch_confirm_frames)
        if self.pending_count >= required:
            self.confirmed_key = key
            self.confirmed_confidence = confidence
            self.pending_key = None
            self.pending_count = 0
            return self._decision(now, state, aspect, confidence, reason, meta,
                                  route_mode)
        return self._decision(now, "UNKNOWN", "UNKNOWN", 0.0,
                              "WAITING_CONFIRMATION", meta, route_mode)

    def _decision(self, now, state, aspect, confidence, reason, meta,
                  route_mode):
        yolo, rgb = self.yolo, self.rgb
        diagnostics = {
            "stamp": (None if yolo is None and rgb is None else int(max(
                item.stamp for item in (yolo, rgb) if item is not None)*1.0e9)),
            "fused_state": state, "fused_aspect": aspect,
            "fused_confidence": float(confidence), "fusion_reason": reason,
            "yolo_state": "UNKNOWN" if yolo is None else yolo.state,
            "yolo_aspect": "UNKNOWN" if yolo is None else yolo.aspect,
            "yolo_confidence": 0.0 if yolo is None else yolo.confidence,
            "yolo_age_sec": meta["yolo_age"],
            "yolo_fresh": meta["yolo_fresh"],
            "yolo_class_name": "" if yolo is None else yolo.class_name,
            "rgb_state": "UNKNOWN" if rgb is None else rgb.state,
            "rgb_aspect": "UNKNOWN" if rgb is None else rgb.aspect,
            "rgb_confidence": 0.0 if rgb is None else rgb.confidence,
            "rgb_age_sec": meta["rgb_age"], "rgb_fresh": meta["rgb_fresh"],
            "stamp_delta_sec": meta["stamp_delta"],
            "confidence_gap": meta["confidence_gap"],
            "fusion_conflict_margin": self.config.fusion_conflict_margin,
            "positions_match": meta["position_match"],
            "sources_agree": meta["agree"],
            "sources_conflict": meta["conflict"],
            "single_source_used": meta["single"],
            "confirmation_count": self.pending_count,
            "rejection_reasons": dict(self.rejections),
            "route_mode": route_mode,
            "rgb_green_down_verified": bool(
                state == "G" and aspect == "GREEN_DOWN" and rgb is not None and
                meta["rgb_fresh"] and meta["rgb_reliable"] and
                rgb.aspect == "GREEN_DOWN" and
                not (meta["yolo_reliable"] and yolo.state == "R")),
        }
        return FusionDecision(state, aspect, float(confidence), reason,
                              diagnostics)
