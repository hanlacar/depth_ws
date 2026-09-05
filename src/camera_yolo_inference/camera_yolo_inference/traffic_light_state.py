"""ROS-independent temporal traffic-light state estimation."""

from dataclasses import dataclass
import math

import numpy as np


UNKNOWN = "UNKNOWN"
ROLE_TO_STATE = {
    "red_light": "R",
    "green_light": "G",
}
PUBLISHED_STATES = frozenset(("R", "G", UNKNOWN))


@dataclass(frozen=True)
class TrafficLightConfig:
    traffic_light_confirmation_frames: int = 3
    traffic_light_lost_timeout_sec: float = 0.5
    traffic_light_minimum_confidence: float = 0.5
    traffic_light_conflict_score_margin: float = 0.15
    traffic_light_mask_area_weight: float = 0.2
    traffic_light_mask_threshold: float = 0.5

    def validate(self):
        if self.traffic_light_confirmation_frames < 1:
            raise ValueError("traffic_light_confirmation_frames must be at least 1")
        finite = (
            self.traffic_light_lost_timeout_sec,
            self.traffic_light_minimum_confidence,
            self.traffic_light_conflict_score_margin,
            self.traffic_light_mask_area_weight,
            self.traffic_light_mask_threshold,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("traffic-light parameters must be finite")
        if self.traffic_light_lost_timeout_sec <= 0.0:
            raise ValueError("traffic_light_lost_timeout_sec must be positive")
        if not 0.0 <= self.traffic_light_minimum_confidence <= 1.0:
            raise ValueError("traffic_light_minimum_confidence must be in [0, 1]")
        if self.traffic_light_conflict_score_margin < 0.0:
            raise ValueError("traffic_light_conflict_score_margin must be nonnegative")
        if self.traffic_light_mask_area_weight < 0.0:
            raise ValueError("traffic_light_mask_area_weight must be nonnegative")
        if not 0.0 <= self.traffic_light_mask_threshold <= 1.0:
            raise ValueError("traffic_light_mask_threshold must be in [0, 1]")


@dataclass(frozen=True)
class TrafficLightEvidence:
    state: str
    confidence: float
    mask_area: float
    score: float


@dataclass(frozen=True)
class TrafficLightDecision:
    state: str
    candidate: str | None
    reason: str
    confirmation_count: int
    evidences: tuple


def _instance_area(instance, threshold):
    mask = instance.get("mask")
    if mask is not None:
        array = np.asarray(mask, dtype=np.float64)
        if array.ndim == 2 and np.isfinite(array).all():
            return float(np.count_nonzero(array >= threshold))
    xyxy = instance.get("xyxy", ())
    try:
        if len(xyxy) == 4:
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            if all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                return max(0.0, x2-x1)*max(0.0, y2-y1)
    except (TypeError, ValueError):
        pass
    return 0.0


def collect_traffic_light_evidence(instances, role_class_ids,
                                   config=TrafficLightConfig()):
    """Aggregate the strongest valid instance for each R/G semantic role.

    The 11-class model labels physical red and yellow lamps alike as
    ``R_light``.  Both therefore produce the same stop state, ``R``.
    """
    config.validate()
    class_to_state = {}
    for role, state in ROLE_TO_STATE.items():
        for class_id in role_class_ids.get(role, ()):
            class_to_state[int(class_id)] = state

    raw = []
    for instance in instances:
        try:
            state = class_to_state.get(int(instance["class_id"]))
            confidence = float(instance.get("confidence", float("nan")))
        except (KeyError, TypeError, ValueError):
            continue
        if (state is None or not math.isfinite(confidence) or
                confidence < config.traffic_light_minimum_confidence):
            continue
        area = _instance_area(instance, config.traffic_light_mask_threshold)
        if not math.isfinite(area) or area <= 0.0:
            continue
        raw.append((state, confidence, area))

    if not raw:
        return ()
    maximum_area = max(item[2] for item in raw)
    strongest = {}
    for state, confidence, area in raw:
        normalized_area = area/maximum_area if maximum_area > 0.0 else 0.0
        score = confidence + config.traffic_light_mask_area_weight*normalized_area
        evidence = TrafficLightEvidence(state, confidence, area, score)
        previous = strongest.get(state)
        if previous is None or evidence.score > previous.score:
            strongest[state] = evidence
    return tuple(sorted(strongest.values(), key=lambda item: item.score,
                        reverse=True))


def choose_candidate(evidences, confirmed_state, config=TrafficLightConfig()):
    """Resolve multi-color frames without a fixed color priority."""
    if not evidences:
        return None, "no_reliable_detection"
    if len(evidences) == 1:
        return evidences[0].state, "single_color"

    top, second = evidences[:2]
    if top.score-second.score >= config.traffic_light_conflict_score_margin:
        return top.state, "conflict_score_winner"
    if confirmed_state in ROLE_TO_STATE.values():
        previous = next(
            (item for item in evidences if item.state == confirmed_state), None)
        if (previous is not None and
                top.score-previous.score <=
                config.traffic_light_conflict_score_margin):
            return confirmed_state, "conflict_keep_confirmed"
    return None, "ambiguous_multi_color"


class TrafficLightFilter:
    """N-frame confirmation plus lost-detection timeout state machine."""

    def __init__(self, config=TrafficLightConfig()):
        config.validate()
        self.config = config
        self.confirmed_state = UNKNOWN
        self.pending_state = None
        self.confirmation_count = 0
        self.last_reliable_detection_at = None
        self.last_reason = "not_observed"
        self.last_evidences = ()

    def update(self, instances, role_class_ids, now):
        now = float(now)
        if not math.isfinite(now):
            return self.invalidate("nonfinite_time")
        evidences = collect_traffic_light_evidence(
            instances, role_class_ids, self.config)
        candidate, reason = choose_candidate(
            evidences, self.confirmed_state, self.config)
        self.last_evidences = evidences
        self.last_reason = reason

        if candidate is None:
            self.pending_state = None
            self.confirmation_count = 0
            return self.tick(now)

        if candidate == self.confirmed_state:
            self.last_reliable_detection_at = now
            self.pending_state = None
            self.confirmation_count = 0
        elif candidate == self.pending_state:
            self.confirmation_count += 1
        else:
            self.pending_state = candidate
            self.confirmation_count = 1

        if self.confirmation_count >= self.config.traffic_light_confirmation_frames:
            self.confirmed_state = candidate
            self.last_reliable_detection_at = now
            self.pending_state = None
            self.confirmation_count = 0
            self.last_reason = "state_confirmed"
        elif (self.confirmed_state != UNKNOWN and
              self.last_reliable_detection_at is not None and
              now-self.last_reliable_detection_at >
              self.config.traffic_light_lost_timeout_sec):
            return self.tick(now)
        return self.decision(candidate)

    def tick(self, now):
        now = float(now)
        if (not math.isfinite(now) or self.last_reliable_detection_at is None or
                now-self.last_reliable_detection_at >
                self.config.traffic_light_lost_timeout_sec):
            self.confirmed_state = UNKNOWN
            self.pending_state = None
            self.confirmation_count = 0
            self.last_reason = "detection_timeout"
        return self.decision(None)

    def invalidate(self, reason="invalid_perception"):
        self.confirmed_state = UNKNOWN
        self.pending_state = None
        self.confirmation_count = 0
        self.last_reliable_detection_at = None
        self.last_reason = str(reason)
        self.last_evidences = ()
        return self.decision(None)

    def decision(self, candidate=None):
        state = (self.confirmed_state if self.confirmed_state in PUBLISHED_STATES
                 else UNKNOWN)
        return TrafficLightDecision(
            state, candidate, self.last_reason, self.confirmation_count,
            self.last_evidences)
