"""Independent VSLAM/user START A/B double-validation policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StartValidationDecision:
    final: bool
    passed: bool
    stop: bool
    detected_branch: str
    reason: str


class StartDoubleValidator:
    """Latch exactly one terminal result from a VSLAM route classifier."""

    def __init__(self, user_branch, classifier, timeout_s=30.0):
        branch = str(user_branch).strip().upper()
        if branch not in ("A", "B"):
            raise ValueError("user branch must be A or B")
        self.user_branch = branch
        self.classifier = classifier
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0.0:
            raise ValueError("START validation timeout must be positive")
        self.started_at = None
        self.result = None

    def update(self, pose, tracking_valid, now):
        now = float(now)
        if self.result is not None:
            return self.result
        if not bool(tracking_valid) or pose is None:
            return StartValidationDecision(
                False, False, True, "",
                "waiting for VSLAM START classification")
        # Sensor/MCU launches are intentionally split. Do not start an
        # unavailable timer before the first real visual-localization pose.
        if self.started_at is None:
            self.started_at = now
        classified = self.classifier.classify(pose)
        if classified.branch in ("A", "B"):
            passed = classified.branch == self.user_branch
            reason = (
                f"START_{self.user_branch} double validated" if passed else
                f"VSLAM={classified.branch} USER={self.user_branch}")
            self.result = StartValidationDecision(
                True, passed, not passed, classified.branch, reason)
            return self.result
        if now - self.started_at >= self.timeout_s:
            reason = "VSLAM START branch ambiguous"
            self.result = StartValidationDecision(
                True, False, True, "", reason)
            return self.result
        return StartValidationDecision(
            False, False, True, "", "waiting for VSLAM START classification")
