"""토픽 rate 계산 순수 로직 — ROS 의존 없음. pytest 로 검증한다.

fps_monitor_node 는 이 클래스를 import 만 한다. 부작용이 없어
차/카메라 없이도 테스트할 수 있다 (MCU 스택의 순수함수 분리와 동일 원칙).
"""

from collections import deque


class TopicRate:
    def __init__(self, window_s: float):
        self.window_s = float(window_s)
        self.stamps = deque()
        self.min_rate = float("inf")

    def tick(self, now: float) -> None:
        self.stamps.append(now)
        cutoff = now - self.window_s
        while self.stamps and self.stamps[0] < cutoff:
            self.stamps.popleft()

    def rate(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        if span <= 0.0:
            return 0.0
        r = (len(self.stamps) - 1) / span
        if r > 0.0:
            self.min_rate = min(self.min_rate, r)
        return r
