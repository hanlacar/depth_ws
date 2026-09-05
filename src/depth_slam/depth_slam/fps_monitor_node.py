#!/usr/bin/env python3
"""
fps_monitor_node.py  —  토픽별 실측 발행 rate 를 롤링 윈도우로 측정한다.

왜 필요한가
    "출력 45~50 FPS" 목표가 실제로 달성되는지 눈으로 확인할 계측기다.
    /odom(=VO 출력)이 주 판정 지표이고, rgb/depth 입력이 60 이 실제로
    들어오는지도 같이 본다. 합성값이 아니라 실측만 신뢰한다.

    ros2 topic hz 와 달리, 여러 토픽을 한 화면에서 동시에, 최소/평균까지
    보여줘 정지 시험의 순간 dip 을 잡아낸다.

측정 방식
    각 토픽 메시지 도착 시각을 window_s 초 버퍼에 쌓고,
    rate = (버퍼 내 샘플 수 - 1) / (마지막-처음 시간) 로 계산.
    타입에 무관하게 헤더를 안 봐도 되도록 도착시각(수신 rate)만 센다.
"""

import time

import rclpy
from rosidl_runtime_py.utilities import get_message

from depth_slam.rate_math import TopicRate
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class FpsMonitor(Node):
    def __init__(self):
        super().__init__("fps_monitor")
        self.declare_parameter("topics", ["/odom"])
        self.declare_parameter("window_s", 2.0)
        self.declare_parameter("report_period_s", 1.0)

        topics = list(self.get_parameter("topics").value)
        window_s = float(self.get_parameter("window_s").value)
        report_period = float(self.get_parameter("report_period_s").value)

        # 타입을 미리 모르므로, 그래프에서 타입을 조회해 generic 구독한다.
        # 구독 시점에 아직 퍼블리셔가 없으면 잠시 후 재시도한다.
        self.window_s = window_s
        self.meters = {t: TopicRate(window_s) for t in topics}
        self.subs = {}
        self._pending = list(topics)

        # best-effort 센서 QoS (카메라 스트림 호환)
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_timer(0.5, self._try_subscribe)
        self.create_timer(report_period, self._report)

    def _try_subscribe(self):
        if not self._pending:
            return
        name_types = dict(self.get_topic_names_and_types())
        still = []
        for t in self._pending:
            types = name_types.get(t)
            if not types:
                still.append(t)
                continue
            try:
                msg_type = get_message(types[0])
            except Exception:
                still.append(t)
                continue
            self.subs[t] = self.create_subscription(
                msg_type, t,
                lambda _msg, key=t: self.meters[key].tick(time.monotonic()),
                self.sensor_qos,
            )
            self.get_logger().info("구독 시작: %s (%s)" % (t, types[0]))
        self._pending = still

    def _report(self):
        parts = []
        for t, m in self.meters.items():
            cur = m.rate()
            mn = m.min_rate if m.min_rate != float("inf") else 0.0
            parts.append("%s: %5.1f Hz (min %4.1f)" % (t, cur, mn))
        self.get_logger().info(" | ".join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = FpsMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
