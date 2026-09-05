"""fps_monitor 의 순수 로직(TopicRate) 단위 테스트. ROS 없이 돈다."""

from depth_slam.rate_math import TopicRate


def test_empty_rate_is_zero():
    tr = TopicRate(window_s=2.0)
    assert tr.rate() == 0.0


def test_single_sample_rate_is_zero():
    tr = TopicRate(window_s=2.0)
    tr.tick(100.0)
    assert tr.rate() == 0.0


def test_steady_60hz():
    tr = TopicRate(window_s=2.0)
    t = 0.0
    for _ in range(120):          # 2초 동안 60Hz
        tr.tick(t)
        t += 1.0 / 60.0
    r = tr.rate()
    assert 58.0 <= r <= 62.0, r


def test_window_drops_old_samples():
    tr = TopicRate(window_s=1.0)
    # 오래된 샘플 2개 + 최근 60Hz 1초
    tr.tick(0.0)
    tr.tick(0.5)
    t = 10.0
    for _ in range(60):
        tr.tick(t)
        t += 1.0 / 60.0
    # 윈도우(1초) 밖의 0.0, 0.5 는 빠지고 최근 것만 남아야 한다
    assert all(s >= 9.0 for s in tr.stamps)


def test_min_rate_tracks_lowest():
    tr = TopicRate(window_s=5.0)
    # 30Hz 로 시작
    t = 0.0
    for _ in range(150):
        tr.tick(t)
        t += 1.0 / 30.0
    _ = tr.rate()
    assert tr.min_rate <= 31.0
