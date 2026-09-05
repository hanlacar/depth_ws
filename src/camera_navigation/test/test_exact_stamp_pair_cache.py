from camera_navigation.timestamp_sync import ExactStampPairCache


def test_reverse_arrival_order_matches_same_stamp_once():
    cache = ExactStampPairCache(4, 0.1)
    assert cache.add_state(10, {"state": "DEGRADED"}, 1_000) is None
    pair = cache.add_path(10, [(1.0, 0.2)], 2_000)
    assert pair.stamp_ns == 10 and pair.path == [(1.0, 0.2)]
    assert cache.add_path(10, [], 3_000) is None
    assert cache.stats()["matched"] == 1


def test_different_stamps_never_match():
    cache = ExactStampPairCache(4, 0.1)
    assert cache.add_path(10, [], 1_000) is None
    assert cache.add_state(11, {}, 2_000) is None
    assert cache.stats()["matched"] == 0


def test_stale_data_is_expired():
    cache = ExactStampPairCache(4, 0.000001)
    cache.add_path(10, [], 1_000)
    assert cache.expire(3_000) == [10]
    assert len(cache) == 0 and cache.stats()["expired"] == 1


def test_cache_is_bounded():
    cache = ExactStampPairCache(2, 1.0)
    for stamp in (1, 2, 3):
        cache.add_path(stamp, [], stamp)
    assert len(cache) == 2
    assert cache.stats()["capacity_dropped"] == 1


def test_invalid_discard_prevents_old_path_reuse():
    cache = ExactStampPairCache(4, 1.0)
    cache.add_path(10, [(1, 1)], 1)
    cache.discard_through(10)
    assert cache.add_state(10, {"state": "DEGRADED"}, 2) is None
    assert cache.stats()["matched"] == 0
