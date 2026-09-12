import pytest

from depth_hybrid_slam.current_mcu_contract import (
    MODE_OWNERSHIP, ownership_for_mode)


def test_all_modes_zero_through_eleven_are_documented():
    assert set(MODE_OWNERSHIP) == set(range(12))


@pytest.mark.parametrize("mode,drive,wheel", [
    (0, "CAMERA_IF_AUTH_ELSE_GPS", "CAMERA_IF_AUTH_ELSE_GPS"),
    (1, "CAMERA_IF_AUTH_ELSE_GPS", "CAMERA_IF_AUTH_ELSE_GPS"),
    (2, "CAMERA_THEN_GPS", "CAMERA_THEN_GPS"),
    (3, "CAMERA_IF_AUTH_ELSE_GPS", "CAMERA_IF_AUTH_ELSE_GPS"),
    (4, "CAMERA_STOP_OR_GPS", "GPS"),
    (5, "LIDAR_IF_ACTIVE_ELSE_CAMERA_OR_GPS",
     "LIDAR_IF_ACTIVE_ELSE_CAMERA_OR_GPS"),
    (6, "CAMERA_STOP_OR_GPS", "GPS"),
    (7, "LIDAR", "LIDAR"),
    (8, "CAMERA_IF_AUTH_ELSE_GPS", "CAMERA_IF_AUTH_ELSE_GPS"),
    (9, "LIDAR_WITH_OPTIONAL_CAMERA_LIMIT", "CAMERA_THEN_GPS"),
    (10, "LIDAR", "LIDAR"),
    (11, "CAMERA_IF_AUTH_ELSE_GPS", "CAMERA_IF_AUTH_ELSE_GPS"),
])
def test_mode_owner_matches_current_manager_source(mode, drive, wheel):
    row = ownership_for_mode(str(mode))
    assert row.drive_owner == drive
    assert row.wheel_owner == wheel
    assert row.stop_owner == "ANY_FRESH_SOURCE_OR_ESTOP"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ownership_for_mode(12)
