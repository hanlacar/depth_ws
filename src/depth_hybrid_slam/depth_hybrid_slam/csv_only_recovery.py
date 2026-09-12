"""Test-only lateral disturbance and bounded closed-loop recovery helpers."""

from dataclasses import asdict, dataclass
import math

from .models import Pose2D
from .route_follower_core import RouteFollower
from .virtual_mcu_core import VirtualAckermannVehicle


@dataclass(frozen=True)
class RecoveryReport:
    segment_id: str
    point_index: int
    side: str
    offset_m: float
    initial_cte_m: float
    maximum_cte_m: float
    minimum_cte_m: float
    decrease_started_s: float | None
    recovered_025_s: float | None
    recovered_010: bool
    stop_occurred: bool
    stop_reason: str
    wrong_segment_jump: bool
    direction_jump: bool
    cursor_backtrack: bool
    maximum_steering_deg: float
    result: str

    def dictionary(self):
        return asdict(self)


def lateral_offset_xy(x, y, route_yaw, offset_m):
    """Apply signed route-normal offset: positive is route-left."""
    values = tuple(float(value) for value in (x, y, route_yaw, offset_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("lateral disturbance values must be finite")
    x, y, yaw, offset = values
    return x-math.sin(yaw)*offset, y+math.cos(yaw)*offset


def virtual_vehicle():
    return VirtualAckermannVehicle(
        {1: 0.527, 2: 0.791, 3: 1.055}, 0.527,
        wheelbase_m=0.730, counts_per_meter=797.0,
        max_steering_deg=22.0, direction_change_hold_s=3.0,
        encoder_signed=False)


def simulate_lateral_recovery(route, start_index, offset_m, duration_s=10.0,
                              dt_s=0.05, corridor_m=1.0):
    """Run the production follower law and calibrated vehicle after one offset.

    The follower cursor is seeded at the active point.  Consequently every
    nearest query uses its bounded forward/local window; global search is
    never enabled during recovery.
    """
    points = tuple(route)
    index = int(start_index)
    if not 0 <= index < len(points)-1:
        raise ValueError("start_index must address a non-final route point")
    offset = float(offset_m)
    if not math.isfinite(offset) or abs(offset) > 2.1+1.0e-9:
        raise ValueError("test disturbance must be finite and within +/-2.1 m")
    point = points[index]
    follower = RouteFollower(
        wheelbase=0.730, max_steering_deg=22.0, corridor_m=corridor_m,
        max_index_backtrack=0)
    follower.last_index = index
    vehicle = virtual_vehicle()
    vehicle.x, vehicle.y = lateral_offset_xy(
        point.x, point.y, point.yaw, offset)
    vehicle.yaw = point.yaw

    initial = maximum = 0.0
    minimum = float("inf")
    decrease_time = None
    recovery_025 = None
    recovered_010 = False
    stop = False
    stop_reason = ""
    wrong_segment = False
    direction_jump = False
    backtrack = False
    maximum_steering = 0.0
    prior_cursor = index
    iterations = max(1, int(math.ceil(float(duration_s)/float(dt_s))))
    for step in range(iterations):
        now = step*float(dt_s)
        result = follower.compute(
            Pose2D(vehicle.x, vehicle.y, vehicle.yaw, now), points,
            dt=float(dt_s), allow_motion=True, global_search=False)
        cte = abs(float(result.cross_track_error))
        if step == 0:
            initial = cte
        maximum = max(maximum, cte)
        minimum = min(minimum, cte)
        if decrease_time is None and cte <= initial-0.01:
            decrease_time = now
        if recovery_025 is None and cte <= 0.25:
            recovery_025 = now
        recovered_010 |= cte <= 0.10
        stop |= bool(result.stop_required)
        if result.stop_required and not stop_reason:
            stop_reason = result.reason
        maximum_steering = max(maximum_steering, abs(result.steering_deg))
        cursor = int(result.nearest_index)
        backtrack |= cursor < prior_cursor
        active = points[min(max(0, cursor), len(points)-1)]
        wrong_segment |= active.segment_id != point.segment_id
        direction_jump |= active.direction != point.direction
        prior_cursor = cursor
        vehicle.step(
            result.drive, result.steering_deg, result.stop_required,
            float(dt_s))

    inside_policy = abs(offset) <= float(corridor_m)+1.0e-9
    recovered = recovery_025 is not None and recovered_010
    passed = (
        recovered and not stop and not wrong_segment and not direction_jump and
        not backtrack if inside_policy else
        stop and stop_reason == "CORRIDOR_VIOLATION" and not wrong_segment and
        not direction_jump and not backtrack)
    return RecoveryReport(
        point.segment_id, point.point_index,
        "LEFT" if offset >= 0.0 else "RIGHT", offset,
        initial, maximum, minimum, decrease_time, recovery_025,
        recovered_010, stop, stop_reason, wrong_segment, direction_jump,
        backtrack, maximum_steering, "PASS" if passed else "FAIL")

