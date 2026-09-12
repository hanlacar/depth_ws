"""Validated piecewise-SE(2) schedules for offline map-route generation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentZone:
    zone_id: str
    start_s_m: float
    end_s_m: float
    theta_rad: float
    tx_m: float
    ty_m: float


def shortest_angular_distance(start: float, end: float) -> float:
    """Return the signed shortest rotation from start to end, in radians."""
    return math.atan2(math.sin(end - start), math.cos(end - start))


def interpolate_angle(start: float, end: float, alpha: float) -> float:
    return start + float(alpha) * shortest_angular_distance(start, end)


def smoothstep(alpha: float) -> float:
    value = min(1.0, max(0.0, float(alpha)))
    return value * value * (3.0 - 2.0 * value)


def parse_alignment_zones(metadata: dict) -> tuple[AlignmentZone, ...]:
    """Parse a complete, ordered, gap-free piecewise alignment schedule."""
    alignment = metadata.get("alignment", {}) or {}
    raw_zones = alignment.get("zones", []) or []
    if not isinstance(raw_zones, list) or len(raw_zones) < 2:
        raise ValueError("piecewise alignment requires at least two zones")
    zones = []
    for index, raw in enumerate(raw_zones):
        try:
            zone = AlignmentZone(
                str(raw["id"]), float(raw["start_s_m"]),
                float(raw["end_s_m"]), float(raw["theta_rad"]),
                float(raw["tx_m"]), float(raw["ty_m"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid alignment zone {index}: {error}") from error
        values = (zone.start_s_m, zone.end_s_m, zone.theta_rad,
                  zone.tx_m, zone.ty_m)
        if not zone.zone_id or not all(math.isfinite(value) for value in values):
            raise ValueError(f"alignment zone {index} contains an empty id or non-finite value")
        if zone.end_s_m <= zone.start_s_m:
            raise ValueError(f"alignment zone {zone.zone_id} has a non-positive extent")
        if zones:
            difference = zone.start_s_m - zones[-1].end_s_m
            if abs(difference) > 1.0e-6:
                relation = "gap" if difference > 0.0 else "overlap"
                raise ValueError(
                    f"alignment zones {zones[-1].zone_id}/{zone.zone_id} "
                    f"have a {relation} of {abs(difference):.9f} m")
        zones.append(zone)
    if abs(zones[0].start_s_m) > 1.0e-6:
        raise ValueError("first alignment zone must start at route distance zero")
    route_length = alignment.get("route_length_m")
    if route_length is not None and abs(zones[-1].end_s_m - float(route_length)) > 1.0e-6:
        raise ValueError("last alignment zone must end at alignment.route_length_m")
    return tuple(zones)


def transform_at_distance(
    distance_m: float, zones: tuple[AlignmentZone, ...], blend_distance_m: float,
) -> tuple[float, float, float, str]:
    """Evaluate theta/translation with a centered smoothstep boundary blend."""
    if not zones:
        raise ValueError("alignment zones are empty")
    blend = float(blend_distance_m)
    if not math.isfinite(blend) or blend <= 0.0:
        raise ValueError("blend_distance_m must be finite and positive")
    minimum_extent = min(zone.end_s_m - zone.start_s_m for zone in zones)
    if blend >= minimum_extent:
        raise ValueError("blend_distance_m must be smaller than every zone extent")
    distance = min(zones[-1].end_s_m, max(zones[0].start_s_m, float(distance_m)))
    for index in range(len(zones) - 1):
        left, right = zones[index], zones[index + 1]
        boundary = left.end_s_m
        start = boundary - blend / 2.0
        end = boundary + blend / 2.0
        if start <= distance <= end:
            alpha = smoothstep((distance - start) / blend)
            return (
                interpolate_angle(left.theta_rad, right.theta_rad, alpha),
                (1.0 - alpha) * left.tx_m + alpha * right.tx_m,
                (1.0 - alpha) * left.ty_m + alpha * right.ty_m,
                f"{left.zone_id}->{right.zone_id}",
            )
    selected = zones[-1]
    for zone in zones:
        if distance <= zone.end_s_m:
            selected = zone
            break
    return (selected.theta_rad, selected.tx_m, selected.ty_m, selected.zone_id)


def apply_piecewise(
    points, distances, zones: tuple[AlignmentZone, ...], blend_distance_m: float,
):
    """Apply the schedule to Nx2 points and return points plus per-point transforms."""
    import numpy as np

    source = np.asarray(points, dtype=np.float64)
    route_s = np.asarray(distances, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) != len(route_s):
        raise ValueError("points must be Nx2 and match the distance vector")
    output = np.empty_like(source)
    schedule = []
    for index, (point, distance) in enumerate(zip(source, route_s)):
        theta, tx, ty, zone_id = transform_at_distance(
            float(distance), zones, blend_distance_m)
        cosine, sine = math.cos(theta), math.sin(theta)
        output[index] = (
            tx + cosine * point[0] - sine * point[1],
            ty + sine * point[0] + cosine * point[1],
        )
        schedule.append((theta, tx, ty, zone_id))
    return output, schedule
