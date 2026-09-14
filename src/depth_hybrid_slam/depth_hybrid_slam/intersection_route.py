"""Build intersection evidence from the CSV follower's local cursor."""

import json
import math

from .traffic_gate import IntersectionProgress


def cumulative_route_distance(route):
    values = [0.0]
    for first, second in zip(route, route[1:]):
        values.append(values[-1]+math.hypot(
            second.x-first.x, second.y-first.y))
    return tuple(values)


def intersection_progress(route, cumulative, nearest_index,
                          normalized_progress, intersection_modes=(4, 6, 8)):
    if not route or len(cumulative) != len(route):
        return IntersectionProgress()
    index = min(max(0, int(nearest_index)), len(route)-1)
    point = route[index]
    fraction = 0.0
    if len(route) > 1:
        cursor = max(0.0, min(
            float(len(route)-1),
            float(normalized_progress)*float(len(route)-1)))
        if int(cursor) == index:
            fraction = cursor-index
    progress_m = float(cumulative[index])
    if index+1 < len(route):
        progress_m += fraction*(cumulative[index+1]-cumulative[index])

    common = {
        "valid": True,
        "mode": int(point.mode),
        "segment_id": str(point.segment_id),
        "direction": int(point.direction),
        "route_index": index,
        "progress_m": progress_m,
    }
    if int(point.mode) not in set(int(value) for value in intersection_modes):
        return IntersectionProgress(**common)

    stops = [(route_index, candidate) for route_index, candidate in enumerate(route)
             if int(candidate.mode) == int(point.mode) and
             str(candidate.event).upper() == "STOP_LINE"]
    mode_indexes = [route_index for route_index, candidate in enumerate(route)
                    if int(candidate.mode) == int(point.mode)]
    if len(stops) != 1 or not mode_indexes:
        return IntersectionProgress(valid=False, **{
            key: value for key, value in common.items() if key != "valid"})
    stop_route_index, stop = stops[0]
    exit_route_index = mode_indexes[-1]
    return IntersectionProgress(
        **common,
        stop_segment_id=str(stop.segment_id),
        stop_direction=int(stop.direction),
        stop_route_index=stop_route_index,
        stop_index=int(stop.point_index),
        stop_progress_m=float(cumulative[stop_route_index]),
        exit_route_index=exit_route_index,
        exit_progress_m=float(cumulative[exit_route_index]))


def progress_json(value):
    return json.dumps(value.__dict__, separators=(",", ":"), sort_keys=True)


def progress_from_json(document):
    try:
        values = json.loads(document)
        if not isinstance(values, dict):
            return IntersectionProgress()
        fields = IntersectionProgress.__dataclass_fields__
        return IntersectionProgress(**{
            key: values[key] for key in fields if key in values})
    except (TypeError, ValueError, json.JSONDecodeError):
        return IntersectionProgress()
