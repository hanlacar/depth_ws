"""CSV-only route-case assembly and pose/decision branch policies."""

from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path

from .geometry import wrap_angle
from .models import Pose2D
from .route_follower_core import RouteFollower
from .route_io import load_segmented_route
from .stop_editor_network import CHOICE_ORDER, CHOICE_SEGMENTS, route_case_segments


NETWORK_BRANCH_PAIRS = (
    ("START", "START_A", "START_B"),
    ("T", "T_A", "T_B"),
    ("V", "V_A", "V_B"),
    ("END", "END_AA", "END_AB"),
)


def load_csv_only_route_case(route_path, metadata_path, route_case):
    """Assemble one START/T/V/END case from map-coordinate route points."""
    name = str(route_case).strip().upper()
    pools = {}
    for branch in ("A", "B"):
        route = load_segmented_route(
            route_path, metadata_path, branch=branch)
        for point in route.points:
            pools.setdefault(point.segment_id, {})[point.point_index] = point
    order = route_case_segments(name, pools)
    points = []
    for segment in order:
        segment_points = [pools[segment][index]
                          for index in sorted(pools[segment])]
        points.extend(segment_points)
    points = [replace(point, index=index) for index, point in enumerate(points)]
    # Recompute body yaw across the selected segment boundaries. This matches
    # route_io's convention for both forward and reverse points.
    for index, point in enumerate(points):
        neighbour = None
        before = False
        if index+1 < len(points) and points[index+1].direction == point.direction:
            neighbour = points[index+1]
        elif index > 0 and points[index-1].direction == point.direction:
            neighbour = points[index-1]
            before = True
        if neighbour is None:
            motion_yaw = 0.0
        elif before:
            motion_yaw = math.atan2(
                point.y-neighbour.y, point.x-neighbour.x)
        else:
            motion_yaw = math.atan2(
                neighbour.y-point.y, neighbour.x-point.x)
        body_yaw = motion_yaw if point.direction > 0 else motion_yaw+math.pi
        points[index] = replace(
            point, yaw=math.atan2(math.sin(body_yaw), math.cos(body_yaw)))
    return tuple(points)


def remap_case_progress(previous, new_route, route_case):
    """Map live progress to an existing equivalent case waypoint."""
    segment_choice = {
        segment: position
        for position, choice in enumerate(CHOICE_ORDER)
        for segment in CHOICE_SEGMENTS[choice].values()
    }
    position = segment_choice.get(previous.segment_id)
    equivalent = previous.segment_id
    if position is not None:
        choice = CHOICE_ORDER[position]
        equivalent = CHOICE_SEGMENTS[choice][route_case[position]]
    segment_points = [point for point in new_route
                      if point.segment_id == equivalent]
    candidates = [point for point in segment_points
                  if point.direction == previous.direction]
    if position in (1, 2) and previous.direction > 0:
        first_reverse = next((point.point_index for point in segment_points
                              if point.direction < 0), None)
        if first_reverse is not None:
            candidates = [point for point in candidates
                          if point.point_index < first_reverse]
    if equivalent == previous.segment_id:
        forward = [point for point in candidates
                   if point.point_index >= previous.point_index]
        return None if not forward else forward[0].index
    if not candidates:
        return None
    matched = min(candidates, key=lambda point: (
        math.hypot(point.x-previous.x, point.y-previous.y),
        abs(wrap_angle(point.yaw-previous.yaw)), point.point_index))
    return matched.index


def csv_only_network_segments(route_path, metadata_path):
    """Return every A/B/common segment once, excluding the base reference."""
    result = {}
    for case in ("AAAA", "BBBB"):
        for point in load_csv_only_route_case(route_path, metadata_path, case):
            result.setdefault(point.segment_id, {})[point.point_index] = point
    return {name: tuple(values[index] for index in sorted(values))
            for name, values in result.items()}


def _nearest_distances(first, second):
    return tuple(min(math.hypot(a.x-b.x, a.y-b.y) for b in second)
                 for a in first)


def network_pair_geometry(segments):
    """Measure real separation without assuming equal point counts/indexes."""
    result = {}
    for choice, name_a, name_b in NETWORK_BRANCH_PAIRS:
        a, b = tuple(segments[name_a]), tuple(segments[name_b])
        nearest_a = _nearest_distances(a, b)
        nearest_b = _nearest_distances(b, a)
        symmetric = nearest_a+nearest_b

        def summary(points):
            return {
                "points": len(points),
                "bbox": {
                    "x_min": min(point.x for point in points),
                    "x_max": max(point.x for point in points),
                    "y_min": min(point.y for point in points),
                    "y_max": max(point.y for point in points),
                },
            }

        result[choice] = {
            name_a: summary(a), name_b: summary(b),
            "mean_nearest_m": sum(symmetric)/len(symmetric),
            "maximum_separation_m": max(symmetric),
            "start_distance_m": math.hypot(a[0].x-b[0].x, a[0].y-b[0].y),
            "end_distance_m": math.hypot(a[-1].x-b[-1].x, a[-1].y-b[-1].y),
        }
    return result


def _first_reverse_index(points):
    return next(index for index, point in enumerate(points)
                if point.direction < 0)


def _direction_block(points, start):
    direction = points[start].direction
    stop = start+1
    while stop < len(points) and points[stop].direction == direction:
        stop += 1
    return tuple(points[start:stop])


def _route_switch_result(origin, target_points, corridor_m=1.0,
                         max_steering_deg=22.0):
    """Evaluate an existing-route switch with the production follower law."""
    route = tuple(replace(point, index=index)
                  for index, point in enumerate(target_points))
    follower = RouteFollower(
        wheelbase=0.73, max_steering_deg=89.0, corridor_m=corridor_m,
        steering_rate_deg_s=10000.0)
    result = follower.compute(
        Pose2D(origin.x, origin.y, origin.yaw, 0.0), route, dt=1.0,
        allow_motion=True, global_search=True)
    nearest = route[min(result.nearest_index, len(route)-1)]
    return {
        "nearest_point_index": nearest.point_index,
        "nearest_distance_m": abs(result.cross_track_error),
        "heading_delta_deg": abs(math.degrees(result.heading_error)),
        "required_steering_deg": abs(result.steering_deg),
        "continuous": (
            abs(result.cross_track_error) <= float(corridor_m)+1.0e-9 and
            abs(result.steering_deg) <= float(max_steering_deg)+1.0e-9 and
            abs(result.heading_error) <= follower.max_heading+1.0e-9),
    }


def _endpoint_transition(origin_points, target_points, corridor_m=1.0,
                         max_steering_deg=22.0):
    """Measure an existing segment edge from its incoming path tangent."""
    origin_points, target_points = tuple(origin_points), tuple(target_points)
    origin = origin_points[-1]
    if len(origin_points) > 1:
        before = origin_points[-2]
        motion_yaw = math.atan2(origin.y-before.y, origin.x-before.x)
        body_yaw = (motion_yaw if origin.direction > 0 else
                    motion_yaw+math.pi)
        origin = replace(origin, yaw=wrap_angle(body_yaw))
    result = _route_switch_result(
        origin, _direction_block(target_points, 0), corridor_m,
        max_steering_deg)
    start = target_points[0]
    result.update({
        "from": (origin_points[-1].segment_id,
                 origin_points[-1].point_index),
        "to": (start.segment_id, start.point_index),
        "gap_m": math.hypot(origin.x-start.x, origin.y-start.y),
    })
    return result


def parking_branch_geometry(segments, corridor_m=1.0,
                            max_steering_deg=22.0):
    """Audit the requested late T/V decisions without modifying CSV geometry.

    A V_foword network keeps V pending on the shared approach and audits its
    explicit commit STOP against both branches.  The preserved legacy network
    still reports the latest A-forward fallback from which B is safe.
    """
    output = {}
    for choice in ("T", "V"):
        a = tuple(segments[f"{choice}_A"])
        b = tuple(segments[f"{choice}_B"])
        reverse_a = _first_reverse_index(a)
        reverse_b = _first_reverse_index(b)
        if choice == "V" and "V_foword" in segments:
            common = tuple(segments["V_foword"])
            commit_candidates = [point for point in common
                                 if point.event == "STOP_LINE"]
            commit = commit_candidates[-1] if commit_candidates else common[-1]
            transitions = {}
            for branch, points, reverse in (
                    ("A", a, reverse_a), ("B", b, reverse_b)):
                start = points[0]
                switch = _route_switch_result(
                    commit, _direction_block(points, 0), corridor_m,
                    max_steering_deg)
                switch.update({
                    "start_point_index": start.point_index,
                    "start_xy_m": (start.x, start.y),
                    "gap_m": math.hypot(
                        commit.x-start.x, commit.y-start.y),
                    "heading_delta_deg": abs(math.degrees(wrap_angle(
                        commit.yaw-start.yaw))),
                    "first_reverse_point_index": points[reverse].point_index,
                    "first_reverse_xy_m": (
                        points[reverse].x, points[reverse].y),
                    "start_to_first_reverse_m": math.hypot(
                        start.x-points[reverse].x,
                        start.y-points[reverse].y),
                })
                transitions[branch] = switch
            safe = (commit.event == "STOP_LINE" and
                    all(item["continuous"]
                        for item in transitions.values()))
            topology_transitions = {
                "COMMON_2_TO_V_foword": _endpoint_transition(
                    segments["COMMON_2"], common, corridor_m,
                    max_steering_deg),
                "V_foword_TO_V_A": _endpoint_transition(
                    common, a, corridor_m, max_steering_deg),
                "V_foword_TO_V_B": _endpoint_transition(
                    common, b, corridor_m, max_steering_deg),
            }
            output[choice] = {
                "common_forward": {
                    "segment_id": "V_foword",
                    "points": len(common),
                    "start": (common[0].point_index,
                              common[0].x, common[0].y),
                    "end": (common[-1].point_index,
                            common[-1].x, common[-1].y),
                    "commit": (commit.point_index, commit.x, commit.y),
                    "commit_event": commit.event,
                },
                "first_reverse": {
                    "A": (a[reverse_a].point_index, a[reverse_a].x,
                          a[reverse_a].y),
                    "B": (b[reverse_b].point_index, b[reverse_b].x,
                          b[reverse_b].y),
                },
                "first_direction_transition": {
                    "A": (a[reverse_a-1].point_index, reverse_a),
                    "B": (b[reverse_b-1].point_index, reverse_b),
                },
                "transitions": transitions,
                "topology_transitions": topology_transitions,
                "requested_commit_safe": safe,
                "latest_safe_commit": {
                    "segment_id": "V_foword",
                    "point_index": commit.point_index,
                    "xy_m": (commit.x, commit.y),
                    "kind": "COMMON_FORWARD_BRANCH_STOP",
                } if safe else None,
                "verdict": ("PASS" if safe else
                            "BRANCH_COMMIT_TOO_LATE_FOR_GEOMETRY"),
            }
            continue
        forward_a, forward_b = a[:reverse_a], b[:reverse_b]
        stop_a, stop_b = forward_a[-1], forward_b[-1]
        distances = _nearest_distances(forward_a, forward_b)+\
            _nearest_distances(forward_b, forward_a)
        requested_switch = _route_switch_result(
            stop_a, _direction_block(b, reverse_b), corridor_m,
            max_steering_deg)
        requested_switch.update({
            "a_point_index": stop_a.point_index,
            "b_point_index": stop_b.point_index,
            "a_xy_m": (stop_a.x, stop_a.y),
            "b_xy_m": (stop_b.x, stop_b.y),
            "branch_separation_m": math.hypot(
                stop_a.x-stop_b.x, stop_a.y-stop_b.y),
            "branch_heading_delta_deg": abs(math.degrees(wrap_angle(
                stop_a.yaw-stop_b.yaw))),
            "first_reverse_jump_m": math.hypot(
                stop_a.x-b[reverse_b].x, stop_a.y-b[reverse_b].y),
        })

        safe = []
        for point in forward_a:
            switch = _route_switch_result(
                point, forward_b, corridor_m, max_steering_deg)
            if switch["continuous"]:
                safe.append((point, switch))
        fallback = None
        if safe:
            point, switch = safe[-1]
            fallback = dict(switch)
            fallback.update({
                "a_point_index": point.point_index,
                "a_xy_m": (point.x, point.y),
                "kind": "LATEST_SAFE_FORWARD_ROUTE_SWITCH",
            })
        if requested_switch["continuous"]:
            fallback = dict(requested_switch)
            fallback["kind"] = "FIRST_FORWARD_TO_REVERSE_STOP"

        output[choice] = {
            "forward_start": {
                "A": (forward_a[0].point_index, forward_a[0].x,
                      forward_a[0].y),
                "B": (forward_b[0].point_index, forward_b[0].x,
                      forward_b[0].y),
            },
            "first_reverse": {
                "A": (a[reverse_a].point_index, a[reverse_a].x,
                      a[reverse_a].y),
                "B": (b[reverse_b].point_index, b[reverse_b].x,
                      b[reverse_b].y),
            },
            "first_direction_transition": {
                "A": (stop_a.point_index, reverse_a),
                "B": (stop_b.point_index, reverse_b),
            },
            "forward_nearest_mean_m": sum(distances)/len(distances),
            "forward_nearest_max_m": max(distances),
            "requested_commit": requested_switch,
            "requested_commit_safe": requested_switch["continuous"],
            "latest_safe_commit": fallback,
            "verdict": ("PASS" if requested_switch["continuous"] else
                        "BRANCH_COMMIT_TOO_LATE_FOR_GEOMETRY"),
        }
    return output


def validate_display_correspondence(source_path, display_path):
    """Require a full exact-key display overlay sourced from unmodified XY."""
    def rows(path):
        output = {}
        duplicates = []
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["segment_id"], int(row["point_index"]))
                if key in output:
                    duplicates.append(key)
                output[key] = row
        return output, duplicates

    source, source_duplicates = rows(source_path)
    display, display_duplicates = rows(display_path)
    missing = sorted(source.keys()-display.keys())
    extra = sorted(display.keys()-source.keys())
    source_error = 0.0
    for key in source.keys() & display.keys():
        source_error = max(
            source_error,
            abs(float(display[key]["source_x_m"])-float(source[key]["x_m"])),
            abs(float(display[key]["source_y_m"])-float(source[key]["y_m"])))
    valid = not (source_duplicates or display_duplicates or missing or extra) and \
        source_error <= 1.0e-9
    return {
        "valid": valid,
        "source_rows": len(source), "display_rows": len(display),
        "source_duplicate_keys": len(source_duplicates),
        "display_duplicate_keys": len(display_duplicates),
        "missing_keys": len(missing), "extra_keys": len(extra),
        "source_xy_max_error_m": source_error,
    }


@dataclass(frozen=True)
class StartBranchResult:
    branch: str
    state: str
    score_a: float
    score_b: float
    lateral_a: float
    lateral_b: float
    heading_a: float
    heading_b: float


class StartBranchClassifier:
    """Compare pose against finite START path sections, not one waypoint."""

    def __init__(self, start_a, start_b, path_length_m=12.0,
                 heading_weight_m=1.0, ambiguity_margin=0.35,
                 maximum_lateral_m=2.0):
        self.paths = {
            "A": self._prefix(start_a, path_length_m),
            "B": self._prefix(start_b, path_length_m),
        }
        self.heading_weight = float(heading_weight_m)
        self.margin = float(ambiguity_margin)
        self.maximum_lateral = float(maximum_lateral_m)
        if any(len(path) < 2 for path in self.paths.values()):
            raise ValueError("START classifier requires two non-empty path sections")
        if self.heading_weight < 0.0 or self.margin <= 0.0 or \
                self.maximum_lateral <= 0.0:
            raise ValueError("invalid START classifier thresholds")

    @staticmethod
    def _prefix(points, length):
        selected = [points[0]]
        distance = 0.0
        for before, after in zip(points, points[1:]):
            distance += math.hypot(after.x-before.x, after.y-before.y)
            selected.append(after)
            if distance >= float(length):
                break
        return tuple(selected)

    def _score(self, pose, points):
        scored = []
        for first, second in zip(points, points[1:]):
            dx, dy = second.x-first.x, second.y-first.y
            squared = dx*dx+dy*dy
            projection = 0.0 if squared <= 1.0e-12 else max(
                0.0, min(1.0, ((pose.x-first.x)*dx+(pose.y-first.y)*dy)/squared))
            x, y = first.x+projection*dx, first.y+projection*dy
            lateral = math.hypot(pose.x-x, pose.y-y)
            heading = abs(wrap_angle(first.yaw-pose.yaw))
            scored.append((lateral+self.heading_weight*heading,
                           lateral, heading))
        return min(scored)

    def classify(self, pose):
        a = self._score(pose, self.paths["A"])
        b = self._score(pose, self.paths["B"])
        best = "A" if a[0] < b[0] else "B"
        selected = a if best == "A" else b
        ambiguous = (selected[1] > self.maximum_lateral or
                     abs(a[0]-b[0]) < self.margin)
        return StartBranchResult(
            "" if ambiguous else best,
            "START_BRANCH_AMBIGUOUS" if ambiguous else
            f"START_BRANCH_SELECTED_{best}",
            a[0], b[0], a[1], b[1], a[2], b[2])


class IndependentRouteCaseSelector:
    """Pose START, late parking decisions, and the unchanged END policy."""

    PARKING_ENTRY_SEGMENTS = {
        "T": ("T_foword",), "V": ("V_foword", "V_A", "V_B")}
    PARKING_EXIT_SEGMENTS = {"T": ("COMMON_2",), "V": ("END_common",)}

    def __init__(self, geometry=None, decision_window_s=3.0,
                 post_commit_hold_s=3.0):
        self.choices = {choice: None for choice in CHOICE_ORDER}
        self.requests = {}
        self.request_timestamps = {}
        self.entry_timestamps = {}
        self.commit_timestamps = {}
        self.reverse_timestamps = {}
        self.parking_lifecycle = {"T": "INACTIVE", "V": "INACTIVE"}
        self.decision_stop_timestamps = {}
        self.hold_until = {}
        self.geometry = geometry or {}
        self.decision_window_s = float(decision_window_s)
        self.post_commit_hold_s = float(post_commit_hold_s)
        if self.decision_window_s < 0.0 or self.post_commit_hold_s < 3.0:
            raise ValueError("parking decision/hold timing is invalid")
        self.last_now = 0.0
        self.last_event = ""
        self.state = "WAITING_FOR_START_POSE"

    @property
    def stop(self):
        if self.choices["START"] is None:
            return True
        if any(choice in self.decision_stop_timestamps and
               self.choices[choice] is None for choice in ("T", "V")):
            return True
        return any(self.last_now < release for release in self.hold_until.values())

    @property
    def route_case(self):
        return "".join(self.choices[choice] or "A" for choice in CHOICE_ORDER)

    def set_start(self, result):
        if result.branch in ("A", "B"):
            self.choices["START"] = result.branch
        self.state = result.state

    def request(self, text, now=0.0):
        self.last_now = float(now)
        value = str(text).strip().upper()
        parts = value.split(":", 1)
        if len(parts) != 2 or parts[0] not in ("T", "V", "END") or \
                parts[1] not in ("A", "B"):
            return False
        choice, branch = parts
        if choice in ("T", "V"):
            lifecycle = self.parking_lifecycle[choice]
            if branch != "B" or lifecycle not in ("PENDING", "B_REQUESTED"):
                self.last_event = ("LATE_BRANCH_COMMAND_IGNORED" if
                                   lifecycle.startswith("COMMITTED") or
                                   lifecycle == "REVERSE_STARTED" else
                                   "OUT_OF_WINDOW_BRANCH_COMMAND_IGNORED")
                return False
            self.requests[choice] = "B"
            self.request_timestamps[choice] = self.last_now
            self.parking_lifecycle[choice] = "B_REQUESTED"
            self.state = f"{choice}:B_REQUESTED"
            self.last_event = f"{choice}:B_REQUESTED"
            return True
        if self.choices[choice] is not None:
            return False
        self.requests[choice] = branch
        self.request_timestamps[choice] = self.last_now
        return True

    def _enter_parking(self, choice, now):
        if self.parking_lifecycle[choice] != "INACTIVE":
            return False
        self.parking_lifecycle[choice] = "PENDING"
        self.entry_timestamps[choice] = float(now)
        self.state = f"{choice}:PENDING"
        self.last_event = self.state
        return True

    def _commit(self, choice, now):
        if self.choices[choice] is not None:
            return False
        selected = "B" if self.requests.get(choice) == "B" else "A"
        self.choices[choice] = selected
        self.commit_timestamps[choice] = float(now)
        self.parking_lifecycle[choice] = f"COMMITTED_{selected}"
        self.state = f"{choice}:COMMITTED_{selected}"
        self.last_event = self.state
        if choice in ("T", "V"):
            self.hold_until[choice] = float(now)+self.post_commit_hold_s
        return True

    def enter_segment(self, segment, now=0.0):
        self.last_now = float(now)
        changed = False
        for choice, exits in self.PARKING_EXIT_SEGMENTS.items():
            if segment in exits and self.parking_lifecycle[choice] not in (
                    "INACTIVE", "COMPLETE"):
                self.parking_lifecycle[choice] = "COMPLETE"
                self.requests.pop(choice, None)
                self.request_timestamps.pop(choice, None)
                self.decision_stop_timestamps.pop(choice, None)
                self.hold_until.pop(choice, None)
                changed = True
        for choice, entries in self.PARKING_ENTRY_SEGMENTS.items():
            if segment in entries:
                changed = self._enter_parking(choice, now) or changed
        if segment == "END_common" and self.choices["END"] is None:
            selected = self.requests.get("END", "A")
            self.choices["END"] = selected
            self.state = f"END_BRANCH_SELECTED_{selected}"
            changed = True
        return changed

    def observe_point(self, segment, point_index, direction, stop_key,
                      stop_state, now):
        self.last_now = float(now)
        changed = self.enter_segment(segment, now)
        lifecycle = self.parking_lifecycle["T"]
        if lifecycle in ("PENDING", "B_REQUESTED"):
            geometry = self.geometry.get("T", {})
            transitions = geometry.get("first_direction_transition", {})
            keys = {f"T_{branch}:{values[0]}"
                    for branch, values in transitions.items()}
            if str(stop_key) in keys and str(stop_state) != "IDLE":
                started = self.decision_stop_timestamps.setdefault(
                    "T", self.last_now)
                if self.last_now-started >= self.decision_window_s:
                    changed = self._commit("T", self.last_now) or changed

        lifecycle = self.parking_lifecycle["V"]
        if lifecycle in ("PENDING", "B_REQUESTED"):
            geometry = self.geometry.get("V", {})
            common = geometry.get("common_forward")
            if common:
                commit_index = int(common["commit"][0])
                key = f"V_foword:{commit_index}"
                if (segment == "V_foword" and str(stop_key) == key and
                        str(stop_state) != "IDLE"):
                    started = self.decision_stop_timestamps.setdefault(
                        "V", self.last_now)
                    if self.last_now-started >= self.decision_window_s:
                        changed = self._commit("V", self.last_now) or changed
            elif segment == "V_A":
                # Compatibility for the preserved legacy CSV.  The V_foword
                # candidate never enters V_A while the choice is pending.
                fallback = geometry.get("latest_safe_commit")
                cutoff = (None if fallback is None else
                          fallback.get("a_point_index"))
                if cutoff is not None and int(point_index) >= int(cutoff):
                    changed = self._commit("V", self.last_now) or changed

        if int(direction) < 0:
            for choice in ("T", "V"):
                if segment in (f"{choice}_A", f"{choice}_B"):
                    changed = self.note_reverse_started(choice, self.last_now) or changed
        return changed

    def note_reverse_started(self, choice, now):
        if choice not in ("T", "V") or self.choices[choice] is None:
            return False
        if self.parking_lifecycle[choice] == "REVERSE_STARTED":
            return False
        self.last_now = float(now)
        self.parking_lifecycle[choice] = "REVERSE_STARTED"
        self.reverse_timestamps[choice] = self.last_now
        self.state = f"{choice}:REVERSE_STARTED"
        self.last_event = self.state
        return True

    def advance(self, now):
        self.last_now = float(now)
        for choice in tuple(self.hold_until):
            if self.last_now >= self.hold_until[choice]:
                self.hold_until.pop(choice, None)

    def lifecycle_status(self):
        return dict(self.parking_lifecycle)

    def timestamp_status(self):
        return {
            "entry": dict(self.entry_timestamps),
            "request": dict(self.request_timestamps),
            "commit": dict(self.commit_timestamps),
            "reverse": dict(self.reverse_timestamps),
        }

    def status(self):
        return {choice: self.choices[choice] or "pending"
                for choice in CHOICE_ORDER}
