"""Non-destructive STOP_LINE editing for the segmented competition route."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile

import yaml

from .route_io import load_segmented_route
from .stop_editor_network import (
    CHOICE_ORDER, CHOICE_SEGMENTS, REFERENCE_SEGMENTS, generate_route_cases,
    segment_branch,
)


STOP_EVENT = "STOP_LINE"
EDITABLE_COLUMN = "event"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EditResult:
    accepted: bool
    state: str
    key: tuple | None = None
    distance_m: float = float("inf")


@dataclass(frozen=True)
class EditOperation:
    key: tuple
    before: str
    after: str
    action: str


class StopRouteEditor:
    """Edit only the event field while retaining every source row verbatim."""

    def __init__(self, route_path, metadata_path="", maximum_click_distance_m=0.50,
                 visualization_route_path="", visualization_metadata_path=""):
        self.route_path = Path(route_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve() if metadata_path \
            else self.route_path.with_suffix(".metadata.yaml")
        if not self.route_path.is_file() or not self.metadata_path.is_file():
            raise ValueError("source route CSV and metadata must exist")
        self.maximum_click_distance_m = float(maximum_click_distance_m)
        if not 0.0 < self.maximum_click_distance_m <= 5.0:
            raise ValueError("maximum_click_distance_m must be in (0, 5]")
        with self.metadata_path.open(encoding="utf-8") as stream:
            self.source_metadata = yaml.safe_load(stream) or {}
        with self.route_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.fieldnames = tuple(reader.fieldnames or ())
            if EDITABLE_COLUMN not in self.fieldnames:
                raise ValueError("segmented route requires an event column")
            self.source_rows = tuple(dict(row) for row in reader)
        if not self.source_rows:
            raise ValueError("segmented route is empty")
        self.rows = [dict(row) for row in self.source_rows]
        self._index = {
            self.key(row): position for position, row in enumerate(self.rows)}
        if len(self._index) != len(self.rows):
            raise ValueError("segment_id/point_index keys are not unique")
        self.segments = {}
        for row in self.rows:
            self.segments.setdefault(str(row["segment_id"]), []).append(row)
        self.route_cases = generate_route_cases(self.segments)
        self.case_keys = {
            name: tuple(self.key(row) for segment in segments
                        for row in self.segments[segment])
            for name, segments in self.route_cases.items()
        }
        self.route_network_keys = set().union(
            *(set(keys) for keys in self.case_keys.values()))
        self.reference_keys = {
            self.key(row) for name in REFERENCE_SEGMENTS
            for row in self.segments.get(name, ())}
        self.active_keys = set(self._index)
        self.selected_case = "ALL"
        self.ambiguity_distance_m = 0.15
        self.last_ambiguity = None
        self.map_xy = {self.key(row): self._to_map(row) for row in self.rows}
        self.required_transitions = self._find_required_transitions()
        self.visualization_route_path = None
        self.visualization_metadata_path = None
        self.visualization_metadata = {}
        self.visualization_rows = tuple(self.rows)
        self.visualization_keys = set(self._index)
        self.visualization_branch = "ALL"
        if visualization_route_path:
            self._load_visualization_overlay(
                visualization_route_path, visualization_metadata_path)
        self.history = []
        self.selected_key = None
        self.last_added_key = None
        self.source_stop_count = self.stop_count
        self._ensure_direction_change_stops()

    @staticmethod
    def key(row):
        return str(row["segment_id"]), int(row["point_index"])

    @property
    def alignment_validated(self):
        return bool((self.source_metadata.get("alignment", {}) or {}).get(
            "validated", False))

    @property
    def stop_count(self):
        return sum(str(row[EDITABLE_COLUMN]).strip().upper() == STOP_EVENT
                   for row in self.rows)

    def _to_map(self, row):
        x, y = float(row["x_m"]), float(row["y_m"])
        if str(self.source_metadata.get("route_coordinate_frame", "")) == "map":
            return x, y
        transform = self.source_metadata.get("csv_to_map", {}) or {}
        tx = float(transform["x_m"])
        ty = float(transform["y_m"])
        scale = float(transform["scale"])
        yaw = math.radians(float(transform["yaw_deg"]))
        return (tx + scale*(math.cos(yaw)*x-math.sin(yaw)*y),
                ty + scale*(math.sin(yaw)*x+math.cos(yaw)*y))

    def _find_required_transitions(self):
        transitions = {}
        for case, keys in self.case_keys.items():
            case_rows = [self.rows[self._index[key]] for key in keys]
            for before, after in zip(case_rows, case_rows[1:]):
                before_direction, after_direction = (
                    int(before["direction"]), int(after["direction"]))
                if before_direction == after_direction:
                    continue
                key = self.key(before)
                item = transitions.setdefault(key, {
                    "cases": [], "branches": [], "mode": int(before["mode"]),
                    "segment": before["segment_id"],
                    "point_index": int(before["point_index"]),
                    "from_direction": before_direction,
                    "to_direction": after_direction,
                    "next_segment": after["segment_id"],
                    "next_point_index": int(after["point_index"]),
                    "stop_waypoint": key,
                })
                item["cases"].append(case)
                item["branches"].append(segment_branch(before["segment_id"]))
        for item in transitions.values():
            item["branches"] = sorted(set(item["branches"]))
        return transitions

    def _load_visualization_overlay(self, route_path, metadata_path=""):
        """Use exact-key map coordinates without changing source route data."""
        route_path = Path(route_path).expanduser().resolve()
        metadata_path = (Path(metadata_path).expanduser().resolve()
                         if metadata_path else
                         route_path.with_suffix(".metadata.yaml"))
        if not route_path.is_file() or not metadata_path.is_file():
            raise ValueError("visualization route CSV and metadata must exist")
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream) or {}
        if (str(metadata.get("route_coordinate_frame", "")) != "map" or
                str(metadata.get("frame_id", "")) != "map"):
            raise ValueError("visualization route must contain map-frame coordinates")
        alignment = metadata.get("alignment", {}) or {}
        if alignment.get("validated") is not False:
            raise ValueError("visualization alignment must remain explicitly unvalidated")
        with route_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"segment_id", "point_index", "x_m", "y_m", "mode"}
            if not required.issubset(set(reader.fieldnames or ())):
                raise ValueError("visualization route is missing required columns")
            overlay_rows = tuple(dict(row) for row in reader)
        if not overlay_rows:
            raise ValueError("visualization route is empty")
        keys = tuple(self.key(row) for row in overlay_rows)
        if len(set(keys)) != len(keys):
            raise ValueError("visualization segment_id/point_index keys are not unique")
        unknown = [key for key in keys if key not in self._index]
        if unknown:
            raise ValueError(f"visualization waypoint is absent from source: {unknown[0]}")
        for key, overlay in zip(keys, overlay_rows):
            source = self.rows[self._index[key]]
            if int(overlay["mode"]) != int(source["mode"]):
                raise ValueError(f"visualization mode mismatch at {key}")
        expected = tuple(self.key(row) for row in self.rows)
        if keys != expected or set(keys) != set(self._index):
            raise ValueError(
                "visualization route must exactly cover all source rows in source order")
        network = metadata.get("network", {}) or {}
        if (int(network.get("duplicate_keys", -1)) != 0 or
                int(network.get("missing_keys", -1)) != 0 or
                int(network.get("ambiguous_keys", -1)) != 0):
            raise ValueError("visualization network key audit must be 0/0/0")
        if int(network.get("route_case_count", -1)) != 16:
            raise ValueError("visualization network must declare 16 route cases")
        self.visualization_route_path = route_path
        self.visualization_metadata_path = metadata_path
        self.visualization_metadata = metadata
        self.visualization_rows = tuple(
            self.rows[self._index[key]] for key in keys)
        self.visualization_keys = set(keys)
        self.visualization_branch = "ALL"
        for key, overlay in zip(keys, overlay_rows):
            self.map_xy[key] = float(overlay["x_m"]), float(overlay["y_m"])

    def _ensure_direction_change_stops(self):
        for key in self.required_transitions:
            row = self.rows[self._index[key]]
            if str(row[EDITABLE_COLUMN]).strip().upper() != STOP_EVENT:
                row[EDITABLE_COLUMN] = STOP_EVENT

    def route_rows(self, branch):
        route = load_segmented_route(
            self.route_path, self.metadata_path, branch=branch)
        return [self.rows[self._index[(point.segment_id, point.point_index)]]
                for point in route.points]

    def case_rows(self, case):
        name = str(case).strip().upper()
        if name not in self.case_keys:
            raise ValueError("route case must be one of AAAA..BBBB")
        return [self.rows[self._index[key]] for key in self.case_keys[name]]

    def case_report(self):
        report = []
        for name in self.route_cases:
            rows = self.case_rows(name)
            report.append({
                "case": name,
                **{choice: branch for choice, branch in zip(CHOICE_ORDER, name)},
                "point_count": len(rows),
                "STOP_count": sum(
                    str(row[EDITABLE_COLUMN]).strip().upper() == STOP_EVENT
                    for row in rows),
                "reverse_count": sum(int(row["direction"]) < 0 for row in rows),
            })
        return report

    def set_case(self, case):
        value = str(case).strip().upper()
        if value != "ALL" and value not in self.case_keys:
            return False
        self.selected_case = value
        self.last_ambiguity = None
        return True

    @property
    def click_keys(self):
        return (self.visualization_keys if self.selected_case == "ALL" else
                set(self.case_keys[self.selected_case]))

    @property
    def using_visualization_overlay(self):
        return self.visualization_route_path is not None

    @property
    def visible_stop_count(self):
        visible = self.click_keys
        return sum(
            self.key(row) in visible and
            str(row[EDITABLE_COLUMN]).strip().upper() == STOP_EVENT
            for row in self.rows)

    def nearest(self, x, y, stops_only=False):
        candidates = []
        for row in self.visualization_rows:
            if self.key(row) not in self.click_keys:
                continue
            if stops_only and str(row[EDITABLE_COLUMN]).strip().upper() != STOP_EVENT:
                continue
            key = self.key(row)
            px, py = self.map_xy[key]
            candidates.append((math.hypot(float(x)-px, float(y)-py), key))
        candidates.sort()
        self.last_ambiguity = None
        if candidates and self.selected_case == "ALL":
            best_distance, best_key = candidates[0]
            best_segment = best_key[0]
            best_branch = segment_branch(best_segment)
            if best_branch in ("A", "B"):
                opposite = "B" if best_branch == "A" else "A"
                paired_segment = next(
                    branches[opposite] for branches in CHOICE_SEGMENTS.values()
                    if branches[best_branch] == best_segment)
                paired = next(
                    ((distance, key) for distance, key in candidates
                     if key[0] == paired_segment), None)
                if (paired is not None and
                        paired[0] <= self.maximum_click_distance_m and
                        abs(paired[0] - best_distance) <= self.ambiguity_distance_m):
                    self.last_ambiguity = (best_key, paired[1])
        return candidates[0] if candidates else (float("inf"), None)

    def add(self, x, y):
        distance, key = self.nearest(x, y)
        if self.last_ambiguity:
            return EditResult(False, "AMBIGUOUS_BRANCH_CLICK:SELECT_CASE",
                              key, distance)
        if key is None or distance > self.maximum_click_distance_m:
            return EditResult(False, "CLICK_TOO_FAR_FROM_ROUTE", key, distance)
        row = self.rows[self._index[key]]
        self.selected_key = key
        if str(row[EDITABLE_COLUMN]).strip().upper() == STOP_EVENT:
            return EditResult(False, "STOP_ALREADY_PRESENT", key, distance)
        before = row[EDITABLE_COLUMN]
        row[EDITABLE_COLUMN] = STOP_EVENT
        self.history.append(EditOperation(key, before, STOP_EVENT, "USER_ADD"))
        self.last_added_key = key
        return EditResult(True, "STOP_ADDED", key, distance)

    def remove(self, x, y):
        distance, key = self.nearest(x, y, stops_only=True)
        if self.last_ambiguity:
            return EditResult(False, "AMBIGUOUS_BRANCH_CLICK:SELECT_CASE",
                              key, distance)
        if key is None or distance > self.maximum_click_distance_m:
            return EditResult(False, "NO_STOP_NEAR_CLICK", key, distance)
        self.selected_key = key
        if key in self.required_transitions:
            return EditResult(False, "REMOVE_REJECTED:REQUIRED_DIRECTION_CHANGE_STOP",
                              key, distance)
        row = self.rows[self._index[key]]
        before = row[EDITABLE_COLUMN]
        row[EDITABLE_COLUMN] = "NONE"
        self.history.append(EditOperation(key, before, "NONE", "USER_REMOVE"))
        return EditResult(True, "STOP_REMOVED", key, distance)

    def undo(self):
        if not self.history:
            return EditResult(False, "UNDO_EMPTY")
        operation = self.history.pop()
        row = self.rows[self._index[operation.key]]
        row[EDITABLE_COLUMN] = operation.before
        self.selected_key = operation.key
        self.last_added_key = None
        return EditResult(True, "UNDO_"+operation.action, operation.key, 0.0)

    def selected(self):
        if self.selected_key is None:
            return None
        row = self.rows[self._index[self.selected_key]]
        source = self.source_rows[self._index[self.selected_key]]
        if self.selected_key in self.required_transitions:
            reason = "REQUIRED_DIRECTION_CHANGE"
        elif (str(source[EDITABLE_COLUMN]).strip().upper() != STOP_EVENT and
              str(row[EDITABLE_COLUMN]).strip().upper() == STOP_EVENT):
            reason = "USER"
        else:
            reason = "SOURCE"
        return {
            "mode": int(row["mode"]), "segment": row["segment_id"],
            "point_index": int(row["point_index"]),
            "event": row[EDITABLE_COLUMN],
            "reason": reason,
        }

    def changes(self):
        added, removed = [], []
        for source, current in zip(self.source_rows, self.rows):
            before = str(source[EDITABLE_COLUMN]).strip().upper()
            after = str(current[EDITABLE_COLUMN]).strip().upper()
            if before != STOP_EVENT and after == STOP_EVENT:
                added.append(self.key(current))
            elif before == STOP_EVENT and after != STOP_EVENT:
                removed.append(self.key(current))
        return added, removed

    def validate_geometry_unchanged(self):
        if len(self.source_rows) != len(self.rows):
            raise ValueError("waypoint count changed")
        for position, (source, current) in enumerate(
                zip(self.source_rows, self.rows)):
            for name in self.fieldnames:
                if name != EDITABLE_COLUMN and source[name] != current[name]:
                    raise ValueError(
                        f"non-event field changed at row {position+2}: {name}")
        for key in self.required_transitions:
            row = self.rows[self._index[key]]
            if str(row[EDITABLE_COLUMN]).strip().upper() != STOP_EVENT:
                raise ValueError("required direction-change STOP is missing")
        return True

    def save(self, output_path, output_metadata_path=""):
        self.validate_geometry_unchanged()
        output = Path(output_path).expanduser().resolve()
        metadata_output = (Path(output_metadata_path).expanduser().resolve()
                           if output_metadata_path else
                           output.with_suffix(".metadata.yaml"))
        protected_sources = {self.route_path, self.metadata_path}
        if self.visualization_route_path:
            protected_sources.update((
                self.visualization_route_path,
                self.visualization_metadata_path))
        if output in protected_sources or metadata_output in protected_sources:
            raise ValueError("source CSV and metadata cannot be overwritten")
        if output == metadata_output:
            raise ValueError("CSV and metadata output paths must be different")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", newline="", encoding="utf-8", delete=False,
                    dir=output.parent, prefix=output.name+".", suffix=".tmp") as stream:
                temporary_name = stream.name
                writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, output)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        output_hash = file_sha256(output)
        added, removed = self.changes()
        metadata = dict(self.source_metadata)
        metadata["route_csv"] = str(output)
        metadata["route_sha256"] = output_hash
        metadata["stop_edit"] = {
            "source_csv": str(self.route_path),
            "source_sha256": file_sha256(self.route_path),
            "source_metadata": str(self.metadata_path),
            "source_metadata_sha256": file_sha256(self.metadata_path),
            "added_stops": [list(item) for item in added],
            "removed_stops": [list(item) for item in removed],
            "direction_change_required_stops": [
                dict(value, branches=sorted(value["branches"]))
                for value in self.required_transitions.values()],
            "user_stops": [list(item) for item in added
                           if item not in self.required_transitions],
            "stop_before_count": self.source_stop_count,
            "stop_after_count": self.stop_count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": output_hash,
            "geometry_unchanged": True,
            "editable_columns": [EDITABLE_COLUMN],
            "route_case_count": len(self.route_cases),
            "route_cases": self.case_report(),
        }
        if self.visualization_route_path:
            metadata["stop_edit"]["visualization_only_overlay"] = {
                "route_csv": str(self.visualization_route_path),
                "route_sha256": file_sha256(self.visualization_route_path),
                "metadata": str(self.visualization_metadata_path),
                "metadata_sha256": file_sha256(
                    self.visualization_metadata_path),
                "frame_id": "map",
                "alignment_validated": False,
                "waypoint_count": len(self.visualization_rows),
            }
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_metadata = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", delete=False,
                    dir=metadata_output.parent, prefix=metadata_output.name+".",
                    suffix=".tmp") as stream:
                temporary_metadata = stream.name
                yaml.safe_dump(metadata, stream, sort_keys=False,
                               allow_unicode=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_metadata, metadata_output)
        finally:
            if temporary_metadata and os.path.exists(temporary_metadata):
                os.unlink(temporary_metadata)
        return output, metadata_output, output_hash

    def mode11_stop_analysis(self, branch):
        route = load_segmented_route(
            self.route_path, self.metadata_path, branch=branch)
        results = []
        for index, point in enumerate(route.points):
            key = (point.segment_id, point.point_index)
            row = self.rows[self._index[key]]
            if point.mode != 11 or str(row[EDITABLE_COLUMN]).upper() != STOP_EVENT:
                continue
            before = route.points[index-1] if index else point
            after = route.points[index+1] if index+1 < len(route.points) else point
            required = key in self.required_transitions
            x, y = self.map_xy[key]
            results.append({
                "branch": branch, "segment": point.segment_id,
                "point_index": point.point_index, "x_m": x, "y_m": y,
                "direction": point.direction,
                "previous_direction": before.direction,
                "next_direction": after.direction,
                "segment_boundary": (before.segment_id != point.segment_id or
                                     after.segment_id != point.segment_id),
                "route_end": index == len(route.points)-1,
                "mission_meaning": "STOP_LINE_MISSION_MARKER",
                "classification": "REQUIRED" if required else "REVIEW_REQUIRED",
            })
        return results
