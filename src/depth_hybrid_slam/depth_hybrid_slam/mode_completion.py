"""Route-progress evidence for one-shot competition-mode completion events."""

from dataclasses import dataclass
import json


ROUTE_ONLY = "ROUTE_ONLY"
TRANSIENT_READINESS_FAILURES = frozenset((
    "STALE_POSE", "LOCALIZATION_NOT_STABLE", "SAFETY_NOT_READY"))


@dataclass(frozen=True)
class ModeBoundary:
    """One contiguous mode block in the currently active route."""

    mode: int
    first_index: int
    last_index: int


class RouteModeCompletionTracker:
    """Track mode lifecycle from route cursor evidence, not /drive_mode changes.

    ``route_complete``, ``mission_complete`` and ``mode_complete`` are kept as
    separate maps so a later mission-aware policy can require both without
    changing how route progress is established.  The current policy is
    intentionally route-only.
    """

    def __init__(self, route):
        self.criterion = ROUTE_ONLY
        self.current_mode = None
        self.previous_mode = None
        self.started_modes = set()
        self.completed_modes = set()
        self.invalid_modes = set()
        self.route_complete = {}
        self.mission_complete = {}
        self.mode_complete = {}
        self.course_complete = False
        self.last_error = ""
        self.last_route_index = None
        self.last_progress = 0.0
        self.last_segment_id = ""
        self.last_point_index = -1
        self._route = ()
        self._boundaries = {}
        self._transition_thresholds = {}
        self._mode_order = ()
        self._rebind_pending = False
        self._rebind_settle_index = None
        self.bind_route(route)

    @staticmethod
    def _build_boundaries(route):
        if not route:
            raise ValueError("mode completion requires a non-empty route")
        boundaries = {}
        order = []
        first = 0
        active_mode = int(route[0].mode)
        for index in range(1, len(route)+1):
            next_mode = int(route[index].mode) if index < len(route) else None
            if next_mode == active_mode:
                continue
            if active_mode in boundaries:
                raise ValueError(
                    f"mode {active_mode} is not one contiguous route block")
            boundaries[active_mode] = ModeBoundary(
                active_mode, first, index-1)
            order.append(active_mode)
            first = index
            active_mode = next_mode
        if any(second != first+1 for first, second in zip(order, order[1:])):
            raise ValueError("active route modes must progress consecutively")
        return boundaries, tuple(order)

    def bind_route(self, route, resume_index=None):
        """Bind a selected A/B route without clearing lifecycle history."""
        boundaries, order = self._build_boundaries(route)
        route = tuple(route)
        mapped_index = None
        if resume_index is not None:
            mapped_index = int(resume_index)
            if not 0 <= mapped_index < len(route):
                raise ValueError("mode completion resume index is outside route")
            if (self.current_mode in self.started_modes and
                    int(route[mapped_index].mode) != self.current_mode):
                # A branch decision and the route-controller timer are
                # asynchronous.  The remap can therefore land on the first
                # point of the immediate successor after the controller has
                # already crossed the selected parking exit.  Preserve the
                # old mode until the next healthy observation so ``observe``
                # can validate and emit the normal transition.  Any jump
                # beyond the immediate successor remains invalid.
                try:
                    position = order.index(self.current_mode)
                    successor = (order[position+1]
                                 if position+1 < len(order) else None)
                except ValueError:
                    successor = None
                if int(route[mapped_index].mode) != successor:
                    self.invalid_modes.add(self.current_mode)
                    self.last_error = "BRANCH_REMAP_MODE_JUMP"
        elif self.current_mode in self.started_modes:
            # The parking branches intentionally overlap.  A valid late A/B
            # commit can have no exact point/direction equivalent even though
            # it stays inside the same contiguous mode.  Re-anchor at that
            # mode's first point (never ahead of observed progress), then let
            # the next controller observation prove healthy forward motion.
            # Missing the current mode entirely is still a hard remap error.
            boundary = boundaries.get(self.current_mode)
            if boundary is None:
                self.invalid_modes.add(self.current_mode)
                self.last_error = "BRANCH_REMAP_PROGRESS_LOST"
            else:
                mapped_index = boundary.first_index
        self._route = route
        self._boundaries = boundaries
        self._mode_order = order
        self._transition_thresholds = {}
        for mode, boundary in boundaries.items():
            terminal_segment = route[boundary.last_index].segment_id
            terminal_first = boundary.last_index
            while (terminal_first > boundary.first_index and
                   route[terminal_first-1].segment_id == terminal_segment):
                terminal_first -= 1
            # Parking modes 7/10 span a common approach and a terminal branch.
            # Their overlapping Ackermann geometry may move the accepted
            # cursor directly into the consecutive successor. The healthy,
            # non-backtracking successor crossing is route-only evidence; all
            # other modes retain the strict final-point threshold so an
            # arbitrary cursor jump can never complete them.
            self._transition_thresholds[mode] = (
                boundary.first_index if mode in (7, 10) and
                terminal_first > boundary.first_index else
                max(boundary.first_index, boundary.last_index-1))
        for mode in order:
            self.route_complete.setdefault(mode, False)
            self.mission_complete.setdefault(mode, False)
            self.mode_complete.setdefault(mode, False)
        self.last_route_index = mapped_index
        self._rebind_pending = bool(
            self.current_mode in self.started_modes and
            self.current_mode not in self.invalid_modes)
        self._rebind_settle_index = None

    def mark_mission_complete(self, mode):
        """Record future mission evidence without changing ROUTE_ONLY policy."""
        mode = int(mode)
        if mode not in self._boundaries:
            raise ValueError(f"mode {mode} is not in the active route")
        self.mission_complete[mode] = True

    def _start(self, mode, events):
        if mode not in self.started_modes:
            self.started_modes.add(mode)
            events.append(f"[MODE {mode} START]")

    def _complete(self, mode, events):
        self.route_complete[mode] = True
        # Current policy: mode_complete = route_complete.  mission_complete is
        # deliberately stored independently for the later mission-aware step.
        self.mode_complete[mode] = self.route_complete[mode]
        if self.mode_complete[mode] and mode not in self.completed_modes:
            self.completed_modes.add(mode)
            events.append(
                f"[MODE {mode} COMPLETE] criterion={self.criterion}")

    def _invalidate(self, mode, reason):
        # Startup may legitimately spend several cycles waiting for pose,
        # localization, safety, or branch classification.  A mode cannot fail
        # before it has actually emitted START.
        if mode in self.started_modes:
            self.invalid_modes.add(int(mode))
            self.last_error = str(reason)

    def _successor(self, mode):
        try:
            position = self._mode_order.index(mode)
        except ValueError:
            return None
        return (self._mode_order[position+1]
                if position+1 < len(self._mode_order) else None)

    def observe(self, route_index, progress, *, healthy, stopped=False,
                failure_reason="", controller_reason=""):
        """Consume one follower observation and return new one-shot log lines.

        A non-final mode completes only after the accepted cursor reaches its
        final route segment and a healthy observation crosses into the
        consecutive next mode.  Crossing the boundary is evidence that the
        final waypoint was passed even when a high-rate vehicle advances by
        one waypoint between controller samples.  The final mode additionally
        requires the follower's existing ``ROUTE_COMPLETE`` result.
        """
        # Completion is terminal.  The controller keeps publishing its final
        # stopped observation, and recovery state may subsequently change;
        # neither can invalidate route evidence already accepted for all
        # modes.
        if self.course_complete:
            return ()
        events = []
        index = int(route_index)
        if not 0 <= index < len(self._route):
            self._invalidate(self.current_mode, "ROUTE_INDEX_INVALID")
            return tuple(events)
        point = self._route[index]
        observed_mode = int(point.mode)
        self.last_progress = float(progress)
        self.last_segment_id = str(point.segment_id)
        self.last_point_index = int(point.point_index)

        rebind_observation = self._rebind_pending
        backtrack = (not rebind_observation and
                     self.last_route_index is not None and
                     index < self.last_route_index)
        if backtrack:
            self._invalidate(self.current_mode, "CURSOR_BACKTRACK")
        # Readiness loss pauses evidence collection for this observation.  It
        # is not proof that route progress itself failed, and a later fresh,
        # healthy observation may continue from the unchanged high-water
        # cursor. Geometry/controller failures remain terminal for the mode.
        if failure_reason and failure_reason not in TRANSIENT_READINESS_FAILURES:
            self._invalidate(self.current_mode, failure_reason)

        eligible = bool(healthy) and not stopped and not failure_reason
        if self.current_mode is None:
            self.current_mode = observed_mode
            if eligible:
                self._start(observed_mode, events)
            self.last_route_index = (index if self.last_route_index is None else
                                     max(self.last_route_index, index))
        elif observed_mode != self.current_mode:
            old_mode = self.current_mode
            old_last_index = self.last_route_index
            expected_mode = self._successor(old_mode)
            boundary = self._boundaries.get(old_mode)
            normal_transition = (
                not backtrack and eligible and
                expected_mode == observed_mode and boundary is not None and
                old_last_index is not None and
                old_last_index >= self._transition_thresholds[old_mode] and
                index >= self._boundaries[observed_mode].first_index and
                old_mode not in self.invalid_modes)
            self.previous_mode = old_mode
            self.current_mode = observed_mode
            if normal_transition:
                self._complete(old_mode, events)
            else:
                self._invalidate(old_mode, (
                    "UNSAFE_MODE_TRANSITION" if not eligible else
                    "WAYPOINT_PROGRESS_INCOMPLETE" if
                    expected_mode == observed_mode else
                    "NONCONSECUTIVE_MODE_TRANSITION"))
            if eligible:
                self._start(observed_mode, events)
            self.last_route_index = index
            self._rebind_pending = False
            self._rebind_settle_index = None
        else:
            if eligible:
                self._start(observed_mode, events)
            if rebind_observation:
                self.last_route_index = index
                if (self._rebind_settle_index is None or
                        index < self._rebind_settle_index):
                    self._rebind_settle_index = index
                elif index > self._rebind_settle_index:
                    self._rebind_pending = False
                    self._rebind_settle_index = None
            elif self.last_route_index is None or index > self.last_route_index:
                self.last_route_index = index

        final_mode = self._mode_order[-1]
        final_boundary = self._boundaries[final_mode]
        terminal_progress = (
            self.current_mode == final_mode and
            self.last_route_index is not None and
            self.last_route_index >= max(final_boundary.first_index,
                                         final_boundary.last_index-1))
        if (eligible and controller_reason == "ROUTE_COMPLETE" and
                terminal_progress and final_mode not in self.invalid_modes):
            self._complete(final_mode, events)
            expected = set(self._mode_order)
            if (not self.course_complete and
                    self.completed_modes == expected):
                self.course_complete = True
                events.append("[COURSE COMPLETE]")
        return tuple(events)

    def status(self):
        mode = self.current_mode
        if self.course_complete:
            state = "COURSE_COMPLETE"
        elif mode in self.invalid_modes:
            state = "INVALID"
        elif mode in self.completed_modes:
            state = "COMPLETE"
        elif mode in self.started_modes:
            state = "ACTIVE"
        else:
            state = "NOT_STARTED"
        return {
            "mode": mode,
            "previous_mode": self.previous_mode,
            "state": state,
            "criterion": self.criterion,
            "route_complete": bool(
                mode is not None and self.route_complete.get(mode, False)),
            "mission_complete": bool(
                mode is not None and self.mission_complete.get(mode, False)),
            "mode_complete": bool(
                mode is not None and self.mode_complete.get(mode, False)),
            "started_modes": sorted(self.started_modes),
            "route_complete_modes": sorted(
                mode for mode, value in self.route_complete.items() if value),
            "mission_complete_modes": sorted(
                mode for mode, value in self.mission_complete.items() if value),
            "completed_modes": sorted(self.completed_modes),
            "invalid_modes": sorted(self.invalid_modes),
            "course_complete": self.course_complete,
            "route_index": self.last_route_index,
            "segment_id": self.last_segment_id,
            "point_index": self.last_point_index,
            "progress": self.last_progress,
            "last_error": self.last_error,
        }

    def status_json(self):
        return json.dumps(self.status(), sort_keys=True, separators=(",", ":"))
