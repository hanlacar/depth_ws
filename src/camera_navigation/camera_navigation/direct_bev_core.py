"""ROS-independent direct semantic-mask BEV path planning."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .metric_path_quality import has_self_intersection, maximum_curvature


BOTH = "BOTH"
LEFT_ONLY = "LEFT_ONLY"
RIGHT_ONLY = "RIGHT_ONLY"
ROAD_ONLY = "ROAD_ONLY"
HOLD = "HOLD"
INVALID = "INVALID"
VALID = "VALID"
DEGRADED = "DEGRADED"


@dataclass
class DirectBevConfig:
    x_min_m: float = 0.30
    x_max_m: float = 8.00
    y_min_m: float = -3.00
    y_max_m: float = 3.00
    resolution_m: float = 0.04
    road_open_m: float = 0.08
    road_close_m: float = 0.16
    minimum_component_area_m2: float = 0.04
    hole_max_area_m2: float = 0.35
    hole_max_width_m: float = 0.80
    hole_max_length_m: float = 1.20
    vehicle_width_m: float = 0.80
    lateral_safety_margin_m: float = 0.12
    default_lane_width_m: float = 1.10
    lane_width_min_m: float = 0.55
    lane_width_max_m: float = 2.20
    lane_width_change_m: float = 0.45
    sliding_windows: int = 20
    window_half_width_m: float = 0.45
    window_min_pixels: int = 2
    fitting_degree: int = 2
    fitting_residual_m: float = 0.18
    minimum_path_points: int = 3
    resample_spacing_m: float = 0.20
    near_required_m: float = 1.50
    temporal_lateral_gate_m: float = 0.65
    temporal_heading_gate_deg: float = 35.0
    temporal_curvature_gate_per_m: float = 0.90
    valid_min_safe_coverage: float = 0.999
    valid_max_fitting_residual_m: float = 0.12
    hold_time_sec: float = 0.20
    lookahead_min_m: float = 0.60
    lookahead_default_m: float = 1.20
    lookahead_max_m: float = 2.20
    wheelbase_m: float = 0.73
    maximum_steering_deg: float = 22.0
    road_center_gate_m: float = 0.70
    camera_centered: bool = True
    path_start_lateral_m: float = 0.0

    def validate(self):
        values = tuple(vars(self).values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("BEV parameters must be finite numbers")
        if not self.x_min_m < self.x_max_m or not self.y_min_m < self.y_max_m:
            raise ValueError("invalid BEV bounds")
        if self.resolution_m <= 0.0 or self.vehicle_width_m <= 0.0:
            raise ValueError("resolution and vehicle width must be positive")
        if self.minimum_path_points < 3 or self.sliding_windows < 3:
            raise ValueError("insufficient BEV windows/path points")
        if not 0.0 < self.maximum_steering_deg <= 22.0:
            raise ValueError("maximum steering must be in (0, 22]")


@dataclass
class DirectBevResult:
    points: np.ndarray
    mode: str
    state: str
    valid: bool
    confidence: float
    diagnostics: dict
    road: np.ndarray
    safe_road: np.ndarray
    component: np.ndarray
    left: np.ndarray
    right: np.ndarray
    rejected: np.ndarray


def pure_pursuit_unclipped(points, wheelbase_m, lookahead_m,
                           lookahead_from_path_start=False):
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    array = array[np.isfinite(array).all(axis=1) & (array[:, 0] > 0.0)]
    if not len(array):
        return None, None
    if lookahead_from_path_start:
        # Camera geometry and footprint erosion can make the first observable,
        # safe path point several metres ahead of base_link.  In that case a
        # vehicle-origin lookahead shorter than that distance always selects
        # the forced centre-axis start point and destroys all steering intent.
        distances = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(array, axis=0), axis=1))))
    else:
        distances = np.linalg.norm(array, axis=1)
    index = int(np.argmin(np.abs(distances-float(lookahead_m))))
    target = array[index]
    squared = float(target@target)
    if squared <= 1.0e-9:
        return None, None
    angle = math.degrees(math.atan2(2.0*wheelbase_m*target[1], squared))
    return float(angle), target.copy()


class DirectBevPlanner:
    def __init__(self, config=None):
        self.config = config or DirectBevConfig()
        self.config.validate()
        self.rows = int(math.ceil(
            (self.config.x_max_m-self.config.x_min_m)/self.config.resolution_m))+1
        self.cols = int(math.ceil(
            (self.config.y_max_m-self.config.y_min_m)/self.config.resolution_m))+1
        self.previous = None
        self.previous_timestamp = None
        self.previous_lane_width = None
        self.previous_mode = None
        self._last_ego_seed_fallback = False
        self._kernels = {
            "open": self._kernel(self.config.road_open_m),
            "close": self._kernel(self.config.road_close_m),
        }

    def _kernel(self, size_m):
        pixels = max(1, int(round(size_m/self.config.resolution_m)))
        if pixels % 2 == 0:
            pixels += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))

    def reset(self):
        self.previous = None
        self.previous_timestamp = None
        self.previous_lane_width = None
        self.previous_mode = None

    def metric_to_grid(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        row = np.rint((self.config.x_max_m-points[:, 0]) /
                      self.config.resolution_m).astype(int)
        col = np.rint((points[:, 1]-self.config.y_min_m) /
                      self.config.resolution_m).astype(int)
        return np.column_stack((row, col))

    def grid_to_metric(self, rows, cols):
        x = self.config.x_max_m-np.asarray(rows)*self.config.resolution_m
        y = self.config.y_min_m+np.asarray(cols)*self.config.resolution_m
        return np.column_stack((x, y)).astype(float)

    def _remove_small_components(self, mask):
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        minimum = self.config.minimum_component_area_m2 / self.config.resolution_m**2
        output = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum:
                output[labels == label] = 1
        return output

    def _fill_small_holes(self, road, lane):
        closed = cv2.morphologyEx(road, cv2.MORPH_CLOSE, self._kernels["close"])
        inverse = (1-closed).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
        output = closed.copy()
        max_area = self.config.hole_max_area_m2/self.config.resolution_m**2
        max_width = self.config.hole_max_width_m/self.config.resolution_m
        max_height = self.config.hole_max_length_m/self.config.resolution_m
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if x == 0 or y == 0 or x+width == self.cols or y+height == self.rows:
                continue
            marking = np.count_nonzero(lane[labels == label])
            if (area <= max_area and width <= max_width and height <= max_height and
                    (marking > 0 or area <= 0.35*max_area)):
                output[labels == label] = 1
        return output

    def _ego_component(self, road):
        self._last_ego_seed_fallback = False
        count, labels, stats, _ = cv2.connectedComponentsWithStats(road, 8)
        if count <= 1:
            return np.zeros_like(road)
        near_rows = max(1, int(round(0.45/self.config.resolution_m)))
        center_col = int(round((0.0-self.config.y_min_m)/self.config.resolution_m))
        half = max(1, int(round(0.35/self.config.resolution_m)))
        seed = labels[max(0, self.rows-near_rows):,
                      max(0, center_col-half):min(self.cols, center_col+half+1)]
        candidates = [int(v) for v in np.unique(seed) if int(v) > 0]
        if not candidates:
            # A forward camera cannot observe the ground immediately under
            # its bumper. Select the nearest visible road intersecting the
            # center corridor; near-field absence is reported as DEGRADED.
            for row in range(self.rows-1, -1, -1):
                visible = labels[row, max(0, center_col-half):
                                 min(self.cols, center_col+half+1)]
                candidates = [int(v) for v in np.unique(visible) if int(v) > 0]
                if candidates:
                    self._last_ego_seed_fallback = True
                    break
        if not candidates:
            return np.zeros_like(road)
        label = max(candidates, key=lambda value: stats[value, cv2.CC_STAT_AREA])
        return (labels == label).astype(np.uint8)

    def preprocess(self, road, lane):
        road = self._remove_small_components(road)
        lane = self._remove_small_components(lane)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, self._kernels["open"])
        road = self._fill_small_holes(road, lane)
        component = self._ego_component(road)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        clearance = self.config.vehicle_width_m/2.0 + \
            self.config.lateral_safety_margin_m
        safe = (distance*self.config.resolution_m >= clearance).astype(np.uint8)
        return road, lane, component, safe, distance

    def _sample_lane_tracks(self, lane, component):
        cfg = self.config
        lane = lane & component
        left, right = [], []
        previous_left = previous_right = None
        for x_m in np.linspace(cfg.x_min_m, cfg.x_max_m,
                               cfg.sliding_windows):
            row = int(round((cfg.x_max_m-x_m)/cfg.resolution_m))
            half = max(1, int(round(
                0.5*(cfg.x_max_m-cfg.x_min_m) /
                cfg.sliding_windows/cfg.resolution_m)))
            cols = np.flatnonzero(np.any(
                lane[max(0, row-half):row+half+1], axis=0))
            groups = (np.split(cols, np.flatnonzero(np.diff(cols) > 1)+1)
                      if len(cols) else [])
            values = [cfg.y_min_m+float(np.mean(group))*cfg.resolution_m
                      for group in groups
                      if len(group) >= cfg.window_min_pixels]
            for positive, output in ((True, left), (False, right)):
                candidates = [value for value in values
                              if (value > 0.0) == positive and value != 0.0]
                previous = previous_left if positive else previous_right
                if candidates:
                    value = min(candidates, key=lambda y: abs(
                        y-(previous or 0.0)))
                    if (previous is None or
                            abs(value-previous) <= cfg.window_half_width_m):
                        output.append((x_m, value))
                        if positive:
                            previous_left = value
                        else:
                            previous_right = value
        return (np.asarray(left, float).reshape(-1, 2),
                np.asarray(right, float).reshape(-1, 2))

    def _crosscheck_lane_candidate(self, raw, mode, distance, component,
                                   warnings):
        """Extension hook used by offline candidate planners; production noop."""
        return raw, mode

    def _robust_fit(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        rejected = np.empty((0, 2), float)
        if len(points) < self.config.minimum_path_points:
            return None, rejected, float("inf")
        degree = min(self.config.fitting_degree, len(points)-1)
        keep = np.ones(len(points), bool)
        coefficients = None
        for _ in range(3):
            weights = 1.0+2.0*(1.0-(points[keep, 0]-self.config.x_min_m) /
                               (self.config.x_max_m-self.config.x_min_m))
            coefficients = np.polyfit(points[keep, 0], points[keep, 1], degree,
                                      w=weights)
            residual = np.abs(points[:, 1]-np.polyval(coefficients, points[:, 0]))
            candidate = residual <= self.config.fitting_residual_m
            if np.count_nonzero(candidate) < self.config.minimum_path_points:
                break
            if np.array_equal(candidate, keep):
                keep = candidate; break
            keep = candidate
        rejected = points[~keep]
        residual = float(np.mean(np.abs(
            points[keep, 1]-np.polyval(coefficients, points[keep, 0]))))
        return coefficients, rejected, residual

    def _offset_boundary(self, points, left):
        coefficients, rejected, residual = self._robust_fit(points)
        if coefficients is None:
            return np.empty((0, 2)), rejected, residual
        x = np.asarray(points)[:, 0]
        y = np.polyval(coefficients, x)
        derivative = np.polyval(np.polyder(coefficients), x)
        half_width = 0.5*(self.previous_lane_width or
                          self.config.default_lane_width_m)
        norm = np.sqrt(1.0+derivative**2)
        if left:
            x = x+half_width*derivative/norm
            y = y-half_width/norm
        else:
            x = x-half_width*derivative/norm
            y = y+half_width/norm
        return np.column_stack((x, y)), rejected, residual

    def _road_center(self, distance, component):
        points = []
        previous_y = 0.0 if self.previous is None else float(self.previous[0, 1])
        for x_m in np.linspace(self.config.x_min_m, self.config.x_max_m,
                               self.config.sliding_windows):
            row = int(round((self.config.x_max_m-x_m)/self.config.resolution_m))
            if not 0 <= row < self.rows:
                continue
            cols = np.flatnonzero(component[row])
            if not len(cols):
                continue
            ys = self.config.y_min_m+cols*self.config.resolution_m
            gated = np.abs(ys-previous_y) <= self.config.road_center_gate_m
            if np.any(gated):
                cols, ys = cols[gated], ys[gated]
            values = distance[row, cols]
            peak = float(np.max(values))
            choices = np.flatnonzero(values >= 0.92*peak)
            index = choices[np.argmin(np.abs(ys[choices]-previous_y))]
            previous_y = float(ys[index])
            points.append((x_m, previous_y))
        return np.asarray(points, float).reshape(-1, 2)

    def _resample(self, points):
        coefficients, rejected, residual = self._robust_fit(points)
        if coefficients is None:
            return np.empty((0, 2)), rejected, residual
        start = max(self.config.x_min_m, float(np.min(points[:, 0])))
        end = min(self.config.x_max_m, float(np.max(points[:, 0])))
        x = np.arange(start, end+0.5*self.config.resample_spacing_m,
                      self.config.resample_spacing_m)
        return np.column_stack((x, np.polyval(coefficients, x))), rejected, residual

    def _project_inside(self, points, component, safe):
        output, safe_count, road_count = [], 0, 0
        for point in np.asarray(points):
            row, col = self.metric_to_grid([point])[0]
            if not 0 <= row < self.rows:
                continue
            safe_cols = np.flatnonzero(safe[row])
            if not len(safe_cols):
                continue
            selected = int(safe_cols[np.argmin(np.abs(safe_cols-col))])
            output.append(self.grid_to_metric([row], [selected])[0])
            safe_count += 1
            road_count += 1
        return np.asarray(output, float).reshape(-1, 2), safe_count, road_count

    def _steering(self, points):
        ratios = np.linspace(self.config.lookahead_min_m,
                             self.config.lookahead_max_m, 9)
        ratios = sorted(ratios, key=lambda value: abs(
            value-self.config.lookahead_default_m))
        best = None
        for lookahead in ratios:
            angle, target = pure_pursuit_unclipped(
                points, self.config.wheelbase_m, lookahead)
            if angle is None:
                continue
            best = (angle, target, float(lookahead))
            if abs(angle) <= self.config.maximum_steering_deg:
                return best+(True,)
        return (best+(False,)) if best else (None, None, None, False)

    def _center_axis_start_x(self, traversable):
        """Nearest footprint-safe point on the configured vehicle axis."""
        column = int(round((self.config.path_start_lateral_m-
                            self.config.y_min_m)/self.config.resolution_m))
        if not 0 <= column < self.cols:
            return None
        for x_m in np.arange(self.config.x_min_m,
                             self.config.x_max_m+0.5*self.config.resolution_m,
                             self.config.resolution_m):
            row = int(round((self.config.x_max_m-x_m) /
                            self.config.resolution_m))
            if 0 <= row < self.rows and traversable[row, column] > 0:
                return float(self.grid_to_metric([row], [column])[0, 0])
        return None

    @staticmethod
    def _quintic_connector(x_values, start_x, start_y, end_x, end_y,
                           end_slope, end_second):
        """C2 connector with zero initial heading/curvature in base_link."""
        length = float(end_x-start_x)
        if length <= 1.0e-6:
            return np.empty((0, 2), float)
        # y(t)=a0+...+a5*t^5, t=(x-x0)/length. Match y/y'/y''
        # at both ends; the vehicle-axis start has y'=y''=0.
        coefficients = np.zeros(6, float)
        coefficients[0] = float(start_y)
        system = np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]])
        target = np.array([
            float(end_y-start_y), float(end_slope*length),
            float(end_second*length*length),
        ])
        coefficients[3:] = np.linalg.solve(system, target)
        t = np.clip((np.asarray(x_values, float)-start_x)/length, 0.0, 1.0)
        y = sum(coefficients[degree]*t**degree for degree in range(6))
        return np.column_stack((np.asarray(x_values, float), y))

    def _points_inside_component(self, points, component):
        grid = self.metric_to_grid(points)
        inside = ((grid[:, 0] >= 0) & (grid[:, 0] < self.rows) &
                  (grid[:, 1] >= 0) & (grid[:, 1] < self.cols))
        if not np.all(inside):
            return False
        return bool(np.all(component[grid[:, 0], grid[:, 1]] > 0))

    def _connect_center_axis_start(self, points, safe):
        """Connect the first safe center-axis point with a C2 shape."""
        points = np.asarray(points, float).reshape(-1, 2)
        if not self.config.camera_centered:
            return points, [], None
        start_x = self._center_axis_start_x(safe)
        if start_x is None:
            return np.empty((0, 2), float), [], "CENTER_AXIS_ROAD_MISSING"
        usable = points[points[:, 0] >= start_x-0.5*self.config.resolution_m]
        if len(usable) < self.config.minimum_path_points:
            return np.empty((0, 2), float), [], "PATH_START_NOT_REPRESENTABLE"
        usable = usable[np.argsort(usable[:, 0])]
        slopes = np.gradient(usable[:, 1], usable[:, 0], edge_order=1)
        seconds = np.gradient(slopes, usable[:, 0], edge_order=1)
        minimum_join = max(2.0*self.config.resample_spacing_m, 0.40)
        join_indices = [index for index in range(1, len(usable))
                        if usable[index, 0]-start_x >= minimum_join]
        if not join_indices and abs(usable[0, 1]-self.config.path_start_lateral_m) <= \
                self.config.resolution_m:
            candidate = usable.copy()
            candidate[0] = (start_x, self.config.path_start_lateral_m)
            if self._points_inside_component(candidate, safe):
                _, _, _, feasible = self._steering(candidate)
                if feasible:
                    return candidate, [], None
        recovery = []
        for attempt, join_index in enumerate(join_indices):
            join_x, join_y = usable[join_index]
            connector_x = np.arange(
                start_x, join_x+0.5*self.config.resample_spacing_m,
                self.config.resample_spacing_m)
            connector_x[-1] = join_x
            connector = self._quintic_connector(
                connector_x, start_x, self.config.path_start_lateral_m,
                join_x, join_y, slopes[join_index], seconds[join_index])
            candidate = np.vstack((connector, usable[join_index+1:]))
            candidate[0, 1] = self.config.path_start_lateral_m
            if len(candidate) < self.config.minimum_path_points:
                continue
            if not self._points_inside_component(candidate, safe):
                recovery.append("START_CONNECTION_LENGTH_INCREASED")
                continue
            _, _, _, feasible = self._steering(candidate)
            if feasible and not has_self_intersection(candidate):
                if attempt:
                    recovery.extend(("START_CURVATURE_RELAXED",
                                     "START_NEAR_PATH_REFIT"))
                return candidate, list(dict.fromkeys(recovery)), None
            recovery.append("START_CONNECTION_LENGTH_INCREASED")
        return (np.empty((0, 2), float), list(dict.fromkeys(recovery)),
                "PATH_START_CONNECTION_FAILED")

    def plan(self, road_mask, lane_mask, timestamp_sec):
        started = __import__("time").perf_counter()
        road, lane, component, safe, distance = self.preprocess(
            road_mask, lane_mask)
        empty = np.empty((0, 2), float)
        if not np.any(component):
            return self._fallback_or_invalid(
                timestamp_sec, "EGO_ROAD_MISSING", road, safe, component)
        left, right = self._sample_lane_tracks(lane, component)
        rejected = empty
        residual = float("inf")
        warnings = []
        if self._last_ego_seed_fallback:
            warnings.append("NEAR_FIELD_UNOBSERVABLE")
        if (len(left) >= self.config.minimum_path_points and
                len(right) >= self.config.minimum_path_points):
            common_x = np.linspace(max(left[:, 0].min(), right[:, 0].min()),
                                   min(left[:, 0].max(), right[:, 0].max()),
                                   min(len(left), len(right)))
            left_y = np.interp(common_x, left[:, 0], left[:, 1])
            right_y = np.interp(common_x, right[:, 0], right[:, 1])
            widths = left_y-right_y
            sane = ((widths >= self.config.lane_width_min_m) &
                    (widths <= self.config.lane_width_max_m))
            if self.previous_lane_width is not None:
                sane &= np.abs(widths-self.previous_lane_width) <= \
                    self.config.lane_width_change_m
            raw = np.column_stack((common_x[sane],
                                   0.5*(left_y[sane]+right_y[sane])))
            mode = BOTH
            if len(raw) >= self.config.minimum_path_points:
                self.previous_lane_width = float(np.median(widths[sane]))
            else:
                warnings.append("LANE_WIDTH_REJECTED")
        elif len(left) >= self.config.minimum_path_points:
            raw, rejected, residual = self._offset_boundary(left, True)
            mode = LEFT_ONLY
        elif len(right) >= self.config.minimum_path_points:
            raw, rejected, residual = self._offset_boundary(right, False)
            mode = RIGHT_ONLY
        else:
            raw = self._road_center(distance, component)
            mode = ROAD_ONLY
            warnings.append("LANE_UNAVAILABLE")
        if len(raw) < self.config.minimum_path_points:
            raw = self._road_center(distance, component)
            mode = ROAD_ONLY
        raw, mode = self._crosscheck_lane_candidate(
            raw, mode, distance, component, warnings)
        fitted, fit_rejected, fit_residual = self._resample(raw)
        rejected = np.vstack((rejected, fit_rejected)) if len(rejected) else fit_rejected
        residual = min(residual, fit_residual)
        fitted, safe_count, road_count = self._project_inside(fitted, component, safe)
        if len(fitted) < self.config.minimum_path_points or has_self_intersection(fitted):
            return self._fallback_or_invalid(
                timestamp_sec, "PATH_NOT_REPRESENTABLE", road, safe, component,
                left, right, rejected)
        fitted, start_recovery, start_error = self._connect_center_axis_start(
            fitted, safe)
        if start_error is not None:
            return self._invalid(start_error, road, safe, component, left, right,
                                 rejected)
        # The connector changes the near-field samples, so recompute coverage
        # instead of carrying the pre-connector counts forward.
        fitted_grid = self.metric_to_grid(fitted)
        road_count = len(fitted)
        safe_count = int(np.count_nonzero(
            safe[fitted_grid[:, 0], fitted_grid[:, 1]]))
        angle, target, lookahead, feasible = self._steering(fitted)
        recovery = list(start_recovery)
        if not feasible and len(fitted) > self.config.minimum_path_points:
            for trim in range(1, min(6, len(fitted)-self.config.minimum_path_points+1)):
                candidate = fitted[:-trim]
                angle, target, lookahead, feasible = self._steering(candidate)
                recovery.append("TAIL_TRIM")
                if feasible:
                    fitted = candidate; warnings.append("STEERING_RECOVERED"); break
        if not feasible:
            return self._invalid(
                "STEERING_LIMIT_EXCEEDED", road, safe, component, left, right,
                rejected, angle=angle, lookahead=lookahead)
        safe_ratio = safe_count/max(1, road_count)
        near = fitted[fitted[:, 0] <= self.config.near_required_m]
        near_safe = all(safe[tuple(self.metric_to_grid([point])[0])]
                        for point in near if np.all(
                            (self.metric_to_grid([point])[0] >= 0)))
        if safe_ratio < self.config.valid_min_safe_coverage:
            return self._invalid(
                "SAFE_ROAD_COVERAGE_LOW", road, safe, component, left, right,
                rejected, angle=angle, lookahead=lookahead)
        if residual > self.config.valid_max_fitting_residual_m:
            warnings.append("FIT_RESIDUAL_HIGH")
        if not near_safe:
            return self._invalid(
                "NEAR_CLEARANCE_LOW", road, safe, component, left, right,
                rejected, angle=angle, lookahead=lookahead)
        curvature = maximum_curvature(fitted)
        temporal_ok = True
        if self.previous is not None:
            current_y = float(np.interp(self.config.near_required_m,
                                        fitted[:, 0], fitted[:, 1]))
            previous_y = float(np.interp(self.config.near_required_m,
                                         self.previous[:, 0], self.previous[:, 1]))
            if abs(current_y-previous_y) > self.config.temporal_lateral_gate_m:
                temporal_ok = False; warnings.append("TEMPORAL_LATERAL_JUMP")
            if len(fitted) >= 2 and len(self.previous) >= 2:
                current_heading = math.degrees(math.atan2(
                    fitted[1, 1]-fitted[0, 1],
                    fitted[1, 0]-fitted[0, 0]))
                previous_heading = math.degrees(math.atan2(
                    self.previous[1, 1]-self.previous[0, 1],
                    self.previous[1, 0]-self.previous[0, 0]))
                heading_delta = (current_heading-previous_heading+180.0) % 360.0-180.0
                if abs(heading_delta) > self.config.temporal_heading_gate_deg:
                    temporal_ok = False; warnings.append("TEMPORAL_HEADING_JUMP")
            previous_curvature = maximum_curvature(self.previous)
            if abs(curvature-previous_curvature) > \
                    self.config.temporal_curvature_gate_per_m:
                temporal_ok = False; warnings.append("TEMPORAL_CURVATURE_JUMP")
        if not temporal_ok:
            return self._fallback_or_invalid(
                timestamp_sec, warnings[-1], road, safe, component,
                left, right, rejected)
        state = VALID if not warnings and mode == BOTH else DEGRADED
        confidence = float(np.clip(
            0.35+0.35*(mode == BOTH)+0.2*safe_ratio+
            0.1*max(0.0, 1.0-residual/self.config.fitting_residual_m), 0.0, 1.0))
        diagnostics = {
            "mode": mode, "state": state, "reasons": warnings,
            "path_points": len(fitted),
            "path_length_m": float(np.sum(np.linalg.norm(
                np.diff(fitted, axis=0), axis=1))),
            "fitting_residual_m": float(residual),
            "minimum_clearance_m": float(np.min([
                distance[tuple(self.metric_to_grid([p])[0])]
                for p in fitted]))*self.config.resolution_m,
            "safe_road_coverage": safe_ratio,
            "road_connectivity": float(np.count_nonzero(component) /
                                        max(1, np.count_nonzero(road))),
            # Stage-diagnostics: component/safe are masks already computed
            # by preprocess() above -- np.count_nonzero() only, no new mask
            # work and no change to safe_ratio/valid_min_safe_coverage logic.
            "ego_component_pixels": int(np.count_nonzero(component)),
            "safe_road_pixels": int(np.count_nonzero(safe)),
            "selected_branch": ("CONTINUITY_CLEARANCE_TIEBREAK"
                                if mode == ROAD_ONLY else "LANE_TRACK"),
            "required_steering_deg": float(angle),
            "maximum_steering_deg": self.config.maximum_steering_deg,
            "lookahead_m": float(lookahead), "target_point": target.tolist(),
            "steering_recovery": recovery,
            "curvature_per_m": float(curvature),
            "near_field_coverage": self._near_field_coverage(road, component),
            "left_boundary_points": int(len(left)),
            "right_boundary_points": int(len(right)),
            "corridor_width_m": (None if self.previous_lane_width is None else
                                 float(self.previous_lane_width)),
            "gate_results": {
                "ego_road": True,
                "near_field_observable": not self._last_ego_seed_fallback,
                "road_continuity": True,
                "corridor_containment": safe_ratio >=
                    self.config.valid_min_safe_coverage,
                "both_lane_boundaries": mode == BOTH,
                "temporal_consistency": temporal_ok,
                "path_point_count": len(fitted) >= self.config.minimum_path_points,
                "curvature_feasible": feasible,
            },
            "processing_ms": (__import__("time").perf_counter()-started)*1000.0,
        }
        self.previous = fitted.copy()
        self.previous_timestamp = float(timestamp_sec)
        self.previous_mode = mode
        return DirectBevResult(fitted, mode, state, True, confidence,
                               diagnostics, road, safe, component, left, right,
                               rejected)

    def _fallback_or_invalid(self, timestamp, reason, road, safe, component,
                             left=None, right=None, rejected=None):
        age = (float("inf") if self.previous_timestamp is None else
               max(0.0, float(timestamp)-self.previous_timestamp))
        if self.previous is not None and age <= self.config.hold_time_sec:
            if np.any(safe) and not self._points_inside_component(
                    self.previous, safe):
                return self._invalid(
                    "HOLD_PATH_UNSAFE", road, safe, component,
                    left, right, rejected)
            diagnostics = {"mode": HOLD, "state": DEGRADED,
                           "reasons": [reason], "hold_age_sec": age,
                           "path_points": len(self.previous),
                           "ego_component_pixels": int(np.count_nonzero(component)),
                           "safe_road_pixels": int(np.count_nonzero(safe)),
                           "safe_road_coverage": (0.0 if not np.any(safe)
                                                  else 1.0)}
            diagnostics.update(self._common_gate_diagnostics(
                road, component, left, right, reason))
            return DirectBevResult(
                self.previous.copy(), HOLD, DEGRADED, True, 0.35, diagnostics,
                road, safe, component,
                empty_points(left), empty_points(right), empty_points(rejected))
        return self._invalid(reason, road, safe, component, left, right, rejected)

    def _invalid(self, reason, road, safe, component, left=None, right=None,
                 rejected=None, angle=None, lookahead=None):
        diagnostics = {"mode": INVALID, "state": INVALID,
                       "reasons": [reason], "path_points": 0,
                       "required_steering_deg": angle,
                       "lookahead_m": lookahead,
                       "ego_component_pixels": int(np.count_nonzero(component)),
                       "safe_road_pixels": int(np.count_nonzero(safe))}
        diagnostics.update(self._common_gate_diagnostics(
            road, component, left, right, reason))
        return DirectBevResult(
            np.empty((0, 2), float), INVALID, INVALID, False, 0.0,
            diagnostics, road, safe, component, empty_points(left),
            empty_points(right), empty_points(rejected))

    def _near_field_coverage(self, road, component):
        first_row = int(np.clip(round(
            (self.config.x_max_m-self.config.near_required_m) /
            self.config.resolution_m), 0, self.rows))
        near_road = int(np.count_nonzero(road[first_row:]))
        return float(np.count_nonzero(component[first_row:])/max(1, near_road))

    def _common_gate_diagnostics(self, road, component, left, right, reason):
        left_count = len(empty_points(left)); right_count = len(empty_points(right))
        return {
            "near_field_coverage": self._near_field_coverage(road, component),
            "road_connectivity": float(np.count_nonzero(component) /
                                        max(1, np.count_nonzero(road))),
            "left_boundary_points": int(left_count),
            "right_boundary_points": int(right_count),
            "corridor_width_m": (None if self.previous_lane_width is None else
                                 float(self.previous_lane_width)),
            "gate_results": {
                "ego_road": bool(np.any(component)),
                "near_field_observable": not self._last_ego_seed_fallback,
                "road_continuity": bool(np.any(component)),
                "corridor_containment": reason not in (
                    "SAFE_ROAD_COVERAGE_LOW", "NEAR_CLEARANCE_LOW"),
                "both_lane_boundaries": (
                    left_count >= self.config.minimum_path_points and
                    right_count >= self.config.minimum_path_points),
                "temporal_consistency": not str(reason).startswith("TEMPORAL_"),
                "path_point_count": False,
                "curvature_feasible": reason != "STEERING_LIMIT_EXCEEDED",
            },
        }


def empty_points(value):
    if value is None:
        return np.empty((0, 2), float)
    return np.asarray(value, float).reshape(-1, 2)
