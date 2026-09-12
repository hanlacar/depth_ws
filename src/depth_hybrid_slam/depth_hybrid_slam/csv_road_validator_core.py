"""Metric BEV CSV-road containment validation with lane-optional evidence.

The input path is always expressed in REP-103 ``base_link`` metres.  Semantic
pixels are converted to the same metric grid by camera_navigation's calibrated
ground-plane projection before this core is called; there is deliberately no
pixel-per-metre fallback here.
"""

from dataclasses import dataclass, asdict
import math

import cv2
import numpy as np


VALID_ROAD_AND_LANE = "VALID_ROAD_AND_LANE"
VALID_ROAD_ONLY = "VALID_ROAD_ONLY"
DEGRADED_LANE_UNCERTAIN = "DEGRADED_LANE_UNCERTAIN"
INVALID_OUTSIDE_ROAD = "INVALID_OUTSIDE_ROAD"
INVALID_INSUFFICIENT_ROAD = "INVALID_INSUFFICIENT_ROAD"
INVALID_GEOMETRY = "INVALID_GEOMETRY"
CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
PATH_UNAVAILABLE = "PATH_UNAVAILABLE"
STALE_INPUT = "STALE_INPUT"


@dataclass(frozen=True)
class ValidatorConfig:
    forward_min_m: float = 0.5
    forward_max_m: float = 4.0
    y_min_m: float = -3.0
    y_max_m: float = 3.0
    resolution_m: float = 0.04
    vehicle_width_m: float = 0.78
    minimum_center_inside_ratio: float = 0.90
    minimum_vehicle_corridor_inside_ratio: float = 0.75
    minimum_visible_path_ratio: float = 0.70
    minimum_road_confidence: float = 0.70
    path_sample_spacing_m: float = 0.05
    corridor_lateral_samples: int = 5
    lane_min_pixels: int = 20
    lane_min_support_ratio: float = 0.35
    lane_max_crossing_ratio: float = 0.10
    lane_max_distance_m: float = 1.60
    lane_heading_tolerance_deg: float = 20.0

    @property
    def x_min_m(self):
        return self.forward_min_m

    @property
    def x_max_m(self):
        return self.forward_max_m

    @property
    def rows(self):
        return int(math.ceil((self.x_max_m-self.x_min_m) /
                             self.resolution_m))+1

    @property
    def cols(self):
        return int(math.ceil((self.y_max_m-self.y_min_m) /
                             self.resolution_m))+1

    def validate(self):
        numeric = (self.forward_min_m, self.forward_max_m, self.y_min_m,
                   self.y_max_m, self.resolution_m, self.vehicle_width_m,
                   self.minimum_center_inside_ratio,
                   self.minimum_vehicle_corridor_inside_ratio,
                   self.minimum_visible_path_ratio,
                   self.minimum_road_confidence,
                   self.path_sample_spacing_m, self.lane_min_support_ratio,
                   self.lane_max_crossing_ratio, self.lane_max_distance_m,
                   self.lane_heading_tolerance_deg)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("validator parameters must be finite")
        if (self.forward_min_m < 0.0 or
                self.forward_max_m <= self.forward_min_m or
                self.y_max_m <= self.y_min_m or self.resolution_m <= 0.0 or
                self.vehicle_width_m <= 0.0 or
                self.path_sample_spacing_m <= 0.0 or
                self.corridor_lateral_samples < 3 or self.lane_min_pixels < 1):
            raise ValueError("invalid validator geometry")
        for value in (self.minimum_center_inside_ratio,
                      self.minimum_vehicle_corridor_inside_ratio,
                      self.minimum_visible_path_ratio,
                      self.minimum_road_confidence,
                      self.lane_min_support_ratio,
                      self.lane_max_crossing_ratio):
            if not 0.0 <= value <= 1.0:
                raise ValueError("ratios must be in [0,1]")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    state: str
    center_inside_ratio: float = 0.0
    vehicle_corridor_inside_ratio: float = 0.0
    visible_path_ratio: float = 0.0
    road_confidence: float = 0.0
    lane_visible: bool = False
    lane_consistent: bool = False
    lane_support_ratio: float = 0.0
    lane_crossing_ratio: float = 0.0
    lane_heading_error_deg: float = 180.0
    evaluated_center_points: int = 0
    evaluated_corridor_points: int = 0
    reason: str = ""

    def diagnostics(self):
        return asdict(self)


def unavailable_result(state, reason):
    if state not in (INVALID_GEOMETRY, CAMERA_UNAVAILABLE,
                     PATH_UNAVAILABLE, STALE_INPUT):
        raise ValueError("invalid unavailable state")
    return ValidationResult(False, state, reason=str(reason))


def classify_input_state(mount_valid, camera_available, semantic_available,
                         path_available, camera_age_s, semantic_age_s,
                         path_age_s, camera_timeout_s=5.0,
                         semantic_timeout_s=0.5, path_timeout_s=0.5):
    """Return a fail-closed availability result, or ``None`` when ready."""
    if not mount_valid:
        return unavailable_result(INVALID_GEOMETRY,
                                  "camera mount not commissioned")
    if not camera_available:
        return unavailable_result(CAMERA_UNAVAILABLE, "CameraInfo unavailable")
    if not path_available:
        return unavailable_result(PATH_UNAVAILABLE, "path unavailable")
    if not semantic_available:
        return unavailable_result(CAMERA_UNAVAILABLE, "semantic unavailable")
    ages = (camera_age_s, semantic_age_s, path_age_s)
    if not all(math.isfinite(float(age)) and float(age) >= 0.0 for age in ages):
        return unavailable_result(STALE_INPUT, "invalid input timestamp")
    if (camera_age_s > camera_timeout_s or
            semantic_age_s > semantic_timeout_s or path_age_s > path_timeout_s):
        return unavailable_result(STALE_INPUT, "semantic, path, or CameraInfo stale")
    return None


def _binary(mask, shape, name):
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != {shape}")
    return (array > 0).astype(np.uint8)


def metric_to_grid(points, config):
    points = np.asarray(points, dtype=float)
    rows = np.rint((config.x_max_m-points[:, 0]) /
                   config.resolution_m).astype(int)
    cols = np.rint((points[:, 1]-config.y_min_m) /
                   config.resolution_m).astype(int)
    inside = ((rows >= 0) & (rows < config.rows) &
              (cols >= 0) & (cols < config.cols))
    return rows, cols, inside


def grid_to_metric(rows, cols, config):
    rows, cols = np.asarray(rows), np.asarray(cols)
    return np.column_stack((
        config.x_max_m-rows*config.resolution_m,
        config.y_min_m+cols*config.resolution_m))


def densify_path(path_xy_yaw, spacing_m):
    """Densify an ordered Nx2/Nx3 path; derive tangent yaw when omitted."""
    path = np.asarray(path_xy_yaw, dtype=float)
    if path.ndim != 2 or path.shape[1] not in (2, 3) or not len(path):
        return np.empty((0, 3), dtype=float)
    if not np.all(np.isfinite(path)):
        return np.empty((0, 3), dtype=float)
    if path.shape[1] == 2:
        if len(path) == 1:
            yaw = np.zeros(1)
        else:
            delta = np.gradient(path[:, :2], axis=0)
            yaw = np.arctan2(delta[:, 1], delta[:, 0])
        path = np.column_stack((path, yaw))
    if len(path) == 1:
        return path.copy()
    output = []
    for index in range(len(path)-1):
        first, second = path[index], path[index+1]
        distance = float(np.linalg.norm(second[:2]-first[:2]))
        count = max(1, int(math.ceil(distance/spacing_m)))
        yaw_delta = math.atan2(
            math.sin(second[2]-first[2]),
            math.cos(second[2]-first[2]))
        for step in range(count):
            ratio = step/count
            xy = first[:2]+ratio*(second[:2]-first[:2])
            output.append((xy[0], xy[1], first[2]+ratio*yaw_delta))
    output.append(tuple(path[-1]))
    return np.asarray(output, dtype=float)


def ego_connected_road(road_mask, config):
    """Keep only road connected to the near-field vehicle neighbourhood."""
    road = (np.asarray(road_mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road, 8)
    if count <= 1:
        return np.zeros_like(road)
    near_points = np.array([
        [config.x_min_m, 0.0],
        [config.x_min_m+0.2, 0.0],
        [config.x_min_m, -0.25],
        [config.x_min_m, 0.25],
    ])
    rows, cols, inside = metric_to_grid(near_points, config)
    candidates = [int(labels[row, col]) for row, col, ok in
                  zip(rows, cols, inside) if ok and labels[row, col] > 0]
    if candidates:
        selected = max(set(candidates), key=candidates.count)
    else:
        selected = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # A remote component cannot establish a road connection to the ego.
        component_rows = np.flatnonzero(np.any(labels == selected, axis=1))
        if not len(component_rows) or component_rows.max() < config.rows-8:
            return np.zeros_like(road)
    return (labels == selected).astype(np.uint8)


def _road_confidence(component, visibility):
    visible_rows = np.any(visibility > 0, axis=1)
    if not np.any(visible_rows):
        return 0.0
    supported = np.any((component > 0) & (visibility > 0), axis=1)
    return float(np.count_nonzero(supported & visible_rows) /
                 np.count_nonzero(visible_rows))


def _sample(mask, points, config):
    rows, cols, inside = metric_to_grid(points, config)
    result = np.zeros(len(points), dtype=bool)
    indexes = np.flatnonzero(inside)
    result[indexes] = mask[rows[indexes], cols[indexes]] > 0
    return result, inside


def _lane_evidence(lane, path, config):
    lane_rows, lane_cols = np.nonzero(lane)
    if len(lane_rows) < config.lane_min_pixels:
        return False, False, 0.0, 0.0, 180.0
    lane_xy = grid_to_metric(lane_rows, lane_cols, config)
    supports, crossings, heading_points = [], [], []
    half_width = config.vehicle_width_m*0.5
    stride = max(1, int(round(0.10/config.path_sample_spacing_m)))
    for point in path[::stride]:
        delta = lane_xy-point[:2]
        cosine, sine = math.cos(point[2]), math.sin(point[2])
        longitudinal = cosine*delta[:, 0]+sine*delta[:, 1]
        lateral = -sine*delta[:, 0]+cosine*delta[:, 1]
        nearby = np.abs(longitudinal) <= max(0.12, config.resolution_m*2.0)
        candidates = lateral[nearby]
        if not len(candidates):
            supports.append(False)
            crossings.append(False)
            continue
        acceptable = ((np.abs(candidates) >= half_width*0.90) &
                      (np.abs(candidates) <= config.lane_max_distance_m))
        supports.append(bool(np.any(acceptable)))
        crossings.append(bool(np.any(np.abs(candidates) < half_width*0.90)))
        chosen = candidates[acceptable]
        if len(chosen):
            lateral_value = chosen[np.argmin(np.abs(chosen))]
            normal = np.array((-sine, cosine))
            heading_points.append(point[:2]+normal*lateral_value)
    if not supports:
        return True, False, 0.0, 0.0, 180.0
    support_ratio = float(np.mean(supports))
    crossing_ratio = float(np.mean(crossings))
    heading_error = 180.0
    if len(heading_points) >= 3:
        samples = np.asarray(heading_points)
        centered = samples-np.mean(samples, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        lane_heading = math.atan2(vh[0, 1], vh[0, 0])
        path_heading = math.atan2(np.mean(np.sin(path[:, 2])),
                                  np.mean(np.cos(path[:, 2])))
        difference = abs(math.degrees(math.atan2(
            math.sin(lane_heading-path_heading),
            math.cos(lane_heading-path_heading))))
        heading_error = min(difference, abs(180.0-difference))
    consistent = (support_ratio >= config.lane_min_support_ratio and
                  crossing_ratio <= config.lane_max_crossing_ratio and
                  heading_error <= config.lane_heading_tolerance_deg)
    return True, bool(consistent), support_ratio, crossing_ratio, heading_error


def validate_metric_bev(road_mask, lane_mask, visibility_mask,
                        path_xy_yaw, config=ValidatorConfig()):
    config.validate()
    shape = (config.rows, config.cols)
    try:
        road = _binary(road_mask, shape, "road")
        lane = _binary(lane_mask, shape, "lane")
        visibility = _binary(visibility_mask, shape, "visibility")
    except ValueError as error:
        return unavailable_result(INVALID_GEOMETRY, str(error))
    path = densify_path(path_xy_yaw, config.path_sample_spacing_m)
    if not len(path):
        return unavailable_result(PATH_UNAVAILABLE, "no finite path points")
    forward = ((path[:, 0] >= config.forward_min_m) &
               (path[:, 0] <= config.forward_max_m))
    path = path[forward]
    if not len(path):
        return unavailable_result(PATH_UNAVAILABLE, "no points in forward window")

    component = ego_connected_road(road, config)
    road_confidence = _road_confidence(component, visibility)
    visible_samples, geometrically_inside = _sample(visibility, path[:, :2], config)
    visible = visible_samples & geometrically_inside
    visible_ratio = float(np.mean(visible))

    center_road, _ = _sample(component, path[:, :2], config)
    visible_count = int(np.count_nonzero(visible))
    center_ratio = (float(np.count_nonzero(center_road & visible)/visible_count)
                    if visible_count else 0.0)

    offsets = np.linspace(-config.vehicle_width_m*0.5,
                          config.vehicle_width_m*0.5,
                          config.corridor_lateral_samples)
    corridor = []
    for point in path:
        normal = np.array((-math.sin(point[2]), math.cos(point[2])))
        corridor.extend(point[:2]+offset*normal for offset in offsets)
    corridor = np.asarray(corridor)
    corridor_road, corridor_inside = _sample(component, corridor, config)
    corridor_ratio = float(np.mean(corridor_road & corridor_inside))

    lane_visible, lane_consistent, lane_support, lane_crossing, lane_heading = \
        _lane_evidence(lane & visibility, path[visible], config) \
        if visible_count else (False, False, 0.0, 0.0, 180.0)

    common = dict(
        center_inside_ratio=center_ratio,
        vehicle_corridor_inside_ratio=corridor_ratio,
        visible_path_ratio=visible_ratio,
        road_confidence=road_confidence,
        lane_visible=lane_visible,
        lane_consistent=lane_consistent,
        lane_support_ratio=lane_support,
        lane_crossing_ratio=lane_crossing,
        lane_heading_error_deg=lane_heading,
        evaluated_center_points=len(path),
        evaluated_corridor_points=len(corridor))
    if road_confidence < config.minimum_road_confidence:
        return ValidationResult(False, INVALID_INSUFFICIENT_ROAD,
                                reason="road continuity below threshold", **common)
    if visible_ratio < config.minimum_visible_path_ratio:
        return ValidationResult(False, INVALID_GEOMETRY,
                                reason="path visibility below threshold", **common)
    if (center_ratio < config.minimum_center_inside_ratio or
            corridor_ratio < config.minimum_vehicle_corridor_inside_ratio):
        return ValidationResult(False, INVALID_OUTSIDE_ROAD,
                                reason="path or vehicle corridor outside road", **common)
    if not lane_visible:
        return ValidationResult(True, VALID_ROAD_ONLY,
                                reason="road valid; lane evidence absent", **common)
    if lane_consistent:
        return ValidationResult(True, VALID_ROAD_AND_LANE,
                                reason="road and optional lane evidence agree", **common)
    return ValidationResult(True, DEGRADED_LANE_UNCERTAIN,
                            reason="road valid; lane evidence uncertain", **common)


def render_bev_overlay(road_mask, lane_mask, visibility_mask, path_xy_yaw,
                       result, config=ValidatorConfig(), scale=3):
    """Render road/lane, vehicle corridor, and point-level containment."""
    road = _binary(road_mask, (config.rows, config.cols), "road")
    lane = _binary(lane_mask, road.shape, "lane")
    visibility = _binary(visibility_mask, road.shape, "visibility")
    image = np.zeros((*road.shape, 3), dtype=np.uint8)
    image[visibility > 0] = (24, 24, 24)
    image[road > 0] = (45, 105, 45)
    image[lane > 0] = (230, 210, 40)
    path = densify_path(path_xy_yaw, config.path_sample_spacing_m)
    path = path[(path[:, 0] >= config.forward_min_m) &
                (path[:, 0] <= config.forward_max_m)] if len(path) else path
    component = ego_connected_road(road, config)
    for lateral, color in ((-config.vehicle_width_m*0.5, (255, 80, 20)),
                           (config.vehicle_width_m*0.5, (255, 80, 20))):
        edge = []
        for point in path:
            normal = np.array((-math.sin(point[2]), math.cos(point[2])))
            edge.append(point[:2]+lateral*normal)
        if edge:
            rows, cols, inside = metric_to_grid(np.asarray(edge), config)
            poly = np.column_stack((cols[inside], rows[inside])).astype(np.int32)
            if len(poly) > 1:
                cv2.polylines(image, [poly], False, color, 1, cv2.LINE_AA)
    if len(path):
        inside_road, inside_grid = _sample(component, path[:, :2], config)
        rows, cols, _ = metric_to_grid(path[:, :2], config)
        for row, col, on_road, in_grid in zip(rows, cols, inside_road, inside_grid):
            if in_grid:
                cv2.circle(image, (int(col), int(row)), 2,
                           (0, 255, 0) if on_road else (0, 0, 255), -1)
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
    lines = [
        result.state,
        (f"center={result.center_inside_ratio:.3f} "
         f"corridor={result.vehicle_corridor_inside_ratio:.3f}"),
        (f"visible={result.visible_path_ratio:.3f} "
         f"road={result.road_confidence:.3f}"),
        (f"lane visible={result.lane_visible} "
         f"consistent={result.lane_consistent}"),
    ]
    for index, line in enumerate(lines):
        cv2.putText(image, line, (8, 20+18*index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image
