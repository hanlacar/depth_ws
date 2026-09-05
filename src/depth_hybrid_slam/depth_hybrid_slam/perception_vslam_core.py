"""Pure geometry and evidence checks for camera/VSLAM cross-validation."""

from dataclasses import dataclass
import math

import numpy as np


TRAFFIC_CLASS_STATE = {
    "r_light": "R",
    "red_light": "R",
    "g_light": "G",
    "green_light": "G",
    "left": "G",
    "left_light": "G",
    "etc_light": "G",
    "other_light": "G",
}


@dataclass(frozen=True)
class DepthProjection:
    valid: bool
    reason: str
    optical_xyz_m: tuple = ()
    median_depth_m: float = math.nan
    depth_mad_m: float = math.nan
    valid_pixels: int = 0


def json_stamp_seconds(document):
    """Read either nanosecond or ROS sec/nanosec JSON stamps."""
    if "stamp" in document and document["stamp"] is not None:
        return float(document["stamp"]) * 1.0e-9
    stamp = document["timestamp"]
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1.0e-9


def select_traffic_detection(document, expected_state=""):
    """Select the strongest model-named traffic-light bbox."""
    candidates = []
    for item in document.get("detections", ()):
        name = str(item.get("class_name", "")).strip().lower()
        state = TRAFFIC_CLASS_STATE.get(name)
        try:
            confidence = float(item.get("confidence", math.nan))
            bbox = tuple(float(value) for value in item.get("xyxy", ()))
        except (TypeError, ValueError):
            continue
        if (state is None or len(bbox) != 4 or
                not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0 or
                not all(math.isfinite(value) for value in bbox) or
                bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            continue
        if expected_state and state != str(expected_state).upper():
            continue
        candidates.append((confidence, dict(item), state, bbox))
    if not candidates:
        return None
    confidence, item, state, bbox = max(candidates, key=lambda value: value[0])
    return {"class_name": str(item.get("class_name", "")),
            "confidence": confidence, "state": state, "bbox": bbox}


def depth_to_metres(array, encoding):
    values = np.asarray(array)
    name = str(encoding).upper()
    if name in ("16UC1", "MONO16"):
        return values.astype(np.float32) * 0.001
    if name == "32FC1":
        return values.astype(np.float32)
    raise ValueError(f"unsupported depth encoding: {encoding}")


def robust_bbox_projection(depth, encoding, camera_matrix, bbox,
                           minimum_pixels=12, minimum_depth_m=0.2,
                           maximum_depth_m=20.0, maximum_mad_m=0.75):
    """Project a robust inner-bbox depth sample into the optical frame."""
    metric = depth_to_metres(depth, encoding)
    if metric.ndim != 2:
        return DepthProjection(False, "DEPTH_SHAPE_INVALID")
    try:
        fx, fy = float(camera_matrix[0]), float(camera_matrix[4])
        cx, cy = float(camera_matrix[2]), float(camera_matrix[5])
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (IndexError, TypeError, ValueError):
        return DepthProjection(False, "CALIBRATION_OR_BBOX_INVALID")
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy, x1, y1, x2, y2)):
        return DepthProjection(False, "CALIBRATION_OR_BBOX_INVALID")
    if fx <= 0.0 or fy <= 0.0 or x2 <= x1 or y2 <= y1:
        return DepthProjection(False, "CALIBRATION_OR_BBOX_INVALID")
    width, height = x2-x1, y2-y1
    # Lamp housings often surround the actual illuminated pixels. Use the
    # central half of the agreed 2D box and reject high-spread depth patches.
    left = max(0, int(math.floor(x1+0.25*width)))
    right = min(metric.shape[1], int(math.ceil(x2-0.25*width)))
    top = max(0, int(math.floor(y1+0.25*height)))
    bottom = min(metric.shape[0], int(math.ceil(y2-0.25*height)))
    if right <= left or bottom <= top:
        return DepthProjection(False, "BBOX_OUTSIDE_DEPTH")
    patch = metric[top:bottom, left:right]
    valid = (np.isfinite(patch) & (patch >= float(minimum_depth_m)) &
             (patch <= float(maximum_depth_m)))
    count = int(np.count_nonzero(valid))
    if count < int(minimum_pixels):
        return DepthProjection(False, "DEPTH_PIXELS_INSUFFICIENT",
                               valid_pixels=count)
    samples = patch[valid]
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples-median)))
    if not math.isfinite(mad) or mad > float(maximum_mad_m):
        return DepthProjection(False, "DEPTH_SPREAD_TOO_HIGH",
                               median_depth_m=median, depth_mad_m=mad,
                               valid_pixels=count)
    tolerance = max(0.08, 3.0*1.4826*mad)
    rows, columns = np.nonzero(valid & (np.abs(patch-median) <= tolerance))
    if len(rows) < int(minimum_pixels):
        return DepthProjection(False, "DEPTH_INLIERS_INSUFFICIENT",
                               median_depth_m=median, depth_mad_m=mad,
                               valid_pixels=len(rows))
    # Array indices address pixel cells; add half a pixel for the geometric
    # centre used by the pinhole calibration coordinates.
    u = float(np.median(columns+left+0.5))
    v = float(np.median(rows+top+0.5))
    z = float(np.median(patch[rows, columns]))
    xyz = ((u-cx)*z/fx, (v-cy)*z/fy, z)
    if not all(math.isfinite(value) for value in xyz):
        return DepthProjection(False, "PROJECTION_NONFINITE")
    return DepthProjection(True, "OK", xyz, z, mad, len(rows))


def transform_point(point, translation, quaternion):
    """Apply a geometry_msgs-style transform without a tf2 Python add-on."""
    px, py, pz = (float(value) for value in point)
    tx, ty, tz = (float(value) for value in translation)
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("invalid transform quaternion")
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    # q * p * conjugate(q), written as a compact vector rotation.
    ux, uy, uz = qx, qy, qz
    dot_uv = ux*px+uy*py+uz*pz
    dot_uu = ux*ux+uy*uy+uz*uz
    cross_x = uy*pz-uz*py
    cross_y = uz*px-ux*pz
    cross_z = ux*py-uy*px
    rx = 2.0*dot_uv*ux+(qw*qw-dot_uu)*px+2.0*qw*cross_x
    ry = 2.0*dot_uv*uy+(qw*qw-dot_uu)*py+2.0*qw*cross_y
    rz = 2.0*dot_uv*uz+(qw*qw-dot_uu)*pz+2.0*qw*cross_z
    return tx+rx, ty+ry, tz+rz


def vslam_ready(mapping_state, localization_state):
    return (str(mapping_state).split(":", 1)[0] == "READY" or
            str(localization_state) in ("TRACKING", "RELOCALIZED"))


def traffic_validation_reasons(fusion, detection_stamp, depth_stamp,
                               camera_publishers, tracking_valid,
                               mapping_state, localization_state,
                               projection_valid, transform_valid,
                               maximum_stamp_delta_sec=0.08,
                               minimum_confidence=0.60):
    """Return every failed gate; an empty list is a final-valid result."""
    reasons = []
    if int(camera_publishers) != 1:
        reasons.append(f"CAMERA_PUBLISHER_COUNT:{int(camera_publishers)}")
    if not tracking_valid:
        reasons.append("CUVSLAM_TRACKING_INVALID")
    if not vslam_ready(mapping_state, localization_state):
        reasons.append("VSLAM_NOT_READY")
    if not bool(fusion.get("sources_agree")):
        reasons.append("TWO_DIMENSIONAL_SOURCES_NOT_AGREED")
    if fusion.get("positions_match") is not True:
        reasons.append("TWO_DIMENSIONAL_BBOX_NOT_MATCHED")
    if bool(fusion.get("single_source_used")):
        reasons.append("TWO_DIMENSIONAL_SINGLE_SOURCE")
    if str(fusion.get("fused_state", "UNKNOWN")) not in ("R", "G"):
        reasons.append("TRAFFIC_STATE_UNCONFIRMED")
    try:
        confidence = float(fusion.get("fused_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < float(minimum_confidence):
        reasons.append("TRAFFIC_CONFIDENCE_LOW")
    try:
        fusion_stamp = json_stamp_seconds(fusion)
        deltas = (abs(fusion_stamp-float(detection_stamp)),
                  abs(float(depth_stamp)-float(detection_stamp)))
        if max(deltas) > float(maximum_stamp_delta_sec):
            reasons.append("OBSERVATION_TIMESTAMP_MISMATCH")
    except (KeyError, TypeError, ValueError):
        reasons.append("OBSERVATION_TIMESTAMP_INVALID")
    if not projection_valid:
        reasons.append("DEPTH_PROJECTION_INVALID")
    if not transform_valid:
        reasons.append("MAP_TRANSFORM_INVALID")
    return reasons
