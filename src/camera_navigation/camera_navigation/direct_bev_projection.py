"""Ground-plane BEV remap construction from calibrated camera geometry."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .ground_plane_calibration import OPTICAL_TO_MECHANICAL


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    distortion_model: str = "plumb_bob"


def _validate_projection_inputs(camera, rotation, position):
    rotation = np.asarray(rotation, dtype=float)
    position = np.asarray(position, dtype=float)
    matrix = np.asarray(camera.matrix, dtype=float)
    distortion = np.asarray(camera.distortion, dtype=float).reshape(-1)
    if (rotation.shape != (3, 3) or position.shape != (3,) or
            matrix.shape != (3, 3) or
            not np.all(np.isfinite(rotation)) or
            not np.all(np.isfinite(position)) or
            not np.all(np.isfinite(matrix)) or
            not np.all(np.isfinite(distortion))):
        raise ValueError("nonfinite or malformed camera transform")
    return rotation, position, matrix, distortion


def _project_optical_points(optical, camera, matrix, distortion):
    model = str(camera.distortion_model or "").lower()
    object_points = optical.reshape(-1, 1, 3)
    if model in ("", "none") or not len(distortion):
        normalized = optical[:, :2]/optical[:, 2:3]
        return normalized @ matrix[:2, :2].T + matrix[:2, 2]
    if model in ("plumb_bob", "rational_polynomial"):
        pixels, _ = cv2.projectPoints(
            object_points, np.zeros(3), np.zeros(3), matrix, distortion)
        return pixels.reshape(-1, 2)
    if model == "equidistant":
        if len(distortion) < 4:
            raise ValueError("equidistant distortion requires four coefficients")
        pixels, _ = cv2.fisheye.projectPoints(
            object_points, np.zeros(3), np.zeros(3), matrix, distortion[:4])
        return pixels.reshape(-1, 2)
    raise ValueError(f"unsupported distortion model: {model}")


def build_ground_remap(config, camera, rotation, position):
    """Return image sampling maps for each metric BEV raster cell."""
    rotation, position, matrix, distortion = _validate_projection_inputs(
        camera, rotation, position)
    rows = int(math.ceil((config.x_max_m-config.x_min_m) /
                         config.resolution_m))+1
    cols = int(math.ceil((config.y_max_m-config.y_min_m) /
                         config.resolution_m))+1
    row, col = np.indices((rows, cols), dtype=np.float64)
    x = config.x_max_m-row*config.resolution_m
    y = config.y_min_m+col*config.resolution_m
    base = np.column_stack((x.ravel(), y.ravel(), np.zeros(rows*cols)))
    mechanical = (rotation.T @ (base-position).T).T
    optical = (OPTICAL_TO_MECHANICAL.T @ mechanical.T).T
    valid = optical[:, 2] > 1.0e-6
    map_x = np.full(len(base), -1.0, np.float32)
    map_y = np.full(len(base), -1.0, np.float32)
    indices = np.flatnonzero(valid)
    if len(indices):
        pixels = _project_optical_points(
            optical[indices], camera, matrix, distortion)
        inside = ((pixels[:, 0] >= 0.0) & (pixels[:, 0] < camera.width) &
                  (pixels[:, 1] >= 0.0) & (pixels[:, 1] < camera.height))
        selected = indices[inside]
        map_x[selected] = pixels[inside, 0]
        map_y[selected] = pixels[inside, 1]
    return map_x.reshape(rows, cols), map_y.reshape(rows, cols)


def project_mask_to_bev(mask, map_x, map_y):
    return (cv2.remap(
        (np.asarray(mask) > 0).astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0) > 0).astype(np.uint8)
