"""D456 ground-plane calibration: base mount config + boot-time IMU attitude
lock + pixel->metric ground-plane projection.

Architecture (matches the production requirement, not re-derived here):

    [user-measured BASE MOUNT] + [boot-time IMU pitch/roll delta]
        -> effective camera extrinsic (locked at boot, not re-adjusted while
           driving)
        -> pixel -> ground-plane metric projection (only while calibration is
           VALID)

Only pitch/roll are auto-corrected. x, y, z and yaw are never touched by this
module; they come only from the user-configured base mount. This module owns
no ROS I/O; ``camera_metric_path_node.py`` is the ROS adapter.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np

# --- Calibration state machine -------------------------------------------
NOT_CONFIGURED = "NOT_CONFIGURED"
CALIBRATION_INIT = "CALIBRATION_INIT"
CALIBRATION_VALID = "CALIBRATION_VALID"
CALIBRATION_DEGRADED = "CALIBRATION_DEGRADED"
CALIBRATION_INVALID = "CALIBRATION_INVALID"


# --- Base mount config -----------------------------------------------------
@dataclass
class CameraMountConfig:
    """User-measured, one-time, physical mounting calibration.

    ``configured`` must be explicitly set true by the user in YAML after they
    have actually measured the values; the loader below defaults it false so
    an un-edited config never silently produces a metric path (section 20).
    """
    configured: bool = False
    position_x_m: float = 0.0
    position_y_m: float = 0.0
    height_z_m: float = 0.0
    reference_roll_deg: float = 0.0
    reference_pitch_deg: float = 0.0
    reference_yaw_deg: float = 0.0

    def is_usable(self):
        if not self.configured:
            return False
        values = (self.position_x_m, self.position_y_m, self.height_z_m,
                   self.reference_roll_deg, self.reference_pitch_deg, self.reference_yaw_deg)
        return all(math.isfinite(v) for v in values) and self.height_z_m > 0.0


# --- Rotation helpers (numpy only, no scipy) --------------------------------
def rotation_matrix_rpy(roll_deg, pitch_deg, yaw_deg):
    """Physical mount RPY to base rotation for ground projection.

    The mount configuration defines a camera looking down as negative pitch.
    REP-103/T870_MCU describes the same rotation as positive pitch, so the
    physical pitch is negated at this explicit interface boundary.
    """
    r = math.radians(roll_deg)
    p = -math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def quaternion_from_matrix(matrix):
    """Rotation matrix -> normalized (x, y, z, w) quaternion."""
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def compose_effective_orientation(reference_rpy_deg, delta_pitch_deg, delta_roll_deg):
    """Compose base-mount reference orientation with a small IMU-measured
    pitch/roll delta via rotation-matrix multiplication (not naive Euler
    addition), matching section 12. Yaw is never modified by delta.

    Returns (effective_roll_deg, effective_pitch_deg, effective_yaw_deg,
    rotation_matrix, quaternion_xyzw).
    """
    reference_matrix = rotation_matrix_rpy(*reference_rpy_deg)
    delta_matrix = rotation_matrix_rpy(delta_roll_deg, delta_pitch_deg, 0.0)
    effective_matrix = delta_matrix @ reference_matrix
    roll, pitch, yaw = matrix_to_rpy(effective_matrix)
    quat = quaternion_from_matrix(effective_matrix)
    return roll, pitch, yaw, effective_matrix, quat


def matrix_to_rpy(matrix):
    """Inverse using the physical mount convention (downward is negative)."""
    m = matrix
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 0.999999:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        yaw = 0.0
    return math.degrees(roll), -math.degrees(pitch), math.degrees(yaw)


# --- Stationary boot attitude estimation (sections 4-5) ---------------------
@dataclass
class InitConfig:
    window_sec: float = 1.0
    min_samples: int = 15
    max_samples: int = 400
    max_accel_stddev_mps2: float = 0.25
    gravity_norm_min_mps2: float = 8.5
    gravity_norm_max_mps2: float = 11.0
    max_gyro_norm_rad_s: float = 0.08
    low_pass_alpha: float = 0.2
    outlier_mad_k: float = 3.5


@dataclass
class AttitudeSample:
    accel: np.ndarray
    gyro: np.ndarray
    stamp: float


def reject_outliers(vectors, k=3.5):
    """Robust outlier rejection on 3-axis samples via per-axis MAD."""
    array = np.asarray(vectors, dtype=float)
    if len(array) < 4:
        return array
    median = np.median(array, axis=0)
    mad = np.median(np.abs(array - median), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    deviation = np.abs(array - median) / (1.4826 * mad)
    keep = np.all(deviation <= k, axis=1)
    kept = array[keep]
    return kept if len(kept) >= 3 else array


def low_pass_mean(vectors, alpha):
    """Exponential low-pass filter applied along the sample sequence, then
    averaged, biasing toward the most recent (settled) samples."""
    array = np.asarray(vectors, dtype=float)
    if len(array) == 0:
        return np.full(3, np.nan)
    state = array[0].copy()
    for sample in array[1:]:
        state = alpha * sample + (1.0 - alpha) * state
    return state


def gravity_roll_pitch_deg(gravity_vector_base):
    """Roll/pitch of the base-mount frame from a base-frame gravity vector
    using the ROS convention x=forward, y=left, z=up, +Z gravity at rest."""
    x, y, z = gravity_vector_base
    roll = math.degrees(math.atan2(-y, z))
    pitch = math.degrees(math.atan2(x, math.hypot(y, z)))
    return roll, pitch


@dataclass
class StationaryCheckResult:
    stationary: bool
    reasons: list = field(default_factory=list)


def check_stationary(accel_samples, gyro_samples, config: InitConfig, speed_mps=None):
    """Sections 5: only accept the init window if the vehicle is judged
    stationary. If a speed source is later wired in, pass speed_mps and it
    takes priority; otherwise fall back to IMU-only variance/gravity checks."""
    reasons = []
    if speed_mps is not None:
        if abs(speed_mps) > 0.05:
            reasons.append("vehicle_speed_nonzero")
            return StationaryCheckResult(False, reasons)
    accel = np.asarray(accel_samples, dtype=float)
    gyro = np.asarray(gyro_samples, dtype=float)
    if len(accel) == 0 or len(gyro) == 0:
        return StationaryCheckResult(False, ["no_samples"])
    norm_mean = float(np.linalg.norm(np.mean(accel, axis=0)))
    if not (config.gravity_norm_min_mps2 <= norm_mean <= config.gravity_norm_max_mps2):
        reasons.append(f"gravity_norm_out_of_range:{norm_mean:.3f}")
    accel_std = float(np.max(np.std(accel, axis=0)))
    if accel_std > config.max_accel_stddev_mps2:
        reasons.append(f"accel_variance_too_high:{accel_std:.4f}")
    gyro_norm_mean = float(np.mean(np.linalg.norm(gyro, axis=1)))
    if gyro_norm_mean > config.max_gyro_norm_rad_s:
        reasons.append(f"gyro_norm_too_high:{gyro_norm_mean:.4f}")
    return StationaryCheckResult(len(reasons) == 0, reasons)


class StationaryAttitudeEstimator:
    """Collects one boot-time IMU window and produces a locked pitch/roll
    delta. Call ``add_sample`` for every accel+gyro pair; once ``ready`` is
    true, call ``finalize`` once and never feed new samples into the result
    again (section 6: correction lock)."""

    def __init__(self, config: InitConfig):
        self.config = config
        self.samples = []
        self.started_monotonic = time.monotonic()
        self.locked = False
        self.result = None

    def add_sample(self, accel, gyro, stamp):
        if self.locked:
            return
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(gyro)) and math.isfinite(stamp)):
            return
        self.samples.append(AttitudeSample(accel, gyro, stamp))
        if len(self.samples) > self.config.max_samples:
            self.samples.pop(0)

    @property
    def elapsed_sec(self):
        return time.monotonic() - self.started_monotonic

    @property
    def sample_count(self):
        return len(self.samples)

    def ready(self, speed_mps=None):
        if self.locked:
            return True
        if self.elapsed_sec < self.config.window_sec:
            return False
        if self.sample_count < self.config.min_samples:
            return False
        return True

    def finalize(self, speed_mps=None):
        """Compute and lock the delta pitch/roll from the collected window.
        Returns a dict with measured roll/pitch, sample stats, and pass/fail
        stationary check. Idempotent after the first call."""
        if self.locked:
            return self.result
        accel = np.array([s.accel for s in self.samples]) if self.samples else np.empty((0, 3))
        gyro = np.array([s.gyro for s in self.samples]) if self.samples else np.empty((0, 3))
        stationary = check_stationary(accel, gyro, self.config, speed_mps=speed_mps)
        clean_accel = reject_outliers(accel, self.config.outlier_mad_k) if len(accel) else accel
        filtered = low_pass_mean(clean_accel, self.config.low_pass_alpha) if len(clean_accel) else np.full(3, np.nan)
        roll, pitch = (gravity_roll_pitch_deg(filtered) if np.all(np.isfinite(filtered))
                       else (float("nan"), float("nan")))
        self.result = {
            "sample_count": int(len(self.samples)),
            "duration_sec": float(self.elapsed_sec),
            "stationary": bool(stationary.stationary),
            "stationary_reasons": stationary.reasons,
            "measured_roll_deg": roll,
            "measured_pitch_deg": pitch,
            "gravity_vector": filtered.tolist() if np.all(np.isfinite(filtered)) else [None, None, None],
        }
        self.locked = True
        return self.result


# --- Calibration validity (sections 8-10) -----------------------------------
@dataclass
class ValidityConfig:
    max_runtime_pitch_correction_deg: float = 5.0
    max_runtime_roll_correction_deg: float = 5.0
    max_calibration_age_sec: float = 3600.0


def evaluate_calibration_state(*, mount: CameraMountConfig, imu_available: bool,
                                init_result, validity: ValidityConfig,
                                calibration_age_sec):
    """Pure function implementing the CALIBRATION_* state machine
    (sections 8-10). Never returns VALID unless every check passes."""
    if not mount.is_usable():
        return CALIBRATION_INVALID, ["NOT_CONFIGURED: base mount position/reference orientation not set"]
    if not imu_available:
        return CALIBRATION_INVALID, ["imu_unavailable"]
    if init_result is None:
        return CALIBRATION_INIT, ["initialization_in_progress"]
    reasons = []
    if not init_result["stationary"]:
        reasons.extend(f"init_{r}" for r in init_result["stationary_reasons"])
    measured_pitch = init_result["measured_pitch_deg"]
    measured_roll = init_result["measured_roll_deg"]
    if not (math.isfinite(measured_pitch) and math.isfinite(measured_roll)):
        reasons.append("measured_attitude_non_finite")
        return CALIBRATION_INVALID, reasons
    delta_pitch = measured_pitch - mount.reference_pitch_deg
    delta_roll = measured_roll - mount.reference_roll_deg
    if abs(delta_pitch) > validity.max_runtime_pitch_correction_deg:
        reasons.append(f"delta_pitch_exceeds_limit:{delta_pitch:.3f}")
    if abs(delta_roll) > validity.max_runtime_roll_correction_deg:
        reasons.append(f"delta_roll_exceeds_limit:{delta_roll:.3f}")
    if calibration_age_sec is not None and calibration_age_sec > validity.max_calibration_age_sec:
        reasons.append(f"calibration_stale:{calibration_age_sec:.1f}s")
    if reasons:
        return CALIBRATION_INVALID, reasons
    if not init_result["stationary"]:
        return CALIBRATION_DEGRADED, ["not_stationary_at_init"]
    return CALIBRATION_VALID, []


# --- Ground-plane pixel -> metric projection (section 13) -------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def pixel_ray_camera_frame(pixel_x, pixel_y, intrinsics: Intrinsics):
    """Pixel -> unit ray in the camera OPTICAL frame (x=right, y=down,
    z=forward), per the standard pinhole model."""
    x = (pixel_x - intrinsics.cx) / intrinsics.fx
    y = (pixel_y - intrinsics.cy) / intrinsics.fy
    z = 1.0
    vector = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


# ROS optical -> camera mechanical/base convention used throughout this repo
# (x=forward, y=left, z=up); matches imu_manager README's documented axis
# candidate. Kept identical here so both modules agree on one convention.
OPTICAL_TO_MECHANICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def project_ray_to_ground(ray_optical, camera_rotation_matrix, camera_position_base):
    """Intersect a camera-optical-frame ray with the base_link ground plane
    (z=0), given the camera's rotation (mechanical/base-aligned axes ->
    base_link) and its mounting position. Returns None if the ray does not
    hit the ground plane in front of the vehicle (e.g. points at/above the
    horizon)."""
    ray_mechanical = OPTICAL_TO_MECHANICAL @ ray_optical
    ray_base = camera_rotation_matrix @ ray_mechanical
    origin = camera_position_base
    if ray_base[2] >= -1e-6:
        return None
    t = -origin[2] / ray_base[2]
    if t <= 0.0:
        return None
    point = origin + t * ray_base
    return point


def project_pixel_path_to_metric(pixel_points, intrinsics: Intrinsics,
                                  camera_rotation_matrix, camera_position_base,
                                  max_range_m=30.0):
    """pixel points (list of (x_px, y_px)) -> list of (x_m, y_m) in
    base_link. Points that do not intersect the ground plane, or fall beyond
    max_range_m, are dropped (not fabricated)."""
    output = []
    for px, py in pixel_points:
        ray = pixel_ray_camera_frame(px, py, intrinsics)
        point = project_ray_to_ground(ray, camera_rotation_matrix, camera_position_base)
        if point is None:
            continue
        if math.hypot(point[0] - camera_position_base[0], point[1] - camera_position_base[1]) > max_range_m:
            continue
        output.append((float(point[0]), float(point[1])))
    return output


def load_camera_mount_config(params):
    """Build CameraMountConfig from a flat dict of ROS parameters (already
    resolved, no ROS types)."""
    return CameraMountConfig(
        configured=bool(params.get("configured", False)),
        position_x_m=float(params.get("position_x_m", 0.0)),
        position_y_m=float(params.get("position_y_m", 0.0)),
        height_z_m=float(params.get("height_z_m", 0.0)),
        reference_roll_deg=float(params.get("reference_roll_deg", 0.0)),
        reference_pitch_deg=float(params.get("reference_pitch_deg", 0.0)),
        reference_yaw_deg=float(params.get("reference_yaw_deg", 0.0)),
    )
