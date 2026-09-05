"""ROS-independent IMU filtering primitives, suitable for deterministic tests."""

from dataclasses import dataclass
import math

import numpy as np


CALIBRATING, READY, CALIBRATION_FAILED = "CALIBRATING", "READY", "FAILED"


def calibration_gate_ready(startup_enabled, startup_state, static_validated):
    """Separate runtime startup readiness from legacy static calibration."""
    return (startup_state == READY if startup_enabled else bool(static_validated))

# Candidate ROS optical (right/down/forward) -> camera mechanical
# (forward/left/up) mapping. Calibration movements must confirm its signs.
OPTICAL_TO_CAMERA_CANDIDATE = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


@dataclass
class FilterConfig:
    complementary_alpha: float = 0.98
    max_dt_sec: float = 0.1
    gravity_mps2: float = 9.80665
    stationary_gyro_threshold: float = 0.03
    stationary_accel_tolerance: float = 0.35
    bias_sample_count: int = 200


@dataclass
class SlopeStateConfig:
    uphill_enter_pitch_deg: float = 3.0
    uphill_exit_pitch_deg: float = 1.5
    uphill_confirmation_frames: int = 10


class SlopeStateDetector:
    """Detect sustained positive (nose-up) pitch with hysteresis."""

    def __init__(self, config=None):
        self.config = config or SlopeStateConfig()
        enter = self.config.uphill_enter_pitch_deg
        exit_pitch = self.config.uphill_exit_pitch_deg
        frames = self.config.uphill_confirmation_frames
        if (not math.isfinite(enter) or not math.isfinite(exit_pitch)
                or enter <= exit_pitch or exit_pitch < 0.0):
            raise ValueError(
                "uphill thresholds must be finite with enter > exit >= 0")
        if isinstance(frames, bool) or int(frames) != frames or frames < 1:
            raise ValueError("uphill confirmation frames must be a positive integer")
        self.config.uphill_confirmation_frames = int(frames)
        self.state = False
        self._candidate_frames = 0

    def invalidate(self):
        self.state = False
        self._candidate_frames = 0

    def update(self, pitch_deg, imu_valid=True):
        if not imu_valid or not math.isfinite(pitch_deg):
            self.invalidate()
            return self.state

        pitch = float(pitch_deg)
        if self.state:
            candidate = pitch <= self.config.uphill_exit_pitch_deg
        else:
            # Deliberately signed: level and every downhill pitch remain false.
            candidate = pitch >= self.config.uphill_enter_pitch_deg

        self._candidate_frames = self._candidate_frames + 1 if candidate else 0
        if self._candidate_frames >= self.config.uphill_confirmation_frames:
            self.state = not self.state
            self._candidate_frames = 0
        return self.state


@dataclass
class StartupCalibrationConfig:
    duration_sec: float = 3.0
    timeout_sec: float = 15.0
    gyro_still_threshold_rad_s: float = 0.05
    accel_stddev_limit_mps2: float = 0.10
    accel_norm_min_mps2: float = 8.5
    accel_norm_max_mps2: float = 11.0


@dataclass
class StartupCalibrationResult:
    gravity_axis_mean: np.ndarray
    mounting_roll_deg: float
    mounting_pitch_deg: float
    mounting_yaw_deg: float
    gyro_bias_axis: np.ndarray
    gravity_norm: float
    duration_sec: float
    restart_count: int


class StartupCalibrator:
    """Collect one continuous stationary startup window in sensor time."""

    def __init__(self, config=None, started_monotonic=0.0):
        self.config = config or StartupCalibrationConfig()
        values = np.asarray(list(vars(self.config).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("startup calibration parameters must be finite and positive")
        if self.config.accel_norm_min_mps2 >= self.config.accel_norm_max_mps2:
            raise ValueError("startup acceleration norm range is invalid")
        self.started_monotonic = float(started_monotonic)
        self.state = CALIBRATING
        self.restart_count = 0
        self.accel_samples = []
        self.gyro_samples = []
        self.last_accel_stamp = None
        self.last_gyro_stamp = None
        self.failure_reason = ""

    def _reset_window(self):
        if self.accel_samples or self.gyro_samples:
            self.restart_count += 1
        self.accel_samples = []
        self.gyro_samples = []

    def check_timeout(self, now_monotonic):
        if (self.state == CALIBRATING
                and now_monotonic - self.started_monotonic > self.config.timeout_sec):
            self.state = CALIBRATION_FAILED
            self.failure_reason = (
                f"timeout after {self.config.timeout_sec:.3f}s; "
                f"accepted accel={len(self.accel_samples)} gyro={len(self.gyro_samples)} "
                f"restarts={self.restart_count}")
        return self.state

    @staticmethod
    def _sample(vector, stamp):
        value = np.asarray(vector, dtype=float)
        return value if value.shape == (3,) and np.all(np.isfinite(value)) and math.isfinite(stamp) else None

    def add_accel(self, vector, stamp, now_monotonic=0.0):
        if self.check_timeout(now_monotonic) != CALIBRATING:
            return False
        value = self._sample(vector, stamp)
        if value is None or (self.last_accel_stamp is not None and stamp <= self.last_accel_stamp):
            return False
        self.last_accel_stamp = float(stamp)
        norm = float(np.linalg.norm(value))
        if not self.config.accel_norm_min_mps2 <= norm <= self.config.accel_norm_max_mps2:
            self._reset_window()
            return False
        self.accel_samples.append((float(stamp), value.copy()))
        return True

    def add_gyro(self, vector, stamp, now_monotonic=0.0):
        if self.check_timeout(now_monotonic) != CALIBRATING:
            return False
        value = self._sample(vector, stamp)
        if value is None or (self.last_gyro_stamp is not None and stamp <= self.last_gyro_stamp):
            return False
        self.last_gyro_stamp = float(stamp)
        if float(np.linalg.norm(value)) > self.config.gyro_still_threshold_rad_s:
            self._reset_window()
            return False
        self.gyro_samples.append((float(stamp), value.copy()))
        return True

    def try_complete(self, now_monotonic, mounting_yaw_deg=0.0):
        if self.check_timeout(now_monotonic) != CALIBRATING:
            return None
        if len(self.accel_samples) < 2 or len(self.gyro_samples) < 2:
            return None
        accel_duration = self.accel_samples[-1][0] - self.accel_samples[0][0]
        gyro_duration = self.gyro_samples[-1][0] - self.gyro_samples[0][0]
        duration = min(accel_duration, gyro_duration)
        if duration < self.config.duration_sec:
            return None
        accel = np.asarray([sample for _, sample in self.accel_samples])
        if float(np.max(np.std(accel, axis=0))) > self.config.accel_stddev_limit_mps2:
            self._reset_window()
            return None
        gravity = np.mean(accel, axis=0)
        alignment = compute_level_alignment(
            gravity,
            gravity_range=(self.config.accel_norm_min_mps2,
                           self.config.accel_norm_max_mps2),
        )
        if not alignment.physical_sanity_passed:
            self.state = CALIBRATION_FAILED
            self.failure_reason = (
                f"physical sanity check failed: {'; '.join(alignment.errors)}; "
                f"gravity_axis_mean={gravity.tolist()} norm={float(np.linalg.norm(gravity)):.6f}; "
                f"correction_roll_deg={alignment.correction_roll_deg:.6f} "
                f"correction_pitch_deg={alignment.correction_pitch_deg:.6f}")
            return None
        gyro_bias = np.mean([sample for _, sample in self.gyro_samples], axis=0)
        self.state = READY
        return StartupCalibrationResult(
            gravity_axis_mean=gravity,
            mounting_roll_deg=alignment.correction_roll_deg,
            mounting_pitch_deg=alignment.correction_pitch_deg,
            mounting_yaw_deg=float(mounting_yaw_deg),
            gyro_bias_axis=gyro_bias,
            gravity_norm=float(np.linalg.norm(gravity)),
            duration_sec=duration,
            restart_count=self.restart_count,
        )


def angle_message_fields(stamp_sec, frame_id, roll_deg, pitch_deg, yaw_deg):
    """Return ROS stamp fields, frame, and unrounded RPY without ROS imports."""
    values = np.asarray([stamp_sec, roll_deg, pitch_deg, yaw_deg], dtype=float)
    if not np.all(np.isfinite(values)) or float(stamp_sec) < 0.0:
        raise ValueError("angle message fields must be finite with non-negative time")
    seconds = int(math.floor(stamp_sec))
    nanoseconds = int(round((float(stamp_sec) - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return seconds, nanoseconds, str(frame_id), tuple(float(value) for value in values[1:])


def euler_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Return active ZYX rotation from mounting RPY in degrees."""
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
        math.cos(yaw), math.sin(yaw),
    )
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def normalize_angle_deg(angle):
    """Normalize an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _rotation_magnitude_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _matrix_to_zyx_rpy(rotation):
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(normalize_angle_deg(math.degrees(value)) for value in (roll, pitch, yaw))


@dataclass
class LevelAlignment:
    gravity_before: np.ndarray
    gravity_after: np.ndarray
    measured_roll_deg: float
    measured_pitch_deg: float
    correction_roll_deg: float
    correction_pitch_deg: float
    correction_rotation_magnitude_deg: float
    selected_euler_branch: str
    physical_sanity_passed: bool
    errors: list


@dataclass
class TransformStages:
    """Explicit column-vector stages; matrices are row-major when serialized."""
    raw_sensor_vector: np.ndarray
    configured_sensor_axis_matrix: np.ndarray
    axis_transformed_vector: np.ndarray
    mounting_rotation_matrix: np.ndarray
    base_link_vector: np.ndarray


@dataclass
class AxisMatrixValidation:
    valid: bool
    determinant: float
    reflection: bool
    orthogonality_error: float
    signed_permutation: bool
    errors: list


def axis_matrix_from_row_major(values):
    """Decode exactly nine YAML/log row-major values without transpose/inverse."""
    flat = np.asarray(values, dtype=float)
    if flat.size != 9:
        raise ValueError("sensor axis matrix requires exactly nine values")
    return flat.reshape((3, 3), order="C")


def validate_rotation_matrix(matrix, *, allow_reflection=True, epsilon=1e-9):
    """Validate a finite orthogonal signed-permutation axis transform."""
    value = np.asarray(matrix, dtype=float)
    errors = []
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        return AxisMatrixValidation(False, math.nan, False, math.inf, False,
                                    ["matrix must be finite 3x3"])
    determinant = float(np.linalg.det(value))
    orthogonality_error = float(np.max(np.abs(value.T @ value - np.eye(3))))
    element_valid = bool(np.all(np.isclose(np.abs(value), 0.0, atol=epsilon)
                                | np.isclose(np.abs(value), 1.0, atol=epsilon)))
    row_axes = np.sum(np.isclose(np.abs(value), 1.0, atol=epsilon), axis=1)
    column_axes = np.sum(np.isclose(np.abs(value), 1.0, atol=epsilon), axis=0)
    signed_permutation = bool(element_valid and np.all(row_axes == 1) and np.all(column_axes == 1))
    reflection = determinant < 0.0
    if orthogonality_error > epsilon:
        errors.append("M.T @ M is not identity")
    if not signed_permutation:
        errors.append("matrix is not a signed permutation")
    if not math.isclose(abs(determinant), 1.0, abs_tol=epsilon):
        errors.append("determinant is not +1 or -1")
    if reflection and not allow_reflection:
        errors.append("reflection matrix is not allowed")
    return AxisMatrixValidation(not errors, determinant, reflection,
                                orthogonality_error, signed_permutation, errors)


def apply_sensor_axis_transform(matrix, raw_vector):
    """Apply configured row-major matrix to a column vector: matrix @ vector."""
    validation = validate_rotation_matrix(matrix)
    vector = np.asarray(raw_vector, dtype=float)
    if not validation.valid:
        raise ValueError("invalid sensor axis matrix: " + "; ".join(validation.errors))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("sensor vector must contain three finite values")
    return np.asarray(matrix, dtype=float) @ vector


def apply_mounting_rotation(mounting_rotation_matrix, axis_transformed_vector):
    """Apply the explicit mounting rotation to a column vector."""
    rotation = np.asarray(mounting_rotation_matrix, dtype=float)
    vector = np.asarray(axis_transformed_vector, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("mounting rotation must be finite 3x3")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("axis-transformed vector must contain three finite values")
    return rotation @ vector


def transform_sensor_to_base(raw_vector, axis_matrix, mounting_rpy_deg):
    """Return all stages of R_mount @ M_axis @ raw column-vector conversion."""
    raw = np.asarray(raw_vector, dtype=float)
    axis = np.asarray(axis_matrix, dtype=float)
    axis_vector = apply_sensor_axis_transform(axis, raw)
    mounting_rotation = euler_rotation_matrix(*mounting_rpy_deg)
    base_vector = apply_mounting_rotation(mounting_rotation, axis_vector)
    return TransformStages(raw.copy(), axis.copy(), axis_vector,
                           mounting_rotation, base_vector)


def compute_level_alignment(gravity, *, allow_inverted_mount=False,
                            plausible_mount_angle_limit_deg=45.0,
                            xy_tolerance_mps2=0.05,
                            gravity_range=(7.0, 12.0)):
    """Select the smallest yaw-free ZYX correction that aligns gravity to +Z."""
    vector = np.asarray(gravity, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("gravity vector must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if not gravity_range[0] <= norm <= gravity_range[1]:
        raise ValueError("gravity norm outside safety range")
    if not math.isfinite(plausible_mount_angle_limit_deg) or plausible_mount_angle_limit_deg <= 0.0:
        raise ValueError("plausible mount angle limit must be positive and finite")

    x, y, z = vector
    base_roll = math.atan2(y, z)
    candidates = []
    for index, roll in enumerate((base_roll, base_roll + math.pi)):
        # After Rx: y must be zero; use the actual intermediate z to select the
        # correct atan2 pitch branch rather than assuming hypot(y,z) is positive.
        z_after_roll = math.sin(roll) * y + math.cos(roll) * z
        pitch = math.atan2(-x, z_after_roll)
        roll_deg = normalize_angle_deg(math.degrees(roll))
        pitch_deg = normalize_angle_deg(math.degrees(pitch))
        rotation = euler_rotation_matrix(roll_deg, pitch_deg, 0.0)
        corrected = rotation @ vector
        if (abs(corrected[0]) <= xy_tolerance_mps2
                and abs(corrected[1]) <= xy_tolerance_mps2
                and corrected[2] > 0.0):
            candidates.append((
                _rotation_magnitude_deg(rotation), index, roll_deg, pitch_deg,
                rotation, corrected,
            ))
    if not candidates:
        raise ValueError("no Euler branch aligns gravity to positive Z")
    magnitude, index, roll_deg, pitch_deg, rotation, corrected = min(candidates, key=lambda item: item[0])

    errors = []
    if not allow_inverted_mount and (abs(roll_deg) >= 90.0 or abs(pitch_deg) >= 90.0):
        errors.append("inverted mounting candidate requires allow_inverted_mount=true")
    if abs(roll_deg) > plausible_mount_angle_limit_deg or abs(pitch_deg) > plausible_mount_angle_limit_deg:
        errors.append("mounting correction exceeds plausible angle limit")
    if corrected[2] <= 0.0:
        errors.append("corrected gravity does not point to positive Z")
    if abs(corrected[0]) > xy_tolerance_mps2 or abs(corrected[1]) > xy_tolerance_mps2:
        errors.append("corrected gravity XY exceeds tolerance")

    # Measured attitude is the inverse orientation; correction is the active
    # vector rotation applied by Rz(0) Ry(pitch) Rx(roll).
    measured_roll, measured_pitch, _ = _matrix_to_zyx_rpy(rotation.T)
    return LevelAlignment(
        gravity_before=vector.copy(),
        gravity_after=corrected,
        measured_roll_deg=measured_roll,
        measured_pitch_deg=measured_pitch,
        correction_roll_deg=roll_deg,
        correction_pitch_deg=pitch_deg,
        correction_rotation_magnitude_deg=magnitude,
        selected_euler_branch=f"atan2_branch_{index}_minimum_rotation",
        physical_sanity_passed=not errors,
        errors=errors,
    )


def validate_axis_matrix(matrix):
    """Compatibility boolean for the strict signed-permutation validator."""
    return validate_rotation_matrix(matrix).valid


def level_mounting_rpy(gravity_in_camera_frame, yaw_deg=0.0):
    """Find mounting roll/pitch that rotates stationary gravity onto base +Z.

    Gravity cannot determine mounting yaw, so yaw is preserved as a manual
    calibration input.
    """
    if not math.isfinite(yaw_deg):
        raise ValueError("mounting yaw must be finite")
    alignment = compute_level_alignment(
        gravity_in_camera_frame,
        allow_inverted_mount=True,
        plausible_mount_angle_limit_deg=180.0,
    )
    return alignment.correction_roll_deg, alignment.correction_pitch_deg, float(yaw_deg)


def quaternion_from_rpy(roll_deg, pitch_deg, yaw_deg):
    """Create a normalized quaternion consistent with active ZYX RPY."""
    if not np.all(np.isfinite([roll_deg, pitch_deg, yaw_deg])):
        raise ValueError("RPY must be finite")
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg]) / 2.0
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    value = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    return value / np.linalg.norm(value)


def vehicle_pitch_from_accel(accel_base_link):
    """Return nose-up/down pitch from a base_link acceleration vector only."""
    vector = np.asarray(accel_base_link, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("base_link acceleration must contain three finite values")
    ax, ay, az = vector
    return math.degrees(math.atan2(-ax, math.hypot(ay, az)))


def validate_level_samples(accel_samples, gyro_samples, gravity_range=(7.0, 12.0),
                           max_accel_stddev=0.25, motion_gyro_threshold=0.08):
    """Return calibration safety errors without changing any configuration."""
    accel = np.asarray(accel_samples, dtype=float)
    gyro = np.asarray(gyro_samples, dtype=float)
    errors = []
    if accel.ndim != 2 or accel.shape[1:] != (3,) or not np.all(np.isfinite(accel)):
        errors.append("invalid accel samples")
    if gyro.ndim != 2 or gyro.shape[1:] != (3,) or not np.all(np.isfinite(gyro)):
        errors.append("invalid gyro samples")
    if errors:
        return errors
    norm = float(np.linalg.norm(np.mean(accel, axis=0)))
    if not gravity_range[0] <= norm <= gravity_range[1]:
        errors.append("gravity norm outside safety range")
    if float(np.max(np.std(accel, axis=0))) > max_accel_stddev:
        errors.append("stationary accel standard deviation too large")
    if bool(np.any(np.linalg.norm(gyro, axis=1) > motion_gyro_threshold)):
        errors.append("motion detected during gyro bias collection")
    return errors


class ImuFilter:
    def __init__(self, config=None, axis_matrix=None, mounting_rpy_deg=(0.0, 0.0, 0.0)):
        self.config = config or FilterConfig()
        axis = np.asarray(axis_matrix, dtype=float) if axis_matrix is not None else np.array([])
        mounting = np.asarray(mounting_rpy_deg, dtype=float)
        self.transform_valid = validate_axis_matrix(axis) and mounting.shape == (3,) and np.all(np.isfinite(mounting))
        self.axis_matrix = axis if self.transform_valid else None
        self.mounting_rpy_deg = mounting if self.transform_valid else None
        self.mounting_rotation = euler_rotation_matrix(*mounting) if self.transform_valid else None
        self.reset_all()

    def apply_startup_calibration(self, mounting_rpy_deg, gyro_bias_axis):
        """Apply one in-memory startup correction; never writes configuration."""
        mounting = np.asarray(mounting_rpy_deg, dtype=float)
        bias_axis = np.asarray(gyro_bias_axis, dtype=float)
        if (self.axis_matrix is None or mounting.shape != (3,) or bias_axis.shape != (3,)
                or not np.all(np.isfinite(mounting)) or not np.all(np.isfinite(bias_axis))):
            return False
        self.mounting_rpy_deg = mounting
        self.mounting_rotation = euler_rotation_matrix(*mounting)
        self.transform_valid = True
        self.reset_all()
        self.gyro_bias = self.mounting_rotation @ bias_axis
        self.bias_ready = True
        return True

    def reset_all(self):
        self.gyro_bias = np.zeros(3)
        self._bias_samples = []
        self.bias_ready = False
        self.accel = None
        self.gyro = None
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_total_deg = 0.0
        self.yaw_reference_deg = 0.0
        self.last_gyro_stamp = None
        self.last_accel_stamp = None
        self.last_attitude_stamp = None
        self.stationary = False

    @staticmethod
    def finite_vector(vector):
        value = np.asarray(vector, dtype=float)
        return value.shape == (3,) and bool(np.all(np.isfinite(value)))

    def validate_timestamp(self, stamp, previous):
        if not math.isfinite(stamp) or stamp <= 0.0:
            return False, None
        if previous is None:
            return True, None
        dt = stamp - previous
        # Header nanoseconds converted to float can land a few ulps over the limit.
        return 0.0 < dt <= self.config.max_dt_sec + 1e-9, dt

    def transform_sensor_to_vehicle_frame(self, vector):
        if not self.transform_valid or not self.finite_vector(vector):
            return None
        # One shared column-vector path for manager and calibration.
        return transform_sensor_to_base(
            vector,
            self.axis_matrix,
            self.mounting_rpy_deg,
        ).base_link_vector

    def update_accel(self, vector, stamp):
        valid, _ = self.validate_timestamp(stamp, self.last_accel_stamp)
        transformed = self.transform_sensor_to_vehicle_frame(vector)
        if (not valid and transformed is not None and self.last_accel_stamp is not None
                and stamp > self.last_accel_stamp + self.config.max_dt_sec):
            self.last_accel_stamp = stamp
            self.accel = transformed
            return False
        if not valid or transformed is None:
            return False
        self.accel = transformed
        self.last_accel_stamp = stamp
        self.detect_stationary_state()
        return True

    def update_gyro(self, vector, stamp):
        valid, dt = self.validate_timestamp(stamp, self.last_gyro_stamp)
        transformed = self.transform_sensor_to_vehicle_frame(vector)
        if (not valid and transformed is not None and self.last_gyro_stamp is not None
                and stamp > self.last_gyro_stamp + self.config.max_dt_sec):
            self.last_gyro_stamp = stamp
            self.gyro = transformed
            return False
        if not valid or transformed is None:
            return False
        self.gyro = transformed
        self.last_gyro_stamp = stamp
        self.detect_stationary_state()
        self.estimate_stationary_bias()
        if dt is not None and self.bias_ready:
            corrected = transformed - self.gyro_bias
            self.update_complementary_filter(corrected, dt)
            self.integrate_relative_yaw(corrected[2], dt)
            self.last_attitude_stamp = stamp
        return True

    def detect_stationary_state(self):
        if self.accel is None or self.gyro is None:
            self.stationary = False
        else:
            self.stationary = (
                np.linalg.norm(self.gyro) <= self.config.stationary_gyro_threshold
                and abs(np.linalg.norm(self.accel) - self.config.gravity_mps2)
                <= self.config.stationary_accel_tolerance
            )
        return self.stationary

    def estimate_stationary_bias(self):
        if self.bias_ready or not self.stationary:
            return self.bias_ready
        self._bias_samples.append(self.gyro.copy())
        if len(self._bias_samples) >= self.config.bias_sample_count:
            self.gyro_bias = np.mean(self._bias_samples, axis=0)
            self.bias_ready = True
        return self.bias_ready

    def update_complementary_filter(self, corrected_gyro, dt):
        gx, gy, _ = corrected_gyro
        self.roll_deg += math.degrees(gx * dt)
        self.pitch_deg += math.degrees(gy * dt)
        if self.accel is not None:
            ax, ay, az = self.accel
            accel_roll = math.degrees(math.atan2(ay, az))
            accel_pitch = vehicle_pitch_from_accel(self.accel)
            alpha = self.config.complementary_alpha
            self.roll_deg = alpha * self.roll_deg + (1.0 - alpha) * accel_roll
            self.pitch_deg = alpha * self.pitch_deg + (1.0 - alpha) * accel_pitch

    def integrate_relative_yaw(self, yaw_rate_rad_s, dt):
        self.yaw_total_deg += math.degrees(yaw_rate_rad_s * dt)

    @property
    def relative_yaw_deg(self):
        return self.yaw_total_deg - self.yaw_reference_deg

    def reset_reference(self):
        self.yaw_reference_deg = self.yaw_total_deg

    def handle_mode_transition(self, old_mode, new_mode):
        if new_mode != old_mode:
            self.reset_reference()

    def is_fresh(self, now, stale_timeout):
        return (
            self.last_gyro_stamp is not None and self.last_accel_stamp is not None
            and 0.0 <= now - self.last_gyro_stamp <= stale_timeout
            and 0.0 <= now - self.last_accel_stamp <= stale_timeout
        )
