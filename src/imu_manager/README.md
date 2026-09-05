# imu_manager

`imu_manager` subscribes to IMU messages from the one D456 instance owned by
`camera_bringup`. It never starts librealsense or `realsense2_camera_node` and
does not publish motor, steering, `/cam_drive`, or `/cam_wheel` commands.

## Inputs and system mode contract

- `/camera/camera/gyro/sample` — `sensor_msgs/msg/Imu`, configured at 200 Hz
- `/camera/camera/accel/sample` — `sensor_msgs/msg/Imu`, configured at 100 Hz
- `/vehicle_mode` — `std_msgs/msg/String`

Wrapper 4.58.1 supports `unite_imu_method` values 0 (none), 1 (copy), and 2
(linear interpolation). This system uses 0 and consumes the two raw topics.

The mode contract was taken from the legacy MCU command manager:
`IDLE`, `NORMAL`, `PARALLEL_PARK`, `T_PARK`, and `SLOPE`. There are no numeric
values. `INTERSECTION` is not defined by that package, so this package does not
invent it; any undefined string is treated as `NORMAL`. When the shared mode
contract later adds intersection, its exact upstream spelling can be adopted.

Every actual mode transition resets relative yaw, preventing accumulation
across missions. Parking controllers can also call `/imu_reset` at each stage.
IMU distance and absolute position are intentionally not estimated.

## Filter and frames

Samples with NaN/Inf, duplicate/reversed timestamps, or excessive `dt` are
dropped. Every process starts in `CALIBRATING` and requires a continuous
three-second stationary window. During this state `/imu_valid=false`,
`/imu/slope=false`, and relative Yaw is not integrated. Movement discards the
window and starts it again. Failure to finish within 15 seconds leaves the node
invalid instead of silently using zero corrections.

Coordinate conversion has two explicit, identical stages for accel and gyro:

```text
optical vector --sensor_axis_matrix--> camera mechanical vector
camera mechanical vector --Rz(yaw) Ry(pitch) Rx(roll)--> base_link vector
```

All vectors are column vectors. YAML and result matrices are row-major 3x3,
and the only axis expression is `axis_transformed = matrix @ raw_vector`.
No transpose or inverse is implicitly applied. Both runtime manager and
calibration call the same `transform_sensor_to_base()` implementation, which
exposes `raw_sensor_vector`, `configured_sensor_axis_matrix`,
`axis_transformed_vector`, `mounting_rotation_matrix`, and `base_link_vector`.

ROS optical axes are `x=right, y=down, z=forward`; the intended camera
mechanical/base convention is `x=forward, y=left, z=up`. The initial candidate
is therefore:

```text
[ 0  0  1 ]
[-1  0  0 ]
[ 0 -1  0 ]
```

The axis candidate still requires commissioning validation. At startup its
axis-transformed mean gravity is aligned to base `+Z` with the smallest Roll/
Pitch correction, and the stationary mean Gyro becomes the X/Y/Z bias. These
values are applied in process memory only and never written to YAML. Gravity
cannot observe mounting Yaw, so `mounting_yaw_deg` remains fixed in YAML. The
camera forward axis must be mechanically aligned with vehicle forward. If that
Yaw changes between runs, startup gravity cannot reliably separate vehicle
Pitch and Roll.

The published quaternion is normalized and generated from exactly the filtered
roll, pitch, and relative yaw. Orientation covariance uses non-zero roll/pitch
variance and a much larger yaw variance because yaw has no absolute reference.
The raw messages' `orientation.w=0` is ignored because the D456 raw streams do
not provide an orientation estimate.

## Outputs

Mission-facing outputs are `/imu_pitch`, `/imu_roll`, `/imu_yaw`,
`/imu_yaw_rate` (`Float32`) plus `/imu_valid` and `/imu/slope` (`Bool`).
Angles and yaw rate are rounded to two decimals only when published. Filtering
and slope decisions use the unrounded Python/NumPy float value. This
format does not claim 0.01-degree physical accuracy. `/imu_reset` resets only
relative Yaw; it never reruns or clears startup Roll/Pitch calibration.

Legacy `/imu/...` data, RPY, angle, validity, stationary, relative-Yaw, roll,
pitch, yaw-rate topics and reset services remain as compatibility publishers.
No `/slope/stop`, drive, wheel, or MCU command is published.

## Build and run

Camera terminal:

```bash
cd ~/camera_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch camera_bringup d456_bringup.launch.py
```

Independent IMU terminal (do not source `camera_ws` here):

```bash
cd ~/imu_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch imu_manager imu_manager.launch.py
```

Reset relative Yaw when required:

```bash
ros2 service call /imu_reset std_srvs/srv/Trigger "{}"
```

At node startup, obey the warning:

```text
Keep vehicle stationary on level ground during startup calibration.
```

Keep the complete vehicle and D456 stationary for at least three seconds,
confirm `/imu_valid=true` and approximately `/imu_pitch=0.00`, then drive. An
IMU cannot prove that the startup surface is level; that is an operator
precondition.

Operational order is strict: park on level ground, start `camera_bringup`,
start `imu_manager`, do not move for at least three seconds, verify validity and
near-zero Pitch/Roll, and only then enable vehicle motion. Never start or
restart `imu_manager` on a slope or while driving. The launch action explicitly
sets `respawn=False`, so a crash cannot silently recalibrate on the road.

With `startup_calibration_enabled=true`, runtime `READY` is the calibration
gate and the legacy `calibration_validated` value is ignored. The static flag is
used only when startup calibration is explicitly disabled; it cannot override
a runtime timeout or failed startup calibration.

If the Motion Module reports permission denied below `/sys/.../iio:device*/`,
the host's RealSense/IIO device permissions must be corrected by an administrator;
this package deliberately does not modify system udev rules.

## Calibration procedure

Calibration only subscribes to the already-running `camera_bringup` topics. It
never starts or opens a second RealSense device. Each run lasts at least 10
seconds and writes `~/.config/imu_manager/imu_calibration_result.yaml`; it never
edits `config/imu_manager.yaml`. The directory is created when needed. Review
the result before copying candidates.

Start the single camera node first, then run one calibration movement:

```bash
ros2 launch imu_manager imu_calibration.launch.py calibration_mode:=LEVEL duration_sec:=10
ros2 launch imu_manager imu_calibration.launch.py calibration_mode:=PITCH duration_sec:=10
ros2 launch imu_manager imu_calibration.launch.py calibration_mode:=ROLL duration_sec:=10
ros2 launch imu_manager imu_calibration.launch.py calibration_mode:=YAW duration_sec:=10
```

- `LEVEL`: park on a level surface for 5–10 seconds. It averages acceleration,
  checks 7–12 m/s² gravity, stationary variance, and motion during gyro-bias
  collection. Motion clears the window and restarts collection. The result
  reports measured level attitude separately from the active mounting
  correction. Both Euler branches are applied with
  `Rz(0) Ry(correction_pitch) Rx(correction_roll)`; the smallest rotation that
  maps gravity to positive base `+Z` is selected. Yaw is recorded as `null`
  because gravity cannot observe it.
- `PITCH`: raise the vehicle nose. Confirm positive/negative pitch direction
  and that roll coupling is small. The run fails if transformed gyro Y does not
  exceed the minimum motion rate or gyro X coupling is too large.
- `ROLL`: raise one vehicle side. Confirm roll direction and small pitch
  coupling. The run fails if transformed gyro X motion is too small or gyro Y
  coupling is too large.
- `YAW`: rotate horizontally left, then right in separate runs. Confirm
  transformed angular-velocity Z and relative-yaw signs are opposite. A
  positive left-turn convention should be selected consistently. A run without
  sufficient transformed gyro Z motion fails validation.

The result includes raw/transformed gravity, before/after roll and pitch,
selected axis matrix, mounting candidate, gyro bias, yaw sign, validation
errors, and per-stream count/Hz/min-max dt/duplicate-or-reverse/100 ms gap/drop
estimates. Low measured FPS is reported rather than treated as calibration
failure. `INTERSECTION` behavior is intentionally unchanged in this stage.

LEVEL physical sanity additionally requires corrected X/Y gravity within
`level_xy_tolerance_mps2`, corrected Z positive, and roll/pitch within
`plausible_mount_angle_limit_deg` (45° by default). A candidate at or beyond
90° is rejected while `allow_inverted_mount=false`. Set that flag true only
after physically confirming an inverted camera; doing so does not bypass the
separate plausible-angle limit. Angles are normalized to `[-180°, 180°)`.

The result distinguishes `measured_level_roll_deg` and
`measured_level_pitch_deg` (the observed camera attitude) from
`mounting_correction_roll_deg` and `mounting_correction_pitch_deg` (the active
rotation to put gravity on +Z). It also records before/after gravity, correction
rotation magnitude, selected Euler branch, inverted-mount policy, physical
sanity, and the LEVEL validation result. No correction is copied into the
system YAML automatically.

Every calibration result also records `matrix_convention: column_vector`, the
configured matrix, axis-transformed gravity, mounting matrix, determinant,
reflection policy, orthogonality error, and signed-permutation status. It
independently recomputes `configured_matrix @ raw_gravity_mean`; any per-axis
difference above `transform_consistency_epsilon` records
`TRANSFORM_CONSISTENCY_ERROR`, fails calibration, and suppresses mounting-angle
candidate calculation.

Motion restart diagnostics retain the unchanged gyro threshold and record each
restart sensor timestamp/norm, the maximum restart norm, restart count, and the
continuous stationary time actually used for the final bias window.

## Current slope state contract

`/imu/slope` is a current-condition `Bool`, not a stop command. It uses the
unrounded, startup-level-relative base-link Pitch. In REP-103 base axes a
nose-up vehicle has positive Pitch, so only sustained positive Pitch can enter
the uphill state; level and negative/downhill Pitch are false. The legacy
`/slope_state` topic publishes the same value for compatibility.

The default entry condition is Pitch `>= 3.0` degrees for 10 consecutive
attitude updates. Once true, the state remains true through threshold noise and
exits only after Pitch `<= 1.5` degrees for 10 consecutive updates. Configure
`uphill_enter_pitch_deg`, `uphill_exit_pitch_deg`, and
`uphill_confirmation_frames` in `imu_manager.yaml` for the vehicle and road.
The entry threshold exceeds five times the approximately 0.58-degree tilt
equivalent of the configured 0.10 m/s^2 startup acceleration-noise limit.

NaN/Inf, calibration failure, stale gyro/accel receipt, or any other invalid
IMU state immediately clears the state to false. `/imu_valid` remains the
separate diagnostic validity signal. Roll, Yaw, quaternion angle, combined
tilt, and camera mounting Pitch are never slope inputs. There is no five-second
stop timer, drive command, behavior-arbiter link, or mode dependency.
