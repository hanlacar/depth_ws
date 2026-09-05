# Mission decision validation

The decision node publishes `/camera/mission/drive_override*` and never owns
`/camera_drive`, `/camera_wheel`, or an MCU topic. In production the single
`camera_command_selector_node` applies these decisions to the final topics. The
production planner defaults remain `production`, `none`, `none` for planner,
line tracking, and road-boundary fallback respectively.

## Contracts confirmed from source and Domain 12

- The deployed route-mode contract is `/mcu/current_mode`
  (`std_msgs/msg/String`), already consumed by the reference-path adapter.
  `/camera/mission/section` (`std_msgs/msg/String`) remains a legacy/manual
  compatibility input; both numeric strings and legacy section names work.
- `imu_manager`'s `/vehicle_mode` is not reused: its known modes are `IDLE`,
  `NORMAL`, `PARALLEL_PARK`, `T_PARK`, and `SLOPE`, which are IMU operating
  modes rather than route sections.
- Drive stages are `STOP=0`, `SLOW=1`, `CRUISE=2`, `FAST=3`.
- `/camera_wheel` is `Int32`; `/camera/bev/diagnostics` contains the
  unquantized `required_steering_deg` used here.
- Direct-BEV geometric steering is positive LEFT. The decision adapter
  converts it exactly once to the final vehicle contract: positive RIGHT,
  negative LEFT, matching `/camera_wheel`.
- `/camera/bev/valid` and `/camera/bev/diagnostics.state` provide planner
  validity. `INVALID`, `INPUT_TIMEOUT`, and `CALIBRATION_INVALID` always
  reduce the hypothetical effective command to zero.

## Real D456 validation (motors disconnected)

Run every terminal with this prefix:

```bash
source /opt/ros/jazzy/setup.bash
source /home/qor/camera_ws/install/setup.bash
export ROS_DOMAIN_ID=12
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_LOCALHOST_ONLY
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

Terminal 1 — camera, unchanged YOLO, and production BEV inputs:

```bash
ros2 launch camera_navigation camera_bev_standalone.launch.py \
  planner_variant:=production line_track_mode:=none \
  road_boundary_fallback:=none debug:=false
```

Terminal 2 — calibrated IMU fusion:

```bash
ros2 launch imu_manager imu_manager.launch.py
```

Terminal 3 — advisory mission nodes and overlay:

```bash
ros2 launch camera_navigation camera_mission_validation.launch.py \
  debug_overlay:=true allow_offset_fallback:=false \
  front_axle_frame:=front_axle
```

Before stop-line ranging, the measured transform must succeed:

```bash
ros2 run tf2_ros tf2_echo front_axle camera_color_optical_frame
```

If it does not, stop the distance test and report `CALIBRATION_INVALID`. Do
not enable the 0.26 m fallback as a substitute for a measured transform.

Terminal 4 — select a section (change the data value per scenario):

```bash
ros2 topic pub --once /camera/mission/section std_msgs/msg/String \
  "{data: SLOPE}"
```

Terminal 5 — record once (the RGB/depth streams are not duplicated):

```bash
cd /home/qor/camera_ws
src/camera_navigation/tools/record_camera_mission_validation.sh \
  validation/mission_d456_$(date +%Y%m%d_%H%M%S)
```

Terminal 6 — inspect the advisory outputs and overlay:

```bash
ros2 run rqt_image_view rqt_image_view /camera/mission/overlay
ros2 topic echo /camera/mission/decision_diagnostics
```

For stop-line checks, measure front-axle-to-line distance at 0.5, 1.0, and
2.0 m and put bag timestamps and measurements into a copy of
`tools/mission_validation_measurements.csv`. Analyze after recording:

```bash
python3 /home/qor/camera_ws/src/camera_navigation/tools/analyze_camera_mission_bag.py \
  BAG_DIRECTORY --measurements-csv MEASUREMENTS.csv \
  --output mission_validation_report.json
```

The physical sequence is flat pose, `traffic20`, red, green, stop lines,
nose-up beyond 15 degrees, and downhill. Keep motor power off throughout.
Actual range accuracy, transition latency, pitch thresholds, and physical
three-second stopping remain NOT RUN until a D456 and measured TF are present.

Encoder-aware late-red parameters are present, but
`calibrated_deceleration_mps2` defaults to `0.0`. This intentionally disables
`LATE_RED_COMMIT_TO_CROSS` until worst-case braking and total latency have been
measured. The expected bridge topics are `/mcu/encoder` (Int32 signed cumulative
count), `/mcu/speed_mps` (Float32 m/s), `/mcu/distance_m` (Float32 absolute
cumulative metres), and `/mcu/speed_valid` (Bool), normally polled at 5 Hz.
In the deployed MCU YAML, bridge odometry is `/mcu/odom`; `/odom` is explicitly
owned by an external `wheel_odom` implementation that is not included in the
MCU repository. Therefore this workspace does not infer an MCU `/odom`
frequency or use `/odom` to reconstruct distance.

Reset at any time with:

```bash
ros2 service call /camera/mission/reset_decision std_srvs/srv/Trigger '{}'
```
