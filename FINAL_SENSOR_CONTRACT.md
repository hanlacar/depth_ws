# Final sensor contract

이 문서는 `depth_ws` production/HIL launch와 실제 node source를 대조한 최종 계약이다. 센서가 없는 상태에서 source, launch parsing, build, unit/core test로 확인했으며 센서 rate와 실제 검출 성능은 측정하지 않았다.

## Production sensor inputs

| Sensor | Driver / owner | Native topic | Production topic | ROS type | Frame | Direct consumers |
|---|---|---|---|---|---|---|
| Front RPLIDAR A2M12 | `front_rplidar_node` (`rplidar_ros/rplidar_node`) | `scan` | `/front/scan` | `sensor_msgs/msg/LaserScan` | `front_laser` | `lidar_perception` |
| Rear RPLIDAR A2M12 | `rear_rplidar_node` (`rplidar_ros/rplidar_node`) | `scan` | `/rear/scan` | `sensor_msgs/msg/LaserScan` | `rear_laser` | `lidar_perception` (Mode 7/10 gate only) |
| D456 RGB | `realsense2_camera_node` | `/camera/camera/color/image_raw` | `/camera/image_raw` | `sensor_msgs/msg/Image` | `camera_color_optical_frame` | YOLO-Seg, RGB traffic-light detector; camera mission debug when enabled |
| D456 camera info | `realsense2_camera_node` | `/camera/camera/color/camera_info` | `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | `camera_color_optical_frame` | YOLO-Seg, camera mission, CSV road validator |
| D456 aligned depth | `realsense2_camera_node` | `/camera/camera/aligned_depth_to_color/image_raw` | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | `camera_color_optical_frame` | camera mission, opt-in yellow-line depth helper |
| D456 gyro | `realsense2_camera_node` | same as production | `/camera/camera/gyro/sample` | `sensor_msgs/msg/Imu` | `camera_gyro_optical_frame` | `imu_manager` |
| D456 accel | `realsense2_camera_node` | same as production | `/camera/camera/accel/sample` | `sensor_msgs/msg/Imu` | `camera_accel_optical_frame` | `imu_manager` |
| Vehicle ODOM | external real vehicle odometry owner | `/odom` | `/odom` | `nav_msgs/msg/Odometry` | `odom` → `base_link` | `odom_localization`, LiDAR perception, maneuver manager |

`dual_rplidar.launch.py` is the only place that maps driver-native `scan` to the two public LiDAR names. Production consumers do not subscribe to `/scan`, `/scan_front`, `/scan_rear`, or `/front_lidar/scan`.

The rear scan is subscribed continuously but its safety assessment, parking-slot output, STOP, and maneuver influence are enabled only when `mode_gates()` returns rear-active for Mode 7 or Mode 10. In every other mode its published rear emergency value is false and its slot evidence is inactive.

## Derived camera and IMU topics

| Function | Publisher | Topic | Type | Consumer |
|---|---|---|---|---|
| Annotated detector image | YOLO-Seg | `/camera/debug/annotated` | `sensor_msgs/msg/Image` | rqt/debug capture |
| Refined road | YOLO-Seg | `/perception/refined/road` | `sensor_msgs/msg/Image` | debug/validation; compact semantic frame is the control input |
| Refined white line | YOLO-Seg | `/perception/refined/white_line` | `sensor_msgs/msg/Image` | debug/validation |
| Refined yellow line | YOLO-Seg | `/perception/refined/yellow_line` | `sensor_msgs/msg/Image` | debug/validation |
| Refined unknown line | YOLO-Seg | `/perception/refined/unknown_line` | `sensor_msgs/msg/Image` | debug/validation |
| Compact perception frame | YOLO-Seg | `/perception/semantic_path_frame` | `race_interfaces/msg/SemanticPathFrame` | camera mission, CSV road validator |
| Fused traffic state | traffic-light fusion | `/camera/traffic_light_fused/state` | `std_msgs/msg/String` | mission manager, signal exit |
| Fused traffic aspect | traffic-light fusion | `/camera/traffic_light_fused/aspect` | `std_msgs/msg/String` | mission manager, signal exit |
| Stop-line detected | camera mission | `/camera/mission/stop_line_detected` | `std_msgs/msg/Bool` | mission manager |
| Stop-line distance | camera mission | `/camera/mission/stop_line_distance_m` | `std_msgs/msg/Float32` | mission manager |
| CSV/camera validation | CSV road validator | `/depth_slam/camera/csv_validation` | `std_msgs/msg/String` | LiDAR perception/diagnostics |
| Mode 11 exit branch | signal exit | `/camera/exit_branch_signal` | `std_msgs/msg/String` | maneuver manager |
| Filtered IMU | `imu_manager` | `/imu/data` | `sensor_msgs/msg/Imu` | camera mission |
| Vehicle pitch | `imu_manager` | `/imu/pitch_deg` | `std_msgs/msg/Float32` | mission manager Mode 2 criterion |
| IMU validity | `imu_manager` | `/imu/valid` | `std_msgs/msg/Bool` | camera mission, mission manager |

Mode 2 completion uses calibrated, nose-up-positive `/imu/pitch_deg >= +5.0` while `/imu/valid` is true. No duplicate pitch adapter is in the production graph.

## ODOM ownership

Production does not create synthetic motion and does not auto-start a test ODOM source. Exactly one external real owner must publish `/odom` with `header.frame_id=odom`, `child_frame_id=base_link`, and the matching `odom -> base_link` TF. `odom_localization` transforms that stream through the available `map -> odom` TF and publishes `/depth_slam/localization/pose` for the route follower; LiDAR/rejoin nodes also consume canonical `/odom` directly.

`test_odom_publisher` is the sole exception and is **TEST ONLY, before real vehicle integration**. It requires `test_only_acknowledged:=true`, consumes actual `/mcu/encoder` and `/mcu/steer_a0`, and publishes only when both measurements are fresh. `/cmd_drive` supplies direction only; it cannot create distance. Never run it with a real `/odom` owner.

## Control and data flow

- Camera: D456 RGB/depth/IMU → YOLO and RGB detector → traffic/road/lane and camera mission → mission gates → command arbiter.
- Front LiDAR: real A2M12 → `/front/scan` → LiDAR perception/ROI/safety → maneuver manager → command arbiter.
- Rear LiDAR: real A2M12 → `/rear/scan` → Mode 7/10 parking and reverse safety only.
- ODOM: real owner, or explicitly isolated TEST ONLY owner → `/odom` and `odom -> base_link` → localization pose plus LiDAR/rejoin → route follower.
- Route: canonical CSV → route follower candidate topics → command arbiter → `/cmd_drive`, `/cmd_wheel`.
- Mission: drive mode and real perception → holds, branches, completion and safety decisions → command arbiter.

Only `command_arbiter` publishes final `/cmd_drive` (`std_msgs/msg/Float32`) and `/cmd_wheel` (`std_msgs/msg/Int32`). Candidate command topics remain internal and cannot directly own vehicle commands.

## Frames and protected geometry

- `base_link -> front_laser`: x `0.730`, y `0.0`, z `0.105`, yaw `0.0` rad.
- `base_link -> rear_laser`: x `-0.680`, y `0.0`, z `0.155`, yaw `pi` rad.
- General obstacle ROI is evaluated in `front_laser`; the CSV swept footprint is evaluated in `base_link`.
- Wheelbase is `0.73 m`; planner limit is `21 deg`; physical/hard limit is `±22 deg`.

## Fail-closed production prerequisite

The canonical CSV hash is `e308f6e8be749d6372b0814ad105fba78058d9419c4c1874efac42c84bd03160`. Its checked-in metadata still records `alignment.validated: false`. Therefore the production route follower deliberately refuses control until a real map/route alignment validation updates that evidence. The Mode 4→5 and Mode 8→9 HIL launches use the explicit prehardware alignment override and TEST ONLY ODOM; that override is not enabled in production.
