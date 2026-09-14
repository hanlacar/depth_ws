# depth_ws production guide

ROS 2 Jazzy 기반의 실제 Intel RealSense D456, 전·후방 SLAMTEC RPLIDAR A2M12, 외부 차량 ODOM, CSV 경로 및 Mode 1~11 주행 스택이다. Gazebo와 Camera/LiDAR/IMU/vehicle 가짜 ROS runtime은 포함하지 않는다. 유일한 예외는 바퀴를 띄운 HIL을 위한 TEST ONLY ODOM이다.

확정된 전체 토픽·frame·publisher/consumer 계약은 [FINAL_SENSOR_CONTRACT.md](FINAL_SENSOR_CONTRACT.md)를 기준으로 한다.

## 환경과 빌드

모든 터미널에서 다음을 실행한다.

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
```

처음 빌드하거나 소스가 바뀐 경우:

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
(
  export PYTHONPATH=/usr/lib/python3/dist-packages:${PYTHONPATH}
  colcon build --symlink-install
)
source install/setup.bash
```

테스트는 사용자 Python package를 사용할 수 있어야 하므로 build용 `PYTHONPATH` 변경을 현재 shell에 남기지 않는다.

```bash
colcon test
colcon test-result --verbose
python3 tools/audit_depth_ws_self_contained.py
git diff --check
```

`tools/ros_network_env.sh`는 기본 route interface를 자동 선택하고 Docker bridge를 제외한다. 같은 subnet의 원격 RViz는 각 장치에서 이 파일을 source한 뒤 같은 `ROS_DOMAIN_ID`를 사용한다. 필요한 경우 source 전에 `ROS_STATIC_PEER=<IP>`를 지정한다.

## 실제 센서 실행

### D456 production stack

카메라 driver와 IMU filter, YOLO, RGB 신호등, camera mission을 함께 실행한다.

```bash
ros2 launch camera_bringup d456_production.launch.py \
  launch_camera:=true serial_no:=338122302896
```

D456 공개 토픽을 확인한다.

```bash
ros2 topic info -v /camera/image_raw
ros2 topic info -v /camera/camera_info
ros2 topic info -v /camera/aligned_depth_to_color/image_raw
ros2 topic info -v /camera/camera/gyro/sample
ros2 topic info -v /camera/camera/accel/sample
ros2 topic info -v /imu/pitch_deg
```

RGB는 driver-native `/camera/camera/color/image_raw`에서 `/camera/image_raw`으로 remap된다. aligned depth는 `/camera/aligned_depth_to_color/image_raw`이다.

### Front A2M12 하나

기본 포트는 `/dev/ttyUSB0`이며 공개 출력은 반드시 `/front/scan`이다.

```bash
ros2 launch depth_hybrid_slam dual_rplidar.launch.py \
  launch_lidar_drivers:=true \
  launch_rear_lidar_driver:=false \
  front_serial_port:=/dev/ttyUSB0
```

```bash
ros2 topic info -v /front/scan
ros2 topic hz /front/scan
ros2 run tf2_ros tf2_echo base_link front_laser
```

### Front + Rear A2M12

후방 기본 포트는 `/dev/ttyUSB1`이며 공개 출력은 `/rear/scan`이다. Rear LiDAR 판단은 Mode 7/10에서만 활성화된다.

```bash
ros2 launch depth_hybrid_slam dual_rplidar.launch.py \
  launch_lidar_drivers:=true \
  launch_rear_lidar_driver:=true \
  front_serial_port:=/dev/ttyUSB0 \
  rear_serial_port:=/dev/ttyUSB1
```

```bash
ros2 topic info -v /front/scan
ros2 topic info -v /rear/scan
ros2 topic hz /rear/scan
ros2 run tf2_ros tf2_echo base_link rear_laser
```

Driver-native 출력과 과거의 legacy aliases는 production 공개 토픽이 아니다.

## 시각화와 상태 확인

Camera annotated image:

```bash
ros2 run rqt_image_view rqt_image_view /camera/debug/annotated
```

LiDAR ROI RViz:

```bash
rviz2 -d ~/depth_ws/install/depth_hybrid_slam/share/depth_hybrid_slam/config/lidar_roi_debug.rviz
```

저장 지도와 canonical route RViz:

```bash
ros2 launch depth_hybrid_slam map_route_view.launch.py \
  map_path:=~/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  route_path:=~/depth_ws/routes/network/route_network_segmented_stop_edited_vforward.csv \
  start_rviz:=true
```

핵심 perception/mission 출력을 확인한다.

```bash
ros2 topic echo /camera/traffic_light_fused/state
ros2 topic echo /camera/traffic_light_fused/aspect
ros2 topic echo /camera/mission/stop_line_detected
ros2 topic echo /camera/mission/stop_line_distance_m
ros2 topic echo /depth_slam/camera/csv_validation
ros2 topic echo /depth_slam/lidar/perception
```

## TEST ONLY ODOM

이 publisher는 **실차 ODOM 연결 전, 차량을 공중에 띄운 HIL 전용**이다. 실제 `/mcu/encoder`와 `/mcu/steer_a0`가 모두 fresh일 때만 `/odom` 및 `odom -> base_link`를 내보내며 임의 motion은 만들지 않는다. 실제 ODOM owner와 절대 동시에 실행하지 않는다.

```bash
ros2 run depth_hybrid_slam test_odom_publisher --ros-args \
  -p test_only_acknowledged:=true
```

```bash
ros2 topic info -v /odom
ros2 topic echo /depth_slam/test_odom/state
```

## 실제 센서 HIL

아래 launch는 실제 D456/A2M12와 TEST ONLY ODOM을 사용한다. 최종 `/cmd_drive`, `/cmd_wheel`도 발행하므로 차량을 공중에 띄우거나 구동계를 분리한 상태에서만 실행한다.

Mode 4 → 5:

```bash
ros2 launch depth_hybrid_slam prehardware_csv_camera_lidar.launch.py \
  start_mode:=4 end_mode:=5 \
  use_rear_lidar:=false start_rviz:=true
```

Mode 8 → 9:

```bash
ros2 launch depth_hybrid_slam prehardware_csv_camera_lidar.launch.py \
  start_mode:=8 end_mode:=9 \
  use_rear_lidar:=false start_rviz:=true
```

Mode 11 real-camera HIL:

```bash
ros2 launch depth_hybrid_slam prehardware_csv_camera_lidar.launch.py \
  start_mode:=11 end_mode:=11 \
  use_rear_lidar:=false start_rviz:=true
```

Mode 11의 pure-core 회귀만 실행하려면:

```bash
pytest -q src/depth_hybrid_slam/test/test_signal_exit_integration.py
```

## Full route production

Production launch는 실제 D456과 전·후방 A2M12를 시작하지만 ODOM은 만들지 않는다. 먼저 차량의 유일한 production ODOM owner가 `/odom`과 `odom -> base_link`를 발행해야 한다.

현재 canonical metadata는 `alignment.validated: false`이므로 production route control은 의도적으로 fail-closed된다. 실제 지도-경로 정합 검증 없이 이 값을 바꾸거나 HIL override를 production에 사용하지 않는다.

센서/ODOM/경로 상태만 확인하는 안전 실행:

```bash
ros2 launch depth_hybrid_slam production_ready.launch.py \
  enable_control:=false dry_run:=true user_approved:=false
```

실제 정합 검증이 완료되고 차량을 주행시킬 때만 다음 세 플래그를 함께 변경한다.

```bash
ros2 launch depth_hybrid_slam production_ready.launch.py \
  enable_control:=true dry_run:=false user_approved:=true
```

## Command ownership 확인

최종 차량 명령의 유일한 owner는 `depth_command_arbiter`다.

```bash
ros2 topic echo /cmd_drive
ros2 topic echo /cmd_wheel
ros2 topic info -v /cmd_drive
ros2 topic info -v /cmd_wheel
```

각 `ros2 topic info -v` 결과에서 publisher count를 확인한다.

- `/front/scan`: `front_rplidar_node` 1개
- `/rear/scan`: `rear_rplidar_node` 1개(dual 실행 시)
- `/camera/image_raw`: RealSense driver 1개
- `/odom`: real owner 1개 또는 TEST ONLY 환경의 test owner 1개
- `/cmd_drive`: `depth_command_arbiter` 1개
- `/cmd_wheel`: `depth_command_arbiter` 1개

## 실제 차량 실행 순서

1. 차량을 정지시키고 E-stop을 준비한다.
2. Front/Rear serial port와 D456 serial을 확인한다.
3. production ODOM owner 하나만 시작하고 `/odom`, `odom -> base_link`를 확인한다.
4. `enable_control:=false dry_run:=true user_approved:=false`로 production launch를 시작한다.
5. `/front/scan`, `/rear/scan`, D456, `/imu/valid`, localization tracking, route binding, mission 및 safety 상태를 확인한다.
6. `/cmd_drive`, `/cmd_wheel` publisher가 arbiter 하나뿐인지 확인한다.
7. canonical map-route alignment가 실제 검증된 경우에만 세 control flag를 활성화한다.
8. Mode 11 종료 후 `/camera/exit_branch_signal`과 최종 정지 상태를 확인한다.

## 보호된 production 기준

- Mode 1~11과 16개 branch case를 유지한다.
- Mode 2는 `/imu/pitch_deg >= +5.0`과 valid IMU를 사용한다.
- Mode 5 planner limit은 `21 deg`, hard steering limit은 `±22 deg`, wheelbase는 `0.73 m`다.
- Mode 9 emergency는 장애물 제거 뒤 1초 연속 clear 후에만 해제된다.
- Mode 11은 GREEN→A, RED→B, UNKNOWN→default A, 5초 STOP이며 늦은 반대 신호를 무시한다.
- Canonical CSV SHA-256은 `e308f6e8be749d6372b0814ad105fba78058d9419c4c1874efac42c84bd03160`이다.
