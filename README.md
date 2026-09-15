# depth_ws 실차 실행

## 실행 방법

### 1. A 시작

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
# 대회장 외
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py start_branch:=A start_mode:=1 end_mode:=11 enable_vslam:=false front_serial_port:=/dev/ttyUSB0 enable_control:=true user_approved:=true enable_rosbag:=true
# 대회장
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py start_branch:=A start_mode:=1 end_mode:=11 enable_vslam:=true front_serial_port:=/dev/ttyUSB0 enable_control:=true user_approved:=true enable_rosbag:=true
```

### 2. B 시작

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
# 대회장 외
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py start_branch:=B start_mode:=1 end_mode:=11 enable_vslam:=false front_serial_port:=/dev/ttyUSB0 enable_control:=true user_approved:=true enable_rosbag:=true
# 대회장
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py start_branch:=B start_mode:=1 end_mode:=11 enable_vslam:=true front_serial_port:=/dev/ttyUSB0 enable_control:=true user_approved:=true enable_rosbag:=true
```

### 3. MCU/Arduino

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
ros2 launch t870_mcu_simple mcu.launch.py port:=auto
```

### 4. RViz2

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
rviz2 -d ~/depth_ws/install/depth_hybrid_slam/share/depth_hybrid_slam/config/prehardware_csv_front_lidar.rviz
```

### 5. 최종 명령 확인

터미널 A:

```bash
ros2 topic echo /cmd_drive
```

터미널 B:

```bash
ros2 topic echo /cmd_wheel
```

## 부분 주행

`start_mode:=N end_mode:=M`

- 5번만: `start_mode:=5 end_mode:=5`
- 3~7번: `start_mode:=3 end_mode:=7`
- 1~2번 A/VSLAM 미사용:

```bash
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py start_branch:=A start_mode:=1 end_mode:=2 enable_vslam:=false front_serial_port:=/dev/ttyUSB0 enable_control:=true user_approved:=true enable_rosbag:=true
```

## rosbag

- 위치: `~/depth_ws/rosbags/`
- 이름: `run_YYYYMMDD_HHMMSS`
- 최대: 완료된 주행 3개
- 새 bag 시작 시: 완료된 bag 중 가장 오래된 것을 자동 삭제하며 현재 기록은 삭제하지 않음

## 문제 확인 토픽

- 주행 위치: `/odom`
- 속도/조향: `/cmd_drive`, `/cmd_wheel`
- 실제 조향값: `/mcu/steer_deg`
- 현재 구간: `/drive_mode`, `/depth_slam/route/active_segment`
- 현재 경로 주체: `/depth_slam/path_owner`
- LiDAR 정지 원인: `/depth_slam/lidar/safety_event`
- Camera 판단: `/depth_slam/camera/csv_validation`, `/camera/traffic_light_fused/state`, `/camera/exit_branch_signal`
- VSLAM/센서 상태: `/depth_slam/localization/watchdog`, `/depth_slam/runtime/watchdog`
