# depth_ws 실차 실행

## 1. 대회장 — A 경로 전체 주행 / VSLAM 사용

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh

ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  start_mode:=1 \
  end_mode:=11 \
  enable_vslam:=true \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true \
  enable_rosbag:=true
```

## 2. 대회장 외 — A 경로 1번부터 2번까지 / VSLAM 미사용

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh

ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  start_mode:=1 \
  end_mode:=2 \
  enable_vslam:=false \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true \
  enable_rosbag:=true
```

## 3. MCU / Arduino 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh

ros2 launch t870_mcu_simple mcu.launch.py port:=auto
```

## 4. 속도 확인

새 터미널:

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh

ros2 topic echo /cmd_drive
```

## 5. 조향각 확인

새 터미널:

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh

ros2 topic echo /cmd_wheel
```

## rosbag

* 위치: `~/depth_ws/rosbags/`
* 이름: `run_YYYYMMDD_HHMMSS`
* 최대: 완료된 주행 3개
* 새 bag 시작 시: 완료된 bag 중 가장 오래된 것을 자동 삭제하며 현재 기록은 삭제하지 않음
