# depth_ws 실차 실행

## 1. A 시작

### 대회장 외 — VSLAM OFF

```bash
cd ~/depth_ws
source setup_depth.sh

cd ~/depth_ws
source setup_depth.sh
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  start_mode:=7 \
  end_mode:=7 \
  enable_vslam:=false \
  enable_parking_slam:=false \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true \
  enable_rosbag:=true
```

`~/depth_ws`는 문서 예시입니다. 다른 위치나 사용자 계정에서도
workspace 루트의 `setup_depth.sh`를 source하면 됩니다.


## 1. A 시작

### 대회장  — VSLAM ON

```bash
cd ~/depth_ws
source setup_depth.sh

ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  start_mode:=7 \
  end_mode:=7 \
  enable_vslam:=true \
  enable_parking_slam:=true \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true \
  enable_rosbag:=true
```


## 2. MCU 시작


```bash
cd ~/depth_ws
source setup_depth.sh

ros2 launch t870_mcu_simple mcu.launch.py port:=auto
```



## 3. RVIZ2 시작


```bash
cd ~/depth_ws
source setup_depth.sh

rviz2 -d ~/depth_ws/install/depth_hybrid_slam/share/depth_hybrid_slam/config/prehardware_csv_front_lidar.rviz
```


## 4. cmd_drive


```bash
cd ~/depth_ws
source setup_depth.sh

ros2 topic echo /cmd_drive
```


## 5. cmd_wheel


```bash
cd ~/depth_ws
source setup_depth.sh

ros2 topic echo /cmd_wheel
```
