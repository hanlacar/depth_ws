# depth_ws 실차 실행

## 1. A 시작

### 대회장 외 — VSLAM OFF

```bash
cd ~/depth_ws
source setup_depth.sh

ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  start_mode:=1 \
  end_mode:=11 \
  enable_vslam:=false \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true \
  enable_rosbag:=true
```

`~/depth_ws`는 문서 예시입니다. 다른 위치나 사용자 계정에서도
workspace 루트의 `setup_depth.sh`를 source하면 됩니다.
