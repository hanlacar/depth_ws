# depth_ws 실행 방법

Intel RealSense D456, Isaac ROS cuVSLAM, RTAB-Map을 사용해 맵과 차량 주행 경로를 함께 저장하고, 저장한 맵에서 위치를 찾은 뒤 경로를 불러오는 ROS 2 Jazzy 워크스페이스입니다.

- 워크스페이스: `/home/qor/depth_ws`
- ROS 도메인: `41`
- RMW: `rmw_fastrtps_cpp`
- 차량 제어 기본값: `enable_control=false`, `dry_run=true`, `user_approved=false`
- 기본 상태에서는 실제 `/slam_drive`, `/slam_wheel` 제어를 활성화하지 않습니다.

## 1. 처음 설치 및 빌드

```bash
cd ~
git clone https://github.com/hanlacar/depth_ws.git
cd ~/depth_ws

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

colcon build --symlink-install
source ~/depth_ws/install/setup.bash
```

이미 저장소가 있으면 다음과 같이 최신 내용을 반영합니다.

```bash
cd ~/depth_ws
git pull origin main

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

colcon build --symlink-install
source ~/depth_ws/install/setup.bash
```

## 2. 공통 환경

모든 터미널에서 실행합니다.

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash

export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
```

## 3. 맵과 차량 경로 저장

### 터미널 1: D456 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

./scripts/run_d456_host.sh
```

### 터미널 2: cuVSLAM Docker 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

./scripts/run_cuvslam_container.sh
```

### 터미널 3: 새 매핑 세션 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

ros2 launch depth_hybrid_slam hybrid_mapping.launch.py \
  auto_session_name:=true \
  high_density:=true \
  enable_control:=false \
  dry_run:=true \
  start_rviz:=true
```

### 터미널 4: 경로 기록 노드 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

ros2 launch depth_hybrid_slam route_recording.launch.py \
  high_density:=true \
  enable_control:=false \
  dry_run:=true
```

### 터미널 5: 경로 기록 시작

차량을 움직이기 직전에 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 service call /depth_slam/route/start std_srvs/srv/Trigger '{}'
```

차량을 수동으로 천천히 주행합니다. 출발 지점 또는 특징이 충분한 기존 구간을 다시 통과하면 루프 클로저 검증에 유리합니다.

### 경로 기록 종료

차량을 완전히 정지한 뒤 실행합니다.

```bash
ros2 service call /depth_slam/route/stop std_srvs/srv/Trigger '{}'
```

이후 다음 순서로 종료합니다.

1. 경로 기록 터미널에서 `Ctrl+C`
2. 매핑 터미널에서 `Ctrl+C`
3. RTAB-Map과 관련 노드가 완전히 종료될 때까지 대기
4. cuVSLAM과 D456 종료

## 4. 저장 세션 확인 및 최종 경로 생성

가장 최근 세션 폴더를 확인합니다.

```bash
cd ~/depth_ws

ls -1dt maps/*/
ls -1dt routes/*/
ls -1dt reports/*/
```

방금 생성된 동일한 폴더명을 `SESSION_ID`에 입력합니다.

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

export SESSION_ID="실제_세션_ID"

ros2 run depth_hybrid_slam mapping_session_finalize \
  --map-path "$PWD/maps/$SESSION_ID/rtabmap.db" \
  --route-path "$PWD/routes/$SESSION_ID/route.csv" \
  --graph-path "$PWD/routes/$SESSION_ID/rtabmap_graph.json" \
  --final-route-path "$PWD/routes/$SESSION_ID/route_final.csv" \
  --metadata-path "$PWD/routes/$SESSION_ID/route_metadata.yaml" \
  --session-id "$SESSION_ID" \
  --quality-report "$PWD/reports/$SESSION_ID/mapping_quality.json"
```

최종 파일을 확인합니다.

```bash
ls -lh \
  "$PWD/maps/$SESSION_ID/rtabmap.db" \
  "$PWD/routes/$SESSION_ID/route.csv" \
  "$PWD/routes/$SESSION_ID/route_final.csv" \
  "$PWD/routes/$SESSION_ID/route_metadata.yaml" \
  "$PWD/reports/$SESSION_ID/mapping_quality.json"
```

주요 결과물:

- `maps/$SESSION_ID/rtabmap.db`: 저장된 RTAB-Map 지도
- `routes/$SESSION_ID/route.csv`: 주행 당시 원본 경로
- `routes/$SESSION_ID/route_final.csv`: 맵 최적화가 반영된 최종 경로
- `routes/$SESSION_ID/route_metadata.yaml`: 맵·경로 세션 및 checksum 정보
- `reports/$SESSION_ID/mapping_quality.json`: 매핑 품질 보고서

## 5. 저장한 맵과 경로 불러오기

먼저 터미널 1과 터미널 2의 D456 및 cuVSLAM을 실행합니다.

그다음 저장할 때 사용한 동일한 세션 ID를 설정합니다.

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

export SESSION_ID="실제_세션_ID"
```

### 터미널 3: 저장 맵 localization 실행

```bash
ros2 launch depth_hybrid_slam hybrid_localization.launch.py \
  case_id:="$SESSION_ID" \
  map_path:="$PWD/maps/$SESSION_ID/rtabmap.db" \
  route_path:="$PWD/routes/$SESSION_ID/route_final.csv"
```

차량을 움직이기 전에 localization 상태를 확인합니다.

```bash
ros2 topic echo /depth_slam/localization/state
```

`TRACKING` 또는 `RELOCALIZED` 상태가 안정적으로 유지되기 전에는 경로 추종을 시작하지 않습니다.

### 터미널 4: 저장 경로 dry-run 추종

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export SESSION_ID="실제_세션_ID"

ros2 launch depth_hybrid_slam route_follower_dry_run.launch.py \
  map_path:="$PWD/maps/$SESSION_ID/rtabmap.db" \
  route_path:="$PWD/routes/$SESSION_ID/route_final.csv"
```

상태와 계산된 명령을 확인합니다.

```bash
ros2 topic echo /depth_slam/route/controller_state
ros2 topic echo /depth_slam/route/cross_track_error
ros2 topic echo /depth_slam/dry_run/drive
ros2 topic echo /depth_slam/dry_run/wheel
ros2 topic echo /depth_slam/rejoin/state
```

## 6. 종료

다음 순서로 각 실행 터미널에서 한 번씩 `Ctrl+C`를 누릅니다.

1. 경로 추종
2. localization
3. cuVSLAM
4. D456

강제 종료하지 말고 각 노드가 정상적으로 종료될 때까지 기다립니다.
