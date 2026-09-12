# VSLAM 지도 + 2D RGB 인식 교차검증

이 구성은 `/home/qor/depth_ws`가 D456, cuVSLAM, RTAB-Map을 소유하고
`depth_ws/src`에 포함된 YOLO/RGB/mission perception을 사용한다. 통합 launch는
RealSense, controller, MCU, Arduino, motor, Gazebo를 시작하지 않는다.

## 원본 감사 결과

### camera_ws

| 기능 | 기존 구현 및 출력 |
|---|---|
| CUDA YOLO-Seg | `camera_yolo_inference_node`; `/perception/detections_json` (`std_msgs/String`), `/perception/semantic_path_frame` (`race_interfaces/SemanticPathFrame`), `/camera/perception_overlay_image` (`sensor_msgs/Image`) |
| RGB 신호등 | `rgb_traffic_light_node`; state/aspect/diagnostics (`std_msgs/String`), confidence (`std_msgs/Float32`), overlay (`sensor_msgs/Image`) |
| YOLO+RGB 융합 | `traffic_light_fusion_node`; `/camera/traffic_light_fused/{state,aspect,confidence,diagnostics}` |
| 정지선 거리 | `camera_mission_perception_node`; 정지선 RLE mask와 aligned depth를 같은 stamp로 결합하고 `base_link`로 변환한 뒤 `front_axle_x=0.365 m`를 빼서 거리 계산 |
| mission overlay | `/camera/mission/debug_overlay` (`sensor_msgs/Image`) |
| 장착값 | `camera_mount.yaml`: x=0.015 m, z=0.970 m, 물리 pitch=-5°; T870 REP-103 TF의 pitch=+0.0872665 rad와 같은 하향 자세 |
| RealSense TF | camera driver가 `camera_link→stream optical frame`만 발행; 차량 mount와 odometry TF는 발행하지 않음 |

### depth_ws

| 기능 | 기존 구현 및 출력 |
|---|---|
| D456 소유자 | `scripts/run_d456_host.sh`; serial `338122302896`, native `/camera/camera/*` topics |
| cuVSLAM | container의 `/visual_slam/tracking/odometry`; bridge의 `/depth_slam/cuvslam/odometry`, tracking/reset/FPS |
| RTAB-Map | `/rtabmap/mapData`, `/rtabmap/mapGraph`, `/rtabmap/mapPath`, `/rtabmap/info`, `/rtabmap/localization_pose` |
| localization | `/depth_slam/localization/{pose,odometry,state,confidence}` |
| TF 소유권 | RTAB-Map `map→odom`, cuVSLAM `odom→base_link`, depth launch `base_link→camera_link`, RealSense 내부 optical TF |
| RViz | RTAB MapCloud, MapGraph, optimized path, TF, current pose |
| rosbag | `competition_bag_topics.yaml`; raw camera/cuVSLAM/SLAM 문맥과 선택적 aligned depth/MapData |

T870 차량 기준은 wheelbase 0.730 m, `base_link`는 차축 높이의 네 바퀴 중심,
`base_link→camera_link=(0.015, 0, 0.835, pitch=+0.0872665 rad)`이다.
ground-plane projection의 0.970 m는 지면 기준 광학 중심 높이이며
`0.835 + wheel_radius 0.135`와 일치한다.

## 교차검증 계약

신호등 최종 결과는 다음 조건이 모두 참인 관측에서만
`/depth_slam/perception/traffic_light/result`로 발행된다.

1. RGB publisher가 정확히 하나다.
2. YOLO와 RGB detector가 같은 상태에 합의한다.
3. 두 bbox가 일치하고 source timestamp 차이가 0.08초 이하다.
4. 같은 timestamp의 aligned depth bbox에 충분하고 안정적인 depth가 있다.
5. 그 timestamp의 optical-frame 점을 `map`으로 변환할 수 있다.
6. cuVSLAM tracking이 유효하고 mapping `READY` 또는 localization
   `TRACKING/RELOCALIZED` 상태다.
7. 융합 confidence가 0.60 이상이다.

정지선도 기존 2D mask/depth/front-axle 결과와 같은 timestamp의 `map←base_link` TF가
모두 유효할 때만 `/depth_slam/perception/stop_line/result`로 발행된다. 실패 중에는
`validated=false`와 원인이 status/overlay/RViz의 노란 marker로 표시된다. 신호등
R/G 상태는 2D overlay와 휘발성 result topic에만 존재하며 RTAB-Map DB에 기록하지 않는다.

## 실행

모든 터미널은 다음 환경을 사용한다.

```bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
```

### 터미널 1 — 유일한 D456

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./scripts/run_d456_host.sh
```

### 터미널 2 — cuVSLAM + bridge

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
./scripts/run_cuvslam_container.sh
```

### 터미널 3A — 새 지도 mapping

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch depth_hybrid_slam hybrid_mapping.launch.py \
  camera_serial:=338122302896 auto_session_name:=true high_density:=true \
  mapping_mode:=true localization_mode:=false start_rviz:=false \
  enable_control:=false dry_run:=true
```

저장 지도를 읽기 전용으로 표시하며 위치추정할 때는 3A 대신 다음을 사용한다.

```bash
ros2 launch depth_hybrid_slam hybrid_localization.launch.py \
  map_path:=/home/qor/depth_ws/maps/SESSION/rtabmap.db \
  mapping_mode:=false localization_mode:=true start_rviz:=false \
  enable_control:=false dry_run:=true
```

### 터미널 4 — 기존 인식 + 교차검증 + 두 화면

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam perception_vslam_view.launch.py \
  graph_guard:=true require_cuda:=true \
  start_rviz:=true start_image_view:=true
```

이 launch에는 카메라 노드가 없다. graph guard는 시작 전에 유일한 D456 RGB,
cuVSLAM/RTAB-Map publisher와 모든 `/mcu/cmd_*` 및 금지 drive/wheel publisher를
확인하고 조건이 어긋나면 전체 통합 시작을 거부한다.

### 상태 확인

```bash
ros2 topic echo /depth_slam/perception/cross_validation_status
ros2 topic echo /depth_slam/perception/traffic_light/validated
ros2 topic echo /depth_slam/perception/traffic_light/result
ros2 topic echo /depth_slam/perception/stop_line/validated
ros2 topic echo /depth_slam/perception/stop_line/result
ros2 topic info /camera/camera/color/image_raw -v
ros2 topic info /depth_slam/perception/markers -v
```

2D 창은 `/depth_slam/perception/debug_overlay`를 표시한다. RViz는
`/rtabmap/mapData`, `/rtabmap/mapGraph`, `/rtabmap/mapPath`, 차량 live path와 현재 위치,
신호등·정지선의 휘발성 marker를 동시에 표시한다.

종료 순서는 2D/RViz 통합 launch → RTAB-Map → cuVSLAM → D456이다.
