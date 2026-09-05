# 고밀도 RTAB-Map 촬영·경로 저장·오프라인 확인

이 절차는 ROS 2 Jazzy, Domain 41, D456 `338122302896` 전용이다. RGB/Depth/IR
640×480 약 60 Hz와 IMU 약 400 Hz 설정은 변경하지 않는다. `high_density:=false`는
기존 RTAB-Map 5 Hz/0.05 m/0.03 rad/15 inlier 설정을 그대로 사용하고,
`high_density:=true`만 10 Hz/0.03 m/0.02 rad/15 inlier를 적용한다.

`/rtabmap/mapPath`는 지도 촬영 중 만들어진 최적화된 SLAM 궤적이다. 이것을 차량
추종 경로로 자동 사용하지 않는다. `/depth_slam/route/path`와 원본 `route.csv`는 READY
이후 명시적으로 기록한 차량 기준 경로이며, 실제 추종에는 최종 graph로 재정합한
`route_final.csv`만 사용한다.

다음 두 디렉터리는 영구 보호 대상이다. 내부 파일이 현재 없더라도 mapping 대상으로
쓸 수 없고 삭제 옵션도 적용되지 않는다.

- `/home/qor/depth_ws/maps/classroom_test`
- `/home/qor/depth_ws/maps/corridor_hand_test`

## 자동 날짜·시간 세션(권장)

`map_path`와 `session_id`를 생략하면 mapping launch가 시작되는 현지 시각을 한 번만
읽어 `map_YYYYMMDD_HHMMSS`를 만든다. 같은 초에 이미 예약된 이름이 있으면 `_01`,
`_02` 순으로 붙이며 기존 디렉터리나 파일을 삭제하지 않는다.

### 터미널 3 — 이름 지정 없이 고밀도 mapping

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam hybrid_mapping.launch.py \
  camera_serial:=338122302896 \
  auto_session_name:=true high_density:=true \
  mapping_mode:=true localization_mode:=false \
  record_bag:=false enable_control:=false dry_run:=true
```

콘솔의 `=== NEW MAPPING SESSION ===` 블록에서 모든 출력 경로와 timezone을 확인한다.
다른 터미널에서는 다음 transient-local 토픽으로 같은 값을 다시 확인할 수 있다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 topic echo /depth_slam/mapping/session_id --once \
  --qos-durability transient_local
ros2 topic echo /depth_slam/mapping/map_path --once \
  --qos-durability transient_local
ros2 topic echo /depth_slam/mapping/route_path --once \
  --qos-durability transient_local
ros2 topic echo /depth_slam/mapping/final_route_path --once \
  --qos-durability transient_local
ros2 topic echo /depth_slam/mapping/graph_path --once \
  --qos-durability transient_local
```

자동 세션의 route recorder는 경로 인수를 다시 계산하지 않고 위 토픽을 받는다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam route_recording.launch.py \
  high_density:=true enable_control:=false dry_run:=true
```

`record_bag:=true`이면 동일한 ID의 `bags/<session_id>/`를 사용한다. 세션 시작 시간과
경로는 `reports/<session_id>/session_metadata.yaml`에 ISO 8601 local/UTC 시간,
timezone, UTC offset, `automatic_name`과 함께 원자적으로 저장된다.

## 새 고밀도 지도 생성

모든 터미널의 공통 환경은 아래와 같다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
```

아래는 이름을 사용자가 직접 지정해야 할 때의 호환 절차다.

### 터미널 0 — 명시적 세션 이름과 기존 프로세스 확인

세션 이름은 모든 터미널에서 같은 값으로 설정한다. 존재하는 결과를 재사용하거나
덮어쓰지 않는다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
export SESSION_ID=high_density_$(date +%Y%m%d_%H%M%S)
for target in \
  "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db" \
  "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db-wal" \
  "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db-shm" \
  "/home/qor/depth_ws/maps/$SESSION_ID/checksums.sha256" \
  "/home/qor/depth_ws/routes/$SESSION_ID/route.csv" \
  "/home/qor/depth_ws/routes/$SESSION_ID/route_final.csv" \
  "/home/qor/depth_ws/routes/$SESSION_ID/rtabmap_graph.json" \
  "/home/qor/depth_ws/routes/$SESSION_ID/route_metadata.yaml" \
  "/home/qor/depth_ws/reports/$SESSION_ID/mapping_quality.json"; do
  if test -e "$target"; then
    echo "STOP: existing session output: $target" >&2
    return 1 2>/dev/null || exit 1
  fi
done
ros2 node list | sort
ros2 topic info /depth_slam/cuvslam/odometry -v || true
```

`/rtabmap/rtabmap`, `/localization_fusion`이 이미 있거나 cuVSLAM odometry publisher가
정확히 1개가 아니면 mapping launch가 중단된다.

### 터미널 1 — D456

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
/home/qor/depth_ws/scripts/run_d456_host.sh
```

### 터미널 2 — cuVSLAM과 포함된 bridge

bridge를 별도로 실행하지 않는다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
/home/qor/depth_ws/scripts/run_cuvslam_container.sh
```

### 터미널 3 — 새 DB 고밀도 mapping

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
export SESSION_ID=앞에서_정한_동일한_값
ros2 launch depth_hybrid_slam hybrid_mapping.launch.py \
  map_path:="/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db" \
  session_id:="$SESSION_ID" \
  mapping_mode:=true localization_mode:=false \
  high_density:=true start_monitor:=true record_bag:=false \
  enable_control:=false dry_run:=true delete_test_db:=false
```

선택적으로 디스크 용량을 확인한 후 `record_bag:=true`를 사용할 수 있다. 이 경우
explicit `camera_slam_with_aligned_depth` MCAP/zstd 프로파일이 별도 프로세스로
시작되며 bag 실패가 RTAB-Map 프로세스를 강제 종료하지 않는다.

### 터미널 4 — 저부하 RViz

mapping 중에는 기존 보수적 표시값(decimation 4, voxel 0.03 m)을 사용한다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam visualization.launch.py \
  start_rviz:=true start_image_view:=false enable_control:=false dry_run:=true
```

### 터미널 5 — 품질과 이동 준비 상태

quality monitor는 터미널 3에서 함께 시작되므로 중복 실행하지 않는다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 topic echo /depth_slam/mapping/state
ros2 topic echo /depth_slam/mapping/diagnostics
ros2 topic echo /depth_slam/mapping/recording_allowed
ros2 topic echo /depth_slam/mapping/recording_hold_reason
```

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /depth_slam/cuvslam/odometry
```

`READY` 전에는 이동 및 route 저장을 시작하지 않는다. blur/depth/과노출/저노출은
`/depth_slam/mapping/diagnostics`와
`reports/<session>/mapping_quality.json`에 raw 값과 통계로 남는다. 현장 정지/느린
이동 자료로 보정하기 전에는 blur/depth/과노출/저노출/inlier 임계값을 임의 적용하지
않으며 보고서에 `UNCALIBRATED` 항목으로 남고 최종 판정은 최대 `PARTIAL_PASS`다.
reset 증가 후 상태는 세션 끝까지 `INVALID_SESSION`이다.

### 터미널 6 — 차량 기준 경로 recorder 실행

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
export SESSION_ID=앞에서_정한_동일한_값
ros2 launch depth_hybrid_slam route_recording.launch.py \
  map_path:="/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db" \
  route_path:="/home/qor/depth_ws/routes/$SESSION_ID/route.csv" \
  session_id:="$SESSION_ID" \
  high_density:=true \
  route_min_distance_m:=0.05 route_min_angle_rad:=0.03 \
  enable_control:=false dry_run:=true
```

### 터미널 7 — 경로 저장 시작·상태·종료

READY와 map-frame pose가 확인된 후 시작한다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 service call /depth_slam/route/start std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/route/status std_srvs/srv/Trigger '{}'
```

주행 기준 경로 기록을 끝낼 때:

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 service call /depth_slam/route/stop std_srvs/srv/Trigger '{}'
```

추적 손실/reset 구간은 `valid=false`이고 좌표를 연결하거나 보간하지 않는다. 아직
연결되지 않은 speed/steering/encoder/wheel odom/IMU 자세 필드는 빈 문자열이다.
큰 map 보정, pose jump, 빠른 회전, 보정된 영상 품질 한계 위반도 해당 시각의 기록을
보류한다. 마지막 재방문과 loop closure가 끝나 `/rtabmap/mapGraph`가 갱신된 것을
확인한 뒤 route stop을 호출해야 최신 graph snapshot이 저장된다.

### 정상 종료와 최종 checksum

종료 순서는 route recorder → RViz → RTAB-Map mapping → cuVSLAM → D456이다. 각 실행
터미널에서 `Ctrl+C`를 한 번 누르고 종료를 기다린다. RTAB-Map 종료 뒤 `-wal`, `-shm`이
없을 때만 다음을 실행한다.

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
export SESSION_ID=앞에서_정한_동일한_값
test -f "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db"
test ! -e "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db-wal"
test ! -e "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db-shm"
ros2 run depth_hybrid_slam mapping_session_finalize \
  --map-path "/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db" \
  --route-path "/home/qor/depth_ws/routes/$SESSION_ID/route.csv" \
  --graph-path "/home/qor/depth_ws/routes/$SESSION_ID/rtabmap_graph.json" \
  --final-route-path "/home/qor/depth_ws/routes/$SESSION_ID/route_final.csv" \
  --metadata-path "/home/qor/depth_ws/routes/$SESSION_ID/route_metadata.yaml" \
  --session-id "$SESSION_ID" \
  --quality-report "/home/qor/depth_ws/reports/$SESSION_ID/mapping_quality.json"
(cd "/home/qor/depth_ws/maps/$SESSION_ID" && sha256sum -c checksums.sha256)
```

finalizer는 DB sidecar가 없는 완전 종료 상태에서 SQLite integrity를 검사하고,
node-relative 원본 점을 최종 optimized node pose로 재계산한 뒤 0.05 m로 재표본화한다.
비정상 jump/yaw 반전을 제거하고 route NaN/Inf, timestamp 단조 증가, 누적거리 단조
비감소를 확인한다. 원본은 덮어쓰지 않으며 기존 `route_final.csv`나 checksum이 있으면
실패한다.

## 카메라 없이 지도와 경로 확인

D456, cuVSLAM, mapping/localization 프로세스가 모두 종료된 상태에서 실행한다.
원본 DB는 기본적으로 `/tmp/depth_map_view.*`에 복사되며 종료 시 복사본만 제거된다.

### 터미널 1 — 임시 DB·지도·TF·경로·RViz 통합 실행

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
export SESSION_ID=확인할_세션
ros2 launch depth_hybrid_slam map_route_view.launch.py \
  map_path:="/home/qor/depth_ws/maps/$SESSION_ID/rtabmap.db" \
  route_path:="/home/qor/depth_ws/routes/$SESSION_ID/route_final.csv" \
  use_temporary_db_copy:=true publish_map_tf_for_view:=true \
  highest_density_view:=false start_rviz:=true \
  enable_control:=false dry_run:=true
```

이 launch 하나가 임시 DB 복사본 RTAB-Map, 오프라인 전용 TF, route CSV publisher,
RViz, 전체 지도 publish 요청을 함께 시작한다.

### 터미널 2 — 지도·그래프·두 경로 확인

```bash
set -eo pipefail
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 topic info /rtabmap/mapData -v
ros2 topic info /rtabmap/mapGraph -v
ros2 topic info /rtabmap/mapPath -v
ros2 topic echo /depth_slam/route/path --once --qos-durability transient_local
ros2 service call /rtabmap/rtabmap/publish_map \
  rtabmap_msgs/srv/PublishMap \
  '{global_map: true, optimized: true, graph_only: false}'
```

launch는 3초 뒤 설치본의 실제 서비스 `/rtabmap/rtabmap/publish_map`을 요청한다.
RViz Fixed Frame은 `map`이며 다음을
서로 다른 색으로 표시한다.

- Saved Map Cloud: `/rtabmap/mapData`, decimation 2, voxel 0.02 m
- Saved Map Graph: `/rtabmap/mapGraph`
- SLAM Capture Trajectory(초록): `/rtabmap/mapPath`
- Vehicle Reference Route(자홍): `/depth_slam/route/path`

최고밀도 오프라인 표시(decimation 1, voxel 0.01 m)는
`highest_density_view:=true`로 실행한다.

표시용 `map→base_link` static TF는 이 launch에서만 조건부로 게시된다. 실행 중인 실제
mapping/localization을 발견하면 중복 TF를 만들지 않고 launch가 중단된다.

## 촬영 주의

- 천천히 이동하고 급회전·렌즈 가림·케이블 장력을 피한다.
- 잠시 정지는 가능하지만 정지한 채 빠르게 방향을 바꾸지 않는다.
- 강한 햇빛과 먼 거리에서는 D456 depth valid ratio가 급락할 수 있다.
- 야외 진단은 depth valid ratio, 과노출/저노출, blur, RTAB inlier, tracking을 본다.
- GPS가 연결되지 않은 상태에서 GPS 값을 만들거나 자동 활성화하지 않는다.
- wheel odometry, steering, encoder, GPS는 실제 토픽과 타입을 확인한 뒤 bag YAML에만
  추가한다.
