# depth_hybrid_slam

고밀도 새 지도 촬영, 명시적 map-frame route 저장, 원본 DB를 복사해서 사용하는
카메라 없는 지도+경로 RViz 절차는
[`docs/classroom_mapping_manual.md`](docs/classroom_mapping_manual.md)를 따른다.

Competition MCAP recording, timestamp audit, and hardware-free replay are
documented in [`docs/competition_bag_manual.md`](docs/competition_bag_manual.md).

D456을 직접 들고 교실 지도를 수동 생성하는 터미널별 절차는
[`docs/classroom_mapping_manual.md`](docs/classroom_mapping_manual.md)에 정리되어 있다.

Isaac ROS cuVSLAM 4.6가 D456 IR stereo+IMU로 `odom→base_link`를 만들고,
RTAB-Map 0.22.1이 RGB-D를 낮은 주기로 처리하여 loop closure와
`map→odom`을 담당한다. `base_link→camera_link`는 고정 장착값
`(0.015, 0, 0.970 m, REP-103 pitch=+5°)` 하나만 발행한다. RTAB-Map의 별도
RGB-D odometry는 항상 꺼져 있다.

## 지원 경계와 요구사항

- 검증 환경: Ubuntu 24.04, ROS 2 Jazzy, x86_64, RTX 5060 Laptop,
  driver 595.84, CUDA 13.2 runtime, Docker 29.8, NVIDIA Container Toolkit 1.19.1.
- 설치 image: `depth/isaac_ros-cuvslam:4.6-d456`.
- Isaac ROS 4.6의 공식 RealSense 예시는 D455/D435i를 열거한다. D456은
  공식 예제 명시 대상이 아니므로 이 저장소의 결과는 실제 D456 호환성 시험 결과다.
- 모든 터미널은 `ROS_DOMAIN_ID=41`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`를 사용한다.
- GPU/Isaac ROS 기준은 [Isaac ROS 4.6 시작 문서](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html)와
  [Visual SLAM 문서](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html)를 따른다.

## 빌드와 실행

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
PYTHONNOUSERSITE=1 colcon build --base-paths src \
  --packages-select depth_hybrid_slam --symlink-install
source install/setup.bash
```

터미널 1에서 호스트 RealSense, 터미널 2에서 GPU 컨테이너를 시작한다.
USB 장치는 컨테이너에 직접 넘기지 않는다. host DDS와 container가 같은 uid,
host network/IPC/PID/hostname/machine-id를 사용해야 Fast DDS shared memory가 동작한다.

```bash
./scripts/run_d456_host.sh
./scripts/run_cuvslam_container.sh
```

기본 매핑은 시작 현지 시각으로 `map_YYYYMMDD_HHMMSS` 세션을 원자적으로 예약한다.
기존 DB를 자동 삭제하는 옵션은 없다.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_hybrid_slam hybrid_mapping.launch.py \
  auto_session_name:=true high_density:=true \
  enable_control:=false dry_run:=true start_rviz:=false
```

명시적 `map_path` 또는 `session_id`는 자동 이름보다 우선한다. 결정된 ID와 지도/경로
경로는 `/depth_slam/mapping/session_id`, `map_path`, `route_path`,
`final_route_path`, `graph_path` transient-local
토픽 및 `reports/<session_id>/session_metadata.yaml`에서 확인한다.

기존 case localization은 checksum 검증 뒤 `bwrap`의 read-only bind로 DB 디렉터리를
보호한 상태에서 연다. 설치된 RTAB-Map 0.22.1에는 `Mem/LocalizationReadOnly`가 없으므로
존재하지 않는 파라미터에 의존하지 않는다.

```bash
ros2 launch depth_hybrid_slam hybrid_localization.launch.py \
  case_id:=case_1 map_path:=$PWD/maps/case_1/rtabmap.db \
  route_path:=$PWD/maps/case_1/route.csv
```

`map→odom` correction은 smoothing된다. 0.5 m 또는 20°를 넘는 보정은
정지/안전 승인 없이 보류된다. 저장 지도 match, 최소 15 inlier, likelihood,
5회 연속 correction 일관성이 확인되기 전 상태는 READY가 아니다.

## 4-case와 경로

case 의미와 실제 슬롯은 여기서 만들지 않는다. 빈 디렉터리 준비와 완성된
지도/경로 봉인은 다음과 같다.

```bash
ros2 run depth_hybrid_slam case_manager init --root maps
ros2 run depth_hybrid_slam case_manager seal case_1 --root maps \
  --db /path/new.db --route /path/route.csv \
  --metadata src/depth_hybrid_slam/config/case_metadata_template.yaml
ros2 run depth_hybrid_slam case_manager verify case_1 --root maps
```

기록은 `map` pose 변화와 상태 변화를 기준으로 선별하고 각 행을
`fsync`한다. 비정상 종료 때 `.partial`이 남아 복구할 수 있고 정상 종료 때만
atomic rename한다. `route.csv`는 당시 pose와 RTAB-Map node-relative pose를 가진
원본이고, RTAB-Map 종료 뒤 finalizer가 최종 graph로 `route_final.csv`를 별도 생성한다.

```bash
ros2 launch depth_hybrid_slam route_recording.launch.py \
  route_path:=$PWD/maps/session_route.csv
ros2 launch depth_hybrid_slam route_follower_dry_run.launch.py \
  map_path:=$PWD/maps/case_1/rtabmap.db \
  route_path:=$PWD/maps/case_1/route.csv
```

dry-run 출력은 `/depth_slam/dry_run/drive`, `/depth_slam/dry_run/wheel`이다.
기본 실행은 실제 `/slam_drive`, `/slam_wheel`을 발행하지 않는다. 실제 출력에는
`enable_control=true`, `dry_run=false`, localization/route/safety READY, 그리고
별도의 `user_approved=true`가 모두 필요하다. 기본값은 계속 `false`이며 map/route
실제 SHA256 검증 실패 시 승인값과 관계없이 주행하지 않는다.

T870 제원은 `config/vehicle_navigation.yaml`의 `wheelbase_m=0.73`,
`max_steering_deg=22.0`, `minimum_turning_radius_m=1.8068134`로 전달된다. 1 m 초과
이탈 시 정지→재진입 계획→저속 추종→합류 검증 상태 머신을 사용한다. 새 토픽과
안전 계약은 [`docs/rejoin_topics.md`](docs/rejoin_topics.md)에 정리했다.

## 미션과 화면

`camera_ws`의 fused light, stop line, sign, uphill, section과 MCU steering만
구독한다. detector를 복제하지 않는다. 실제 ramp/acceleration/finish section ID는
`config/mission.yaml`이 비어 있으므로 코스 정보를 받은 뒤 입력해야 한다.

```bash
ros2 launch depth_hybrid_slam mission_dry_run.launch.py
ros2 launch depth_hybrid_slam visualization.launch.py start_rviz:=true
ros2 launch depth_hybrid_slam production_ready.launch.py \
  route_path:=$PWD/maps/case_1/route.csv
```

GUI는 기본 비활성이다. 원본 영상은 `/camera/camera/color/image_raw`를 직접 보며
Python relay를 사용하지 않는다.

## 측정, rosbag, 백업

```bash
ros2 launch depth_hybrid_slam cuvslam_performance.launch.py
./scripts/record_hybrid_mapping_bag.sh bags/test_mapping
cp --reflink=auto maps/case_1/rtabmap.db /backup/location/rtabmap.db
sha256sum maps/case_1/rtabmap.db maps/case_1/route.csv
```

bag 스크립트는 실행 시 존재하는 토픽만 기록한다. 영상 기록 부하는 성능 시험과
분리한다. 종료는 follower/RTAB-Map, cuVSLAM container, RealSense 순서로 각각
`Ctrl-C`를 사용한다. DB 복구는 검증된 백업을 새 case 디렉터리에 복사한 뒤
metadata/checksum을 다시 봉인하는 방식으로만 한다.

## 자주 생기는 오류

- odometry가 0 Hz: host/container의 domain, RMW, uid, hostname, machine-id, IPC 확인.
- 6 Hz 안팎: UDP fallback 병목 가능성. `run_cuvslam_container.sh`의 SHM 설정 사용.
- `frame delta >19 ms`: 일부 stereo frame 누락. 평균/p95와 reset을 함께 판정.
- localization이 `MAP_MISMATCH`: case 경로, metadata, 두 checksum을 수정 없이 재확인.
- 카메라 장착 위치가 바뀜: 기존 map/route를 사용하지 말고 네 case를 다시 기록.
- tracking loss/pose stale/mission UNKNOWN: fail-closed 정지 상태가 정상 동작이다.

LiDAR 장애물 회피·S자 우회·주차는 extension point일 뿐 이번 패키지에 구현하지
않았다. 가짜 LiDAR publisher도 production launch에 없다.
