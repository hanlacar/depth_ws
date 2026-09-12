# VSLAM map-frame A 경로 수동 기록

이 기능은 `/depth_slam/localization/pose`만 경로 좌표로 사용한다. GPS→map
변환과 TF fallback은 사용하지 않으며 차량 명령 publisher를 만들지 않는다.

## 계약

- node: `map_route_recorder`
- launch: `map_route_recording.launch.py`
- 기본 출력: `/home/qor/depth_ws/routes/recorded_map`
- pose: `/depth_slam/localization/pose`
  (`geometry_msgs/msg/PoseWithCovarianceStamped`)
- state: `/depth_slam/localization/state` (`std_msgs/msg/String`)
- confidence: `/depth_slam/localization/confidence` (`std_msgs/msg/Float32`)
- 선택적 direction 입력: `/slam_drive` (`std_msgs/msg/Float32`).
  `use_drive_command_direction:=false`가 기본이며, 켤 때도 구독만 한다.
- visualization: `/depth_slam/recorded_route/raw_path`,
  `/depth_slam/recorded_route/resampled_path` (`nav_msgs/msg/Path`, `map`)
- debug: `/depth_slam/recorded_route/status` (`std_msgs/msg/String`)

`TRACKING`과 `RELOCALIZED`만 최종 경로 후보가 된다. 현재 localization 구현의
다른 상태는 `NOT_LOCALIZED`, `STOP_REQUIRED`, `LOST`, `DEGRADED`,
`MAP_MISMATCH`이며 RAW에는 pose와 상태가 남지만 최종 경로에서는 빠진다. pose
timeout과 상태 이탈은 dropout으로 센다. 복구 위치가 기본 0.75 m보다 멀거나
direction이 바뀌면 `RECORDED_A_###` segment를 새로 시작한다.

## 실제 기록

모든 터미널은 `ROS_DOMAIN_ID=41`, `rmw_fastrtps_cpp`를 사용한다.

터미널 1 — D456 소유 프로세스:

```bash
cd /home/qor/depth_ws
./scripts/run_d456_host.sh
```

터미널 2 — cuVSLAM과 bridge:

```bash
cd /home/qor/depth_ws
./scripts/run_cuvslam_container.sh
```

터미널 3 — 보호된 DB를 read-only bind로 여는 RTAB-Map localization:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam hybrid_localization.launch.py \
  map_path:=/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  route_path:=/home/qor/depth_ws/routes/network/route_network_segmented.csv
```

터미널 4 — recorder(실제 제어 없음):

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs
ros2 launch depth_hybrid_slam map_route_recording.launch.py \
  output_directory:=/home/qor/depth_ws/routes/recorded_map \
  resample_spacing_m:=0.10 start_rviz:=false \
  use_drive_command_direction:=false
```

터미널 5 — 키보드 제어:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 run depth_hybrid_slam map_route_keyboard
```

키는 `s` 시작, `p` pause/resume, `F` forward, `R` reverse, `?` 상태,
`q` finish/save이다. 키보드 없이 다음 Trigger service를 직접 호출해도 된다.

```bash
ros2 service call /depth_slam/map_route_record/start std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/map_route_record/pause std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/map_route_record/resume std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/map_route_record/set_forward std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/map_route_record/set_reverse std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/map_route_record/finish std_srvs/srv/Trigger '{}'
```

터미널 6 — live RViz:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_hybrid_slam visualization.launch.py \
  start_rviz:=true start_image_view:=false
```

시작 전에 다음을 확인한다. state가 안정된 `TRACKING` 또는 `RELOCALIZED`가
아니거나 pose가 0.5초보다 오래됐거나 frame이 `map`이 아니면 `s`가 거부된다.

```bash
ros2 topic hz /depth_slam/localization/pose
ros2 topic echo /depth_slam/localization/state
ros2 topic info /depth_slam/localization/pose -v
ros2 node info /map_route_recorder
ros2 topic info /slam_drive -v
ros2 topic info /slam_wheel -v
ros2 topic info /camera_drive -v
ros2 topic info /camera_wheel -v
ros2 topic info /gps_drive -v
ros2 topic info /gps_wheel -v
```

`ros2 node info`의 recorder publisher에는 visualization/status와 ROS 기본
토픽만 있어야 하며 위 여섯 drive/wheel 토픽에는 recorder가 publisher로
나오면 안 된다.

## 출력과 승인

한 번의 주행은 다음 파일을 만든다.

```text
/home/qor/depth_ws/routes/recorded_map/map_route_raw_YYYYMMDD_HHMMSS.csv
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS.csv
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS.metadata.yaml
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS.quality.json
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS_overview.png
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS_s_curve.png
/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS_gps_reference.png
```

GPS plot은 좌우 native-coordinate 구조 비교일 뿐 alignment 또는 실패 지표가
아니다. RAW는 covariance 36개를 모두 포함한다. final은 0.10 m 거리 재샘플,
정지 중복 제거, 임시 `mode=0`, `event=NONE`, `drive_level=1.0`을 사용한다.

저장 직후 자동 검사에는 raw/route 수, 거리, 평균/최대 간격, yaw jump,
dropout, segment break, covariance 통계, wheelbase 0.73 m 및 ±22° 기하,
S자 특징, 기존 `RouteFollower`의 backtrack 0/deviation 0/clamp/`ROUTE_COMPLETE`
검사가 포함된다. 그러나 metadata는 사람이 RViz를 볼 때까지 승인되지 않는다.

live 프로세스를 종료한 뒤 정적으로 지도와 경로를 다시 연다.

```bash
export ROUTE=/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS.csv
export META=/home/qor/depth_ws/routes/recorded_map/map_route_A_YYYYMMDD_HHMMSS.metadata.yaml
ros2 launch depth_hybrid_slam map_route_view.launch.py \
  map_path:=/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  route_path:="$ROUTE" start_rviz:=true
```

지도 위 도로 형상, 누락/반대 진입, segment break를 확인한 뒤에만 승인한다.

```bash
ros2 run depth_hybrid_slam map_route_validate \
  --metadata "$META" --route "$ROUTE" \
  --map /home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  --approve-rviz
```

승인 명령은 자동 품질 PASS 및 route/DB SHA256 일치를 다시 확인하고 나서
`route_validation.validated`와 alignment의 `validated`를 true로 바꾼다.
그 전에는 기존 `verify_route_binding()`이 `MAP_ROUTE_NOT_VALIDATED`로 닫힌다.

실제 출력에 연결되지 않는 follower dry-run은 다음과 같다.

```bash
ros2 launch depth_hybrid_slam route_follower_dry_run.launch.py \
  map_path:=/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  route_path:="$ROUTE" route_metadata_path:="$META" \
  piecewise_preview_path:='' piecewise_preview_metadata_path:=''
```

다음 단계는 기존 GPS CSV 의미를 최근접 좌표로 복사하는 것이 아니라, 검증된
진행도/거리 대응으로 `mode`, `segment`, `event`를 이 map-frame 경로에 이식하는
것이다. 이후 T주차와 평행주차는 별도 recording segment로 추가할 수 있다.
