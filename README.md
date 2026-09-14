# depth_ws 실차 운용·구간 점검표

ROS 2 Jazzy 기반 HENES T870 최종 구성이다. Intel RealSense D456, 전방
SLAMTEC RPLIDAR A2M12, Arduino MCU의 실제 encoder/steering `/odom`, 저장
RTAB-Map과 A/B CSV 경로를 사용한다. production launch에는 fake/synthetic
ODOM이나 Gazebo가 없다. 최종 `/cmd_drive`, `/cmd_wheel` 발행자는
`depth_command_arbiter` 하나이고 MCU bridge가 두 명령의 유일한 소비자이자
`/odom` 및 `odom -> base_link`의 소유자다.

## 실행 전

- 차량을 띄우거나 E-stop을 즉시 누를 수 있는 상태에서 저속 확인한다.
- D456과 라이다가 같은 허브에 있어도 라이다가 `/dev/ttyUSB0`인지 확인한다.
- START는 반드시 `START_BRANCH=A` 또는 `START_BRANCH=B`로 명시한다.
- 터미널 1은 실제 `/odom`이 들어오기 전까지 fail-safe 정지한다. 터미널 2의
  MCU가 READY/armed가 되고 RTAB-Map VSLAM START 검증까지 완료되면 움직일 수 있다.
- 저장 지도는
  `maps/merged_competition_level_aligned_v10/rtabmap.db`를 읽기 전용으로 쓰며,
  RTAB-Map 재위치화가 실패하면 주행하지 않는다.

## 실행 방법 3개

### 1. 터미널 1 — CSV + D456/RTAB VSLAM + 전방 라이다

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
START_BRANCH=A  # B 시작이면 이 값만 B로 변경
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=${START_BRANCH} \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true
```

`A` 실행은 `START_BRANCH=A`, `B` 실행은 `START_BRANCH=B`다. VSLAM이 판단한
START와 이 값이 다르면 `[SEGMENT 1] FAIL`을 한 번 출력하고 계속 정지한다.

### 2. 터미널 2 — 실차 MCU + 실제 ODOM

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
ros2 launch t870_mcu_simple mcu.launch.py port:=auto
```

`serial opened` → `Arduino READY` → `bridge_ready:true` →
`firmware_armed:true`를 확인한다. `/mcu/command_diagnostics`의 commanded/applied
stage와 PWM이 일치해야 한다. PWM 계약은 후진 -1=50, 1단=50, 2단=75,
3단=100이다.

### 3. 터미널 3 — RViz2

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
rviz2 -d ~/depth_ws/install/depth_hybrid_slam/share/depth_hybrid_slam/config/prehardware_csv_front_lidar.rviz
```

`map -> odom -> base_link -> front_laser/camera_link`가 끊기지 않아야 한다.
정적 장애물은 파랑, 동적 장애물은 초록, 현재 ROI 내부 장애물은 빨강이다.

## 센서·명령 확인

다음 토픽이 실제 장치 주기로 갱신되는지 확인한다.

| 대상 | 토픽 |
|---|---|
| MCU ODOM/조향 | `/odom`, `/mcu/steer_deg`, `/mcu/command_diagnostics` |
| D456/VSLAM | `/camera/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`, `/camera/camera/imu`, `/depth_slam/vslam/tracking_valid` |
| 전방 LiDAR | `/front_lidar/scan`, `/depth_slam/lidar/perception` |
| 도로·차선 | `/perception/semantic_path_frame`, `/depth_slam/camera/csv_validation` |
| 신호등 | `/camera_traffic_light`, `/camera/traffic_light_rgb/state`, `/camera/traffic_light_fused/state` |
| CSV 진행 | `/drive_mode`, `/depth_slam/route/active_segment`, `/depth_slam/route/active_index` |
| 최종 명령 | `/cmd_drive`, `/cmd_wheel`, `/depth_slam/command_arbiter/diagnostics` |

## 최종 안전 우선순위

`Emergency STOP > 0.5 m 장애물 STOP > mission/branch/계획 STOP >
LiDAR 거리 감속 > 조향 감속 > mission 속도` 순서다. 전 구간에서 장애물이
`<=0.5 m`면 0, `0.5 m < 거리 <=1.0 m`면 최대 1단이다. 안전/실측 조향 입력이
0.5초 이상 stale이면 정지한다.

실제 `/mcu/steer_deg`의 절댓값이 `>=10°`로 0.5초 연속 유지되면 전 구간에서
1단으로 감속한다. `<10°`가 1초 연속 유지돼야 해제된다. 일반 구간은 원래
2단, 구간 9는 원래 3단으로 복귀하며 두 상태가 서로 덮어써 진동하지 않는다.

## A/B 선택

- 구간 1: VSLAM START 분류와 사용자의 `start_branch`가 모두 A 또는 모두 B일
  때만 통과한다. 한쪽만 사용하지 않는다.
- 구간 7/10: 전방 LiDAR가 실제 A/B 빈 공간을 판정한 뒤 해당 A/B CSV 주차
  경로를 선택한다. 슬롯 없이 기본 A를 선택하지 않는다. 선택은 freshness가
  끝나도 해당 주차가 끝날 때까지 유지된다.
- 구간 11: 5초 관찰 중 명확한 A 신호만 A다. RED, UNKNOWN, 무신호,
  불충분/분할 표결은 B fallback이다. commit 후 반대 신호는 무시한다.

## 구간 1~11 판정표

각 최종 판정은 메인 launch 터미널에 `[SEGMENT N] COMPLETE - ...` 또는
`[SEGMENT N] FAIL - ...` 형식으로 구간당 최대 한 번 출력된다.

| 구간 | 동작 | COMPLETE 기준 / FAIL 기준 |
|---:|---|---|
| 1 START | D456 RGB-D RTAB-Map 위치로 START A/B 분류 | VSLAM=USER면 COMPLETE, mismatch/분류 timeout이면 FAIL+정지 |
| 2 SLOPE | CSV STOP에서 실제 4초 정지 | 정지 시작 순간 유효한 `abs(IMU pitch)>=5°`; 4.9°/invalid는 FAIL |
| 3 ROAD | CSV를 카메라 metric BEV로 검증 | road 내부, 흰/노란 통합 차선 crossing=0, LiDAR `>0.5 m` 모두 충족 |
| 4 INTERSECTION | CSV STOP 최소 3초 | R은 3초 후에도 정지, G/UNKNOWN은 최소 3초 뒤 release |
| 5 AVOIDANCE | CSV 충돌 예상 장애물만 짧게 우회 | 오직 구간 5, 정지→2초 확인→전체 계획→전 점 `<=20°` 검증→1단 주행→CSV 재합류를 2회 성공 |
| 6 INTERSECTION | 두 번째 CSV STOP | 구간 4와 동일 |
| 7 T-PARK | LiDAR A/B 슬롯 후 해당 CSV 주차 | 슬롯 확인→선택 경로 주차→CSV 재합류; 슬롯 없는 완료는 FAIL |
| 8 INTERSECTION | 세 번째 CSV STOP | 구간 4와 동일 |
| 9 ACCEL | 정상 3단 | `<=1.0 m` 1단, `<=0.5 m` 정지; hard stop 뒤 `>1.5 m` clear가 1초 연속일 때만 3단 복귀 |
| 10 V-PARK | LiDAR A/B 슬롯 후 해당 CSV 주차 | 구간 7과 동일 |
| 11 EXIT | CSV STOP 및 출구 결정 | 최소 5초 정지, 명확한 A만 A, 그 외 B, 선택 경로 완료 |

## 교차로 신호 규칙

모든 CSV `STOP_LINE`은 최소 3초 정지하며 그동안에도 신호 토픽을 계속 읽는다.
fresh한 원본/fusion 입력 중 R 또는 Y가 하나라도 있으면 G보다 우선하고 R이
유지되는 동안 무기한 정지한다. R이 실제로 G/UNKNOWN으로 바뀐 뒤 이미 3초를
채웠다면 출발한다. 교차로를 0.35 m 이상 통과해 commit된 뒤에는 뒤늦은
신호로 교차로 내부에서 급정지하지 않는다.

## 5구간 우회

- 전방 ROI는 최대 1.5 m이며 2 m 물체 때문에 정지하지 않는다.
- 현재 CSV 차량 footprint와 충돌이 예상될 때 먼저 정지한다. 장애물이 2초
  연속 확인되고 실제 `/odom` 정지가 0.3초 확인된 뒤 한 번에 계획한다.
- 마지막 관측 거리가 1.0 m 미만이면 새 경로를 만들지 않고 정지한다.
- 장애물 envelope만큼만 CSV 옆으로 이동한 최단 후보부터 검사한다. 차량 폭,
  margin, 연석 경계를 만족하고 전체 점의 요구 조향이 `abs<=20°`인 경로만
  실행한다. 물리 명령 한계 `abs<=22°`는 그대로 유지한다.
- `STOP_FOR_PLANNING → LIDAR_PATH_TRACKING(1단) → CSV_REJOIN →
  CSV_TRACKING` 순서이며 재합류 index는 같은 segment의 전방 점만 허용한다.

## CSV 상태 관리

progress index는 역행하지 않고 현재 segment/direction의 앞쪽만 검색한다.
STOP key는 통과 후 완료 집합에 들어가 재발동하지 않는다. A/B 변경은 같은
segment와 같거나 큰 point index로만 remap하며 반대 branch로 전역 snap하지
않는다. 구간 5 재합류와 구간 7/10 주차 재합류도 동일 segment, 전방 window,
거리 0.25 m, heading 10° 기준을 사용한다.

## 현장 확인이 남는 항목

자동 테스트는 로직·launch·CSV 계약만 검증한다. 실제 차량에서는 USB 포트,
VSLAM 재위치화, 지도/CSV 물리 정합, encoder 부호와 counts-per-meter, steering
center/부호, 제동거리, 신호등 토픽 지속 주기를 반드시 낮은 속도로 확인한다.
실센서 없이 얻은 결과는 `PASS_OFFLINE`이며 `REAL-VEHICLE PASS`가 아니다.
