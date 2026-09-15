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
  MCU가 READY/armed가 되고 localization이 안정화되면 움직일 수 있다. START
  A/B는 VSLAM 재검증 없이 사용자가 준 `start_branch`로 확정한다.
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

`A` 실행은 `START_BRANCH=A`, `B` 실행은 `START_BRANCH=B`다. 이 값만으로
START_A/START_B를 확정하며 `[SEGMENT 1] COMPLETE`를 한 번 출력한다. VSLAM은
START 분기 선택기가 아니라 이후 현재 위치 localization 보조다.

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
| 현재 경로 소유자 | `/depth_slam/path_owner` (`STOP/CSV/CAMERA/LIDAR/PARKING`) |

조향 토픽의 production 계약은 `/wheel`, `/cmd_wheel`, `/mcu/steer_deg` 모두
`LEFT=+`, `RIGHT=-`, `CENTER=0`이다. 물리 명령 범위는 `±22°`이며 경로 계획은
안전 여유를 둔 `abs(required_steering)<=20°`만 허용한다. CSV curvature부터
`/cmd_wheel`, MCU 명령과 측정 ADC까지 부호를 반전하는 중간 adapter는 없다.

## 최종 안전 우선순위

경로 소유권은 `STOP > LiDAR/PARKING > CAMERA > CSV`이며 한 번에 하나뿐이다.
그 안에서 `Emergency STOP > 0.5 m 장애물 STOP > mission/branch/계획 STOP >
LiDAR 거리 감속 > 조향 감속 > mission 속도` 순서다. 전 구간에서 장애물이
`<=0.5 m`면 0, `0.5 m < 거리 <=1.0 m`면 최대 1단이다. 안전/실측 조향 입력이
0.5초 이상 stale이면 정지한다.

실제 `/mcu/steer_deg`의 절댓값이 `>=10°`로 1초 연속 유지되면 전 구간에서
1단으로 감속한다. `<10°`가 1초 연속 유지돼야 해제된다. 일반 구간은 원래
2단, 구간 9는 원래 3단으로 복귀하며 두 상태가 서로 덮어써 진동하지 않는다.

## A/B 선택

- 구간 1: 사용자가 준 `start_branch:=A` 또는 `start_branch:=B`로 즉시
  확정한다. VSLAM은 분기 double validation에 사용하지 않는다.
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
| 1 START | 사용자 `start_branch`로 START A/B 확정 | A/B 유효값이면 COMPLETE; VSLAM double validation 없음 |
| 2 SLOPE | CSV STOP에서 실제 4초 정지 | 유효한 상대 `abs(IMU pitch)>=4.5°`가 0.5초 연속이고 4초 정지를 완료하면 COMPLETE |
| 3 ROAD | CSV를 카메라 metric BEV로 검증 | road 내부, 흰/노란 통합 차선 crossing=0, LiDAR `>0.5 m` 모두 충족 |
| 4 INTERSECTION | CSV STOP 최소 3초 | R은 3초 후에도 정지, G/UNKNOWN은 최소 3초 뒤 release |
| 5 AVOIDANCE | CSV 충돌 예상 장애물만 짧게 우회 | 오직 구간 5, 정지→2초 확인→전체 계획→전 점 `<=20°` 검증→1단 주행→CSV 재합류를 2회 성공 |
| 6 INTERSECTION | 두 번째 CSV STOP | 구간 4와 동일 |
| 7 T-PARK | LiDAR A/B 슬롯 후 해당 CSV 주차 | 슬롯 확인→선택 경로 주차→CSV 재합류; 슬롯 없는 완료는 FAIL |
| 8 INTERSECTION | 세 번째 CSV STOP | 구간 4와 동일 |
| 9 ACCEL | 정상 3단 | `<=1.0 m` 1단, `<=0.5 m` 정지; hard stop 뒤 `>1.5 m` clear가 1초 연속일 때만 3단 복귀 |
| 10 V-PARK | LiDAR A/B 슬롯 후 해당 CSV 주차 | 구간 7과 동일 |
| 11 EXIT | CSV STOP 및 출구 결정 | 최소 5초 정지, 화면 x순 3등이 `G R R`일 때만 A; `R G R`, `R R G`, `R R R`, UNKNOWN/미검출은 B |

## 교차로 신호 규칙

모든 CSV `STOP_LINE`은 최소 3초 정지하며 그동안에도 신호 토픽을 계속 읽는다.
fresh한 원본/fusion 입력 중 R 또는 Y가 하나라도 있으면 G보다 우선하고 R이
유지되는 동안 무기한 정지한다. G는 STOP 시작부터 최소 3초 뒤 해제되고,
UNKNOWN은 연속 3초 확인된 뒤 해제된다. 교차로를 0.35 m 이상 통과해 commit된 뒤에는 뒤늦은
신호로 교차로 내부에서 급정지하지 않는다.

## 5구간 우회

- 전방 ROI는 최대 1.5 m이며 2 m 물체 때문에 정지하지 않는다.
- 현재 CSV 차량 footprint와 충돌이 예상될 때 먼저 정지한다. 장애물이 2초
  연속 확인되고 실제 `/odom` 정지가 0.3초 확인된 뒤 한 번에 계획한다.
- 마지막 관측 거리가 1.0 m 미만이면 새 경로를 만들지 않고 정지한다.
- 장애물 envelope만큼만 CSV 옆으로 이동한 최단 후보부터 검사한다. 차량 폭,
  margin, 모든 pose의 차량 footprint-연석 clearance, 정확한 CSV 재합류와
  전체 점의 요구 조향 `abs<=20°`를 만족한 경로만 실행한다. 우회 중 차선
  crossing은 허용하지만 연석 collision은 허용하지 않는다. curb-safe 후보가
  하나도 없으면 억지로 우회하지 않고 STOP을 유지한다. 물리 명령 한계
  `abs<=22°`는 그대로 유지한다.
- `STOP_FOR_PLANNING → LIDAR_PATH_TRACKING(1단) → CSV_REJOIN →
  CSV_TRACKING` 순서이며 재합류 index는 같은 segment의 전방 점만 허용한다.

정상 코너의 안쪽 연석처럼 가까워도 현재 CSV swept footprint를 막지 않는
cluster는 `NORMAL_CORNER`로 취급해 감속만 할 수 있고 Mode 5 경로 생성 원인이
되지 않는다. 실제 CSV corridor를 막는 장애물만 Mode 5 우회 후보가 된다.

## CSV·VSLAM·Camera·LiDAR 역할

- CSV는 변경하지 않는 기본 global reference path다.
- VSLAM과 실제 `/odom`은 현재 차량 pose localization에만 사용한다.
- Camera는 CSV 앞쪽 3~5 m의 네 바퀴/차량 footprint와 실제 차선을 검증한다.
- LiDAR는 전역 안전, Mode 5 우회, Mode 7/10 주차를 담당한다.

Camera 상태는 `TRUE/UNKNOWN/FAIL`이다. `TRUE`와 `UNKNOWN`은 모두 CSV를
유지하며, 단순 차선 미검출·낮은 confidence·정상 코너는 경로 생성 사유가
아니다. 차선 geometry가 충분하고 예상 wheel/footprint 충돌이 0.4초 연속인
경우에만 `FAIL`이다. 차선 track은 ID, 색, geometry, curve, heading,
confidence, age, last_seen, missing_duration을 유지하며 약 0.20초 이내의 짧은
검출 손실을 보완한다. 진행방향과 평행한 연속 선은 lane, 수직 단일 선은
stop line, 반복 수직 stripe는 crosswalk로 geometry를 함께 사용한다.

Camera correction은 최후 수단이다:

```text
persistent FAIL -> STOP -> 실제 정지 확인 -> 3초 연속 재검증
-> 3~5 m local path 생성 -> 전체 path/curb/curvature/<=20° 검사
-> 1단 follow -> 원래 CSV endpoint 재합류
```

3초 동안 FAIL/geometry confidence가 유지되지 않거나 전체 경로가 안전하지
않으면 출발하지 않는다. LiDAR maneuver 중에는 검증 기록만 가능하고 Camera
계획/소유권 획득은 금지된다. Camera 경로 주행 중 LiDAR emergency 또는
maneuver가 들어오면 Camera 경로를 즉시 폐기하고 LiDAR가 우선한다.

## CSV 상태 관리

progress index는 역행하지 않고 현재 segment/direction의 앞쪽만 검색한다.
STOP waypoint 도달은 base_link 또는 상수 offset이 아니라 live
`map -> front_laser` TF 위치로만 판정한다. STOP key는 통과 후 완료 집합에
들어가 재발동하지 않는다. A/B 변경은 같은
segment와 같거나 큰 point index로만 remap하며 반대 branch로 전역 snap하지
않는다. 구간 5 재합류와 구간 7/10 주차 재합류도 동일 segment, 전방 window,
거리 0.25 m, heading 10° 기준을 사용한다.

## 현장 확인이 남는 항목

자동 테스트는 로직·launch·CSV 계약만 검증한다. 실제 차량에서는 USB 포트,
VSLAM 재위치화, 지도/CSV 물리 정합, encoder 부호와 counts-per-meter, steering
center/부호, 제동거리, 신호등 토픽 지속 주기를 반드시 낮은 속도로 확인한다.
실센서 없이 얻은 결과는 `PASS_OFFLINE`이며 `REAL-VEHICLE PASS`가 아니다.
