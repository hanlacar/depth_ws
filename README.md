# depth_ws 실차 운용·구간 점검표

ROS 2 Jazzy 기반 HENES T870 실차 구성이다. 하나의 Intel RealSense D456, 전방 SLAMTEC RPLIDAR A2M12, Arduino MCU의 실제 encoder/steering ODOM, A/B CSV 경로를 사용한다. 후방 라이다와 가짜 ODOM은 이 최종 실행 구성에서 사용하지 않는다. 전체 토픽·frame 계약은 [FINAL_SENSOR_CONTRACT.md](FINAL_SENSOR_CONTRACT.md)를 기준으로 한다.

## 실행 전 전제

- 차량을 띄우거나 E-stop을 즉시 누를 수 있는 상태에서 처음 확인한다.
- D456과 전방 라이다가 USB 허브에 연결돼 있어도 전방 라이다 장치가 `/dev/ttyUSB0`인지 확인해야 한다.
- 시작 경로는 터미널 1의 `start_branch:=A` 또는 `start_branch:=B`로 반드시 명시한다. 기본값은 A지만 현장에서는 생략하지 않는다.
- 터미널 1은 카메라·라이다·CSV 제어를 먼저 올리며 실제 `/odom`이 없으면 정지 대기한다. 터미널 2에서 MCU가 READY가 되는 순간 주행 조건이 모두 충족되면 차가 바로 움직일 수 있다.
- 최종 `/cmd_drive`, `/cmd_wheel` 발행자는 `depth_command_arbiter` 하나이고, 두 토픽과 `/odom`의 실차 소비·발행자는 `t870_mcu_simple_bridge` 하나여야 한다.

## 실행 방법 3개

### 1. 터미널 1 — CSV + D456 + 전방 라이다

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
ros2 launch depth_hybrid_slam depth_csv_camera_lidar.launch.py \
  start_branch:=A \
  front_serial_port:=/dev/ttyUSB0 \
  enable_control:=true \
  user_approved:=true
```

B 경로는 위 명령의 `start_branch:=A`만 `start_branch:=B`로 바꾼다.

### 2. 터미널 2 — 실차 MCU + 실제 ODOM

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
ros2 launch t870_mcu_simple mcu.launch.py port:=auto
```

정상 연결은 `serial opened`, `Arduino READY`, `bridge_ready:true`, `firmware_armed:true` 순서로 확인한다. `/mcu/command_diagnostics`에서 요청 stage와 applied stage/PWM이 같아야 한다. PWM은 후진 -1=50, 1단=50, 2단=75, 3단=100이다. ARM 또는 재부팅 때 encoder가 0으로 바뀌는 샘플은 ODOM 이동량으로 적분하지 않는다.

### 3. 터미널 3 — RViz2

```bash
cd ~/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
rviz2 -d ~/depth_ws/install/depth_hybrid_slam/share/depth_hybrid_slam/config/prehardware_csv_front_lidar.rviz
```

RViz의 기준 경로와 차량 TF가 겹쳐 시작하고, 이동 중 `map → odom → base_link → front_laser/camera_link`가 끊기지 않아야 한다. 장애물 marker는 정적 파랑, 동적 초록, 현재 ROI 내부 위험점 빨강이다.

## 공통 제어 로직

명령 우선순위는 `0.5 m 이내 전방 긴급정지 → mission/branch/LiDAR hold → LiDAR 우회 명령 → CSV 명령`이다. 안전·mission·branch·LiDAR hold 입력 중 하나가 0.5초 이상 끊기면 fail-safe 정지한다. 카메라 도로영역 판정은 보조 입력이며 CSV가 기준이다.

9구간을 제외한 모든 구간은 실제 `/mcu/steer_deg` 절댓값이 10도 초과로 1초 유지되면 1단으로 감속한다. 10도 이하가 다시 1초 유지돼야 원래 stage로 복귀한다. 9구간은 이 감속을 무시하고 전진 3단을 고정한다.

모든 CSV `STOP_LINE`은 최소 3초 정지한다. 신호 교차로인 4·6·8구간에서는 현재 입력 중 하나라도 R 또는 Y이면 계속 정지하고 UNKNOWN 타이머도 초기화한다. 모든 R/Y가 사라진 뒤 G이면 즉시 출발 허가, UNKNOWN이면 3초 연속 유지 후 출발 허가다. 정지선을 0.35 m 이상 정상 통과해 교차로 진입이 commit된 뒤에는 뒤늦은 신호로 교차로 안에서 급정지하지 않는다.

## A/B 경로 구성 확인

| 시작 선택 | 순서 | 총 점 수 | 반대 경로 포함 여부 |
|---|---|---:|---|
| A | START_A → COMMON_1 → T_foword → T_A → COMMON_2 → V_foword → V_A → END_common → END_AA | 3,242 | START_B/T_B/V_B/END_AB 제외 |
| B | START_B → COMMON_1 → T_foword → T_B → COMMON_2 → V_foword → V_B → END_common → END_AB | 3,259 | START_A/T_A/V_A/END_AA 제외 |

시작 후 `/depth_slam/route/active_branch`는 선택한 A 또는 B, `/depth_slam/route/active_segment`는 위 순서의 첫 segment, `/depth_slam/route/selected_branch`도 같은 값이어야 한다. 반대 branch 선이 RViz에 참고용으로 보일 수 있어도 follower의 활성 경로에는 들어가지 않는다.

## 구간별 현장 점검표

| 구간 | A/B CSV 범위 | 기대 동작 | 확인할 핵심 상태 |
|---:|---|---|---|
| 1 | START_A/B 0–101 | 전진 2단 CSV 추종 | `/drive_mode=1`, arbiter `CSV_TRACKING`, MCU applied stage 2 |
| 2 | A 102–230, B 102–241; 정지점 A:173/B:178 | 전진 2단, 정지점에서 실제 4초 정지 | stop key가 START_A:173 또는 START_B:178, mission reason `MODE2_4S_STOP`; `/imu/pitch_deg≥5`와 valid는 mission 완료 진단 조건 |
| 3 | A 231–600, B 242–580 | 전진 2단 CSV 추종 | 조향 10도 1초 조건이면 `STEERING_SUSTAINED_SLOWDOWN`과 stage 1, 해제도 1초 확인 |
| 4 | START_A 601–652 또는 START_B 581–634, 이어 COMMON_1 0–156; 정지점 25 | 첫 신호 교차로 | R/Y=`STOP_LINE_HOLD·RED_HOLD`, G=`RELEASE_PENDING`, UNKNOWN=`UNKNOWN_HOLD` 3초 뒤 release |
| 5 | COMMON_1 157–409 | 전진 2단, 반복 가능한 정적 장애물 우회 | 아래 ‘5구간 우회’의 상태 순서와 planned rejoin index 확인 |
| 6 | COMMON_1 410–659; 정지점 554 | 두 번째 신호 교차로 | 4구간과 동일하며 R이 지속되는 동안 `/cmd_drive=0` 유지 |
| 7 | T_foword 0–95 후 T_A 또는 T_B | 전방 라이다 구성에서는 기록 CSV 주차: 전진→정지→1단 후진→정지→전진 | A 전환점 4/80, B 전환점 5/66; maneuver `T_CSV_FALLBACK`; 후방 slot 없이도 CSV 완료가 mission 완료로 인정됨 |
| 8 | COMMON_2 0–707; 정지점 397 | 세 번째 신호 교차로 | R/Y 우선. R 없는 GREEN_LEFT는 허용. R+G 또는 R+GREEN_LEFT는 정지 |
| 9 | COMMON_2 708–1216 | 전 구간 무조건 전진 3단 고정 | arbiter `MODE9_FIXED_SPEED`, applied stage 3; 0.5 m 긴급정지는 예외 |
| 10 | V_foword 0–107 후 V_A 또는 V_B | 전방 라이다 구성에서는 기록 CSV 주차: 전진→정지→1단 후진→정지→전진 | A 전환점 11/66, B 전환점 0/70; maneuver `V_CSV_FALLBACK`; 후방 slot 불필요 |
| 11 | END_common 0–145 후 END_AA 0–78 또는 END_AB 0–90 | 진입 시 5초 정지하고 출구 선택, 마지막 정지점 145에서 최소 3초 정지 | GREEN→A, RED→B, UNKNOWN/무신호→A; commit 뒤 늦은 반대 신호 무시 |

구간 진행은 `/drive_mode`, `/depth_slam/route/active_segment`, `/depth_slam/route/active_index`, `/depth_slam/route/mode_status`로 확인한다. 최종 실제 명령은 `/cmd_drive`, `/cmd_wheel`, 명령 소유자는 `/depth_slam/command_arbiter/diagnostics`, MCU 적용 결과는 `/mcu/command_diagnostics`, 위치는 `/odom`으로 대조한다.

## 5구간 우회 확인

- 우회 판단 ROI는 front_lidar 기준 최대 1.5 m이며 갑자기 2 m 이상으로 늘어나지 않는다.
- 현재 CSV 진행방향의 차량 swept footprint가 장애물과 충돌할 때만 정지·우회 대상으로 삼는다. 옆 연석은 장애물로 보더라도 CSV 충돌선 밖이면 우회 트리거가 아니다.
- 같은 장애물이 최소 2초 확인되고 차량 정지가 확인된 뒤 경로를 만든다. 마지막 확인 장애물이 front_lidar에서 1.0 m보다 가까워졌으면 새 경로를 만들지 않고 정지한다.
- planner는 장애물 옆으로 필요한 만큼만 CSV를 평행 이동한 짧은 경로를 찾고, 같은 CSV segment의 정확한 앞쪽 점으로 재합류한다. 우회 중에는 1단으로 서행한다.
- 정상 상태 순서는 `CSV_TRACKING → STOP_CONFIRM_OBSTACLE → PLAN → LIDAR_PATH → CSV_REJOIN → CSV_TRACKING`이다. 두 번째 이후 우회도 매번 새 `/depth_slam/lidar/planned_rejoin_index`를 직접 검증하므로 진행 index가 앞서가도 재합류가 막히지 않아야 한다.
- `/depth_slam/lidar/perception`에서 mode 5 range가 1.5, `/depth_slam/lidar/maneuver`에서 hold/owner/state, `/depth_slam/lidar/csv_rejoin_valid`에서 재합류 판정을 확인한다.

## 9구간 긴급정지 확인

전방 장애물이 0.5 m 이내면 stage 3보다 긴급정지가 우선해 `/cmd_drive=0`이 된다. 한 번 정지한 뒤에는 1.5 m 이내에 장애물이 전혀 없는 상태가 연속 1초 유지돼야 `ACCEL_TRACKING`으로 돌아가고 다시 stage 3을 적용한다. 라이다 scan이 0.5초 이상 stale이어도 정지한다.

## 신호등 오판 점검

신호 원본은 `/camera_traffic_light`, `/camera/traffic_light_rgb/state`, fusion 결과는 `/camera/traffic_light_fused/state`다. 세 입력 중 fresh한 R/Y가 하나라도 있으면 confidence와 G보다 우선한다. 실제 R이 계속 보이는데 차가 움직인다면 `/depth_slam/mission/traffic_gate_state`, `/depth_slam/mission/traffic_gate_reason`, `/camera/traffic_light_fused/diagnostics`에서 각각 RED_HOLD 여부와 `valid_red_present`를 대조한다. R publisher 자체가 0.5초 이상 끊긴 경우에는 R이 아니라 UNKNOWN으로 처리되고 3초 뒤 출발할 수 있으므로 카메라 토픽 주기도 함께 확인한다.

## 검사 결과와 남은 현장 항목

- A/B topology, 반대 branch 제외, 모든 STOP_LINE과 전·후진 경계는 CSV 로더 기준으로 일치한다.
- `/cmd_drive`와 `/cmd_wheel`은 arbiter 단일 publisher, `/odom`은 MCU bridge 단일 owner 계약으로 구성돼 있다.
- 7·10구간의 전방 라이다 CSV fallback은 후방 slot 검출 없이도 완료 진단이 닫히도록 정합화했다.
- 코드 회귀검사는 ROS-independent unit/integration test, 전체 colcon test, self-contained audit를 기준으로 한다.
- 실제 차량에서는 USB 포트, encoder 방향·counts-per-meter, steering center/부호, 제동거리, 각 신호등 토픽의 지속 주기를 반드시 저속으로 재확인해야 한다.
- 현재 metadata의 저장 RTAB-Map 정합은 `validated:false`, symmetric RMS 1.83 m로 허용 1.0 m를 넘는다. 이 실행은 첫 실제 ODOM pose를 CSV 원점에 맞추는 odom-relative 방식이라 주행을 막지는 않지만, RViz의 저장 지도와 CSV가 정밀하게 일치한다고 간주하면 안 된다.
