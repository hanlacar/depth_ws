# 저장 경로 추종·재진입 토픽 계약

차량 제원과 기본값은 `config/vehicle_navigation.yaml`이 단일 실행 설정이다.
wheelbase 0.73 m, 조향 한계 ±22°, 최소 회전반경 1.8068134 m이며 T870 명령 부호는
왼쪽 양수, 오른쪽 음수다. 모든 실제 명령은 기존 `/slam_drive`와 `/slam_wheel`만 사용한다.

기본 `dry_run=true`, `enable_control=false`, `user_approved=false`에서는 실제 명령을
발행하지 않는다. 실제 제어에는 최신 pose, `TRACKING`/`RELOCALIZED` 안정화, map/route
SHA256 일치, 현재 footprint의 known-free map 포함, safety READY, mission 통과 및
사용자 승인이 모두 필요하다. mission/safety stop은 재진입보다 항상 우선한다.

## 상태

`/depth_slam/rejoin/state` (`std_msgs/String`):

`FOLLOW_ROUTE → ROUTE_DEVIATION_STOP → PLAN_REJOIN → FOLLOW_REJOIN_PATH →
VERIFY_REJOIN → FOLLOW_ROUTE`

충돌 없는 Ackermann 경로가 없으면 `PLAN_REJOIN → REJOIN_NO_FEASIBLE_PATH`이며 정지를
유지한다. 최초 localization 위치가 1 m 넘게 떨어진 경우에도 localization 안정화와
map/route 검증 뒤 같은 상태 머신을 사용한다. reverse connector는 route point의
`direction=-1`과 `reverse_rejoin_allowed=true`가 동시에 지정된 구간에서만 생성한다.

## 시각화와 진단

- `/depth_slam/route/reference_path` (`nav_msgs/Path`, 주황): metadata의
  `csv_to_map`을 적용한 active A 경로
- `/depth_slam/route/raw_csv_path` (`nav_msgs/Path`, 자홍): 변환 전 GPS-local ENU
  수치 좌표를 RViz의 map 축에 그대로 그린 **정합 전/후 비교 전용** 경로. 물리적인
  map-frame 경로 또는 제어 입력으로 해석하면 안 된다.
- `/depth_slam/route/piecewise_path` (`nav_msgs/Path`, 파랑): 오프라인 생성한 piecewise
  SE(2) 후보. 검증 전에는 비교 표시만 하며 follower의 제어 reference로 승격하지 않는다.
- `/depth_slam/rejoin/path` (`nav_msgs/Path`): 선택된 Dubins/Reeds–Shepp-subset 연결 경로
- `/depth_slam/rejoin/candidates` (`visualization_msgs/MarkerArray`): 방향 검사를 통과한
  합류 후보와 선택점
- `/depth_slam/route/controller_diagnostics`: 선택 index/reason, 후보 수, 경로 길이,
  최대 곡률, footprint 충돌검사, map/route 검증 결과
- `/depth_slam/route/map_route_verified` (`std_msgs/Bool`): metadata와 실제 파일 SHA256
- `/depth_slam/route/within_map` (`std_msgs/Bool`): 현재 차량 footprint가 known-free인지

기존 `/depth_slam/route/{reference_path,target_point,cross_track_error,heading_error,
progress,controller_state,controller_diagnostics}`는 유지한다. 횡오차는 이제 최근접
점 거리가 아니라 상태 기반으로 선택된 선분에 대한 signed projection 거리다.

Occupancy Grid 기본 토픽은 `/map`이다. `allow_unknown=false`가 기본이며 footprint에
unknown, map 외부, occupancy 50 이상이 하나라도 포함되면 후보 경로를 폐기한다.
