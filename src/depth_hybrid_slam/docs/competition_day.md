# 대회 당일 선택·재국소화·비상 절차

1. 차량 구동은 비활성 상태로 camera와 cuVSLAM을 먼저 시작한다.
2. 실제 주차 상태에 맞는 case를 사람이 선택하고 `case_manager verify`를 실행한다.
3. 검증된 DB/route 절대경로로 localization을 시작한다. checksum 검증 후 `bwrap`의
   read-only bind로 원본 DB 디렉터리를 보호하며, RTAB-Map은 그 경로를 읽는다.
4. `RELOCALIZED`, confidence, 연속 pose, 시작 경로 거리/heading을 확인한다.
5. 한 번의 match로 READY 처리하지 않는다. 가림 후 회복과 다른 방향 재인식을 정지 상태에서 확인한다.
6. route follower와 mission을 dry-run으로 확인한 뒤에만 별도 사용자 승인 절차를 진행한다.

재국소화 실패 시 차량을 움직이지 말고 시야를 확보한 뒤 global localization을 다시 요청한다.
tracking loss, 큰 correction, timeout, map/route mismatch가 있으면 원인을 해결하기 전 계속 주행하지 않는다.
비상 시 MCU의 물리적 비상정지 절차가 최우선이며, ROS 명령이나 이 패키지를 복구 수단으로 간주하지 않는다.

DB가 손상되면 실행 중 파일을 덮어쓰지 않는다. 백업을 새로운 디렉터리로 복원하고 checksum과 metadata를 다시 검증한다.
