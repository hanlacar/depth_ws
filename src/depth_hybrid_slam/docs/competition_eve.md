# 대회 전날 4-case 기록 절차

1. 카메라 장착값과 D456 serial `338122302896`을 확인하고 기존 case를 별도 저장소에 백업한다.
2. `case_manager init`으로 `case_1`~`case_4` 디렉터리만 준비한다.
3. 사용자에게 받은 실제 주차 상태 정의를 각 case 메모에 기록한다. 임의 슬롯 의미를 부여하지 않는다.
4. case마다 새 DB 경로로 camera, cuVSLAM, hybrid mapping을 시작한다.
5. 수동 주행 중 route recorder를 시작하고 정지선·section·mission marker를 확인한다.
6. 정상 종료 후 loop closure, reset 수, 시작/끝 pose와 세 저장소 commit을 metadata에 채운다.
7. `case_manager seal`로 map/route를 복사하고 checksum을 생성한다.
8. `case_manager verify`를 통과한 뒤 원본 기록과 봉인 case를 각각 백업한다.
9. 네 case 모두 별도 프로세스/DB로 반복한다. DB를 이어 쓰지 않는다.

카메라 mount, firmware, calibration이 달라지면 기존 네 묶음은 대회용으로 승인하지 않는다.
