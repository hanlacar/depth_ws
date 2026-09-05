#!/usr/bin/env bash
# save_map.sh — 매핑 중 DB 를 안전하게 백업(flush)한다.
#
# rtabmap 은 정상 종료(Ctrl-C) 시 DB 를 flush 한다. 하지만 종료 전에
# 중간 스냅샷을 남기고 싶을 때 이 스크립트를 쓴다.
#
# 사용: ./save_map.sh   (mapping.launch.py 가 떠 있는 상태에서)
set -euo pipefail

echo "== 현재 DB 를 백업(flush) 한다 =="
# rtabmap 의 backup 서비스: DB 를 .back 으로 복제하고 현재 상태를 flush.
ros2 service call /rtabmap/backup std_srvs/srv/Empty "{}"

DB="${1:-$HOME/depth_ws/maps/rtabmap.db}"
echo
echo "== DB 파일 상태 =="
ls -lh "$DB" 2>/dev/null || echo "DB 파일이 아직 없다: $DB"
echo
echo "완료. 대회 주행 단계에서는 이 DB 를 로컬라이제이션 모드로 로드한다:"
echo "  Mem/IncrementalMemory:=false 로 실행 (별도 launch 에서)."
