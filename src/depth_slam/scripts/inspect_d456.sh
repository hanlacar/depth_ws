#!/usr/bin/env bash
# inspect_d456.sh — D456 가 640x480@60 을 실제로 지원하는지 확인한다.
# 추측 금지: enumerate 결과로만 판정한다.
set -euo pipefail

echo "== 연결된 RealSense 장치 =="
rs-enumerate-devices -s || { echo "장치 미검출"; exit 1; }

echo
echo "== 640x480@60 프로파일 존재 여부 (Color / Depth) =="
# 전체 프로파일에서 640x480 60fps 라인만 추린다.
rs-enumerate-devices | grep -Ei "640x480.*60|60.*640x480" || \
  echo "!! 640x480@60 프로파일이 목록에 없다 — 펌웨어/USB 확인 필요"

echo
echo "== USB 연결 속도 (5Gbps 이상이어야 60fps 안정) =="
lsusb -t | grep -i "5000M\|10000M" || echo "!! USB3 미확인 — USB2 면 60fps 불가"

echo
echo "판정 기준:"
echo "  - 위에 640x480@60 이 Color/Depth 둘 다 나오면 목표 프로파일 가능."
echo "  - 안 나오면 출력 상한이 30fps 로 내려가며, 이는 HW 한계다."
