#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qor/depth_ws
python3 "$ROOT/tools/publish_ply_cloud.py" \
  "$ROOT/analysis/level_aligned_v8/final/merged_competition_level_aligned_v8.ply" &
PUB_PID=$!
trap 'kill "$PUB_PID" 2>/dev/null || true' EXIT INT TERM
taskset -c 0-1 rviz2 -d "$ROOT/analysis/level_aligned_v8/final/view_full_cloud.rviz"
