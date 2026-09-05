#!/usr/bin/env bash
set -euo pipefail

output="${1:-a6_preflight_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$output"

ros2 node list | sort | tee "$output/nodes.txt"
ros2 topic list -t | sort | tee "$output/topics_and_types.txt"

topics=(
  /camera_drive /camera_wheel
  /mcu/cmd_drive /mcu/cmd_wheel
  /mcu/active_drive_source /mcu/active_wheel_source
  /mcu/safety_state /mcu/steer_deg /mcu/speed_mps /odom
)
for topic in "${topics[@]}"; do
  safe_name="${topic#/}"
  safe_name="${safe_name//\//_}"
  ros2 topic info -v "$topic" >"$output/topic_${safe_name}.txt" 2>&1 || true
  timeout 3s ros2 topic echo "$topic" --once \
    >"$output/value_${safe_name}.txt" 2>&1 || true
done

ros2 param get /direct_bev_planner_node planner_variant \
  >"$output/planner_variant.txt" 2>&1 || true
ros2 param get /camera_yolo_inference_node line_track_mode \
  >"$output/line_track_mode.txt" 2>&1 || true

wheel_owner="$(tr -d '[:space:]' <"$output/value_mcu_active_wheel_source.txt" || true)"
drive_owner="$(tr -d '[:space:]' <"$output/value_mcu_active_drive_source.txt" || true)"
if [[ "$wheel_owner" != *camera* ]]; then
  printf 'FAIL: active wheel source is not camera: %s\n' "$wheel_owner" >&2
  exit 3
fi
if [[ "$drive_owner" != *camera* ]]; then
  printf 'FAIL: active drive source is not camera: %s\n' "$drive_owner" >&2
  exit 4
fi
printf 'PASS: camera owns both drive and wheel. Logs: %s\n' "$output"
