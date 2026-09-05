#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [OUTPUT_BAG_DIRECTORY]" >&2
  exit 2
fi

output="${1:-mission_validation_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing path: $output" >&2
  exit 2
fi

exec ros2 bag record -o "$output" \
  /camera/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/camera_info \
  /imu/data /imu/valid /tf /tf_static \
  /mcu/encoder /mcu/speed_mps /mcu/distance_m /mcu/speed_valid \
  /perception/semantic_path_frame /perception/detections_json \
  /perception/refined/stop_line \
  /camera/mission/section \
  /camera/mission/stop_line_detected \
  /camera/mission/stop_line_distance_m \
  /camera/mission/stop_line_count \
  /camera/mission/stop_line_distances_m \
  /camera/mission/sign_detected \
  /camera/mission/traffic_light \
  /camera/mission/uphill_detected \
  /camera/mission/diagnostics \
  /camera/mission/decision_state \
  /camera/mission/drive_override_active \
  /camera/mission/drive_override \
  /camera/mission/decision_diagnostics \
  /camera/mission/overlay \
  /camera/bev/valid /camera/bev/diagnostics \
  /camera_drive /camera_wheel
