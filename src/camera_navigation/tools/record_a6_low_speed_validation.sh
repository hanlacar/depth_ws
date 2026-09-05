#!/usr/bin/env bash
set -euo pipefail

output="${1:-a6_low_speed_$(date +%Y%m%d_%H%M%S)}"
preflight="${output}.preflight.txt"

topics=(
  /camera/image_raw
  /camera/camera_info
  /camera/perception_overlay_image
  /camera/bev/overlay_image
  /camera/bev/camera_overlay
  /camera/bev/diagnostics
  /camera/bev/controller_diagnostics
  /camera/bev_drive_diagnostics
  /camera/bev/path
  /camera/bev/state
  /camera/path
  /perception/semantic_path_frame
  /camera_drive
  /camera_wheel
  /mcu_drive
  /mcu_wheel
  /mcu/cmd_drive
  /mcu/cmd_wheel
  /mcu/cmd_stop
  /mcu/current_mode
  /mcu/active_drive_source
  /mcu/active_wheel_source
  /mcu/safety_state
  /mcu/ready
  /mcu/steer_deg
  /mcu/speed_mps
  /mcu/speed_valid
  /mcu/telemetry_ok
  /mcu/fault
  /mcu/fault_text
  /mcu/fw_state
  /mcu/iface_state
  /mcu/estop_latched
  /mcu/hard_stop_active
  /odom
  /imu/relative_yaw_deg
  /imu/valid
  /manual_drive
  /manual_wheel
  /manual_stop
  /estop_lock
  /a6_validation/marker
  /tf
  /tf_static
)

required=(
  /camera/image_raw /camera/bev/path /camera/bev/state
  /camera_drive /camera_wheel
  /mcu/cmd_drive /mcu/cmd_wheel
  /mcu/active_drive_source /mcu/active_wheel_source /mcu/safety_state
  /mcu/steer_deg /mcu/speed_mps /odom
  /imu/relative_yaw_deg /imu/valid
)

ros2 topic list -t | sort | tee "$preflight"
available="$(ros2 topic list)"
missing=()
for topic in "${required[@]}"; do
  if ! grep -Fxq "$topic" <<<"$available"; then
    missing+=("$topic")
  fi
done
if ((${#missing[@]})); then
  printf 'ERROR: required topics missing:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Start the camera, IMU, MCU manager and bridge before recording.\n' >&2
  exit 2
fi

printf 'Recording %s\n' "$output"
printf 'Preflight topic/type snapshot: %s\n' "$preflight"
exec ros2 bag record -o "$output" --include-unpublished-topics \
  --topics "${topics[@]}"
