#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

requested=(
  /camera/camera/color/image_raw
  /camera/camera/color/camera_info
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/infra1/image_rect_raw
  /camera/camera/infra1/camera_info
  /camera/camera/infra2/image_rect_raw
  /camera/camera/infra2/camera_info
  /camera/camera/imu
  /visual_slam/status
  /depth_slam/cuvslam/odometry
  /rtabmap/info
  /depth_slam/localization/pose
  /tf /tf_static /manual_drive /manual_wheel /mcu/current_mode
  /camera/traffic_light_fused/state /camera/traffic_light_fused/aspect
  /camera/mission/stop_line_detected /camera/mission/sign_detected
  /depth_slam/cuvslam/diagnostics /depth_slam/mission/diagnostics
)
mapfile -t live < <(ros2 topic list)
present=()
for topic in "${requested[@]}"; do
  if printf '%s\n' "${live[@]}" | grep -Fxq "${topic}"; then
    present+=("${topic}")
  fi
done
if ((${#present[@]} == 0)); then
  echo "No requested topics are live; refusing to create an empty bag." >&2
  exit 2
fi
output="${1:-bags/hybrid_mapping_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "${output}")"
exec ros2 bag record -o "${output}" "${present[@]}"
