#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /home/qor/camera_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=12
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_LOCALHOST_ONLY
# Jazzy replacement for deprecated ROS_LOCALHOST_ONLY.  Keep validation camera
# frames on this host unless the operator deliberately supplies another range.
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

topics=(
  /camera/image_raw
  /perception/detections_image
  /camera/perception_overlay_image
  /perception/masks/road
  /perception/refined/road
  /perception/masks/white_line
  /perception/masks/yellow_line
  /camera/bev/safe_road_mask
  /camera/bev/overlay_image
  /camera/bev/camera_overlay
)

declare -A live=()
for attempt in 1 2 3 4 5; do
  while read -r topic type; do
    [[ "$type" == "[sensor_msgs/msg/Image]" ]] && live["$topic"]=1
  done < <(ros2 topic list --no-daemon --spin-time 2 -t)
  ((${#live[@]} > 0)) && break
  printf 'Waiting for live Image topics (%d/5)...\n' "$attempt"
  sleep 1
done

selected=()
for topic in "${topics[@]}"; do
  if [[ -n "${live[$topic]:-}" ]]; then
    printf 'FOUND   %s\n' "$topic"
    selected+=("$topic")
  else
    printf 'MISSING %s\n' "$topic"
  fi
done

if ((${#selected[@]} == 0)); then
  printf 'No live sensor_msgs/msg/Image topics were found.\n' >&2
  exit 2
fi

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for topic in "${selected[@]}"; do
  ros2 run rqt_image_view rqt_image_view "$topic" &
  pids+=("$!")
done
printf 'Opened %d Image View windows. Close them or press Ctrl-C here.\n' \
  "${#selected[@]}"
wait
