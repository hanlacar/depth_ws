#!/usr/bin/env bash
set -eo pipefail

case_name="${1:-AAAA}"
spawn_branch="${case_name:0:1}"
domain_id="${2:-181}"
controller_hz="${3:-100.0}"
timeout_s="${4:-900.0}"
if [[ ! "${case_name}" =~ ^[AB]{4}$ ]]; then
  echo "case must be exactly four A/B letters" >&2
  exit 2
fi
log_dir="/tmp/depth_ws_ros_${domain_id}_${case_name}"
mkdir -p "${log_dir}"

source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID="${domain_id}"
export ROS_LOG_DIR="${log_dir}"

ros2 launch depth_hybrid_slam prehardware_csv_only_closed_loop.launch.py \
  start_rviz:=false spawn_branch:="${spawn_branch}" \
  controller_hz:="${controller_hz}" \
  >"${log_dir}/launch.stdout.log" 2>&1 &
launch_pid=$!
cleanup() {
  kill -INT "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 2

ros2 run depth_hybrid_slam csv_only_closed_loop_probe --ros-args \
  -p route_path:=/home/qor/depth_ws/routes/network/route_network_segmented_stop_edited_vforward.csv \
  -p route_metadata_path:=/home/qor/depth_ws/routes/network/route_network_segmented_stop_edited_vforward.metadata.yaml \
  -p expected_case:="${case_name}" -p timeout_s:="${timeout_s}"
