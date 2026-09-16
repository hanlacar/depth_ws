#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "${script_dir}/.." && pwd)"
source "${workspace}/setup_depth.sh"
export ROS_LOG_DIR=/tmp/depth_ros_logs

camera_serial="${CAMERA_SERIAL:-}"
serial_argument=()
[[ -z "${camera_serial}" ]] || serial_argument=("serial_no:=_${camera_serial#_}")

exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera "${serial_argument[@]}" \
  enable_color:=true enable_depth:=true enable_infra1:=true enable_infra2:=true \
  enable_gyro:=true enable_accel:=true \
  rgb_camera.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  depth_module.infra_profile:=640x480x60 \
  gyro_fps:=400 accel_fps:=400 unite_imu_method:=2 \
  align_depth.enable:=true enable_sync:=true pointcloud.enable:=false
