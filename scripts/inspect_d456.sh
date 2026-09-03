#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo '=== USB devices and negotiated speed ==='
lsusb
lsusb -t
echo '=== RealSense device summary ==='
rs-enumerate-devices -s
echo '=== RealSense firmware, USB type, and supported stream profiles ==='
rs-enumerate-devices
echo '=== Installed versions ==='
dpkg-query -W -f='${Package}\t${Version}\n' 'librealsense2*' \
  'ros-jazzy-realsense2-camera*' 'rtabmap*' 'ros-jazzy-rtabmap*' 2>&1 || true

