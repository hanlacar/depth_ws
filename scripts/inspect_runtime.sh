#!/usr/bin/env bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo '=== Nodes and typed topics ==='
ros2 node list
ros2 topic list -t
for topic in /odom /tf /tf_static \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/imu; do
  echo "=== $topic ==="
  ros2 topic info -v "$topic" || true
done
echo '=== Measured rates (10 samples each) ==='
for topic in /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/imu /odom /depth_slam/output_image; do
  timeout 15s ros2 topic hz --window 100 "$topic" || true
done
echo '=== TF tree ==='
timeout 8s ros2 run tf2_tools view_frames || true

