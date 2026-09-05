#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_name="$(hostname)"

exec docker run --rm --gpus all --network host --ipc host --pid host \
  --hostname "${host_name}" --user "$(id -u):$(id -g)" \
  -e ROS_LOG_DIR=/tmp/depth_ros_logs -e ROS_DOMAIN_ID=41 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v /etc/machine-id:/etc/machine-id:ro \
  -v "${workspace_root}/isaac_ros_ws:/workspaces/isaac_ros_ws:ro" \
  -v "${workspace_root}/src/depth_hybrid_slam:/workspaces/isaac_ros_ws/src/depth_hybrid_slam:ro" \
  depth/isaac_ros-cuvslam:4.6-d456 bash -lc \
  'source /opt/ros/jazzy/setup.bash && source /workspaces/isaac_ros_ws/install/setup.bash && ros2 launch depth_hybrid_slam cuvslam_only.launch.py'
