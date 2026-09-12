#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/local_setup.bash
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/nav2_csv_route_follower_logs}"
exec ros2 launch nav2_csv_route_follower v10_nav2_csv_follower.launch.py "$@"
