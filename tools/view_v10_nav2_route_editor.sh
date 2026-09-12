#!/usr/bin/env bash
set -euo pipefail

# ROS 2 environment hooks may inspect variables that have not been initialized.
# Keep errexit and pipefail enabled, but temporarily disable nounset while sourcing.
set +u
source /opt/ros/jazzy/setup.bash
source /home/qor/depth_ws/install/local_setup.bash
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/vslam_nav2_route_editor_logs}"
exec ros2 launch vslam_nav2_route_editor v10_route_editor.launch.py "$@"
