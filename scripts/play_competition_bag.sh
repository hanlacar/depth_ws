#!/usr/bin/env bash
set -euo pipefail
workspace=/home/qor/depth_ws
[[ $# == 1 ]] || { echo "usage: $0 BAG_SESSION_PATH" >&2; exit 2; }
bag_path="$(realpath -e "$1")"
[[ "$bag_path" == "$workspace/bags/"* ]] || { echo "bag must be under $workspace/bags" >&2; exit 2; }
set +u
source /opt/ros/jazzy/setup.bash
source "$workspace/install/setup.bash"
set -u
export ROS_DOMAIN_ID=41 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
(cd "$bag_path" && sha256sum --check checksums.sha256)
ros2 bag play "$bag_path" --storage mcap --disable-keyboard-controls
(cd "$bag_path" && sha256sum --check checksums.sha256)
