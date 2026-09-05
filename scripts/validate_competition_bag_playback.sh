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
export ROS_LOG_DIR="$workspace/log/competition_bag_replay"
mkdir -p "$ROS_LOG_DIR"
if ros2 node list --no-daemon --spin-time 3 2>/dev/null | \
   grep -Eq '/camera/camera$|visual_slam_node|cuvslam_bridge'; then
  echo "live camera/cuVSLAM nodes still exist; stop hardware stack first" >&2
  exit 2
fi
report="$(mktemp /tmp/competition_bag_replay.XXXXXX.json)"
python3 -m depth_hybrid_slam.bag_replay_probe --output "$report" &
probe_pid=$!
trap 'kill -TERM "$probe_pid" 2>/dev/null || true' EXIT
sleep 2
"$workspace/scripts/play_competition_bag.sh" "$bag_path"
sleep 2
kill -TERM "$probe_pid" 2>/dev/null || true
wait "$probe_pid"
trap - EXIT
echo "REPLAY_REPORT=$report"
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); raise SystemExit(0 if data["pass"] else 1)' "$report"
