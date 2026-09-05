#!/usr/bin/env bash
set -euo pipefail

workspace=/home/qor/depth_ws
case_id=""
purpose=competition
profile=camera_slam
duration=0
session_id="$(date +%Y%m%d_%H%M%S)"
map_path=""
route_path=""
requested_session_path=""

usage() {
  echo "usage: $0 --case CASE [--purpose mapping|localization|competition] [--duration SEC] [--session-id ID] [--session-path PATH] [--profile camera_slam|camera_slam_with_mapdata] [--map PATH] [--route PATH]"
}
while (($#)); do
  case "$1" in
    --case) case_id="$2"; shift 2 ;;
    --purpose) purpose="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --session-id) session_id="$2"; shift 2 ;;
    --session-path) requested_session_path="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --map) map_path="$2"; shift 2 ;;
    --route) route_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$case_id" ]] || { usage >&2; exit 2; }
[[ "$case_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || { echo "unsafe case id" >&2; exit 2; }
[[ "$session_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || { echo "unsafe session id" >&2; exit 2; }
[[ "$purpose" =~ ^(mapping|localization|competition)$ ]] || { echo "invalid purpose" >&2; exit 2; }

set +u
source /opt/ros/jazzy/setup.bash
source "$workspace/install/setup.bash"
set -u
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR="$workspace/log/competition_bag"
mkdir -p "$ROS_LOG_DIR" "$workspace/bags/.locks"

bag_root="$workspace/bags"
if [[ -n "$requested_session_path" ]]; then
  session_path="$(realpath -m -- "$requested_session_path")"
  [[ "$session_path" == "$bag_root/"* ]] || {
    echo "session path must be below $bag_root" >&2
    exit 2
  }
else
  session_path="$bag_root/$case_id/$session_id"
fi
[[ ! -e "$session_path" ]] || { echo "refusing to overwrite: $session_path" >&2; exit 2; }
[[ -n "$requested_session_path" ]] || mkdir -p "$bag_root/$case_id"

exec 9>"$bag_root/.locks/$case_id.lock"
flock -n 9 || { echo "another recorder holds case lock: $case_id" >&2; exit 2; }

temporary="$(mktemp -d /tmp/depth_bag_record.XXXXXX)"
preflight="$temporary/preflight.yaml"
topics_file="$temporary/topics.txt"
resources="$temporary/resource_statistics.json"
control_guard="$temporary/control_guard.json"
recording_config="$workspace/src/depth_hybrid_slam/config/bag_recording.yaml"
topic_config="$workspace/src/depth_hybrid_slam/config/competition_bag_topics.yaml"
qos_config="$workspace/src/depth_hybrid_slam/config/bag_qos_overrides.yaml"

cleanup_temp() { rm -rf -- "$temporary"; }
trap cleanup_temp EXIT

preflight_args=(
  --topic-config "$topic_config" --recording-config "$recording_config"
  --profile "$profile" --case-id "$case_id" --session-id "$session_id"
  --purpose "$purpose" --bag-root "$bag_root" --report "$preflight"
  --topics-output "$topics_file"
)
[[ -z "$requested_session_path" ]] || preflight_args+=(--session-path "$session_path")
[[ -z "$map_path" ]] || preflight_args+=(--map-path "$map_path")
[[ -z "$route_path" ]] || preflight_args+=(--route-path "$route_path")
ros2 run depth_hybrid_slam bag_preflight "${preflight_args[@]}"
mapfile -t topics < "$topics_file"
((${#topics[@]} > 0)) || { echo "preflight selected no topics" >&2; exit 2; }

node_suffix="${case_id}_${session_id//-/_}"
ros2 bag record --storage mcap --storage-preset-profile fastwrite \
  --compression-mode file --compression-format zstd --compression-threads 4 \
  --max-bag-size 4294967296 --max-bag-duration 900 \
  --max-cache-size 67108864 --disable-keyboard-controls \
  --qos-profile-overrides-path "$qos_config" \
  --node-name "competition_bag_recorder_${node_suffix}" \
  --custom-data "case_id=$case_id" "session_id=$session_id" "purpose=$purpose" \
  --output "$session_path" --topics "${topics[@]}" &
recorder_pid=$!
python3 -m depth_hybrid_slam.bag_resource_monitor --pid "$recorder_pid" \
  --bag-path "$session_path" --output "$resources" &
resource_pid=$!
python3 -m depth_hybrid_slam.bag_control_guard --output "$control_guard" &
guard_pid=$!

stop_helpers() {
  kill -TERM "$resource_pid" "$guard_pid" 2>/dev/null || true
  wait "$resource_pid" 2>/dev/null || true
  wait "$guard_pid" 2>/dev/null || true
}
stop_recorder() {
  if kill -0 "$recorder_pid" 2>/dev/null; then
    # rosbag2 Jazzy in this environment pauses/finalizes promptly on SIGTERM.
    kill -TERM "$recorder_pid" 2>/dev/null || true
    for _attempt in $(seq 1 240); do
      kill -0 "$recorder_pid" 2>/dev/null || return 0
      sleep 0.5
    done
    echo "recorder did not finalize within 120 seconds" >&2
    return 1
  fi
}
trap 'stop_recorder' INT TERM

record_status=0
if [[ "$duration" != 0 ]]; then
  sleep "$duration" || true
  stop_recorder
fi
wait "$recorder_pid" || record_status=$?
stop_helpers
trap - INT TERM

[[ -f "$session_path/metadata.yaml" ]] || {
  echo "recording did not create metadata.yaml (exit=$record_status)" >&2
  exit 2
}
install -m 0644 "$preflight" "$session_path/preflight.yaml"
[[ ! -e "$session_path/resource_statistics.json" ]] &&
  install -m 0644 "$resources" "$session_path/resource_statistics.json"
[[ ! -e "$session_path/control_guard.json" ]] &&
  install -m 0644 "$control_guard" "$session_path/control_guard.json"

normal_flag=()
if [[ "$record_status" == 0 || "$record_status" == 130 ]]; then normal_flag=(--normal-termination); fi
ros2 run depth_hybrid_slam bag_manifest "$session_path" \
  --preflight "$session_path/preflight.yaml" --recording-config "$recording_config" \
  --workspace "$workspace" "${normal_flag[@]}"
ros2 bag info "$session_path"
(cd "$session_path" && sha256sum --check checksums.sha256)
echo "BAG_PATH=$session_path"
