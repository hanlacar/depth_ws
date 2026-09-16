#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "${script_dir}/.." && pwd)"
source "${workspace}/setup_depth.sh"
[[ $# == 1 ]] || { echo "usage: $0 BAG_SESSION_PATH" >&2; exit 2; }
bag_path="$(realpath -e "$1")"
[[ "$bag_path" == "$workspace/bags/"* ]] || { echo "bag must be under $workspace/bags" >&2; exit 2; }
[[ -f "$bag_path/preflight.yaml" ]] || { echo "missing preflight.yaml" >&2; exit 2; }
ros2 run depth_hybrid_slam bag_manifest "$bag_path" \
  --preflight "$bag_path/preflight.yaml" \
  --recording-config "$workspace/src/depth_hybrid_slam/config/bag_recording.yaml" \
  --workspace "$workspace" --normal-termination
ros2 bag info "$bag_path"
(cd "$bag_path" && sha256sum --check checksums.sha256)
