#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "${script_dir}/.." && pwd)"
source "${workspace}/setup_depth.sh"
[[ $# == 1 ]] || { echo "usage: $0 BAG_SESSION_PATH" >&2; exit 2; }
bag_path="$(realpath -e "$1")"
[[ "$bag_path" == "$workspace/bags/"* ]] || { echo "bag must be under $workspace/bags" >&2; exit 2; }
(cd "$bag_path" && sha256sum --check checksums.sha256)
ros2 bag play "$bag_path" --storage mcap --disable-keyboard-controls
(cd "$bag_path" && sha256sum --check checksums.sha256)
