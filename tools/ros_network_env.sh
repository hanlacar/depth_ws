#!/usr/bin/env bash
# Backward-compatible alias. New terminals should source ../setup_depth.sh.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] source tools/ros_network_env.sh (do not execute it)" >&2
  exit 2
fi

_depth_compat_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_depth_compat_dir}/../setup_depth.sh"
unset _depth_compat_dir
