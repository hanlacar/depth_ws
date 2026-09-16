#!/usr/bin/env bash
# Portable depth_ws environment. Source this file once in every ROS terminal.

set -u

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] source setup_depth.sh (do not execute it)" >&2
  exit 2
fi

DEPTH_WS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DEPTH_WS_ROOT

# Capture caller intent before ROS setup files add their own environment-hook
# defaults. Only values that existed in the user's shell count as overrides.
_depth_domain_value="${ROS_DOMAIN_ID-}"
_depth_rmw_value="${RMW_IMPLEMENTATION-}"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
else
  echo "[ERROR] ROS 2 Jazzy not found: /opt/ros/jazzy/setup.bash" >&2
  return 1
fi

if [[ -f "${DEPTH_WS_ROOT}/install/setup.bash" ]]; then
  set +u
  source "${DEPTH_WS_ROOT}/install/setup.bash"
  set -u
else
  echo "[DEPTH WS] install/setup.bash not found; build the workspace first" >&2
fi

if [[ -n "${_depth_domain_value}" ]]; then
  export ROS_DOMAIN_ID="${_depth_domain_value}"
else
  export ROS_DOMAIN_ID=12
fi
if [[ -n "${_depth_rmw_value}" ]]; then
  export RMW_IMPLEMENTATION="${_depth_rmw_value}"
else
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi
export ROS_AUTOMATIC_DISCOVERY_RANGE="${DEPTH_WS_DISCOVERY_RANGE:-LOCALHOST}"

# A stale external URI can pin CycloneDDS to a removed NIC or old DHCP IP.
# depth_ws never requires custom DDS XML. Advanced users can explicitly keep
# their URI for one shell with DEPTH_WS_PRESERVE_CYCLONEDDS_URI=1.
if [[ "${DEPTH_WS_PRESERVE_CYCLONEDDS_URI:-0}" != "1" ]]; then
  unset CYCLONEDDS_URI
fi

printf '%s\n' '[DEPTH WS]'
printf 'root=%s\n' "${DEPTH_WS_ROOT}"
printf 'domain=%s\n' "${ROS_DOMAIN_ID}"
printf 'rmw=%s\n' "${RMW_IMPLEMENTATION}"
printf 'discovery=%s\n' "${ROS_AUTOMATIC_DISCOVERY_RANGE}"
printf 'cyclonedds_uri=%s\n' "${CYCLONEDDS_URI:-UNSET}"

unset _depth_domain_value _depth_rmw_value
