#!/usr/bin/env bash
# Source this file in every ROS 2 terminal.  It intentionally does not modify
# shell startup files and never persists a DHCP address or static peer.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source tools/ros_network_env.sh (do not execute it)" >&2
  exit 2
fi

if ! command -v ip >/dev/null 2>&1; then
  echo "[ROS NET] ERROR: iproute2 'ip' command not found" >&2
  return 1
fi

_depth_net_iface=""
while IFS= read -r _depth_net_route; do
  read -r -a _depth_net_fields <<<"${_depth_net_route}"
  _depth_net_candidate=""
  for ((_depth_net_i=0; _depth_net_i<${#_depth_net_fields[@]}; _depth_net_i++)); do
    if [[ "${_depth_net_fields[_depth_net_i]}" == "dev" &&
          $((_depth_net_i+1)) -lt ${#_depth_net_fields[@]} ]]; then
      _depth_net_candidate="${_depth_net_fields[_depth_net_i+1]}"
      break
    fi
  done
  case "${_depth_net_candidate}" in
    ""|lo|docker*|br-*|veth*) continue ;;
  esac
  _depth_net_iface="${_depth_net_candidate}"
  break
done < <(ip -4 route show default 2>/dev/null)

if [[ -z "${_depth_net_iface}" ||
      ! "${_depth_net_iface}" =~ ^[[:alnum:]_.:-]+$ ]]; then
  echo "[ROS NET] ERROR: no usable non-Docker IPv4 default-route interface" >&2
  return 1
fi

_depth_net_ipv4="$(ip -o -4 addr show dev "${_depth_net_iface}" scope global \
  2>/dev/null | awk 'NR == 1 {split($4, value, "/"); print value[1]}')"
if [[ ! "${_depth_net_ipv4}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "[ROS NET] ERROR: ${_depth_net_iface} has no usable global IPv4" >&2
  return 1
fi

export ROS_DOMAIN_ID=12
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY

_depth_net_peer_xml=""
_depth_net_peer_display="NONE"
if [[ -n "${DEPTH_WS_STATIC_PEERS:-}" ]]; then
  export ROS_STATIC_PEERS="${DEPTH_WS_STATIC_PEERS}"
  _depth_net_peer_display="${ROS_STATIC_PEERS}"
  IFS=';' read -r -a _depth_net_peers <<<"${ROS_STATIC_PEERS}"
  for _depth_net_peer_raw in "${_depth_net_peers[@]}"; do
    _depth_net_peer="${_depth_net_peer_raw//[[:space:]]/}"
    if [[ -z "${_depth_net_peer}" ||
          ! "${_depth_net_peer}" =~ ^[[:alnum:]._-]+(:[0-9]+)?$ ]]; then
      echo "[ROS NET] ERROR: invalid DEPTH_WS_STATIC_PEERS entry: ${_depth_net_peer_raw}" >&2
      unset ROS_STATIC_PEERS
      return 1
    fi
    _depth_net_peer_xml+="        <Peer Address=\"${_depth_net_peer}\"/>"$'\n'
  done
  _depth_net_peer_xml="      <Peers>"$'\n'"${_depth_net_peer_xml}      </Peers>"
else
  unset ROS_STATIC_PEERS
fi

_depth_net_xml="$(mktemp "/tmp/depth_ws_cyclonedds_${UID}_XXXXXX.xml")" || {
  echo "[ROS NET] ERROR: unable to create CycloneDDS runtime XML" >&2
  return 1
}
chmod 600 "${_depth_net_xml}"

{
  printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
  printf '%s\n' '<CycloneDDS xmlns="https://cdds.io/config">'
  printf '%s\n' '  <Domain Id="any">'
  printf '%s\n' '    <General>'
  printf '%s\n' '      <Interfaces>'
  printf '%s\n' '        <NetworkInterface name="lo" priority="2" multicast="false" prefer_multicast="false"/>'
  # The routed interface must be preferred for outbound SPDP. Loopback stays
  # configured as a local unicast locator for same-host participants.
  printf '        <NetworkInterface name="%s" address="%s" priority="10" multicast="true" prefer_multicast="false"/>\n' \
    "${_depth_net_iface}" "${_depth_net_ipv4}"
  printf '%s\n' '      </Interfaces>'
  printf '%s\n' '      <Transport>udp</Transport>'
  printf '%s\n' '      <AllowMulticast>spdp</AllowMulticast>'
  printf '      <MulticastRecvNetworkInterfaceAddresses>%s</MulticastRecvNetworkInterfaceAddresses>\n' \
    "${_depth_net_ipv4}"
  printf '%s\n' '      <RedundantNetworking>false</RedundantNetworking>'
  printf '%s\n' '    </General>'
  printf '%s\n' '    <Discovery>'
  printf '%s\n' '      <ParticipantIndex>auto</ParticipantIndex>'
  printf '%s\n' '      <MaxAutoParticipantIndex>63</MaxAutoParticipantIndex>'
  printf '%s\n' '      <LeaseDuration>10 s</LeaseDuration>'
  [[ -n "${_depth_net_peer_xml}" ]] && printf '%s\n' "${_depth_net_peer_xml}"
  printf '%s\n' '    </Discovery>'
  printf '%s\n' '  </Domain>'
  printf '%s\n' '</CycloneDDS>'
} >"${_depth_net_xml}" || {
  echo "[ROS NET] ERROR: unable to write CycloneDDS runtime XML" >&2
  return 1
}

export CYCLONEDDS_URI="file://${_depth_net_xml}"
export DEPTH_WS_ACTIVE_IFACE="${_depth_net_iface}"
export DEPTH_WS_ACTIVE_IPV4="${_depth_net_ipv4}"

printf '%s\n' '[ROS NET]'
printf 'domain=%s\n' "${ROS_DOMAIN_ID}"
printf 'rmw=%s\n' "${RMW_IMPLEMENTATION}"
printf 'interface=%s\n' "${DEPTH_WS_ACTIVE_IFACE}"
printf 'ipv4=%s\n' "${DEPTH_WS_ACTIVE_IPV4}"
printf 'discovery=%s\n' "${ROS_AUTOMATIC_DISCOVERY_RANGE}"
printf 'static_peers=%s\n' "${_depth_net_peer_display}"
printf 'cyclonedds_uri=%s\n' "${CYCLONEDDS_URI}"

unset _depth_net_route _depth_net_fields _depth_net_candidate _depth_net_i
unset _depth_net_iface _depth_net_ipv4 _depth_net_peer_xml
unset _depth_net_peer_display _depth_net_peers _depth_net_peer_raw
unset _depth_net_peer _depth_net_xml
