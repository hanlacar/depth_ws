# Competition rosbag recording

The recorder is host-only, explicit-topic, MCAP/Zstd, fail-closed tooling. It
does not launch camera, SLAM, route following, MCU, mission or control nodes.
`enable_control=false`, `dry_run=true`, and real drive command publishers are
required to remain absent.

## Build

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
PYTHONNOUSERSITE=1 colcon build --base-paths src/depth_hybrid_slam \
  --packages-select depth_hybrid_slam --symlink-install
source install/setup.bash
```

## Record

Start the existing D456 host, cuVSLAM container and optional RTAB-Map stack
once. Do not start duplicates. The following creates a new path and refuses an
existing path. `--duration 60` performs a normal timed stop; omit it for manual
Ctrl-C termination.

```bash
cd /home/qor/depth_ws
./scripts/record_competition_bag.sh --case camera_stationary_test \
  --purpose competition --duration 60
./scripts/record_competition_bag.sh --case camera_hand_motion_test \
  --purpose competition --duration 60
```

Use `--case case_1` through `case_4` for competition cases. Any safe test ID is
also accepted. Derived aligned depth and `/rtabmap/mapData` are intentionally
excluded from the default profile because raw depth/calibration are retained;
use `camera_slam_with_aligned_depth` or `camera_slam_with_mapdata` to opt in.

Each new session is stored below
`bags/<case_id>/<YYYYmmdd_HHMMSS>/`. Fastwrite MCAP files are compressed with
rosbag2 file-level Zstd using four workers; this sustains the measured raw image
rate better than single-writer MCAP chunk compression. Files split at 4 GiB or
900 seconds, whichever happens first. A successful stop adds `manifest.yaml`, `topic_statistics.json`,
`resource_statistics.json`, `control_guard.json`, and `checksums.sha256`.

## Inspect and recover metadata

```bash
BAG=/home/qor/depth_ws/bags/<case_id>/<session_id>
ros2 bag info "$BAG"
(cd "$BAG" && sha256sum --check checksums.sha256)
```

Do not run `ros2 bag reindex` on a finalized healthy bag: it rewrites
`metadata.yaml` and invalidates the checksum. Only if a crashed, unfinalized
session has no metadata, run this before finalization:

```bash
ros2 bag reindex -s mcap "$BAG"
./scripts/finalize_competition_bag.sh "$BAG"
```

## Hardware-free replay

Stop RViz/RTAB-Map, cuVSLAM, and D456 in that order, then validate on the same
domain. The validation report is written under `/tmp`; the bag remains
unchanged and checksums are checked before and after playback. Playback uses
the recorded stamps and does not publish `/clock` or set `use_sim_time`.

```bash
cd /home/qor/depth_ws
./scripts/validate_competition_bag_playback.sh "$BAG"
```

## Vehicle extension

`competition_bag_topics.yaml` contains disabled MCU, control, mission, safety,
and LiDAR role groups. On the completed vehicle, inspect the live graph, add
only verified topic/type mappings, enable the required groups, and add tests.
Never infer names or publish synthetic data to satisfy preflight.
