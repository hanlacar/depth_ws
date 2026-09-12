# nav2_csv_route_follower

ROS 2 Jazzy test package that uses `/nav2_route/path` as the preferred
geometry, falls back to `route_network_segmented.csv` after sustained path
deviation, and always takes direction/events/speed from the segmented CSV.
The current course contract maps modes 4 and 6 to intersections, mode 7 to
T parking, mode 9 to acceleration, and mode 10 to parallel parking.  Those
sections force `CSV_ONLY`, independent of the deviation selector.  All other
modes use `NAV2` or `CSV_FALLBACK` through the deviation hysteresis.

Only these vehicle-shaped outputs are created:

```text
/route_drive  std_msgs/msg/Float32
/route_wheel  std_msgs/msg/Int32 (degrees, right negative, left positive)
```

It never publishes MCU, camera, GPS or LiDAR command topics. The existing v10
DB, PLY, Nav2 map, `START_A.csv`, `map_ws`, and input route-network files are
read only.

## Current route file contract

The strict parser expects exactly:

```text
segment_id,segment_type,point_index,latitude,longitude,x_m,y_m,
direction,mode,drive_level,event,from_node,to_node
```

`direction=1/-1` becomes F/R. `drive_level=1/2/3` is the requested forward
speed. `event=STOP_LINE` creates a mandatory one-second latched stop. The YAML
provides GPS origin metadata but does not contain a CSV-to-VSLAM transform.

The launch therefore exposes a scale-1 SE(2) registration. Defaults were fit
robustly from `AAA_BASE` to the v10 map START_A geometry:

```text
csv_to_map_x=-1.84276105 m
csv_to_map_y=-1.48548570 m
csv_to_map_yaw_deg=-153.98206441
scale=1.000 (not adjustable)
```

This fit has about 1.83 m symmetric RMS because the v10 path contains local
geometry edits. Tune the three parameters from surveyed anchors before live
vehicle use; the runtime comparison is intended to expose those deviations,
not hide them.

## Source and control policy

Initial source is NAV2. NAV2 switches to CSV after either target position is
at least 1.5 m apart or heading is at least 20 degrees apart for 10 consecutive
samples. CSV returns to NAV2 only after position is at most 0.75 m and heading
at most 10 degrees for 20 consecutive samples. All values are ROS parameters.

Priority is STOP, direction-change wait, reverse, CSV drive level, then
steering safety. Reverse is -1.00. Forward retains CSV 1/2/3 except that an
integer wheel command of at least 10 degrees caps it at 1.00. Steering uses
1.0 m pure-pursuit lookahead, wheelbase 0.73 m, and clamps to [-22,+22].

CSV STOP_LINE, direction changes, and Nav2 STOP metadata are spatially merged
within `stop_merge_distance_m=0.75`; a merged location stops once, not once per
cause. Separate positions each stop independently. The stop controller uses
`RUNNING -> APPROACH_STOP -> HOLD_STOP -> RELEASE -> RUNNING`, triggers at
`stop_trigger_distance_m=0.30`, holds drive and wheel at zero for one second,
then latches the stop against retriggering.

Branch decisions are accepted on `/route_branch/decision` as JSON, for example
`{"stage":"T","value":"B"}`. Valid stages are START, T, PARALLEL and END.
Only an explicit B selects B. Missing input, timeout, UNKNOWN and INVALID select
A, and the selected stage is latched once entered. No perception/decision node
is included.

CSV progress is matched from the current map pose inside the current contiguous
segment and is monotonic.  A later pass through a nearby intersection cannot
steal the current CSV index; a segment change is accepted only near the active
segment end.

Synthetic odometry is visualization-only and defaults to `0.25 m/s`; it does
not alter `/route_drive`. It replays `/route_compare/csv_path` so the full
selected route can enter every special section. `/route_drive=0` freezes the
synthetic pose, positive drive advances it, and negative drive advances along
the stored reverse geometry. Level 1 is shown at half the configured speed;
levels 2/3 and reverse are capped at the configured speed. Override it with
`synthetic_odom:=true synthetic_speed_mps:=0.25`. For targeted testing,
`synthetic_route_start_index` can start the replay at a selected assembled CSV
index without changing the Nav2 follower's `start_index`.

## Run

Build and start at v10 route index 43 with sensor-free odometry:

```bash
cd /home/qor/depth_ws
colcon build --base-paths src/nav2_csv_route_follower \
  --packages-select nav2_csv_route_follower --symlink-install
source install/setup.bash
./tools/view_v10_nav2_csv_follower.sh \
  start_index:=43 synthetic_odom:=true
```

Use measured T870 test odometry instead:

```bash
./tools/view_v10_nav2_csv_follower.sh synthetic_odom:=false
```

The RViz layout shows the VSLAM cloud and Nav2 map plus three distinct paths:
`/nav2_route/path` (cyan), `/route_compare/csv_path` (orange), and
`/route_compare/active_path` (green).  `/route_compare/csv_markers` displays
CSV STOP_LINE positions in red and direction changes in magenta.  The vehicle,
targets, current CSV index/segment/event/drive level, source, stop state and
odometry trail are also enabled automatically.

Inspect the exact input route without changing it:

```bash
ros2 run nav2_csv_route_follower inspect_route_network \
  --csv /home/qor/depth_ws/routes/network/route_network_segmented.csv \
  --yaml /home/qor/depth_ws/routes/network/route_network_segmented.yaml
```

Useful runtime checks:

```bash
ros2 topic echo /route_compare/status
ros2 topic echo /route_compare/csv_index
ros2 topic echo /route_compare/csv_segment
ros2 topic echo /route_compare/csv_event
ros2 topic echo /route_compare/csv_drive_level
ros2 topic echo /route_drive
ros2 topic echo /route_wheel
ros2 topic hz /route_drive
ros2 topic hz /route_wheel
```
