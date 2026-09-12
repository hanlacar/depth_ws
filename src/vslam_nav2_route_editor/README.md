# vslam_nav2_route_editor

Offline-only tools for projecting the saved v10 RTAB-Map scan cloud to a Nav2
trinary map, authoring fixed `map`-frame routes, and checking ODOM-based path
following. Its only vehicle-shaped command outputs are the isolated
`/route_drive` and `/route_wheel` topics; they are not MCU command topics.

Route points are stored in CSV with
`index,segment_id,x,y,yaw_deg,direction,speed,required_steering_deg,event`.
Direction and speed stay in the CSV/metadata because `nav_msgs/Path` carries
pose only. Reverse yaw is the vehicle heading and is never changed merely by
changing `direction` to `R`.

`direction` is only `F` or `R`. `event` is `NONE`, `STOP`, or `ACCEL`.
Legacy eight-column CSV files and legacy `direction=STOP` rows load
automatically; the latter become a STOP event without rotating yaw. The ninth
column is written only on an explicit save.

The launch does not require the configured route CSV to exist. If it exists it
is loaded; otherwise the publisher starts with an empty, unsaved route named
`START_A`. In RViz, set Fixed Frame to `map` and use **Publish Point**. Clicks
append waypoint indices 0, 1, 2, ... and immediately update
`/nav2_route/path`, `/nav2_route/metadata`, and `/nav2_route/markers`. Clicking
does not create or overwrite files.

Save the running route explicitly (this is the only live-editor operation that
creates the CSV and metadata files):

```bash
ros2 run vslam_nav2_route_editor route_editor save --name START_A
```

Start a new empty in-memory route without deleting or modifying an existing
CSV:

```bash
ros2 run vslam_nav2_route_editor route_editor new --name START_A
```

The equivalent low-level services are `/nav2_route/save` and
`/nav2_route/new` (`std_srvs/srv/Trigger`). Reload and delete-last are available
as `/nav2_route/reload` and `/nav2_route/delete_last`; delete-last changes only
the in-memory route until an explicit save.

The launch preloads a minimal RViz layout containing `/map`,
`/vslam_map/cloud`, `/nav2_route/markers`, and `/nav2_route/path` with Fixed
Frame `map`. The markers include a nearly transparent, selectable XY surface
covering the full Nav2 map extent, so **Publish Point** resolves a real `map`
coordinate even where sparse PointCloud2 or unknown occupancy pixels provide no
pickable depth. It does not alter the DB, PLY, OccupancyGrid, TF, or route
coordinates.

Undo the last click or clear all in-memory points without writing files:

```bash
ros2 run vslam_nav2_route_editor route_editor undo --name START_A
ros2 run vslam_nav2_route_editor route_editor clear --name START_A
```

Assign forward/reverse ranges and mandatory direction-change stops in the
running, in-memory route (ranges are inclusive):

```bash
ros2 run vslam_nav2_route_editor route_editor set-direction --name START_A --start 0 --end 10 --direction F
ros2 run vslam_nav2_route_editor route_editor set-stop --name START_A --index 11
ros2 run vslam_nav2_route_editor route_editor set-direction --name START_A --start 12 --end 20 --direction R
ros2 run vslam_nav2_route_editor route_editor show --name START_A
ros2 run vslam_nav2_route_editor route_editor save --name START_A
```

Direct `F` to `R` or `R` to `F` transitions are rejected; a `STOP` waypoint
must separate them. Changing a point to reverse never rotates its stored yaw.
Forward speed is 2.00, or 1.00 when absolute required steering is at least 10
degrees. Reverse speed is -1.00 and STOP speed is 0.00. Steering outside
[-22,+22] degrees is reported as `INVALID` by `show` and prevents saving.

RViz waypoint markers are green spheres for F, blue cubes for R, and orange
cylinders for STOP. Invalid steering is red while retaining the direction's
shape.

## Selecting and marking events

RViz has two Publish Point tools. `/clicked_point` keeps the existing append
behavior. `/nav2_route/select_point` selects the nearest existing waypoint
within 0.50 m; it never adds a point. The selected point gets a large yellow
marker and only its index is shown. The same selection can be made exactly by
map coordinate from the CLI:

```bash
ros2 run vslam_nav2_route_editor route_editor select --name START_A --x 10.0 --y 20.0
ros2 run vslam_nav2_route_editor route_editor selected --name START_A
ros2 run vslam_nav2_route_editor route_editor mark-stop --name START_A
ros2 run vslam_nav2_route_editor route_editor mark-accel --name START_A
ros2 run vslam_nav2_route_editor route_editor clear-event --name START_A
ros2 run vslam_nav2_route_editor route_editor show-indices --name START_A --enabled true
```

STOP forces speed 0.00. ACCEL is persisted as a location only and currently
does not change speed. Use `save` explicitly to write events.

## ODOM-based fixed Path follower

The standard launch starts a test-only odometry chain:

```text
map -> rviz_odom -> rviz_base_link
```

`map -> rviz_odom` is computed from the first START_A pose and the first
received `/rviz_check/odom` pose. It does not publish or replace the production
`map`, `odom`, or `base_link` transforms. The included odometry model follows
the supplied T870 Jazzy reference: 797 count/m, steering `(ADC-484)/18`,
wheelbase 0.730 m, rear-to-base 0.365 m, and steering clamp ±22 degrees.

The follower uses `/nav2_route/path` as its authoritative geometry and
`/rviz_check/odom` as measured motion. `/nav2_route/metadata` is optional
auxiliary input for `F/R` and `NONE/STOP/ACCEL`, because `nav_msgs/Path` has no
fields for them. It subscribes to no camera, LiDAR, or GPS topic.

It uses a progress-aware nearest point and parameterized pure-pursuit
lookahead (default 1.0 m). The Ackermann wheel command is rounded to integer
degrees and clamped to `[-22,+22]`, with right negative and left positive.
Only these vehicle-shaped outputs are published:

```text
/route_drive                       std_msgs/msg/Float32
/route_wheel                       std_msgs/msg/Int32 (degrees)
```

Debug outputs are separated from the command contract:

```text
/route_debug/target_index          std_msgs/msg/Int32
/route_debug/cross_track_error_m   std_msgs/msg/Float32
/route_debug/status                std_msgs/msg/String (JSON)
/route_debug/map_odom_alignment    geometry_msgs/msg/TransformStamped
/route_debug/current_pose          geometry_msgs/msg/PoseStamped
/route_debug/odom_path             nav_msgs/msg/Path
/route_debug/markers               visualization_msgs/msg/MarkerArray
/route_debug/vehicle_marker        visualization_msgs/msg/Marker
/route_debug/synthetic_index       std_msgs/msg/Int32 (synthetic mode only)
```

`/route_test/drive` and `/route_test/wheel` are removed; commands are not
duplicated. No publisher is created for `/mcu/cmd_drive`, `/mcu/cmd_wheel`,
camera, GPS, or LiDAR command topics. ACCEL remains stored metadata and does
not enable a 3.00 speed mode.

Run the complete offline map, route, ODOM, follower, and RViz stack:

```bash
cd /home/qor/depth_ws
./tools/view_v10_nav2_route_editor.sh
```

Start from an existing waypoint instead of index 0 by passing `start_index`.
The selected waypoint's `x`, `y`, and yaw become the initial map pose,
`map -> rviz_odom` is aligned to it, and the follower will never search or
target a lower index. A large red heading arrow on
`/route_debug/vehicle_marker` has exactly the current `rviz_base_link` pose in
`map`; it therefore appears at the selected waypoint even before encoder data
changes. Restarting the launch with another index recalculates the initial
alignment. For example, start START_A at index 43:

```bash
cd /home/qor/depth_ws
./tools/view_v10_nav2_route_editor.sh start_index:=43
```

### Sensor-free synthetic ODOM

Use `synthetic_odom:=true` to move a fake odometry pose along the published
Path without encoder or steering hardware:

```bash
cd /home/qor/depth_ws
./tools/view_v10_nav2_route_editor.sh \
  start_index:=43 \
  synthetic_odom:=true
```

The mode publishes `/rviz_check/odom` at 20 Hz. Path poses are converted to a
relative `rviz_odom` trajectory whose selected start pose is `(0,0,0)`;
therefore the normal `map -> rviz_odom` alignment places the vehicle exactly
on the selected map waypoint. It also publishes the matching dynamic
`rviz_odom -> rviz_base_link` TF. Replay remains at the exact start pose until
the latched `map -> rviz_odom` alignment is received, avoiding a DDS startup
offset. Waypoint segments use linear interpolation and shortest-angle yaw
interpolation. The default replay speed is 1.0 m/s and is independent of
`/route_drive`.

When synthetic mode is enabled, the T870 test odometry node is not started, so
there is exactly one `/rviz_check/odom` publisher. Optional launch settings:

```bash
./tools/view_v10_nav2_route_editor.sh \
  start_index:=43 synthetic_odom:=true \
  synthetic_rate_hz:=20.0 synthetic_speed_mps:=1.0 \
  synthetic_stop_hold_s:=0.5 \
  synthetic_log_path:=/tmp/start_a_synthetic.csv
```

The log columns are `time,current_index,target_index,direction,event,`
`route_drive,route_wheel,cross_track_error`. STOP waypoints are held briefly
so the zero-speed policy is observable. RViz displays the route, vehicle,
measured trajectory, and lookahead target while the synthetic pose advances.

Inspect its command outputs:

```bash
ros2 topic echo /route_drive
ros2 topic echo /route_wheel
ros2 topic hz /route_drive
ros2 topic hz /route_wheel
ros2 topic echo /route_debug/status
```

If the separately supplied T870 odometry package is already running, avoid
duplicate `rviz_odom -> rviz_base_link` authority with:

```bash
./tools/view_v10_nav2_route_editor.sh start_test_odom:=false
```

Use `route_editor --help` for offline coordinate editing, multiple-point
movement, reordering, exact 0.20 m straight generation, smoothing, and
validation. Complex edits are intentionally done with the deterministic CLI.
