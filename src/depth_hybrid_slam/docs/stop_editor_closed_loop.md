# VSLAM STOP editor and pre-hardware closed loop

Both launches are test-only. The STOP editor first performs a real `bubblewrap`
probe. With Ubuntu's `apparmor_restrict_unprivileged_userns=1`, it selects
`TEMP_COPY`, checks that `/tmp` has at least 1.5 times the source DB size, and
uses only `/tmp/depth_stop_editor.<pid>.*/rtabmap.db`. Launch shutdown removes
that directory after normal exit, Ctrl+C, RViz exit, startup failure, or launch
exception. Insufficient space fails before a partial copy with
`INSUFFICIENT_TEMP_SPACE`. If the probe genuinely succeeds, the direct BWRAP
read-only bind remains preferred. No system security setting is changed.

The map requester subscribes to `/rtabmap/map` before requesting the map. It
requires both `/rtabmap/rtabmap/publish_map` and the `/rtabmap/map` publisher
within the 15-second startup timeout. The large saved map may legitimately take
several minutes to materialize, so its non-empty `map`-frame payload has a
separate 900-second timeout. Either failure exits with `RTABMAP_START_FAILED`
instead of waiting forever. The editor never changes the
source CSV or metadata. Its atomic save temporaries live beside the new output
files and are removed even when saving fails. The provisional CSV-to-map
transform remains `alignment.validated: false` in every output metadata file.

The editor automatically pairs the source route with
`route_network_segmented_all_branches_display_aligned.csv`. This is a map-frame,
visualization-only overlay of all 7,463 source rows. Every overlay row is matched
to the source route by the exact `(segment_id, point_index)` key. Duplicate,
missing, and ambiguous key counts must all be zero. Existing active-A candidate
coordinates are retained exactly; B branches and `AAA_BASE` reuse the same
unvalidated six-zone piecewise transform schedule. ADD and REMOVE change only
the matched source row's `event` field in a new output CSV.

`AAA_BASE` is shown faintly as a base/reference recording but is not part of a
competition traversal. The other 12 unique segments form four independent A/B
choices (START, T, V, END), producing the explicit cases `AAAA` through `BBBB`.
The production route follower remains unchanged: no production branch command
still selects its normal A default.

RViz opens with these displays already enabled in the `map` fixed frame:

- `/rtabmap/map` occupancy map and the saved RTAB-Map cloud
- the complete unique route network on `/depth_slam/stop_editor/network`
- mode-colored routes, solid A branches, dashed B branches, and segment labels
- mode 1 through 11 text labels and optional one-case highlight
- red ordinary STOPs, magenta cube direction-change STOPs, and selection marker

The in-map warning remains visible:

```text
CSV OVERLAY: VISUALIZATION ONLY
ALIGNMENT: UNVALIDATED
```

## STOP editor

Terminal 1:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch depth_hybrid_slam stop_editor_vslam.launch.py \
  map_path:=/home/qor/depth_ws/maps/merged_competition_level_aligned_v10/rtabmap.db \
  route_path:=/home/qor/depth_ws/routes/network/route_network_segmented.csv
```

Terminal 2:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 run depth_hybrid_slam stop_editor_keyboard
```

Use `a` for ADD, select RViz **Publish Point**, and click within 0.50 m of the
route. Use `r` and click an existing marker to remove an ordinary STOP. A
magenta direction-change STOP is required and cannot be removed. `u` undoes
the last user edit, `l` lists STOPs, `?` prints status, `s` validates and saves,
and `q` exits only the keyboard helper. Press `c` and enter `ALL`/`0`, a case
name (`AAAA`..`BBBB`), or its number (`1`..`16`). `ALL` is the default. With a
specific case selected, the whole network is dimmed and that route is drawn
thick. If an ALL-mode click cannot safely distinguish overlapping A/B branches,
the editor reports `AMBIGUOUS_BRANCH_CLICK:SELECT_CASE`; select a case and click
again.

The saved files are:

```text
/home/qor/depth_ws/routes/network/route_network_segmented_stop_edited.csv
/home/qor/depth_ws/routes/network/route_network_segmented_stop_edited.metadata.yaml
```

The same modes can be selected without the keyboard:

```bash
ros2 topic pub --once /depth_slam/stop_editor/mode std_msgs/msg/String "{data: ADD}"
ros2 topic pub --once /depth_slam/stop_editor/mode std_msgs/msg/String "{data: REMOVE}"
ros2 topic pub --once /depth_slam/stop_editor/case std_msgs/msg/String "{data: ABBA}"
ros2 service call /depth_slam/stop_editor/undo std_srvs/srv/Trigger '{}'
ros2 service call /depth_slam/stop_editor/save std_srvs/srv/Trigger '{}'
```

## Latest real manager with virtual bridge

The serial bridge must remain off. The manager outputs are additionally
remapped under `/prehardware` so even an accidentally running production
bridge cannot receive test commands.

The virtual bridge loads `config/virtual_vehicle.yaml`. Its measured mapping is
stage `1/2/3` = `0.527/0.791/1.055 m/s` for PWM `50/75/100`. The latest MCU
`CURRENT_VALUES.yaml` declares reverse PWM `-50`, so reverse is `-0.527 m/s`.
The same runtime declares `encoder_signed: false` and `797 count/m`: the
counter therefore increases by `abs(ds)*797` in both directions while the
command stage supplies the odometry sign. Wheelbase is 0.730 m and physical
steering remains `+LEFT/-RIGHT` with a ±22° clamp.

Prepare the unmodified latest runtime once:

```bash
mkdir -p /tmp/t870_mcu_0911_v5
unzip -n /home/qor/Downloads/T870_MCU_TEAM_RUNTIME_0911.zip \
  -d /tmp/t870_mcu_0911_v5
cd /tmp/t870_mcu_0911_v5/T870_MCU_최신통합_0911_v5
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Terminal 1, actual manager only (do not start `bridge` or
`t870_mcu.launch.py`):

```bash
cd /tmp/t870_mcu_0911_v5/T870_MCU_최신통합_0911_v5
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=141
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 run t870_mcu manager --ros-args \
  --params-file src/t870_mcu/config/t870_mcu.yaml \
  --remap /mcu/cmd_drive:=/prehardware/mcu/cmd_drive \
  --remap /mcu/cmd_wheel:=/prehardware/mcu/cmd_wheel \
  --remap /mcu/cmd_stop:=/prehardware/mcu/cmd_stop
```

Terminal 2, no branch command means actual A route:

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=141
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch depth_hybrid_slam prehardware_csv_vslam_closed_loop.launch.py \
  route_path:=/home/qor/depth_ws/routes/network/route_network_segmented.csv \
  branch_command:='' start_mode:=1 end_mode:=11
```

For B, use `branch_command:=B`. To use the saved editor result, replace
`route_path` with `route_network_segmented_stop_edited.csv`. Mode ranges such
as only mode 2 or modes 3 through 7 are selected with `start_mode:=2
end_mode:=2` or `start_mode:=3 end_mode:=7`.

Useful monitoring commands:

```bash
ros2 topic echo /depth_slam/stop_editor/status
ros2 topic echo /depth_slam/route/active_branch
ros2 topic echo /depth_slam/route/active_index
ros2 topic echo /depth_slam/route/stop_waypoint_state
ros2 topic echo /slam_wheel
ros2 topic echo /gps_wheel
ros2 topic echo /prehardware/mcu/cmd_wheel
ros2 topic echo /depth_slam/virtual_mcu/state
ros2 topic echo /odom
```

Expected left sign chain is `/slam_wheel=+12`, `/gps_wheel=-12`, and
`/prehardware/mcu/cmd_wheel=+12`. All physical manager/virtual steering is
clamped to ±22 degrees.
