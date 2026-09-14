# Competition CSV + Camera + LiDAR integration

`competition_csv_camera_lidar.launch.py` is the production graph. It contains
no synthetic node, VSLAM, Nav2, Gazebo, camera path generator, or camera
controller. `prehardware_csv_camera_lidar.launch.py` uses the same real D456
and A2M12 inputs with the explicitly acknowledged TEST ONLY ODOM source.

The production graph includes `dual_rplidar.launch.py` and enables one front
and one rear A2M12 owner by default. Set `use_lidar:=false` only when the real
drivers are already owned externally, or `use_rear_lidar:=false` when the rear
unit is not installed. The launch
remaps the system `rplidar_ros` output explicitly to `/front/scan` and
`/rear/scan`; legacy scan aliases are not created.

## Ownership

CSV follower publishes candidate drive/wheel/stop. Camera publishes semantic,
traffic, stop-line and CSV-road diagnostics only. LiDAR consumes canonical
`/front/scan` and `/rear/scan`, then publishes safety and temporary maneuver
candidates. `depth_command_arbiter` alone publishes `/cmd_drive` (`Float32`)
and `/cmd_wheel` (`Int32`, +left/-right, +/-22 degrees).

Priority is hard emergency, mission/branch/LiDAR hold, valid LiDAR temporary
path, valid CSV candidate, then safe stop. Drive stages are exactly
`{-1,0,1,2,3}` and wheelbase is `0.73 m`.

## Steering-aware collision ROI

`/mcu/steer_deg` is the canonical steering feedback. No command-topic fallback
is used while feedback is stale. The front ROI uses `tan(steering)/0.73` to build an Ackermann arc with
a 0.30 m sensing half-width. Clustered returns cause a hard stop through
0.50 m, CSV-preserving drive-stage-1 slowdown through 1.00 m, and slowdown
through 1.50 m only after fixed-frame temporal tracking confirms motion.
Mode 2 disables front safety; rear safety is active only in Modes 7 and 10.

Mode 5 additionally observes 2.0 m, +/-80 degrees, and approximately +/-1.0 m.
The route follower publishes `/depth_slam/route/collision_preview`, containing
only the current cursor's forward, same-segment, same-direction 2 m samples.
LiDAR transforms it into `base_link` using fresh odometry. Only non-curb
clusters intersecting the 0.80 m vehicle width plus 0.15 m margin set
`PATH_BLOCKED`; curbs constrain the planner instead of triggering it.
`/depth_slam/lidar/roi_markers` shows the three zones, clusters, dynamic
returns, broad ROI, CSV sweep, curbs, and selected local path.

Speed-bump suppression is fail-closed. It requires a commissioned
`mode:first_index:last_index` zone, fresh Camera ROAD validation, valid
negative object evidence, and a confirmed STATIC cluster. No production zone
is commissioned because current Camera diagnostics do not supply validated
object evidence. The deployed default suppresses nothing; there is no blanket
2D-LiDAR exemption.

## Mission completion layer

`/depth_slam/route/mode_status` remains route-only. Semantic LiDAR safety and
maneuver events feed a separate tracker publishing `/depth_slam/mission/event`
and `/depth_slam/mission/mode_status`; its `mode_complete` is route complete
AND mission complete. Evidence includes actual `/cmd_drive` zero-time, valid
`/imu/pitch_deg`, intersection exit, two Mode 5 rejoins, parking slot and
parking/rejoin provenance, applied Mode 9 emergency stop, and the Mode 11
five-second stop plus immutable branch commit.

## State contracts

- Camera CSV validation: `VALID_LANE`, `VALID_ROAD_ONLY`, `NEAR_BOUNDARY`,
  `OUTSIDE_ROAD`, `UNKNOWN`. Boundary proximity remains diagnostic-only.
  A fresh `OUTSIDE_ROAD` result holds the arbiter; stale/UNKNOWN evidence
  falls back to CSV and never creates a Camera steering command.
- Mode 4/6/8 intersection: `APPROACH -> STOP_LINE_HOLD -> RELEASE_PENDING ->
  INTERSECTION_COMMITTED -> INTERSECTION_EXITED`. Fresh R holds indefinitely;
  fresh permitted G releases immediately; UNKNOWN/stale releases only after
  3.0 seconds. Valid green has priority when red/yellow is co-active. Mode 8
  explicitly accepts `GREEN_LEFT`, including `R+GREEN_LEFT` and
  `Y+GREEN_LEFT`, as an immediate release.
  `RELEASE_PENDING` returns to the hold on a new R until the bounded CSV cursor
  passes the active segment's STOP_LINE by `0.35 m`. The committed state is
  latched through the end of that CSV mode, so later Camera R is diagnostic
  only and cannot create a mission hold. Mode 2/7/10/11 stop machines are not
  inputs to this gate.
- Mode 5: `CSV_TRACKING -> STOP_FOR_PLANNING -> LIDAR_PATH_TRACKING ->
  CSV_REJOIN -> CSV_TRACKING`. Planner failure stops with a hard obstacle and
  falls back to CSV only after it clears. Quintic detour candidates preserve
  vehicle-width plus obstacle margin and progressively lengthen the transition
  until the generated curvature is at or below the canonical 21-degree
  planning limit. Exhaustion reports `NO_FEASIBLE_DETOUR`; an infeasible path
  is never made drivable by clamping its wheel command.
- Mode 7/10: slot A/B selects the existing branch command. A valid bounded
  parking path owns the temporary command. Parking arcs that exceed the
  canonical 21-degree planning limit are regenerated with a larger radius and
  proportionally longer arc before point integration. Every generated path is
  independently audited against the hard `<22 deg` Ackermann contract
  (`wheelbase=0.73 m`, derived minimum radius `1.901715 m`). The selected
  branch and audited plan are latched before LiDAR suspends the CSV candidate;
  a three-second zero-speed hold separates the CSV/local-path direction
  change. After reaching its endpoint it keeps a second zero-speed LiDAR hold
  for at least three seconds and until the same-segment/forward-window CSV
  rejoin validator accepts distance, body heading, direction, road state and
  required steering (`<=22 deg`). Its bounded distance is `2.5 m`, which
  includes the first steer-feasible T_A/T_B candidate while retaining every
  other lineage and feasibility gate. The accepted index is transferred atomically
  to the CSV follower on ownership release. Failure uses the selected CSV branch; hard
  safety always stops.
- Mode 9: `ACCEL_TRACKING -> EMERGENCY_STOP -> STEERING_CENTERING ->
  CSV_REJOIN -> ACCEL_TRACKING`; enter is `<=0.50 m`, clear is `>=0.60 m`
  for three confirmed scans.
- Mode 11: stop for 5 seconds, commit Camera 1/A or 2/B, default A for
  unknown/stale, then ignore late opposite evidence. CSV mapping is
  A=`END_AA`, B=`END_AB`.

## Intersection route-progress contract

`route_follower` publishes `/depth_slam/route/intersection_progress` from its
existing local, monotonic cursor; it does not run a second/global nearest-point
search. The context carries active mode, segment, direction, assembled-route
index and metric progress, plus the matching CSV `stop_index`, stop progress
and mode exit index/progress. Commit requires all lineage fields to match, the
route index to be strictly beyond the STOP_LINE, and metric progress to exceed
the line by the configured margin. Backtracking evidence is ignored.

For the shipped vforward route, all three intersections use direction `+1`.
Mode 4 and Mode 6 are on `COMMON_1` at CSV point indexes `25` and `554`;
Mode 8 is on `COMMON_2` at point index `397`. In case `AAAA` their assembled
route indexes are `678`, `1207`, and `1957`; a B start shifts those indexes to
`660`, `1189`, and `1939`. Exit is the last waypoint of the current
intersection mode or the observed transition to the following mode.

The mission node also publishes
`/depth_slam/mission/traffic_stop_allowed` and
`/depth_slam/mission/intersection_committed`. The command arbiter has no Camera
traffic subscription: only the pre-commit mission hold can stop it. LiDAR hard
emergency, physical safety/E-stop wiring, publisher conflict and invalid
command fail-safe remain higher-priority stop causes after commit.
