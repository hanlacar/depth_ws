# Latest T870 MCU integration contract (0911 v5 ZIP)

The audited deployment is `/home/qor/Downloads/T870_MCU_TEAM_RUNTIME_0911.zip`.
It is a separate ROS runtime; depth_ws never imports its Python package and
communicates only through ROS topics.

## Production command flow

`route_follower` publishes `/depth_slam/follower/candidate_*`. The behavior
selector applies localization, local-path, branch, mission and CSV-road gates
and is the sole publisher of the internal `/slam_drive`, `/slam_wheel`,
`/slam_stop` triplet. `depth_slam_mcu_source_adapter` validates freshness,
stages and limits, converts steering at the boundary, and publishes
`/gps_drive`, `/gps_wheel`, `/gps_stop`. The external `mcu_manager` alone owns
`/mcu/cmd_drive`, `/mcu/cmd_wheel`, `/mcu/cmd_stop`; `mcu_bridge` sends these to
Arduino.

The adapter and route follower are disarmed by default. Production requires
all three follower gates plus `enable_mcu_adapter:=true`.

The camera command selector solely owns `/camera_drive`, `/camera_wheel` and
`/camera_stop`. The depth LiDAR source solely owns `/lidar_drive`,
`/lidar_wheel`, `/lidar_stop` plus the current manager's avoidance/path-safety
signals. Both adapters invert the internal physical steering sign exactly once
at the MCU team-topic boundary.

## Pre-hardware route and behavior flow

The branch selector defaults to A, accepts synthetic/production A or B on
`/depth_slam/route/branch_command`, and makes the selected branch alter the
actual segmented route loaded by the follower. The route follower publishes a
monotonic active index; the local-path connector extracts 0.5--4.0 m ahead and
transforms map coordinates to `base_link` without a global nearest search.
The resulting path feeds the CSV-road validator and its state feeds the
behavior selector. Camera-unavailable/invalid-geometry states permit an
explicit unverified CSV fallback; road-containment and path/localization
failures stop.

## Mode ownership from current mode_policy.py

| mode | drive owner | wheel owner | special override |
|---:|---|---|---|
| 0 | camera if authority, else GPS | same | obstacle requires fresh safe GPS |
| 1 | camera if authority, else GPS | same | obstacle requires fresh safe GPS |
| 2 | camera, then GPS fallback | same | slope policy |
| 3 | camera if authority, else GPS | same | obstacle requires fresh safe GPS |
| 4 | camera zero-stop or GPS | GPS | mission-authorized intersection stop |
| 5 | LiDAR if avoidance active, else camera/GPS | same | avoidance switch |
| 6 | camera zero-stop or GPS | GPS | mission-authorized intersection stop |
| 7 | LiDAR | LiDAR | parking |
| 8 | camera if authority, else GPS | same | obstacle requires fresh safe GPS |
| 9 | LiDAR, optionally limited by camera | camera then GPS | LiDAR status/stop hard gate |
| 10 | LiDAR | LiDAR | parking |
| 11 | camera if authority, else GPS | same | obstacle requires fresh safe GPS |

A fresh manual drive+wheel pair overrides every row. Any fresh source stop or
`/estop_lock` overrides ownership and stops the vehicle.

## Steering boundary

| layer | left | right |
|---|---:|---:|
| depth_ws controller and `/slam_wheel` | positive | negative |
| MCU team camera/LiDAR/GPS topics | negative | positive |
| manager internal and `/mcu/cmd_wheel` | positive | negative |
| manual topic | positive | negative |

## Odom and TF ownership

The latest MCU bridge publishes `/odom` and `odom -> base_link`. The separate
optional MCU `vehicle_tf.launch.py` publishes `base_link` sensor transforms;
the default MCU launch intentionally does not start it. The depth_ws
`production_ready.launch.py` starts neither odometry nor static vehicle TF, and
the cuVSLAM camera mount TF is opt-in with
`publish_camera_mount_tf:=true` (default `false`). Production must start the MCU
vehicle TF launch and leave that depth_ws argument false. A standalone
depth_ws-only cuVSLAM session can instead opt in to the camera mount publisher.
Mapping/test launches that create visual odometry or synthetic odometry are not
production launches and must not run alongside the MCU bridge on the same topic.
The cuVSLAM component explicitly sets `publish_odom_to_base_tf=false` and
`publish_map_to_odom_tf=false`; it retains its tracking Odometry topic while
the MCU bridge and RTAB-Map own those two TF edges.

The vehicle TF uses `base_link -> camera_link z=0.835 m`. Camera metric
projection uses optical-centre height `0.970 m` above the ground plane; these
are consistent because `ground_center` is `z=-0.135 m` from `base_link`.
