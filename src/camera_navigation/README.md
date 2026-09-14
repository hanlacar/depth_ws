# camera_navigation

This package is perception/advisory-only. It retains the D456 semantic-frame
consumer, calibrated ground-plane projection helpers, stop-line/sign/uphill
perception, and YOLO+RGB traffic-light fusion. It publishes no driving path,
drive command, or steering command.

Production executables:

- `camera_mission_perception_node`: consumes `SemanticPathFrame`, aligned
  depth, `CameraInfo`, IMU and optional RGB; publishes stop-line, sign,
  traffic and uphill observations.
- `traffic_light_fusion_node`: combines independent YOLO and RGB observations
  into fresh `R`, `G`, or `UNKNOWN` evidence.

The deleted image-path, BEV-path, controller and command-selector stack was a
camera driving owner and is intentionally not part of the competition graph.
CSV is the only global/reference route. Camera semantics validate that route
through `depth_hybrid_slam/csv_road_validator` and never reshape it.

The canonical D456 mount is
`(x,y,z)=(0.32,0.00,0.85) m`, physical
`(roll,pitch,yaw)=(0,-5,0) deg`, relative to rear-axle `base_link`.
