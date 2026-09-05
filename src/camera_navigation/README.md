# camera_navigation

The former projected-view planner, camera geometry, calibration playback, and
their launch paths were removed. No navigation node from this package is
started by the perception validation profile.

`camera_image_path_node` is the original-image-coordinate path generator. It
uses synchronized semantic masks directly in the 640x480 image plane. It does
not use projection, metric coordinates, or publish `nav_msgs/Path`.

The package configuration remains fail-closed for mission integration:
`require_control_mode: true` requires the typed `/mission/control_mode` output
from `course_mission_node`, and only mode 1 enables `CAMERA_PATH_OWNER`.
The standalone `image_path.launch.py` and the full
`camera_pixel_control.launch.py` override that setting to `false`, so the
camera-only pixel stack can run without an external mission-mode publisher.
This changes only the execution gate; path validity and controller safety
gates remain active.

The production hot-path input is one typed
`race_interfaces/SemanticPathFrame` on `/perception/semantic_path_frame`.
It carries lossless binary RLE for the shared refined road, combined refined
white/yellow/unknown lane semantics, and words/stop-line/crosswalk exclusions
from exactly one inference timestamp. It also carries separate refined
white/yellow/unknown masks, restored marking pixels, raw Road, candidate
confidence and refinement diagnostics. Raw YOLO mask topics remain unchanged
for comparison and other consumers. Outputs are:

- `/camera/image_path` (`std_msgs/String` JSON, explicitly `IMAGE_PIXELS`)
- `/camera/image_path_typed` (`race_interfaces/ImagePath`, production contract)
- `/camera/image_path_valid`, `/camera/image_path_confidence`,
  `/camera/image_path_state`
- `/camera/path_ownership`, `/camera/path_metrics`
- `/camera/path_debug_image`
- `/camera/path_overlay_image` (same-stamp RGB background plus final green path)
- `/camera/bev/overlay_image` (same-remap RGB BEV plus semantic/path layers)
- `/camera/candidate/pixel/drive`, `/camera/candidate/pixel/wheel` from the pixel controller
- `/camera_drive`, `/camera_wheel` only from `camera_command_selector_node`

The pixel controller pairs the typed `stop_line_rle` with
`/camera/aligned_depth_to_color/image_raw`. It filters invalid/outlier depth in
the central ROI, subtracts the configurable camera-to-front-bumper offset, and
applies a longitudinal ceiling without replacing path steering. A confirmed
distance at or below 0.7 m latches a zero-drive candidate until a future
behavior layer explicitly releases it; missing depth never creates a new latch.

The pixel controller also consumes `/imu/slope` and `/imu/valid`. A fresh,
valid `False -> True` slope edge starts one monotonic-time stop whose default
duration is `uphill_stop_duration_sec: 5.0`. Valid path steering is preserved
during that longitudinal veto. After the timer completes, the controller
drives again even while slope remains true and re-arms only after a valid false
observation. An invalid IMU cannot create an entry edge. Ordinary stops use a
`0.0` drive candidate; camera_ws does not implement an E-stop topic.

`visualization_only:=true` separates path computation from vehicle ownership:
GPS can remain the mission/control owner while camera pixel-path calculation
and the RQT overlay continue. This mode never publishes vehicle commands.

The canonical path overlay is subscriber-gated and defaults to 45 FPS through
`path_overlay_max_fps`. With no viewer, the node removes its raw RGB
subscription and performs no overlay image copy, drawing, conversion, or
publication. With a viewer, exact-stamp RGB/path pairs enter a latest-only
worker so visualization cannot hold up path publication.

`tools/validate_road_images.py` runs the production TensorRT segmentation,
image planner, and pixel controller against a ZIP file or image directory. It
writes overlays and a JSON report outside the source tree and returns nonzero
when any geometry, temporal-stability, output-range, or fail-safe check fails.

The JSON contains pixel and normalized coordinates. It must not be connected
to a metric controller without a separately reviewed coordinate conversion.

## Lane-free road-center planning

The image planner selects a source independently at every sampled row:
`BOTH_BOUNDARIES`, an observed single boundary constrained by the connected
road width, then `ROAD_CENTER`. Consequently a lane-free but stable,
ego-connected road mask can produce a valid fitted/temporal pixel path.
Near-field road loss, an unconnected far blob, width collapse, non-finite or
discontinuous geometry, and critical branch ambiguity remain fail-closed.

Road edge clipping, near-field coverage/minimum width, local width/center
outliers, branch suspicion, source ratios, containment, and temporal quality
are exposed in `/camera/path_metrics`. Their thresholds are configurable in
`config/image_path.yaml`; they remain pixel/ratio observations and do not
assume an unmeasured metre-per-pixel conversion.

The production pixel controller reuses each typed `ImagePathPoint.source` and
the existing `ImagePath.path_state`; branch-suspected paths are marked degraded.
Confident lane-dominant straight paths may produce a drive candidate of `2.0`, while
valid ROAD_CENTER, sustained single-boundary, temporal-fallback, degraded, or
lower-confidence paths are capped at `1.0`. Invalid or stale paths still publish
zero drive/wheel candidates; steering remains saturated to
`[-22, +22]` degrees.

## Metric reference adapter

The ground-plane projection publishes `/camera/bev/path` as a `nav_msgs/Path`
in `base_link` meters. The commissioned mount is relative to the shared
four-wheel-center `base_link`: `(x, y, z)=(0.015, 0.00, 0.970) m`, with
physical projection `(roll, pitch, yaw)=(0, -5, 0) deg`. This is the same
downward orientation as the T870 REP-103 static TF pitch `+5 deg`; the
projection interface performs the documented sign conversion. Runtime
CameraInfo remains the only
intrinsic source; intrinsic and distortion values are not replaced here.

`metric_path_quality.py` preserves the near-to-far projection order, removes
only consecutive sub-5cm duplicates, trims (without extrapolation) the path at
the first over-4m horizon jump, and rejects non-finite, reversed,
self-intersecting, or shorter-than-1m paths. `/camera/metric_path_status`
reports point count, min/max X, path/forward length, curvature, and the exact
quality result.

`camera_reference_path_adapter_node` consumes `/camera/bev/path`, the MCU mode
on `/mcu/current_mode`, plus an external, timestamp-matched
`odom -> base_link` TF. It is fail-closed until it receives exact mode `"5"`
and publishes only then to `/avoidance/route/reference_path`. It never
subscribes to velocity for integration and never creates odometry. Valid transformed windows are joined
by nearest-point overlap, monotonic overlap indices, heading consistency, tail
continuity, duplicate removal, and jump rejection. It keeps 5m behind the
vehicle, caps the stitched path at 50m, and targets at least 10m forward when
the camera actually supplies it.

Without TF the node remains alive in `waiting_for_tf`, retains any prior
stitched path, and publishes no new reference. Missing or disallowed MCU mode
sets `inactive_mode`; entering or leaving mode `"5"` clears pending and stitched
state so a prior mission segment cannot be reused. A
pending timestamped camera Path is retried for at most 0.5 seconds; it is never
re-stamped or reused after that window. Diagnostics are published on
`/camera/reference_path_adapter_diagnostics`.

Run only the adapter when `/camera/bev/path` already exists:

```bash
ros2 launch camera_navigation camera_reference_path_adapter.launch.py
```

Run the existing metric projector and adapter together when D456, perception,
and pixel path are already active:

```bash
ros2 launch camera_navigation camera_metric_reference.launch.py
```
