# D455 camera-only real-sensor validation — 2026-09-14

Result: **BLOCKED / D455_CAMERA_STAGE_PASS not declared**

This report contains only measurements actually performed with one physically
connected Intel RealSense D455. No LiDAR node or synthetic sensor/vehicle
publisher was run. The requested moving A-route test could not be started
without violating the workspace's TEST ONLY ODOM contract: the only approved
test odometry owner requires fresh real MCU encoder and steering data.

## A. Workspace

```text
HEAD: 8b989131097c9789ebdd775644195d9cf14d8d4b
branch: main
worktree before this report: clean
worktree after this report: reports/d455_camera_real_sensor_2026-09-14.md only
```

`README.md` and `FINAL_SENSOR_CONTRACT.md` were read before starting hardware.
Installed packages and current source/launch/executable names were enumerated.

## B. D455

```text
Model: Intel RealSense D455
Serial: 248622300861
ASIC serial: 302623061802
Firmware: 5.17.0.10
RealSense ROS: 4.58.1
LibRealSense build/runtime: 2.58.1 / 2.58.1
USB mode: 3.2 (PASS; no USB 2 fallback)
RGB sensor: present
RGB profile: 640x480 RGB8 @ 60 Hz
Actual RGB Hz: 59.771 Hz (15-19 ms, stddev 0.39 ms, final 841-message window)
Depth sensor: present
Depth profile: 640x480 Z16 @ 60 Hz, aligned to color
Actual aligned depth Hz: 59.764 Hz (15-18 ms, stddev 0.35 ms, final 841-message window)
Gyro: present, configured 200 Hz, measured 200.052 Hz
Accel: present, configured 100 Hz, measured 100.688 Hz
IMU type: BMI085
```

The kernel log contained no RealSense USB reset, disconnect, reconnect,
bandwidth, UVC, or xHCI error during the inspected interval. The driver did,
however, report one `Motion Module failure` hardware notification at startup,
an IMU calibration-unavailable warning, and repeated
`Asic Temperature value is not valid!` errors. Gyro and accel were still
publishing at the measured rates after the startup warning. These warnings
remain unresolved and prevent an unqualified whole-device stability PASS.

Raw RGB was displayed with `rqt_image_view`. Three real frames were also
captured at distinct timestamps and hashes. A captured frame showed normal
orientation and usable exposure. Continuous human confirmation of no visible
stutter was not explicitly recorded, so that visual sub-check is not promoted
beyond the measured 59.771 Hz timing evidence.

D455 CameraInfo was emitted by the device, not copied into production files:

```text
frame_id: camera_color_optical_frame
width x height: 640 x 480
distortion_model: plumb_bob
fx, fy: 385.4229431152344, 385.03912353515625
cx, cy: 323.7796630859375, 246.8297119140625
```

## C. Actual commands used

Commands below were actually run. Every ROS terminal sourced the Jazzy,
workspace, and checked-in ROS network setup where shown.

### Terminal 1 — workspace/source contract inspection

```bash
cd /home/qor/depth_ws
git rev-parse HEAD
git branch --show-current
git status --short
cat README.md
cat FINAL_SENSOR_CONTRACT.md
find src -type f \( -name '*.launch.py' -o -name '*.yaml' \) | sort
find src -maxdepth 3 -type f | sort
rg -n '/camera/image_raw|/camera/camera_info|traffic|GREEN_LEFT|mode.{0,12}11|command_arbiter|TEST ONLY ODOM' src README.md FINAL_SENSOR_CONTRACT.md
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 pkg list | grep -E 'camera|realsense|yolo|traffic|imu|route|depth'
ros2 pkg executables camera_bringup
ros2 pkg executables camera_navigation
ros2 pkg executables camera_rgb_traffic_light
ros2 pkg executables camera_yolo_inference
ros2 pkg executables depth_hybrid_slam
```

Focused `sed`/`rg` reads were also run against the current camera launch/config,
traffic fusion, mission, Mode 11, command arbiter, prehardware launch,
closed-loop probe, and TEST ONLY ODOM source and tests.

### Terminal 2 — physical USB/device and kernel inspection

```bash
lsusb
command -v rs-enumerate-devices
command -v realsense-viewer
command -v rs-fw-update
rs-enumerate-devices
ls -l /dev/video* /dev/v4l/by-id/*
journalctl -k --since '10 minutes ago' --no-pager | tail -n 200
journalctl -k --since '30 minutes ago' --no-pager | rg -i 'usb|uvc|xhci|realsense|reset|disconnect|bandwidth|error'
```

The first sandboxed USB attempt failed with libusb `-99`/udev-monitor access;
the same read-only commands were rerun with host device permission and produced
the D455 results above.

### Terminal 3 — D455-only RealSense driver

```bash
cd /home/qor/depth_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source tools/ros_network_env.sh
ros2 launch camera_bringup d456_bringup.launch.py \
  serial_no:=248622300861 color_width:=640 color_height:=480 color_fps:=60
```

Despite its historical D456 filename, this official wrapper takes a launch-only
serial override and obtains CameraInfo from the attached device. The production
mount TF launch was not used.

### Terminal 4 — sensor topics and rates

```bash
ros2 node list
ros2 topic list -t
ros2 topic info -v /camera/image_raw
ros2 topic info -v /camera/camera_info
ros2 topic info -v /camera/aligned_depth_to_color/image_raw
ros2 topic info -v /camera/camera/gyro/sample
ros2 topic info -v /camera/camera/accel/sample
timeout 15 ros2 topic hz /camera/image_raw
timeout 15 ros2 topic hz /camera/aligned_depth_to_color/image_raw
timeout 8 ros2 topic hz /camera/camera/gyro/sample
timeout 8 ros2 topic hz /camera/camera/accel/sample
timeout 8 ros2 topic echo /camera/camera_info --once
python3 src/camera_yolo_inference/tools/capture_image_samples.py \
  --topic /camera/image_raw --output /tmp/d455_raw_2026-09-14 \
  --count 3 --interval 2 --timeout 10
```

### Terminal 5 — live RGB and detector viewers

```bash
# This direct invocation was attempted and failed: command not found.
rqt_image_view /camera/image_raw

# Successful invocations:
ros2 run rqt_image_view rqt_image_view /camera/image_raw
ros2 run rqt_image_view rqt_image_view /camera/traffic_light_rgb/overlay_image
```

### Terminal 6 — YOLO, RGB traffic detector, and fusion only

```bash
ros2 launch camera_navigation \
  d456_yolo_rgb_traffic_light_fusion_validation.launch.py \
  launch_camera:=false launch_yolo:=true launch_rgb:=true \
  launch_fusion:=true launch_rqt:=false
```

The launch selected the checked-in YOLO checkpoint on `cuda:0` (NVIDIA
GeForce RTX 5060 Laptop GPU). No mission, route, arbiter, LiDAR, or vehicle
node was launched.

### Terminal 7 — perception graph and samples

```bash
ros2 node list
ros2 topic list -t | sort
ros2 node info /camera_yolo_inference_node
ros2 node info /rgb_traffic_light_node
ros2 node info /traffic_light_fusion_node
timeout 8 ros2 topic echo /perception/detections_json --once
timeout 8 ros2 topic echo /camera/traffic_light_rgb/diagnostics --once
timeout 8 ros2 topic echo /camera/traffic_light_fused/diagnostics --once
timeout 8 ros2 topic echo /camera/traffic_light_fused/state --once
timeout 8 ros2 topic echo /camera/traffic_light_fused/aspect --once
timeout 8 ros2 topic echo /camera/traffic_light_fused/confidence --once
python3 /tmp/d455_signal_observer.py --duration 8
```

The last observer is a temporary read-only subscriber. It did not publish a
camera, signal, odometry, or control topic. Annotated/overlay evidence was
captured with the checked-in `capture_image_samples.py` tool under `/tmp`.

### Terminal 8 — protected production-file evidence

```bash
git status --short
sha256sum src/camera_bringup/config/d456.yaml \
  src/camera_bringup/config/camera_mount.yaml \
  src/depth_hybrid_slam/config/d456_60hz.yaml
```

```text
5616dd94e6333768c958d14b7fc4493e948422e32a3fd40aec643435dea60949  d456.yaml
8a00f4d074f2ca7a6150e1c98f5607ac34cada4a94e6a921b86e41e8373baf45  camera_mount.yaml
2d2010aa9e08a4cfd94b2a58cfb9a86df77bd926d11a033385553b1a078e0d1d  d456_60hz.yaml
```

No route/HIL/RViz launch command was executed because it could not produce
motion without real MCU input and the camera+route HIL launch would also start
the prohibited front LiDAR.

## D. Production topic contract

```text
/camera/image_raw: PASS — Image, camera_color_optical_frame, publisher 1
/camera/camera_info: PASS — CameraInfo, camera_color_optical_frame, publisher 1
/camera/aligned_depth_to_color/image_raw: PASS — Image, publisher 1, 59.764 Hz
/camera/camera/gyro/sample: PASS WITH HARDWARE WARNING — Imu, publisher 1, 200.052 Hz
/camera/camera/accel/sample: PASS WITH HARDWARE WARNING — Imu, publisher 1, 100.688 Hz
```

The RGB and CameraInfo public remaps matched the production contract. No
production D456 topic, serial, intrinsic, or mount configuration was changed.

## E. Perception

Actual graph:

```text
/camera/image_raw
  -> camera_yolo_inference_node
     -> /perception/detections_json
     -> /camera_traffic_light
     -> /camera/debug/annotated
  -> rgb_traffic_light_node
     -> /camera/traffic_light_rgb/state
     -> /camera/traffic_light_rgb/aspect
     -> /camera/traffic_light_rgb/confidence
     -> /camera/traffic_light_rgb/diagnostics
     -> /camera/traffic_light_rgb/overlay_image
YOLO + RGB
  -> traffic_light_fusion_node
     -> /camera/traffic_light_fused/state
     -> /camera/traffic_light_fused/aspect
     -> /camera/traffic_light_fused/confidence
     -> /camera/traffic_light_fused/diagnostics
```

```text
RED:
  Detection: RGB detector PASS; YOLO detected R_light but also image/UI false positives
  Fusion: FAIL (unstable/UNKNOWN)
  Stability: best single-light run was RGB RED 477/477 with 0 transitions;
             YOLO state R 281/281; fusion RED 53/160 and UNKNOWN 107/160,
             74 fusion transitions. RGB confidence mean 0.7735.

YELLOW:
  Detection: NOT TESTED
  Fusion: NOT TESTED
  Stability: NOT TESTED

GREEN:
  Detection: NOT TESTED
  Fusion: NOT TESTED
  Stability: NOT TESTED

UNKNOWN:
  Detection: PASS for the initial real no-signal frame (YOLO detections [])
  Fusion: PASS for sampled state/aspect UNKNOWN, confidence 0.0
  Stability: NOT MEASURED over a dedicated timed interval

GREEN_LEFT:
  Detection: NOT TESTED
  Fusion: NOT TESTED
  Stability: NOT TESTED

Other source-supported aspects (RED_X, GREEN_CIRCLE, GREEN_DOWN,
GREEN_OTHER): NOT TESTED
```

The first RED road image contained multiple real red signals. YOLO and the RGB
detector selected different boxes, producing `positions_match:false` and
`fusion_reason:CONFLICT`; fusion remained UNKNOWN for 160/160 samples. After
cropping to one main signal, a laptop taskbar icon was detected by YOLO as a
low-confidence `G_light` (~0.26-0.30). The fusion correctly failed closed but
oscillated between RED and UNKNOWN. No threshold or production default was
changed.

The user then redirected the test from stationary repeated-aspect validation
to an A-route run where R/G would be shown during motion. That requested motion
could not be started under the no-MCU/no-synthetic-runtime constraints.

Camera mission perception and its mission-consumer topics were not launched,
because the revised request limited camera use to traffic signals and the
route test was blocked before mission integration.

## F. Route/vehicle

```text
TEST ONLY ODOM: NOT RUN
odom -> base_link: NOT AVAILABLE
RViz vehicle motion: NOT TESTED
CSV following: NOT TESTED
abnormally long stop: NOT TESTED
A branch full-route run: BLOCKED
```

The checked-in `test_odom_publisher` is the sole approved test odometry owner.
It subscribes to real `/mcu/encoder` and `/mcu/steer_a0`, and publishes `/odom`
plus `odom -> base_link` only while both are fresh. `/cmd_drive` supplies only
direction and cannot create distance. `csv_only_closed_loop_probe` publishes
only branch requests and observes odometry/encoder/commands; it is not a
motion source. With the real MCU excluded, no official moving bench odometry
exists.

The existing `prehardware_csv_camera_lidar.launch.py` is also unsuitable for
this stage because it unconditionally includes the front LiDAR driver and
LiDAR perception path. No fake `/front/scan` or `/rear/scan` was published.

## G. Mode tests

Current-source policy was inspected before testing. Modes 4, 6, and 8 use the
route-progress-latched intersection gate. Fresh red/yellow holds; permitted
green releases; unknown holds for three seconds then releases; after crossing
the commit boundary a later red cannot re-stop the vehicle. `GREEN_LEFT` is a
special permitted aspect only in Mode 8. Mode 11 directly observes a fixed
image ROI for five seconds and selects GREEN -> A, RED -> B, with insufficient
valid evidence defaulting to A. Branch selection then commits and ignores late
opposite signals.

```text
Mode 4 intersection: NOT TESTED
Mode 6 intersection: NOT TESTED
Mode 8 non-GREEN_LEFT hold: NOT TESTED
Mode 8 GREEN_LEFT: NOT TESTED
Mode 8 post-release CSV following: NOT TESTED
Mode 11 GREEN -> A: NOT TESTED
Mode 11 RED -> B: NOT TESTED
Mode 11 UNKNOWN -> default A: NOT TESTED
Mode 11 5-second observation/stop: NOT TESTED
Mode 11 selected branch actual following: NOT TESTED
```

## H. Response time

```text
RED -> stop: NOT MEASURED
GO signal -> departure: NOT MEASURED
GREEN_LEFT -> mission transition: NOT MEASURED
Mode 11 branch decision: NOT MEASURED
```

## I. Discovered issues

1. **No moving TEST ONLY ODOM under the stated no-MCU constraint**
   - Cause: approved odometry integrates only real encoder/steering samples.
   - Impact: A-route/RViz/mode/command response tests cannot run.

2. **No camera-only full-route launch**
   - Cause: the formal camera+route HIL launch unconditionally includes front
     LiDAR; the CSV-only launch still requires real MCU feedback.
   - Impact: starting the formal full HIL launch would violate this test scope.

3. **RED fusion instability for the displayed laptop content**
   - Cause: multiple signals first caused bbox mismatch; later a taskbar icon
     was misclassified as `G_light`, activating green-over-red safety logic.
   - Impact: fusion oscillated RED/UNKNOWN and cannot yet be used as stable
     mission evidence for this display setup.

4. **D455 motion/temperature warnings**
   - Cause: not established. Driver reported unavailable IMU calibration,
     one Motion Module hardware failure, and invalid ASIC temperature values.
   - Impact: RGB/depth/IMU topics remained live, but whole-device stability is
     not cleanly passing.

## J. Modifications

```text
Production source/config changes: none
D455 production serial persistence: none
D456 intrinsic overwrite: none
D456 mount overwrite: none
Synthetic publishers added/restored: none
Temporary observer: /tmp/d455_signal_observer.py (read-only subscriber)
Report added: reports/d455_camera_real_sensor_2026-09-14.md
```

No perception threshold, fusion safety rule, route alignment evidence, mission
gate, or command ownership policy was changed.

## K. Unresolved items

- Provide fresh real MCU encoder and steering data with wheels safely lifted,
  or explicitly change the prohibition on a motion source. Without one of
  those choices, the requested A-route vehicle motion is impossible.
- If continuing the real-screen detector test, use a fullscreen image without
  browser chrome/taskbar and revalidate R/G fusion before using it for stops.
- Investigate the D455 Motion Module/ASIC-temperature warnings before an
  unqualified sensor-stage PASS.
- YELLOW, GREEN, GREEN_LEFT, camera-to-mission integration, Modes 4/6/8/11,
  RViz movement, branch following, response times, and command ownership in a
  running control graph remain NOT TESTED.

Final declaration: **D455_CAMERA_STAGE_PASS NOT DECLARED**.
