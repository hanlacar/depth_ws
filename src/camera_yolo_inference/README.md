# camera_yolo_inference

This is the official D456 RGB YOLO-Seg inference package. The independent
`race_perception/yolo_camera_node.py` implementation is legacy and is not
included by this package's launch files.

## Perception-only validation

```bash
ros2 launch camera_yolo_inference d456_yolo_perception.launch.py
```

The default profile starts exactly one D456 process and the inference node on
`cuda:0`. CUDA is required and there is no automatic CPU fallback. Optional
depth helpers and RQT are disabled by default. Start RQT with
`launch_rqt:=true`, or start the camera separately and use
`launch_camera:=false`.

The launch defaults every child process to `rmw_cyclonedds_cpp`, the transport
validated for sustained 640x480 RGB8 at 60 FPS on this production host. When
the camera is launched separately, use the same RMW for both processes.

The canonical competition validation output is
`/camera/perception_overlay_image`. It uses the exact source RGB stamp,
current semantic colors, a compact active-class legend, and W/Y/STOP presence
flags. Rendering is subscriber-gated, latest-only, asynchronous, and limited
to `perception_overlay_max_fps` (45 FPS by default), independently of inference.
`/perception/detections_image` remains the detailed development output.

The node also publishes all 11 semantic masks under `/perception/masks/*`, the
three compatibility masks under `/camera/*_mask`, detections JSON, inference
latency/status, `/camera/perception_valid`, and the distinct
`/camera/navigation_mask_available` signal.

The `R_light` and `G_light` segmentation instances also feed a temporal state
filter published on `/camera_traffic_light` as `R`, `G`, or the explicit safe
`UNKNOWN` state. Physical yellow and red lamps are both trained as `R_light`
and therefore both mean stop. Confidence, mask area, recent
confirmed state, N-frame confirmation, and a lost-detection timeout prevent a
single-frame or ambiguous multi-color detection from forcing a state change.
This output is observational only and is not connected to vehicle commands.

## TensorRT deployment profile

The NVIDIA production profile defaults to the static, batch-one, 640-pixel
FP16 TensorRT engine in `models/`. INT8, dynamic shapes, CPU fallback, and
automatic `.pt` fallback are disabled. The engine is an RTX 5060 deployment
artifact and should be rebuilt for a materially different GPU/runtime.

Create an engine explicitly (never during ROS node startup):

```bash
python3 tools/export_tensorrt_engine.py \
  --model models/hanla_yolo11n_seg_best.pt \
  --output models/hanla_yolo11n_seg_best.engine
```

The adjacent `.engine.json` records the source hash, toolchain, GPU, profile,
classes, and creation time. Compare backends and collect pure-backend latency:

```bash
python3 tools/benchmark_backends.py --help
```

`/camera/inference_latency_ms` is the model backend's reported inference stage
only; it excludes image conversion, decode, semantic-mask processing,
rendering, JSON, and ROS publication. `/camera/pipeline_latency_ms` is the full
per-frame pipeline wall time. `/camera/performance_diagnostics` publishes a
periodic JSON summary with bounded-window stage statistics, observed processing
FPS, and overwritten latest-frame count.

The node keeps the real-camera input default at sensor-data BEST_EFFORT QoS.
Saved-video benchmarks may set `input_reliability:=reliable` together with a
RELIABLE publisher to avoid local DDS fragmentation loss; publisher and
subscriber reliability must be changed as a pair.

Road, white-line, and yellow-line masks are always generated and published on
both their semantic and compatibility topics. Other full-resolution semantic
mask messages are subscriber-aware by default and are serialized only while a
subscriber exists; set `publish_optional_masks:=true` to force all 12 topics.
All topic names and message types remain available.
