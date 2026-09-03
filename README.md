# depth_ws — Intel RealSense D456 RGB-D Visual SLAM

ROS 2 Jazzy에서 Intel RealSense D456의 RGB, aligned depth, 통합 IMU와 선택적인 기존 MCU `/odom`을 RTAB-Map에 연결하는 독립 워크스페이스다. 자동주행, 모터 명령, 차선·신호등·주차, Nav2, MCU 수정은 이 저장소 범위가 아니다.

## 2026-09-04 조사 결과와 현재 판정

- OS: Ubuntu 24.04.4 LTS, kernel 7.0.0-30, ROS 2 Jazzy.
- CPU/RAM: Intel Core Ultra 9 275HX (24 logical CPUs), RAM 30 GiB 중 조사 당시 17 GiB 사용.
- GPU: NVIDIA 드라이버와 통신되지 않아 `nvidia-smi` 측정 불가.
- `realsense2_camera`: Debian 패키지 4.58.1 설치됨.
- RTAB-Map: 미설치. apt 후보 버전은 0.22.1이며 아래 설치 명령 필요.
- D456: `rs-enumerate-devices`에서 `No device detected`; USB 목록에도 D456 없음. 내장 Chicony 웹캠만 USB 2.0 480 Mbit/s로 표시됨.
- ROS_DOMAIN_ID=77: `/odom`, `/tf`, `/tf_static` publisher 없음.
- 따라서 코드/정적 테스트 외 실제 스트림 프로파일, 토픽, FPS, latency, drift, 이동 궤적, loop closure, 3D map은 **NOT TESTED**다. 하드웨어 없이 PASS로 판정하지 않는다.

기능 판정: Visual SLAM **PARTIAL PASS (구성·자동 테스트 완료, 하드웨어 미검증)** / RGB 60 FPS **NOT TESTED** / Depth 60 FPS **NOT TESTED** / 출력 60 FPS **NOT TESTED** / 실차 검증 **NOT TESTED**.

## 구성

- `depth_bringup`: D456, RTAB-Map mapping, 저부하 RViz, 전체 bringup launch와 YAML.
- `depth_description`: `base_link -> camera_link` 장착 TF. x=0.32 m, y=0, z=0.85 m, pitch=-5°를 그대로 사용한다.
- `depth_monitor`: 서로 다른 원본 timestamp만 세어 RGB/depth/IMU/VO/map/output FPS, drop, latency, tracking timeout, pose overlay와 표준 진단 토픽을 발행한다.

목표 TF는 `map -> odom -> base_link -> camera_link -> optical frames`다. RGB-D visual odometry는 두 모드 모두 `/visual_odom`에 생성된다. `use_mcu_odom:=true`이면 visual odometry의 TF를 끄고 MCU `/odom`을 RTAB-Map graph 입력으로 쓰므로 MCU만 `odom -> base_link`를 발행한다. RTAB-Map은 `map -> odom`을 발행한다. `false`이면 `/visual_odom`을 graph 입력으로 쓰고 `rgbd_odometry`가 `odom -> base_link`를 발행한다. 두 visual odometry 노드는 launch condition으로 상호 배타적이다. 기존 차량 URDF에 올바른 장착 TF가 있으면 `publish_mount_tf:=false`로 중복을 막는다. RealSense는 `camera_link` 아래 optical TF만 담당한다.

## 설치와 빌드

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
sudo apt update
sudo apt install ros-jazzy-realsense2-camera ros-jazzy-rtabmap-ros \
  ros-jazzy-cv-bridge ros-jazzy-diagnostic-msgs ros-jazzy-tf2-tools \
  ros-jazzy-robot-state-publisher ros-jazzy-xacro python3-pytest python3-yaml
cd ~/depth_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source ~/depth_ws/install/setup.bash
```

모든 새 터미널에서 다음 환경을 설정한다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

## D456 프로파일을 먼저 확정

D456를 USB 3.x 포트에 직접 연결한 뒤 다음을 실행한다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd ~/depth_ws
./scripts/inspect_d456.sh | tee d456_inspection.txt
```

`lsusb -t`의 D456 경로가 `5000M` 이상인지, `rs-enumerate-devices`가 표시한 Color/Depth/Accel/Gyro 프로파일에 선택한 조합이 실제 존재하는지 확인한다. 현재 기본값은 color `640,480,60`, depth `640,480,30`인 보수적 시작값일 뿐 D456 실측 지원 주장이나 60 FPS 판정이 아니다. 장치가 출력한 프로파일만 launch 인자로 지정한다. `gyro_fps:=0 accel_fps:=0`은 드라이버의 유효 기본 프로파일 선택을 요청한다.

단독 실행 예:

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_bringup d456_60fps.launch.py \
  serial_no:="'_YOUR_SERIAL_'" color_profile:=640,480,60 depth_profile:=640,480,30
```

장치가 depth `640,480,60`을 실제로 열 수 있다고 열거한 경우에만 `depth_profile:=640,480,60`으로 바꾼다. IR, point cloud, colorizer와 고비용 필터는 기본 비활성화하며 sync와 depth-to-color alignment는 활성화한다. aligned depth는 CPU 부하와 실제 FPS를 비교해 `align_depth:=false`와 별도 시험할 수 있다.

## Visual SLAM 실행

기존 MCU `/odom`과 `odom -> base_link` TF가 있을 때:

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_bringup visual_slam_full.launch.py \
  use_mcu_odom:=true publish_mount_tf:=true start_rviz:=false
```

MCU odom이 없을 때 D456 단독 visual odometry:

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_bringup visual_slam_full.launch.py \
  use_mcu_odom:=false publish_mount_tf:=true start_rviz:=false
```

기존 URDF가 `base_link -> camera_link`를 이미 발행하면 반드시 `publish_mount_tf:=false`로 실행한다. 실제 토픽명이 기본 remap과 다르면 카메라 실행 후 `ros2 topic list -t`로 확인하고 `visual_slam_mapping.launch.py`의 `rgb_topic`, `depth_topic`, `camera_info_topic`, `imu_topic`, `odom_topic` 인자를 전달한다.

RViz는 전체 처리율에 영향을 주므로 별도 실행을 권장한다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_bringup visual_slam_view.launch.py use_rtabmap_viz:=true
```

## 출력과 측정

예상 카메라 입력은 `/camera/camera/color/image_raw` (`sensor_msgs/msg/Image`), `/camera/camera/color/camera_info` (`sensor_msgs/msg/CameraInfo`), `/camera/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/msg/Image`), `/camera/camera/imu` (`sensor_msgs/msg/Imu`)다. 이는 wrapper 규칙에 따른 기본 remap이며 실제 하드웨어에서 아직 확인되지 않았다. gyro/accel 개별 토픽과 RealSense TF도 연결 후 기록해야 한다.

진단 출력은 다음과 같다.

- `/depth_slam/status` — `std_msgs/msg/String`
- `/depth_slam/{rgb_fps,depth_fps,imu_fps,visual_odom_fps,map_update_fps,output_fps,latency_ms}` — `std_msgs/msg/Float32`
- `/depth_slam/dropped_frames` — `std_msgs/msg/UInt64`
- `/depth_slam/tracking_valid` — `std_msgs/msg/Bool`
- `/depth_slam/diagnostics` — `diagnostic_msgs/msg/DiagnosticArray`
- `/depth_slam/output_image` — `sensor_msgs/msg/Image`

출력 영상 subscriber가 없으면 변환/overlay를 수행하지 않는다. subscriber가 있으면 queue depth가 작은 SensorData QoS로 최신 RGB만 처리하고 원본 header timestamp를 보존한다. 같은 timestamp의 재발행은 거부한다. 더 낮은 부하는 `overlay_enabled:=false`를 사용한다.

실측 명령:

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic hz --window 300 /camera/camera/color/image_raw
ros2 topic hz --window 300 /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz --window 300 /camera/camera/imu
ros2 topic hz --window 300 /odom
ros2 topic hz --window 300 /visual_odom
ros2 topic hz --window 300 /depth_slam/output_image
ros2 topic echo /depth_slam/diagnostics
```

TF/QoS/타입 전체 점검:

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd ~/depth_ws
./scripts/inspect_runtime.sh | tee runtime_inspection.txt
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link camera_link
ros2 topic info -v /tf
ros2 topic info -v /tf_static
```

## rosbag, DB와 안전한 초기화

기본 DB는 `~/depth_ws/maps/rtabmap.db`이며 Git에서 제외된다. 기존 DB는 기본적으로 삭제하지 않는다. 새 mapping이 필요하면 먼저 다른 이름으로 보존한다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
mkdir -p ~/depth_ws/bags
ros2 bag record -o ~/depth_ws/bags/d456_mapping \
  /camera/camera/color/image_raw /camera/camera/color/camera_info \
  /camera/camera/aligned_depth_to_color/image_raw /camera/camera/imu /odom /tf /tf_static
```

새 DB를 명시적으로 만들 때만 다음처럼 별도 경로와 `delete_db:=true`를 함께 쓴다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch depth_bringup visual_slam_full.launch.py \
  database_path:=~/depth_ws/maps/new_mapping.db delete_db:=true
```

종료는 launch 터미널에서 한 번 `Ctrl-C`를 누르고 모든 노드가 종료될 때까지 기다린다.

## 검증 절차와 PASS 기준

1. 정지 60초: RGB/depth/IMU FPS, pose drift, TF 오류, CPU/RAM을 기록한다.
2. 사용자가 수동으로 천천히 전진·좌회전·우회전·복귀한다. 모터 명령은 이 WS가 발행하지 않는다.
3. trajectory 방향, 3D cloud 형상, pose jump, loop closure log를 확인한다.
4. 전체 노드와 RViz를 3분 이상 실행하고 평균/최소 FPS, latency 평균/p95, drop ratio, CPU/RAM 증가를 기록한다.
5. RGB와 출력은 평균 55 FPS 이상일 때만 60 FPS PASS로 판정한다. SLAM 처리율과 영상 출력률은 별개로 보고한다. Depth 60이 장치 프로파일에 없으면 `UNSUPPORTED`다.

Visual SLAM PASS에는 실제 RGB/depth/IMU, pose 변화, 3D 지도, 정상 TF, publisher 중복 없음, 정상 종료와 DB 저장이 모두 필요하다. 현재는 D456와 차량 이동 데이터가 없으므로 이 기준을 충족했다고 주장하지 않는다.

## 자동 테스트

```bash
source /opt/ros/jazzy/setup.bash
source ~/depth_ws/install/setup.bash
export ROS_DOMAIN_ID=77
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd ~/depth_ws
python3 -m compileall -q src
python3 -m pytest -q
PYTHONNOUSERSITE=1 colcon test --event-handlers console_direct+
colcon test-result --verbose
```

이 호스트의 사용자 영역 `setuptools 81.0.0`은 ROS/colcon이 사용하는 `tests_require` 메타데이터를 제거하므로 테스트 명령에 `PYTHONNOUSERSITE=1`을 쓴다. 시스템 `setuptools 68.1.2`에서는 패키지별 pytest가 정상 검색된다.

## 문제 해결

- `No device detected`: 전원/케이블을 확인하고 D456를 USB 3.x에 직접 연결한다. 허브를 제거하고 udev rules/사용자 권한을 확인한다.
- profile start failure: 추정값을 쓰지 말고 `rs-enumerate-devices`가 해당 serial에 표시한 정확한 조합으로 바꾼다.
- RTAB package not found: 위의 `ros-jazzy-rtabmap-ros`를 설치하고 다시 source/build한다.
- IMU sync 경고: 실제 gyro/accel rate와 `/camera/camera/imu` 존재를 확인하고 `unite_imu_method` 및 QoS를 점검한다.
- RGB/depth sync failure: aligned depth와 color header stamp, SensorData QoS compatibility, USB 속도를 확인한다.
- TF conflict: `/tf`와 `/tf_static`의 publisher GID를 확인한다. MCU TF 사용 시 `use_mcu_odom:=true`; 기존 장착 TF 사용 시 `publish_mount_tf:=false`다.
- 렉: RViz를 끄고(`start_rviz:=false`) 먼저 입력/VO를 계측한다. RTAB-Map의 assembled `/rtabmap/cloud_map`은 subscriber가 있을 때만 계산되며, overlay subscriber가 없는 경우 monitor는 영상 변환을 생략한다.
- NVIDIA 오류: 현재 드라이버가 GPU와 통신하지 않는다. 이 구성은 CPU 동작을 기본으로 하며 GPU 가속을 주장하지 않는다.
