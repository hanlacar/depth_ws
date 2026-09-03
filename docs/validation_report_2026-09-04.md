# D456 RGB-D Visual SLAM 검증 보고서 — 2026-09-04

1. **최종 판정** — Visual SLAM: **PARTIAL PASS**(구현·빌드·비하드웨어 시험 완료, RTAB-Map/D456 미연결), RGB 60 FPS: **NOT TESTED**, Depth 60 FPS: **NOT TESTED**, 출력 60 FPS: **NOT TESTED**, 하드웨어 실차: **NOT TESTED**.
2. **GitHub 저장소 상태** — 원격 `origin/main`에 ref가 없는 빈 저장소를 `/home/qor/depth_ws`에 clone했다. 최종 commit/push 결과는 작업 종료 시 갱신한다.
3. **생성·수정 파일** — `.gitignore`, `README.md`, `maps/.gitkeep`, `scripts/inspect_d456.sh`, `scripts/inspect_runtime.sh`, `src/depth_bringup/{package.xml,setup.py,setup.cfg,resource/depth_bringup,launch/*.launch.py,config/*,test/test_configuration.py}`, `src/depth_description/{package.xml,CMakeLists.txt,launch/d456_mount.launch.py,urdf/d456_mount.urdf.xacro}`, `src/depth_monitor/{package.xml,setup.py,setup.cfg,resource/depth_monitor,depth_monitor/*.py,test/test_metrics.py}`, `test/test_workspace.py`, 본 보고서.
4. **패키지/버전** — Ubuntu 24.04.4, ROS Jazzy, librealsense 2.58.1, realsense2_camera 4.58.1, rmw_fastrtps_cpp 8.4.4, RViz2 14.1.22. RTAB-Map은 미설치이고 apt 후보는 0.22.1이다.
5. **D456 지원 스트림 프로파일** — 장치가 연결되지 않아 `rs-enumerate-devices`가 `No device detected`를 반환했다. 해상도/FPS 및 gyro/accel rate는 **NOT TESTED**이며 추정으로 확정하지 않았다.
6. **실제 입력 토픽/타입** — ROS domain 77에는 조사 당시 `/parameter_events`, `/rosout`만 있었고 카메라 및 `/odom`은 없었다. launch 기본 remap은 README에 별도로 명시했으나 실측 토픽이라고 주장하지 않는다.
7. **실제 출력 토픽/타입** — 합성 시험에서 `/depth_slam/diagnostics`(`diagnostic_msgs/msg/DiagnosticArray`) 수신을 확인했다. 나머지 선언 출력과 타입은 README에 있다. 하드웨어 데이터 출력은 미시험이다.
8. **TF/publisher** — domain 77의 `/tf`, `/tf_static` publisher는 없었다. `depth_description` 단독 시험에서 `d456_mount_state_publisher`가 `base_link -> camera_link`를 x=0.320, y=0, z=0.850 m, RPY=(0,-5°,0)로 발행함을 확인했다. MCU 모드는 RGB-D odometry TF를 비활성화하고, 단독 모드만 이를 활성화한다. 실제 전체 tree/중복 검사는 **NOT TESTED**다.
9. **RGB 입력 FPS** — 실제 D456: **NOT TESTED**. 합성 시험: 60.001 FPS(기능 검증용이며 성능 판정에 사용하지 않음).
10. **Depth 입력 FPS** — 실제 D456: **NOT TESTED**. 합성 시험: 29.997 FPS.
11. **출력 영상 FPS** — 실제 영상 subscriber 부하 시험: **NOT TESTED**. 합성 진단 시험에서는 의도적으로 subscriber가 없어 0 FPS였다.
12. **Visual Odometry FPS** — 실제 RTAB-Map: **NOT TESTED**. 합성 `/visual_odom`: 20.002 FPS.
13. **RTAB-Map 갱신 FPS** — **NOT TESTED**. RTAB-Map 미설치라 합성 시험 값은 0이다.
14. **평균/p95 latency** — 실제 D456: **NOT TESTED**. 동일 호스트 합성 메시지에서 평균 0.495 ms, p95 0.729 ms였으며 카메라 지연값으로 간주하지 않는다.
15. **drop 비율** — 실제 D456: **NOT TESTED**. 합성 60/30 FPS 입력에서는 0 frames로 계산되었다.
16. **CPU/GPU/메모리** — Intel Core Ultra 9 275HX, 24 logical CPUs. 30 GiB 중 조사 당시 17 GiB 사용, 12 GiB available. NVIDIA `nvidia-smi`는 driver 통신 실패. 실제 SLAM 부하는 **NOT TESTED**.
17. **정지 drift** — **NOT TESTED**(D456 없음).
18. **이동 궤적** — **NOT TESTED**(D456/사용자 차량 조작 없음).
19. **3D 지도** — **NOT TESTED**. RTAB-Map `cloud_map`은 subscriber가 있을 때만 생성되도록 RTAB-Map 자체 MapsManager 출력을 사용한다.
20. **Loop Closure** — **NOT TESTED**.
21. **자동 테스트** — 루트 `pytest`: 14 passed. 패키지 `colcon test`: 9 tests, 0 errors/failures/skips. flake8 10 files 및 pep257 통과, YAML 로드/launch AST/XML/URDF 검사 통과. 합성 DDS에서 RGB/depth/IMU/visual odom 수신, TRACKING 전환, FPS/latency/drop 진단 발행 확인.
22. **빌드** — `colcon build --symlink-install`: 3 packages finished. 설치 공간에 4 bringup launch, description launch/URDF, config 3개와 monitor executables가 생성됨.
23. **하드웨어 미검증 항목** — firmware, USB 연결 속도, RGB/depth 동시 profile, IMU rates, 실제 토픽/QoS, `/odom` frame/rate/QoS, 전체 TF, 60초 drift, 3분 부하, 실차 trajectory/map/loop closure, DB 쓰기, 실제 shutdown.
24. **남은 문제** — D456가 연결되지 않았고 RTAB-Map 0.22.1 패키지는 sudo 암호가 없어 설치하지 못했다. NVIDIA driver도 동작하지 않는다. 실제 D456 profile에 따라 launch profile 인자를 확정해야 한다.
25. **다음 실행 명령** — README의 설치 후 `./scripts/inspect_d456.sh`; profile 확인 후 `ros2 launch depth_bringup visual_slam_full.launch.py use_mcu_odom:=true start_rviz:=false`. 기존 장착 TF가 있으면 `publish_mount_tf:=false`를 추가한다.
26. **최종 Git commit SHA** — self-reference를 피하기 위해 최종 handoff에서 `git rev-parse HEAD` 실측값을 보고한다.
27. **GitHub push** — 최종 handoff에서 push 명령 결과를 보고한다.

이 보고서의 합성 수치는 monitor 계산과 DDS 연결 검증만을 위한 것이며 실제 센서 성능 PASS 근거가 아니다.
