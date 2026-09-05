#!/usr/bin/env python3
"""
mapping.launch.py  —  D456 640x480@60 매핑 파이프라인

구조 (핵심)
    realsense2_camera ──▶ rgbd_odometry (고속 VO, 목표 30~50 Hz)
                     └──▶ rtabmap       (저속 맵 그래프 1~2 Hz + 3D grid)

    출력 45~50 FPS 의 실체 = /odom 토픽 rate.
    rtabmap 자체 갱신(DetectionRate)은 1~2 Hz 가 정상이다.

이 단계는 "매핑 + DB 저장" 까지만 한다.
키보드 조종/경로저장은 다음 단계에서 붙인다.

사용 예
    ros2 launch depth_slam mapping.launch.py
    ros2 launch depth_slam mapping.launch.py delete_db:=true start_rviz:=true
    # 매핑을 끝내고 DB 저장:  터미널에서 Ctrl-C  (rtabmap 이 종료 시 DB flush)
    # 또는 즉시 저장 트리거:
    #   ros2 service call /rtabmap/backup std_srvs/srv/Empty
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("depth_slam")

    # ---------------- 런치 인자 ----------------
    args = [
        DeclareLaunchArgument("delete_db", default_value="false",
            description="시작 시 기존 DB 삭제 (새 맵 시작 시 true)"),
        DeclareLaunchArgument("database_path",
            default_value=os.path.expanduser("~/depth_ws/maps/rtabmap.db"),
            description="맵 DB 저장 경로"),
        DeclareLaunchArgument("start_rviz", default_value="false"),
        DeclareLaunchArgument("use_imu", default_value="true",
            description="D456 IMU 를 odom guess 로 사용"),
        DeclareLaunchArgument("odom_rate_limit", default_value="0",
            description="0=제한없음(입력rate). CPU 부족 시 30~50 으로 상한"),
    ]

    delete_db = LaunchConfiguration("delete_db")
    database_path = LaunchConfiguration("database_path")
    start_rviz = LaunchConfiguration("start_rviz")
    use_imu = LaunchConfiguration("use_imu")

    cam_cfg = PathJoinSubstitution([pkg, "config", "d456_camera.yaml"])
    rtab_cfg = PathJoinSubstitution([pkg, "config", "rtabmap_mapping.yaml"])

    # 공통 리매핑: realsense 토픽 → rtabmap 이 기대하는 이름
    rgb_topic = "/camera/camera/color/image_raw"
    depth_topic = "/camera/camera/aligned_depth_to_color/image_raw"
    info_topic = "/camera/camera/color/camera_info"
    imu_topic = "/camera/camera/imu"

    # ---------------- 1) RealSense D456 ----------------
    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        name="camera",
        parameters=[cam_cfg],
        output="screen",
    )

    # ---------------- 2) static TF: base_link -> camera_link ----------------
    # 실측 장착값. (x,y,z, roll,pitch,yaw). pitch -5deg = -0.0873 rad.
    # ★ 실제 장착에 맞게 반드시 재측정해 교체할 것.
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera",
        arguments=["0.320", "0", "0.850", "0", "-0.0873", "0",
                   "base_link", "camera_link"],
        output="screen",
    )

    # ---------------- 3) rgbd_odometry (고속 VO) ----------------
    rgbd_odom = Node(
        package="rtabmap_odom",
        executable="rgbd_odometry",
        name="rgbd_odometry",
        output="screen",
        parameters=[rtab_cfg, {
            "wait_imu_to_init": use_imu,
        }],
        remappings=[
            ("rgb/image", rgb_topic),
            ("depth/image", depth_topic),
            ("rgb/camera_info", info_topic),
            ("imu", imu_topic),
        ],
    )

    # ---------------- 4) rtabmap (저속 맵 그래프) ----------------
    # delete_db 인자에 따라 --delete_db_on_start 유무만 다른 두 정의 중
    # 하나만 뜬다 (IfCondition / UnlessCondition 으로 상호배타).
    def _rtabmap(name_suffix, extra_args, cond):
        return Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[rtab_cfg, {
                "database_path": database_path,
                "subscribe_depth": True,
                "subscribe_rgbd": False,
            }],
            remappings=[
                ("rgb/image", rgb_topic),
                ("depth/image", depth_topic),
                ("rgb/camera_info", info_topic),
                ("odom", "/odom"),
            ],
            arguments=extra_args,
            condition=cond,
        )

    rtabmap_del = _rtabmap("del", ["--delete_db_on_start"],
                           IfCondition(delete_db))
    rtabmap_keep = _rtabmap("keep", [], UnlessCondition(delete_db))

    # ---------------- 5) FPS 모니터 ----------------
    fps_monitor = Node(
        package="depth_slam",
        executable="fps_monitor",
        name="fps_monitor",
        output="screen",
        parameters=[{
            "topics": ["/odom", rgb_topic, depth_topic],
            "window_s": 2.0,
        }],
    )

    # ---------------- 6) RViz (선택) ----------------
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        condition=IfCondition(start_rviz),
        arguments=["-d", PathJoinSubstitution([pkg, "config", "mapping.rviz"])],
        output="screen",
    )

    return LaunchDescription(args + [
        realsense,
        static_tf,
        rgbd_odom,
        # rtabmap 은 delete_db 조건으로 둘 중 하나만 뜨게 한다
        rtabmap_del,
        rtabmap_keep,
        fps_monitor,
        rviz,
    ])
