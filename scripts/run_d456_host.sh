#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=41
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR=/tmp/depth_ros_logs

exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera serial_no:=_338122302896 \
  enable_color:=true enable_depth:=true enable_infra1:=true enable_infra2:=true \
  enable_gyro:=true enable_accel:=true \
  rgb_camera.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  depth_module.infra_profile:=640x480x60 \
  gyro_fps:=400 accel_fps:=400 unite_imu_method:=2 \
  align_depth.enable:=true enable_sync:=true pointcloud.enable:=false
