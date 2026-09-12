#!/usr/bin/env python3
"""Publish a binary PCL PLY as one transient-local PointCloud2 message."""

import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


def read_ply(path: Path):
    with path.open("rb") as stream:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line:
                raise ValueError("invalid PLY header")
            header += line
        offset = stream.tell()
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("only binary little-endian PLY is supported")
    count = int(next(line.split()[2] for line in text.splitlines() if line.startswith("element vertex")))
    source_dtype = np.dtype({"names":["x","y","z","nx","ny","nz","r","g","b","curvature"],
                             "formats":["<f4"]*6+["u1"]*3+["<f4"],
                             "offsets":[0,4,8,12,16,20,24,25,26,27],"itemsize":31})
    src = np.memmap(path,dtype=source_dtype,mode="r",offset=offset,shape=(count,))
    out_dtype=np.dtype({"names":["x","y","z","rgb"],"formats":["<f4","<f4","<f4","<u4"],"offsets":[0,4,8,12],"itemsize":16})
    out=np.empty(count,dtype=out_dtype)
    out["x"]=src["x"];out["y"]=src["y"];out["z"]=src["z"]
    out["rgb"]=(src["r"].astype(np.uint32)<<16)|(src["g"].astype(np.uint32)<<8)|src["b"].astype(np.uint32)
    return out


class CloudPublisher(Node):
    def __init__(self,path:Path,topic:str):
        super().__init__("level_aligned_v8_ply_publisher")
        self.points=read_ply(path)
        qos=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub=self.create_publisher(PointCloud2,topic,qos)
        self.timer=self.create_timer(1.0,self.publish_once)
        self.sent=False

    def publish_once(self):
        msg=PointCloud2();msg.header.stamp=self.get_clock().now().to_msg();msg.header.frame_id="map"
        msg.height=1;msg.width=len(self.points);msg.is_bigendian=False;msg.is_dense=False
        msg.fields=[PointField(name="x",offset=0,datatype=PointField.FLOAT32,count=1),
                    PointField(name="y",offset=4,datatype=PointField.FLOAT32,count=1),
                    PointField(name="z",offset=8,datatype=PointField.FLOAT32,count=1),
                    PointField(name="rgb",offset=12,datatype=PointField.UINT32,count=1)]
        msg.point_step=16;msg.row_step=16*len(self.points);msg.data=self.points.tobytes()
        self.pub.publish(msg)
        if not self.sent:
            self.get_logger().info(f"published {len(self.points)} colored points on {self.pub.topic}")
            self.sent=True


def main():
    p=argparse.ArgumentParser();p.add_argument("ply",type=Path);p.add_argument("--topic",default="/level_aligned_v8/full_cloud");a=p.parse_args()
    rclpy.init();node=CloudPublisher(a.ply,a.topic)
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
