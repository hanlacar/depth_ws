import time
import numpy as np
from threading import Lock

def validate_image_contract(image,info,now_sec,max_age_sec,expected_width=640,expected_height=480):
    if image is None or info is None:return False,"missing_image_or_camera_info"
    expected=(int(expected_width),int(expected_height))
    if (image.width,image.height)!=expected or (info.width,info.height)!=expected:return False,"resolution_mismatch"
    if image.encoding.lower() not in ("rgb8","bgr8"):return False,"unsupported_encoding"
    if not image.header.frame_id or image.header.frame_id!=info.header.frame_id:return False,"frame_mismatch"
    if info.distortion_model!="plumb_bob" or len(info.d)!=5:return False,"camera_info_contract"
    if not np.isfinite(np.r_[info.k,info.d]).all():return False,"nonfinite_camera_info"
    stamp=image.header.stamp.sec+image.header.stamp.nanosec*1e-9
    if now_sec-stamp>max_age_sec:return False,"stale_image"
    return True,"ok"

class LatestFrameBuffer:
    def __init__(self):
        self.latest=None;self.last_processed=None;self.dropped_count=0
        self.highest_received=None
        self.received_from_ros=0;self.accepted_unique=0;self.processed=0
        self.duplicate_count=0;self.stale_count=0
        self._lock=Lock()
    def push(self,message):
        with self._lock:
            self.received_from_ros+=1
            incoming=(message.header.stamp.sec,message.header.stamp.nanosec)
            if incoming==self.highest_received:
                self.duplicate_count+=1
                return "duplicate"
            if self.highest_received is not None and incoming<self.highest_received:
                self.stale_count+=1
                return "stale"
            self.highest_received=incoming;self.accepted_unique+=1
            replaced=False
            if self.latest is not None:
                stamp=(self.latest.header.stamp.sec,self.latest.header.stamp.nanosec)
                if stamp!=self.last_processed:
                    self.dropped_count+=1;replaced=True
            self.latest=message
            return "replaced" if replaced else "accepted"
    def take(self):
        with self._lock:
            if self.latest is None:return None
            stamp=(self.latest.header.stamp.sec,self.latest.header.stamp.nanosec)
            if stamp==self.last_processed:return None
            message=self.latest;self.last_processed=stamp;self.processed+=1;return message
    def snapshot(self):
        with self._lock:
            return {"received_from_ros":self.received_from_ros,
                    "accepted_unique":self.accepted_unique,
                    "replaced_before_processing":self.dropped_count,
                    "processed":self.processed,
                    "duplicate":self.duplicate_count,
                    "stale":self.stale_count}
