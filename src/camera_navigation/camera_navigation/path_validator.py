"""Path numerical, range, and freshness validation."""
import numpy as np
from dataclasses import dataclass


def validate_path(path, confidence=1., min_points=2, max_forward_m=30.):
    p=np.asarray(path,float)
    return bool(p.ndim==2 and p.shape[1]==2 and len(p)>=min_points and np.isfinite(p).all() and 0<=confidence<=1 and np.all(p[:,0]>=-0.1) and np.all(p[:,0]<=max_forward_m))


def is_stale(stamp_seconds, now_seconds, timeout_seconds):
    return now_seconds-stamp_seconds > timeout_seconds


@dataclass(frozen=True)
class MaskMeta:
    stamp: float
    frame_id: str
    width: int
    height: int
    encoding: str


def validate_mask_set(metas, camera_size, now, sync_tolerance=.05, stale_timeout=.2, previous_stamp=None):
    """Validate that masks originate from one inference frame."""
    if len(metas)!=3 or any(not np.isfinite(m.stamp) for m in metas): return False, "missing_or_nonfinite"
    stamps=[m.stamp for m in metas]
    if previous_stamp is not None and min(stamps) < previous_stamp: return False, "timestamp_reversal"
    if max(stamps)-min(stamps)>sync_tolerance: return False, "timestamp_skew"
    if now-max(stamps)>stale_timeout: return False, "stale"
    if len({m.frame_id for m in metas})!=1 or not metas[0].frame_id: return False, "frame_mismatch"
    if len({(m.width,m.height) for m in metas})!=1 or (metas[0].width,metas[0].height)!=camera_size: return False, "size_mismatch"
    if len({m.encoding for m in metas})!=1 or metas[0].encoding not in ("mono8","8UC1"): return False, "encoding_mismatch"
    return True, "ok"
