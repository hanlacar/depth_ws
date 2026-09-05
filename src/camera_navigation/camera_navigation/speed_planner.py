"""Curvature/lateral-acceleration speed planning."""
import math
import numpy as np
from .path_generator import BOTH_BOUNDARIES, LEFT_ONLY, RIGHT_ONLY, ROAD_ONLY, TURN_TEMPLATE


def max_curvature(path):
    p=np.asarray(path,float)
    if len(p)<3:return 0.
    d=np.gradient(p,axis=0); dd=np.gradient(d,axis=0)
    return float(np.nanmax(np.abs(d[:,0]*dd[:,1]-d[:,1]*dd[:,0])/np.maximum((d*d).sum(1)**1.5,1e-9)))


def target_speed(path, mode, valid=True, stale=False, turn_abort=False, turn_active=False, normal=1., turn=.4, single=.6, road=.35, max_lateral_accel=1.):
    if not valid or stale or turn_abort:return 0.
    if turn_active:return float(turn)
    cap={TURN_TEMPLATE:turn, LEFT_ONLY:single, RIGHT_ONLY:single, ROAD_ONLY:road}.get(mode,normal)
    k=max_curvature(path)
    return float(min(cap, math.sqrt(max_lateral_accel/k) if k>1e-6 else cap))
