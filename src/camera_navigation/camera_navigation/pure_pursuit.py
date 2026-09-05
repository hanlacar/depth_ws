"""Pure-pursuit steering in degrees."""
import math
import numpy as np


def steering_angle_deg(path, speed_mps, wheelbase_m, min_lookahead_m, max_lookahead_m, lookahead_speed_gain, max_steering_deg):
    p=np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2: return 0.
    p=p[np.isfinite(p).all(axis=1) & (p[:,0] > 0.)]
    if len(p)<1 or not np.isfinite([speed_mps,wheelbase_m,min_lookahead_m,max_lookahead_m,lookahead_speed_gain,max_steering_deg]).all() or wheelbase_m<=0: return 0.
    ld=np.clip(min_lookahead_m+lookahead_speed_gain*max(speed_mps,0.), min_lookahead_m, max_lookahead_m)
    target=p[np.argmin(np.abs(np.linalg.norm(p,axis=1)-ld))]
    l2=float(target@target)
    if l2 < 1e-9: return 0.
    angle=math.degrees(math.atan2(2.*wheelbase_m*target[1], l2))
    return float(np.clip(angle, -max_steering_deg, max_steering_deg))


def rate_limit(target, previous, limit_deg_s, dt):
    if not np.isfinite([target,previous,limit_deg_s,dt]).all() or limit_deg_s<0 or dt<0:return 0.
    return float(np.clip(target, previous-limit_deg_s*dt, previous+limit_deg_s*dt))
