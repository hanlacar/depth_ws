"""Bezier turn templates and non-time-based turn-state tracking."""
from enum import IntEnum
import numpy as np


class TurnState(IntEnum):
    LANE_FOLLOW=0; TURN_PREPARE=1; TURN_ENTER=2; TURN_EXECUTE=3; EXIT_SEARCH=4; LANE_REACQUIRE=5; ABORT=6


def bezier_turn(direction, start_heading_rad=0., min_turn_radius_m=1., samples=40):
    """Quarter-circle cubic Bezier with left positive Y and bounded curvature."""
    direction = -1 if direction > 0 else 1
    forward = np.array([np.cos(start_heading_rad), np.sin(start_heading_rad)])
    left = np.array([-forward[1], forward[0]])
    radius=float(min_turn_radius_m); k=4.*(np.sqrt(2.)-1.)/3.
    p0=np.zeros(2); p1=forward*k*radius; p2=forward*radius+left*direction*(1-k)*radius; p3=forward*radius+left*direction*radius
    t=np.linspace(0,1,samples)[:,None]
    return (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3


class TurnStateMachine:
    def __init__(self, progress_timeout_s=3.): self.state=TurnState.LANE_FOLLOW; self.entry_yaw=None; self.execute_start=None; self.progress_timeout_s=progress_timeout_s; self.confidence=1.
    def update(self, direction, yaw_deg=None, imu_valid=True, intersection=False, exit_visible=False, lane_visible=False, now=0.):
        if self.state == TurnState.LANE_FOLLOW and direction: self.state=TurnState.TURN_PREPARE
        elif self.state == TurnState.TURN_PREPARE and intersection: self.state=TurnState.TURN_ENTER
        elif self.state == TurnState.TURN_ENTER: self.state=TurnState.TURN_EXECUTE; self.entry_yaw=yaw_deg if imu_valid and yaw_deg is not None and np.isfinite(yaw_deg) else None; self.execute_start=now
        elif self.state == TurnState.TURN_EXECUTE:
            yaw_ok=imu_valid and self.entry_yaw is not None and yaw_deg is not None and np.isfinite(yaw_deg)
            if yaw_ok and abs(((yaw_deg-self.entry_yaw+180)%360)-180) > 55: self.state=TurnState.EXIT_SEARCH
            elif not yaw_ok:
                self.confidence=.4
                if self.execute_start is not None and now-self.execute_start>self.progress_timeout_s:self.state=TurnState.ABORT
        elif self.state == TurnState.EXIT_SEARCH and exit_visible: self.state=TurnState.LANE_REACQUIRE
        elif self.state == TurnState.LANE_REACQUIRE and lane_visible: self.state=TurnState.LANE_FOLLOW
        return self.state
