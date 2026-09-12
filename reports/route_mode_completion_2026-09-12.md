# CSV-only route-mode completion validation (2026-09-12)

MODE COMPLETE STRUCTURE: PASS
AAAA MODE 1-11 COMPLETE: PASS
BAAA MODE 1-11 COMPLETE: PASS

## Result matrix

| Mode | AAAA START count | AAAA COMPLETE count | BAAA START count | BAAA COMPLETE count | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1 | 1 | PASS |
| 2 | 1 | 1 | 1 | 1 | PASS |
| 3 | 1 | 1 | 1 | 1 | PASS |
| 4 | 1 | 1 | 1 | 1 | PASS |
| 5 | 1 | 1 | 1 | 1 | PASS |
| 6 | 1 | 1 | 1 | 1 | PASS |
| 7 | 1 | 1 | 1 | 1 | PASS |
| 8 | 1 | 1 | 1 | 1 | PASS |
| 9 | 1 | 1 | 1 | 1 | PASS |
| 10 | 1 | 1 | 1 | 1 | PASS |
| 11 | 1 | 1 | 1 | 1 | PASS |

- AAAA ROUTE_COMPLETE: PASS
- AAAA COURSE COMPLETE: PASS (count 1)
- BAAA ROUTE_COMPLETE: PASS
- BAAA COURSE COMPLETE: PASS (count 1)
- Failure-state hits in both qualifying logs: 0

## Completion contract

`RouteModeCompletionTracker` builds a contiguous boundary for every CSV mode
in the active route. A non-final mode requires a monotonic cursor to reach the
mode's final route segment and then cross its last waypoint into the next
consecutive mode under healthy route/localization/safety/controller gates.
The crossing permits one waypoint of sampling advance at 10x/250 Hz. It does
not accept `/drive_mode`, a segment end, a STOP_LINE, or a branch name as
completion evidence.

STOP/mission holds defer completion. A cursor backtrack, incomplete/unsafe or
nonconsecutive transition, unsafe branch remap, pose/localization failure,
safety failure, corridor/heading failure, or a route-deviation/rejoin state
invalidates completion for the active mode. The final mode additionally
requires the existing follower `ROUTE_COMPLETE` result. COURSE COMPLETE needs
all selected route modes to have completed.

`route_complete`, `mission_complete`, and `mode_complete` are separate state
maps. The current criterion is `ROUTE_ONLY`, so `mode_complete` follows
`route_complete`; no mission-complete input is forced.

## Diagnostic contract

The new `/depth_slam/route/mode_status` topic is
`std_msgs/msg/String` containing compact JSON. It includes current/previous
mode, `state`, `criterion`, current route/mission/mode completion booleans,
started/completed/invalid mode sets, course state, route index, CSV
`segment_id`/`point_index`, progress, and the last rejection reason. No
existing topic type or owner changed.

## Live ROS evidence

Both qualifying runs used the candidate CSV, RViz disabled, 10x simulation,
250 Hz route follower, and 250 Hz virtual vehicle. Each launch was stopped and
cleaned before the next run.

### AAAA

- Log: `/tmp/depth_ws_ros_191_AAAA/launch.stdout.log`
- Selected/active case: AAAA; opposite exclusive branch hits: none
- Existing controller result: ROUTE_COMPLETE
- STOP count: 11/11; minimum observed hold: 2.999947 s (probe threshold 2.90 s)
- Steering range: -22..+22 degrees
- Completion order: 1,2,3,4,5,6,7,8,9,10,11; every START/COMPLETE count 1
- COURSE COMPLETE count: 1

### BAAA

- Log: `/tmp/depth_ws_ros_195_BAAA/launch.stdout.log`
- Spawn/selected/active case: START_B / BAAA / BAAA
- Exclusive hits: START_B, T_A, V_A, END_AA; opposite hits: none
- Existing controller result: ROUTE_COMPLETE
- STOP count: 12/12; minimum observed hold: 2.998900 s (probe threshold 2.90 s)
- Steering range: -22..+21 degrees
- Completion order: 1,2,3,4,5,6,7,8,9,10,11; every START/COMPLETE count 1
- COURSE COMPLETE count: 1

The reusable live-run helper now derives `spawn_branch` from the first letter
of `expected_case`; this corrects verification setup only and does not alter
the A/B selector or route topology. Its default probe timeout is 900 seconds
so the longer START_B path can finish at the stable 10x/250 Hz rate.

## Regression and integrity

- All source-package tests: 798 passed, 2 skipped, 0 failed.
- `depth_hybrid_slam` colcon test: 279 passed, 0 failed.
- Focused completion/CSV contract test: 38 passed, 0 failed.
- Self-contained audit: 19/19 PASS.
- Candidate CSV SHA-256 remains
  `e308f6e8be749d6372b0814ad105fba78058d9419c4c1874efac42c84bd03160`.
- No CSV coordinate, V_foword geometry, A/B topology, parking policy, rejoin
  policy, or production safety threshold was changed.
- No commit or push was performed.
