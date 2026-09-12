# Mode 10 V_foword and CSV rejoin validation (2026-09-12)

MODE10 COMMON FORWARD ROUTE: PASS
CSV 0.5M REJOIN: PASS
CSV OFF-ROUTE SAFETY: PASS
T_B TO COMMON_2 REJOIN: PASS

## Scope and provenance

- Working tree modified: `/home/qor/depth_ws` only.
- Upstream reference repository was fetched read-only at commit
  `fc535402100c19159fc87474143b9a874bd0848e` (`Update mmission_ws`,
  2026-09-12 17:23:33 +0900).
- Reference files:
  `routes/route_network_segmented_10.csv` and
  `src/mission_manager/mission_manager/dr_real_segmented_follower_node.py`.
- The reference CSV and follower both confirm
  `COMMON_2 -> V_foword -> V_A/V_B -> END_common`.
- Candidate:
  `routes/network/route_network_segmented_stop_edited_vforward.csv` with
  7,419 data rows. The original 7,463-row edited CSV is preserved.
- Every non-Mode-10 row is byte-for-byte field-equal to the source row. The
  imported V geometry keeps the reference `x_m/y_m`; only the independently
  verified longitude-origin delta `+0.0000001694 deg` is applied. No point was
  offset or interpolated.

## Mode 10 geometry

Raw CSV coordinates:

| Point | x_m | y_m | Event |
|---|---:|---:|---|
| COMMON_2 end p1216 | -16.298765 | -48.433470 | NONE |
| V_foword start p0 | -16.214647 | -48.189207 | NONE |
| V_foword safe commit p105 | -22.325263 | -32.256993 | STOP_LINE |
| V_foword end p107 | -22.514198 | -32.175668 | NONE |
| V_A start p0 | -22.654269 | -32.085746 | NONE |
| V_B start p0 | -22.652051 | -32.071869 | STOP_LINE / direction boundary |

`V_foword` has 108 points (p0..p107). Endpoint diagnostics use the incoming
common tangent and the production 0.7 m lookahead law:

| Transition | Gap m | Heading delta deg | Raw required steer deg | <=22 deg |
|---|---:|---:|---:|---|
| COMMON_2 end -> V_foword start | 0.258341 | 5.2387 | 9.5925 | yes |
| V_foword end -> V_A start | 0.166451 | 8.8768 | 5.0601 | yes |
| V_foword end -> V_B start | 0.172562 | 13.3050 | 25.6408 | no (endpoint-only) |

The pinned recording's p106->p107 spacing is only about 8 mm, making a branch
decision first applied at p107 too late for V_B. Coordinates were not altered.
Instead, p105 is the latest shared point at which both branch approaches are
inside the 1.0 m corridor and raw steering envelope:

| Commit transition | Gap m | Heading delta deg | Raw required steer deg | Result |
|---|---:|---:|---:|---|
| V_foword p105 -> V_A | 0.370905 | 8.4759 | 3.4368 | PASS |
| V_foword p105 -> V_B | 0.375581 | 13.7059 | 12.8100 | PASS |

The state order is `V:PENDING -> V:B_REQUESTED` (optional) `-> STOP_LINE at
V_foword:p105 -> COMMITTED_A/B -> >=3 s hold -> REVERSE_STARTED`. A command
received after reverse begins is reported as `LATE_BRANCH_COMMAND_IGNORED`.
The established T late-decision geometry and policy are unchanged.

## Deterministic closed-loop disturbance matrix

The same `RouteFollower` and calibrated `VirtualAckermannVehicle` used by the
CSV-only ROS chain were run at 50 ms simulation steps. The cursor was seeded at
the active `COMMON_1:p300` point and `global_search=False` for every recovery.
The production threshold is `normal_corridor_m=1.0`; therefore offsets above
1.0 m must stop. With no OccupancyGrid in CSV-only mode, they are not allowed
to invent a rejoin path.

| Offset | Side | Initial CTE m | Max CTE m | Min CTE m | Decrease s | <=0.25 m s | <=0.10 m | Stop | Result |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| +0.20 | LEFT | 0.200 | 0.200 | 0.000014 | 0.45 | 0.00 | yes | no | PASS |
| -0.20 | RIGHT | 0.200 | 0.219 | 0.000021 | 0.45 | 0.00 | yes | no | PASS |
| **+0.50** | **LEFT** | **0.500** | **0.500** | **0.000118** | **0.60** | **1.65** | **yes** | **no** | **PASS** |
| **-0.50** | **RIGHT** | **0.500** | **0.544** | **0.000061** | **0.65** | **1.90** | **yes** | **no** | **PASS** |
| +0.80 | LEFT | 0.800 | 0.800 | 0.000223 | 0.50 | 2.65 | yes | no | PASS |
| -0.80 | RIGHT | 0.800 | 0.826 | 0.000071 | 0.65 | 2.70 | yes | no | PASS |
| +1.20 | LEFT | 1.200 | 1.200 | 1.200 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |
| -1.20 | RIGHT | 1.200 | 1.200 | 1.200 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |
| +1.80 | LEFT | 1.800 | 1.800 | 1.800 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |
| -1.80 | RIGHT | 1.800 | 1.800 | 1.800 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |
| +2.10 | LEFT | 2.100 | 2.100 | 2.100 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |
| -2.10 | RIGHT | 2.100 | 2.100 | 2.100 | - | - | no | yes, CORRIDOR_VIOLATION | PASS |

All 12 runs had zero wrong-segment jumps, direction jumps, and cursor
backtracks. Maximum commanded steering was bounded to 22 degrees.

Additional +/-0.50 m runs passed at the common straight, a common corner,
COMMON_2 after T_A, COMMON_2 after T_B, V_foword, and END_common. Their worst
maximum CTE was 0.534 m and slowest 0.25 m recovery was 2.65 s.

## Live ROS closed-loop evidence

RViz was disabled during integration; each launch was stopped and its process
list checked before the next run.

- `AAAA`: `ROUTE_COMPLETE`, all 11 expected STOP events, minimum hold
  3.000231 s, wheel -22..+22 deg, no opposite branch hit. A live +0.50 m
  disturbance in COMMON_2 reached max CTE 0.517 m, recovered below 0.10 m,
  and did not stop, jump segment, or backtrack.
- `AABA`: `ROUTE_COMPLETE`, lifecycle
  `PENDING -> B_REQUESTED -> COMMITTED_B -> REVERSE_STARTED -> COMPLETE`, all
  11 expected STOP events, minimum hold 3.000491 s, wheel -22..+22 deg, and
  zero V_A waypoint activation. A live -0.50 m disturbance had max CTE
  0.506302 m, began decreasing at 0.332 s wall time, recovered within 0.25 m
  at 0.805 s wall time and below 0.10 m, with no stop/jump/backtrack.
- `ABBA`: reached `ROUTE_COMPLETE`; T and V both followed
  `PENDING -> B_REQUESTED -> COMMITTED_B -> REVERSE_STARTED -> COMPLETE`.
  At T_B -> COMMON_2 the live controller remained `OK` with observed CTE
  0.0172 m. All 11 stops were >=3.001062 s and wheel remained -22..+22 deg.
  The expected short T_A pending approach is recorded separately and is not a
  wrong branch; V_A remains forbidden and was never activated.

The earlier 10x/100 Hz ABBA failure was reproduced. It crossed the unchanged
1.0 m corridor once and correctly latched `ROUTE_DEVIATION_STOP -> PLAN_REJOIN`.
At 10x speed, 100 Hz wall-clock control is only 10 Hz in simulation time. With
controller and vehicle publishing at 250 Hz (25 Hz effective) the same CSV and
threshold pass. This is a test acceleration/sample-rate issue, not a T_B or
COMMON_2 geometry fault. No production safety threshold was relaxed.

## Regression

- Targeted V_foword/recovery suite: 51 passed.
- `depth_hybrid_slam` package via colcon: 259 passed.
- Full source-package regression: 778 passed, 2 skipped, 0 failed (previous
  baseline 745 passed plus 33 new tests).
- Self-contained audit: 19/19 passed.
- Deterministic complete-route simulations: AAAA, AABA, ABAA and ABBA all
  reached route completion with maximum command magnitude 22 degrees and
  minimum direction-change hold >=3.0 s.
- Production `map_route_verified` remains false; the existing test-only
  alignment override is unchanged.
