# Route-only mode completion

The segmented route follower owns the one-shot competition-mode lifecycle.
It derives that lifecycle from the active CSV route cursor; `/drive_mode` is
an output and is never accepted as completion evidence.

A non-final mode is route-complete only when the follower cursor reaches that
mode's final route segment and subsequently crosses its last waypoint into the
consecutive next mode while the route, localization, safety, controller, and
cursor are healthy. This boundary crossing permits one waypoint of controller
sampling advance at accelerated test rates; it does not trust a mode value by
itself. STOP_LINE
holds, mission stops, direction-change release cycles, route-deviation/rejoin
states, corridor violations, heading failures, unsafe branch remaps, and cursor
backtracks cannot complete a mode. The final selected mode additionally needs
the existing `ROUTE_COMPLETE` controller result. A course completes only after
every selected mode has completed in order.

The implementation deliberately stores `route_complete`,
`mission_complete`, and `mode_complete` independently. Its current criterion
is `ROUTE_ONLY`, so `mode_complete = route_complete`; `mission_complete` is
diagnostic state reserved for a later mission-aware criterion.

The follower logs each event once:

```text
[MODE 1 START]
[MODE 1 COMPLETE] criterion=ROUTE_ONLY
...
[MODE 11 COMPLETE] criterion=ROUTE_ONLY
[COURSE COMPLETE]
```

It also publishes `/depth_slam/route/mode_status` as `std_msgs/msg/String`.
The payload is compact JSON containing the current and previous modes, state,
criterion, route/mission/mode completion booleans, one-shot started/completed
sets, invalid modes, route index, CSV `segment_id`/`point_index`, overall
progress, last error, and course-complete flag. Existing topics and message
contracts are unchanged.
