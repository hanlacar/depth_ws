import math
from pathlib import Path

from depth_hybrid_slam.mcu_source_adapter_core import adapt_slam_command
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.prehardware_core import BranchSelector, StopWaypointMachine
from depth_hybrid_slam.route_follower_core import RouteFollower
from depth_hybrid_slam.route_io import load_segmented_route
from depth_hybrid_slam.stop_editor_core import StopRouteEditor
from depth_hybrid_slam.virtual_mcu_core import VirtualAckermannVehicle
import yaml


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT / "routes/network/route_network_segmented.csv"
METADATA = ROOT / "routes/network/route_network_segmented.metadata.yaml"
VIRTUAL_CONFIG = ROOT / "src/depth_hybrid_slam/config/virtual_vehicle.yaml"


def _calibration():
    values = yaml.safe_load(VIRTUAL_CONFIG.read_text())["depth_virtual_mcu_bridge"][
        "ros__parameters"]
    return values


def _vehicle():
    values = _calibration()
    return VirtualAckermannVehicle(
        {stage: values[f"stage_{stage}_speed_mps"] for stage in (1, 2, 3)},
        values["reverse_speed_mps"], wheelbase_m=values["wheelbase_m"],
        counts_per_meter=values["counts_per_meter"],
        max_steering_deg=values["max_steering_deg"],
        direction_change_hold_s=values["direction_change_hold_s"],
        encoder_signed=values["encoder_signed"])


def test_virtual_encoder_forward_reverse_left_right_and_clamp():
    vehicle = _vehicle()
    forward = vehicle.step(1, 0, False, 1.0)
    assert forward.x == 0.527
    assert forward.encoder == round(0.527*797)
    vehicle.step(0, 0, True, 1.5)
    vehicle.step(0, 0, True, 1.5)
    left = vehicle.step(1, 99, False, 1.0)
    assert left.steer_deg == 22.0 and left.yaw > 0.0
    vehicle.step(0, 0, True, 1.5)
    vehicle.step(0, 0, True, 1.5)
    reverse = vehicle.step(-1, 0, False, 1.0)
    assert reverse.speed_mps == -0.527
    vehicle.step(0, 0, True, 1.5)
    vehicle.step(0, 0, True, 1.5)
    right = vehicle.step(1, -99, False, 1.0)
    assert right.steer_deg == -22.0
    assert right.yaw < left.yaw
    assert right.encoder == round(right.distance_m*797)
    values = _calibration()
    assert [values[f"stage_{stage}_pwm"] for stage in (1, 2, 3)] == [50, 75, 100]
    assert values["reverse_pwm"] == -50 and values["encoder_signed"] is False
    assert values["counts_per_meter"] == 797.0
    for stage, speed in ((1, 0.527), (2, 0.791), (3, 1.055)):
        timed = _vehicle()
        state = None
        for _ in range(100):
            state = timed.step(stage, 0, False, 0.1)
        expected_distance = speed * 10.0
        expected_encoder = round(expected_distance * 797.0)
        assert math.isclose(state.distance_m, expected_distance, rel_tol=0.02)
        assert state.encoder == expected_encoder
        assert state.speed_mps == speed


def test_virtual_direction_change_is_rejected_without_three_second_stop():
    vehicle = _vehicle()
    vehicle.step(1, 0, False, 0.1)
    rejected = vehicle.step(-1, 0, False, 0.1)
    assert rejected.stopped and not rejected.direction_guard_ok


def _key(point):
    return f"{point.segment_id}:{point.point_index}"


def _ceiling(route, follower, stop_machine):
    first = max(0, follower.last_index or 0)
    for index in range(first, len(route)):
        point = route[index]
        direction_change = (index+1 < len(route) and
                            point.direction != route[index+1].direction)
        if ((point.event == "STOP_LINE" or direction_change) and
                _key(point) not in stop_machine.completed):
            return index
    return None


def _stop_ahead(route, nearest, pose):
    distance = 0.0
    previous = route[nearest]
    for point in route[nearest:min(len(route), nearest+100)]:
        if point is not previous:
            distance += math.hypot(point.x-previous.x, point.y-previous.y)
        previous = point
        index = point.index
        direction_change = (index+1 < len(route) and
                            point.direction != route[index+1].direction)
        trigger = 1.0 if direction_change else 0.5
        if ((point.event == "STOP_LINE" or direction_change) and
                (distance <= trigger or
                 math.hypot(point.x-pose.x, point.y-pose.y) <= trigger)):
            return _key(point), True
        if distance > 1.0:
            break
    return "", False


def _simulate(route):
    follower = RouteFollower(corridor_m=1.0, max_index_backtrack=0)
    vehicle = _vehicle()
    vehicle.x, vehicle.y, vehicle.yaw = route[0].x, route[0].y, route[0].yaw
    stops = StopWaypointMachine(3.0)
    target_segments = set()
    left_seen = right_seen = False
    maximum_wheel = 0.0
    minimum_actual_hold = float("inf")
    for step in range(100000):
        pose = Pose2D(vehicle.x, vehicle.y, vehicle.yaw, step*0.1)
        result = follower.compute(
            pose, route, dt=0.1, allow_motion=True, global_search=step == 0,
            progress_ceiling=_ceiling(route, follower, stops))
        target_segments.add(route[result.target_index].segment_id)
        maximum_wheel = max(maximum_wheel, abs(result.steering_deg))
        left_seen |= result.steering_deg > 0.5
        right_seen |= result.steering_deg < -0.5
        if result.reason == "ROUTE_COMPLETE":
            return {
                "complete": True, "segments": target_segments,
                "left": left_seen, "right": right_seen,
                "maximum_wheel": maximum_wheel,
                "stops": stops.completed,
                "minimum_hold": minimum_actual_hold,
                "distance": vehicle.distance,
                "guard": vehicle.direction_guard_ok,
            }
        assert not result.stop_required, result
        waypoint_key, reached = _stop_ahead(route, result.nearest_index, pose)
        decision = stops.update(waypoint_key, reached, False, step*0.1)
        transition_release = False
        if decision.state == "RELEASED" and waypoint_key:
            minimum_actual_hold = min(minimum_actual_hold, vehicle.stop_elapsed)
            for index, point in enumerate(route[:-1]):
                if (_key(point) == waypoint_key and
                        point.direction != route[index+1].direction):
                    follower.last_index = index+1
                    follower.last_steering = 0.0
                    transition_release = True
                    break
        adapted = adapt_slam_command(
            result.drive, result.steering_deg,
            decision.stop or transition_release, (0.0, 0.0, 0.0))
        assert adapted.valid
        # The unmodified manager performs its documented second sign flip.
        manager_wheel = -adapted.wheel
        assert manager_wheel == int(-adapted.wheel)
        state = vehicle.step(
            adapted.drive, manager_wheel, adapted.stop, 0.1)
        assert state.direction_guard_ok
    raise AssertionError("route did not complete")


def test_a_and_b_complete_closed_loop_with_actual_branch_topology(tmp_path):
    editor = StopRouteEditor(ROUTE, METADATA)
    edited, metadata, _ = editor.save(tmp_path/"edited.csv")
    results = {}
    selector = BranchSelector()
    default_branch = selector.evaluate(0.0).branch
    selector.set_command("B", 1.0)
    commanded_branch = selector.evaluate(1.0).branch
    assert (default_branch, commanded_branch) == ("A", "B")
    for branch in (default_branch, commanded_branch):
        route = list(load_segmented_route(
            edited, metadata, branch=branch).points)
        results[branch] = _simulate(route)
        result = results[branch]
        assert result["complete"] and result["guard"]
        assert result["left"] and result["right"]
        assert result["maximum_wheel"] <= 22.0
        assert result["minimum_hold"] >= 3.0
        assert len(result["stops"]) == 10
    assert {"START_A", "T_A", "V_A", "END_AA"} <= results["A"]["segments"]
    assert not ({"START_B", "T_B", "V_B", "END_AB"} & results["A"]["segments"])
    assert {"START_B", "T_B", "V_B", "END_AB"} <= results["B"]["segments"]
    assert not ({"START_A", "T_A", "V_A", "END_AA"} & results["B"]["segments"])


def test_sign_chain_is_physical_left_positive_and_discrete():
    adapted = adapt_slam_command(2.0, 12, False, (0.0, 0.0, 0.0))
    assert adapted.drive == 2.0
    assert adapted.wheel == -12
    manager_wheel = -adapted.wheel
    assert manager_wheel == 12


def test_closed_loop_launch_uses_virtual_not_serial_bridge():
    launch = (ROOT / "src/depth_hybrid_slam/launch/"
              "prehardware_csv_vslam_closed_loop.launch.py").read_text()
    for token in ("virtual_mcu_bridge", "route_follower", "behavior_selector",
                  "mcu_source_adapter", "start_mode", "end_mode",
                  "branch_command", "offline_rtabmap_include",
                  "manager_command_prefix"):
        assert token in launch
    assert "virtual_vehicle.yaml" in launch
    config = VIRTUAL_CONFIG.read_text()
    for token in ("stage_1_speed_mps: 0.527", "stage_2_speed_mps: 0.791",
                  "stage_3_speed_mps: 1.055", "reverse_pwm: -50",
                  "counts_per_meter: 797.0", "encoder_signed: false"):
        assert token in config
    assert 'executable="mcu_bridge"' not in launch
    assert "t870_mcu" not in launch
    rviz = (ROOT / "src/depth_hybrid_slam/config/"
            "prehardware_closed_loop.rviz").read_text()
    for token in ("/rtabmap/map", "/depth_slam/stop_editor/network",
                  "/depth_slam/stop_editor/mode_markers",
                  "/depth_slam/virtual_mcu/trajectory"):
        assert token in rviz
