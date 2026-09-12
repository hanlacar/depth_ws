from pathlib import Path

from depth_hybrid_slam.csv_only_branching import (
    IndependentRouteCaseSelector, StartBranchClassifier,
    csv_only_network_segments, load_csv_only_route_case,
    network_pair_geometry, parking_branch_geometry,
    remap_case_progress, validate_display_correspondence)
from depth_hybrid_slam.models import Pose2D
from depth_hybrid_slam.route_io import load_segmented_route
from depth_hybrid_slam.stop_editor_network import (
    CHOICE_ORDER, CHOICE_SEGMENTS, route_case_segments)
import yaml


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT / "routes/network/route_network_segmented_stop_edited.csv"
METADATA = ROUTE.with_suffix(".metadata.yaml")
PACKAGE = ROOT / "src/depth_hybrid_slam"
DISPLAY = ROOT / "routes/network/route_network_segmented_all_branches_display_aligned.csv"
DISPLAY_METADATA = DISPLAY.with_suffix(".metadata.yaml")


def test_csv_only_default_route_has_exact_a_case_and_stops():
    route = load_segmented_route(ROUTE, METADATA, branch="A")
    assert route.points[0].segment_id == "START_A"
    assert route.segment_order == route_case_segments("AAAA")
    assert len([point for point in route.points if point.event == "STOP_LINE"]) == 10
    assert sum(route.points[index].direction != route.points[index+1].direction
               for index in range(len(route.points)-1)) == 4


def test_csv_only_mixed_cases_contain_only_the_selected_exclusive_segments():
    for case in ("AAAA", "BAAA", "ABAA", "AABA", "AAAB", "BBAA"):
        route = load_csv_only_route_case(ROUTE, METADATA, case)
        segments = {point.segment_id for point in route}
        assert tuple(dict.fromkeys(point.segment_id for point in route)) == \
            route_case_segments(case)
        for choice, selected in zip(CHOICE_ORDER, case):
            other = "B" if selected == "A" else "A"
            assert CHOICE_SEGMENTS[choice][selected] in segments
            assert CHOICE_SEGMENTS[choice][other] not in segments


def test_start_branch_classifier_uses_path_lateral_and_heading():
    segments = csv_only_network_segments(ROUTE, METADATA)
    classifier = StartBranchClassifier(segments["START_A"], segments["START_B"])
    starts = {}
    for branch in "AB":
        point = segments[f"START_{branch}"][0]
        starts[branch] = point
        result = classifier.classify(Pose2D(point.x, point.y, point.yaw, 0.0))
        assert result.branch == branch
        assert result.state == f"START_BRANCH_SELECTED_{branch}"
        assert getattr(result, f"lateral_{branch.lower()}") == 0.0
    middle = Pose2D(
        (starts["A"].x+starts["B"].x)/2.0,
        (starts["A"].y+starts["B"].y)/2.0,
        (starts["A"].yaw+starts["B"].yaw)/2.0, 0.0)
    ambiguous = classifier.classify(middle)
    assert ambiguous.branch == ""
    assert ambiguous.state == "START_BRANCH_AMBIGUOUS"


def test_source_and_display_network_preserve_real_branch_separation_and_keys():
    source = csv_only_network_segments(ROUTE, METADATA)
    display = csv_only_network_segments(DISPLAY, DISPLAY_METADATA)
    source_geometry = network_pair_geometry(source)
    display_geometry = network_pair_geometry(display)
    expected_maximums = {
        "START": 5.326191979441313,
        "T": 2.6726740684898322,
        "V": 4.89855519097754,
        "END": 3.3963187242290753,
    }
    for choice, expected in expected_maximums.items():
        assert abs(source_geometry[choice]["maximum_separation_m"]-expected) < 1e-6
        assert abs(display_geometry[choice]["maximum_separation_m"]-expected) < 1e-6
        assert source_geometry[choice]["mean_nearest_m"] > 0.8
        assert display_geometry[choice]["mean_nearest_m"] > 0.8
    correspondence = validate_display_correspondence(ROUTE, DISPLAY)
    assert correspondence == {
        "valid": True,
        "source_rows": 7463, "display_rows": 7463,
        "source_duplicate_keys": 0, "display_duplicate_keys": 0,
        "missing_keys": 0, "extra_keys": 0,
        "source_xy_max_error_m": 0.0,
    }


def start_a_result(segments):
    segments = csv_only_network_segments(ROUTE, METADATA)
    classifier = StartBranchClassifier(segments["START_A"], segments["START_B"])
    start = segments["START_A"][0]
    return classifier.classify(Pose2D(start.x, start.y, start.yaw, 0.0))


def parking_selector():
    segments = csv_only_network_segments(ROUTE, METADATA)
    selector = IndependentRouteCaseSelector(parking_branch_geometry(segments))
    selector.set_start(start_a_result(segments))
    return selector


def test_parking_geometry_allows_t_stop_commit_but_rejects_v_stop_commit():
    geometry = parking_branch_geometry(
        csv_only_network_segments(ROUTE, METADATA))
    t = geometry["T"]
    assert t["verdict"] == "PASS"
    assert t["first_reverse"]["A"][0] == 5
    assert t["first_reverse"]["B"][0] == 6
    assert abs(t["forward_nearest_mean_m"]-0.10657454613385471) < 1e-9
    assert abs(t["forward_nearest_max_m"]-0.1557953073041695) < 1e-9
    assert abs(t["requested_commit"]["branch_separation_m"]-
               0.1557953073041695) < 1e-9
    assert abs(t["requested_commit"]["branch_heading_delta_deg"]-
               2.87191067692503) < 1e-9
    assert abs(t["requested_commit"]["first_reverse_jump_m"]-
               0.9155940991754984) < 1e-9
    assert abs(t["requested_commit"]["required_steering_deg"]-
               18.212946996912528) < 1e-9
    assert t["requested_commit_safe"]

    v = geometry["V"]
    assert v["verdict"] == "BRANCH_COMMIT_TOO_LATE_FOR_GEOMETRY"
    assert v["first_reverse"]["A"][0] == 91
    assert v["first_reverse"]["B"][0] == 85
    assert abs(v["forward_nearest_mean_m"]-0.7606387009368039) < 1e-9
    assert abs(v["forward_nearest_max_m"]-1.663702698256212) < 1e-9
    assert abs(v["requested_commit"]["branch_separation_m"]-
               1.5862683167563403) < 1e-9
    assert abs(v["requested_commit"]["branch_heading_delta_deg"]-
               13.680626196306077) < 1e-9
    assert abs(v["requested_commit"]["first_reverse_jump_m"]-
               1.9378764915476436) < 1e-9
    assert abs(v["requested_commit"]["required_steering_deg"]-
               39.22630492764032) < 1e-9
    assert not v["requested_commit_safe"]
    safe = v["latest_safe_commit"]
    assert safe["a_point_index"] == 23
    assert safe["nearest_point_index"] == 28
    assert abs(safe["nearest_distance_m"]-0.10978814255191593) < 1e-9
    assert abs(safe["heading_delta_deg"]-3.2532826783562356) < 1e-9
    assert abs(safe["required_steering_deg"]-16.948617924166378) < 1e-9


def test_t_default_a_commits_at_transition_stop_and_holds_after_commit():
    selector = parking_selector()
    assert not selector.request("T:B", 1.0)
    assert selector.last_event == "OUT_OF_WINDOW_BRANCH_COMMAND_IGNORED"
    assert selector.enter_segment("T_foword", 10.0)
    assert selector.lifecycle_status()["T"] == "PENDING"
    selector.observe_point("T_A", 4, 1, "T_A:4", "MINIMUM_3S_HOLD", 20.0)
    assert selector.choices["T"] is None
    assert selector.stop
    selector.observe_point("T_A", 4, 1, "T_A:4", "WAIT_TRAFFIC_RELEASE", 23.0)
    assert selector.choices["T"] == "A"
    assert selector.lifecycle_status()["T"] == "COMMITTED_A"
    assert selector.stop
    selector.advance(25.999)
    assert selector.stop
    selector.advance(26.0)
    assert not selector.stop
    selector.note_reverse_started("T", 26.1)
    assert selector.lifecycle_status()["T"] == "REVERSE_STARTED"


def test_t_b_request_during_forward_commits_b_at_transition_stop():
    selector = parking_selector()
    selector.enter_segment("T_foword", 10.0)
    assert selector.request("T:B", 12.0)
    assert selector.lifecycle_status()["T"] == "B_REQUESTED"
    selector.observe_point("T_A", 4, 1, "T_A:4", "MINIMUM_3S_HOLD", 20.0)
    selector.observe_point("T_A", 4, 1, "T_A:4", "WAIT_TRAFFIC_RELEASE", 23.0)
    assert selector.choices["T"] == "B"
    assert selector.route_case == "ABAA"
    assert selector.lifecycle_status()["T"] == "COMMITTED_B"


def test_t_b_request_during_stop_is_used_but_reverse_late_request_is_ignored():
    selector = parking_selector()
    selector.enter_segment("T_foword", 10.0)
    selector.observe_point("T_A", 4, 1, "T_A:4", "MINIMUM_3S_HOLD", 20.0)
    assert selector.request("T:B", 22.9)
    selector.observe_point("T_A", 4, 1, "T_A:4", "WAIT_TRAFFIC_RELEASE", 23.0)
    assert selector.choices["T"] == "B"
    selector.note_reverse_started("T", 26.1)
    assert not selector.request("T:B", 26.2)
    assert selector.last_event == "LATE_BRANCH_COMMAND_IGNORED"


def test_v_default_and_b_commit_at_latest_safe_forward_point():
    selector = parking_selector()
    selector.enter_segment("V_A", 30.0)
    selector.observe_point("V_A", 22, 1, "", "IDLE", 31.0)
    assert selector.choices["V"] is None
    selector.observe_point("V_A", 23, 1, "", "IDLE", 31.1)
    assert selector.choices["V"] == "A"
    assert selector.lifecycle_status()["V"] == "COMMITTED_A"

    selector = parking_selector()
    selector.enter_segment("V_A", 30.0)
    assert selector.request("V:B", 30.5)
    selector.observe_point("V_A", 23, 1, "", "IDLE", 31.1)
    assert selector.choices["V"] == "B"
    assert selector.route_case == "AABA"
    assert selector.lifecycle_status()["V"] == "COMMITTED_B"


def test_v_stop_or_reverse_b_request_is_late_and_request_lifetime_is_bounded():
    selector = parking_selector()
    selector.enter_segment("V_A", 30.0)
    selector.observe_point("V_A", 23, 1, "", "IDLE", 31.1)
    assert not selector.request("V:B", 40.0)
    assert selector.last_event == "LATE_BRANCH_COMMAND_IGNORED"
    selector.note_reverse_started("V", 50.0)
    assert not selector.request("V:B", 50.1)
    selector.enter_segment("END_common", 60.0)
    assert "V" not in selector.requests
    assert "V" not in selector.request_timestamps
    assert selector.lifecycle_status()["V"] == "COMPLETE"


def test_live_case_remap_uses_only_existing_continuous_forward_waypoints():
    route_a = load_csv_only_route_case(ROUTE, METADATA, "AAAA")
    route_t_b = load_csv_only_route_case(ROUTE, METADATA, "ABAA")
    route_v_b = load_csv_only_route_case(ROUTE, METADATA, "AABA")

    t_origin = next(point for point in route_a
                    if point.segment_id == "T_A" and point.point_index == 4)
    t_index = remap_case_progress(t_origin, route_t_b, "ABAA")
    t_target = route_t_b[t_index]
    assert (t_target.segment_id, t_target.point_index, t_target.direction) == \
        ("T_B", 5, 1)

    v_origin = next(point for point in route_a
                    if point.segment_id == "V_A" and point.point_index == 23)
    v_index = remap_case_progress(v_origin, route_v_b, "AABA")
    v_target = route_v_b[v_index]
    assert (v_target.segment_id, v_target.point_index, v_target.direction) == \
        ("V_B", 29, 1)
    assert ((v_target.x-v_origin.x)**2+(v_target.y-v_origin.y)**2)**0.5 < 0.11


def test_end_request_policy_remains_independent_and_defaults_a():
    selector = parking_selector()
    assert selector.request("END:B", 1.0)
    assert selector.enter_segment("END_common", 2.0)
    assert selector.choices["END"] == "B"

    selector = parking_selector()
    selector.enter_segment("END_common", 2.0)
    assert selector.choices["END"] == "A"


def test_csv_only_launch_contains_only_the_isolated_full_network_chain():
    launch = (PACKAGE / "launch/prehardware_csv_only_closed_loop.launch.py").read_text()
    for executable in (
            'executable="csv_only_branch_selector"',
            'executable="csv_only_network_visualizer"',
            'executable="route_follower"',
            'executable="csv_only_localization_source"',
            'executable="csv_only_virtual_vehicle"', 'executable="rviz2"'):
        assert executable in launch
    for forbidden in (
            "rtabmap", "cuvslam", "nav2", "camera", "lidar", "t870_mcu",
            "mcu_manager", "serial", "behavior_selector", "mcu_source_adapter"):
        assert forbidden not in launch.lower()
    for token in (
            "route_network_segmented_stop_edited_vforward.csv", "spawn_branch",
            "route_network_segmented_all_branches_display_aligned_vforward.csv",
            '"enable_control": True', '"dry_run": False',
            '"user_approved": True',
            '"prehardware_test_override_alignment": True',
            '"prehardware_csv_only_case_selection": True',
            'DeclareLaunchArgument("simulation_speedup", default_value="1.0")'):
        assert token in launch


def test_csv_only_vehicle_uses_direct_candidate_commands_and_measured_config():
    source = (PACKAGE / "depth_hybrid_slam/csv_only_virtual_vehicle_node.py").read_text()
    localization = (
        PACKAGE / "depth_hybrid_slam/csv_only_localization_source_node.py").read_text()
    config = yaml.safe_load((PACKAGE / "config/virtual_vehicle.yaml").read_text())[
        "depth_virtual_mcu_bridge"]["ros__parameters"]
    assert [config[f"stage_{stage}_speed_mps"] for stage in (1, 2, 3)] == [
        0.527, 0.791, 1.055]
    assert config["reverse_speed_mps"] == 0.527
    assert config["wheelbase_m"] == 0.73
    assert config["max_steering_deg"] == 22.0
    assert config["direction_change_hold_s"] == 3.0
    for topic in (
            "/depth_slam/follower/candidate_drive",
            "/depth_slam/follower/candidate_wheel",
            "/depth_slam/follower/candidate_stop", "/mcu/encoder", "/odom"):
        assert topic in source
    assert 'odom.header.stamp, odom.header.frame_id = stamp, "map"' in source
    assert '"spawn_branch": "A"' in source
    assert 'message.header.frame_id != "map"' in localization
    assert '"/depth_slam/csv_only/branch_stop"' in localization
    assert '"/depth_slam/localization/pose"' in localization
    assert 'String(data="TRACKING" if fresh else "STALE")' in localization


def test_csv_only_rviz_has_full_network_and_active_route_in_map_frame():
    config_path = PACKAGE / "config/prehardware_csv_only_closed_loop.rviz"
    config_text = config_path.read_text()
    config = yaml.safe_load(config_text)["Visualization Manager"]
    assert config["Global Options"]["Fixed Frame"] == "map"
    view = config["Views"]["Current"]
    assert view["Scale"] <= 10
    assert view["X"] == 28 and view["Y"] == 30
    assert len(config["Displays"]) == 5
    marker_displays = [display for display in config["Displays"]
                       if display["Class"] == "rviz_default_plugins/MarkerArray"]
    assert len(marker_displays) == 2
    assert all("Topic" in display and "Marker Topic" not in display
               for display in marker_displays)
    for token in (
            "CSV Active Route (Emphasized)", "/depth_slam/route/reference_path",
            "Full A/B Network and STOP Markers",
            "/depth_slam/csv_only/full_network", "STOP_MARKERS",
            "ROUTE_CASE_STATUS", "START_A", "START_B", "T_A", "T_B",
            "V_foword", "V_A", "V_B", "END_AA", "END_AB",
            "injected_disturbance", "recovery_target", "recovery_status",
            "Virtual Vehicle Pose",
            "/depth_slam/csv_only/markers", "Virtual Trajectory",
            "/depth_slam/csv_only/trajectory", "Follower Target",
            "/depth_slam/route/target_point"):
        assert token in config_text
    for forbidden in ("/rtabmap", "/map", "PointCloud2", "Image"):
        assert forbidden not in config_text


def test_csv_only_network_markers_are_separate_solid_a_and_real_dashed_b():
    source = (
        PACKAGE / "depth_hybrid_slam/csv_only_network_visualizer_node.py").read_text()
    for token in (
            'line.ns, line.id = name, marker_id',
            'Marker.LINE_LIST if branch == "B" else Marker.LINE_STRIP',
            'start += dash_m+gap_m', 'label.ns, label.id = f"label_{name}"',
            '"/depth_slam/csv_only/network_debug"',
            'selected == "pending"', 'return 0.095, 0.62',
            'return f"{choice}: B REQUESTED"',
            'return f"{choice}: COMMITTED {phase[-1]}"',
            "if self.dirty:", "if self.debug_pending:"):
        assert token in source
    assert "points[0].x+offset_x" in source
    assert "points[0].y+offset_y" in source


def test_csv_only_live_probe_requires_case_and_opposite_exclusion_contract():
    probe = (PACKAGE / "depth_hybrid_slam/csv_only_closed_loop_probe.py").read_text()
    for token in (
            'state == "ROUTE_COMPLETE"', '"expected_case"',
            '"/depth_slam/route/branch_command"',
            'String(data=f"{choice}:{branch}")',
            "self.expected_segments <= self.segments", "not wrong_hits",
            "encoder_change > 100", "self.wheel_min < 0 < self.wheel_max",
            "len(waypoint_holds) == self.expected_stops",
            "min(waypoint_holds) >= 2.90"):
        assert token in probe


def test_route_follower_case_extension_leaves_production_binding_gate_intact():
    source = (PACKAGE / "depth_hybrid_slam/route_follower_node.py").read_text()
    assert "prehardware_csv_only_case_selection" in source
    assert '"/depth_slam/route/selected_case"' in source
    assert '"/depth_slam/route/active_case"' in source
    assert "self.map_route_verified or\n            (self.test_alignment_override" in source
    assert "self.map_route_verified = True" not in source
    assert '"/depth_slam/route/stop_waypoint_key"' in source
    assert "preserve_transition_stop" in source
    assert "No averaged or offset connector is made" in source
    assert '"/depth_slam/route/mode_status"' in source
    assert "RouteModeCompletionTracker" in source


def test_live_case_runner_spawns_from_the_expected_case_start_branch():
    runner = (ROOT / "tools/run_ros_csv_case.sh").read_text()
    assert 'spawn_branch="${case_name:0:1}"' in runner
    assert 'spawn_branch:="${spawn_branch}"' in runner
