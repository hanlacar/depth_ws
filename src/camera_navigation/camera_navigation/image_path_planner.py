"""Original-image-coordinate path planning; deliberately contains no BEV geometry."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .metric_path_quality import has_self_intersection
from .pixel_lateral_control import lookahead_offset_px, steering_from_offset_deg


VALID, DEGRADED, INVALID, INACTIVE = "VALID", "DEGRADED", "INVALID", "INACTIVE"
BOTH_BOUNDARIES = "BOTH_BOUNDARIES"
LEFT_BOUNDARY = "LEFT_BOUNDARY"
RIGHT_BOUNDARY = "RIGHT_BOUNDARY"
ROAD_CENTER = "ROAD_CENTER"
TEMPORAL_FALLBACK = "TEMPORAL_FALLBACK"
LANE_DOMINANT = "LANE"
MIXED_SOURCES = "MIXED"
ROAD_DOMINANT = "ROAD"


def camera_owns_path(section, intersection_sections=(4, 6, 8, 11)):
    """Ownership is section based; direction is intentionally not an input."""
    return int(section) not in {int(value) for value in intersection_sections}


@dataclass
class PlannerConfig:
    vehicle_center_x_px: float = 320.0
    roi_top: int = 120
    roi_bottom: int = 475
    sample_interval_px: int = 10
    sample_band_half_height_px: int = 2
    seed_half_width_px: int = 80
    seed_height_px: int = 24
    minimum_component_pixels: int = 20
    maximum_lateral_jump_px: float = 90.0
    road_containment_tolerance_px: int = 8
    exclusion_overlap_ratio: float = 0.35
    polynomial_degree: int = 2
    temporal_alpha: float = 0.55
    temporal_straight_alpha: float = 0.28
    temporal_curve_alpha: float = 0.72
    temporal_small_shift_px: float = 6.0
    temporal_hysteresis_frames: int = 3
    temporal_consistent_boost: float = 0.30
    temporal_reacquire_alpha: float = 0.40
    temporal_state_timeout_sec: float = 0.5
    maximum_temporal_shift_px: float = 55.0
    max_temporal_fallback_frames: int = 5
    temporal_fallback_confidence_decay: float = 0.15
    max_single_boundary_fallback_frames: int = 5
    both_weight: float = 1.0
    single_weight: float = 0.72
    road_weight: float = 0.48
    temporal_weight: float = 0.25
    valid_min_points: int = 6
    valid_min_single_boundary_points: int = 5
    # Compatibility/diagnostic knob only; both_ratio is not a validity gate.
    valid_min_both_ratio: float = 0.0
    valid_min_confidence: float = 0.45
    valid_min_continuity_score: float = 0.65
    road_validation_min_rows: int = 3
    valid_min_road_containment_ratio: float = 0.5
    fit_overshoot_margin_px: float = 20.0
    center_direction_outlier_px: float = 30.0
    road_gross_outlier_margin_px: float = 35.0
    polynomial_residual_scale_px: float = 18.0
    curve_moderate_threshold_px: float = 0.2
    curve_sharp_threshold_px: float = 1.5
    lookahead_straight_ratio: float = 0.75
    lookahead_curve_ratio: float = 0.52
    lookahead_sharp_ratio: float = 0.32
    # Single-boundary rows only get a LEFT/RIGHT_BOUNDARY estimate when a
    # learned lane width exists; previously that required the exact sample
    # row to have seen a BOTH_BOUNDARIES detection at some point, so rows
    # that rarely (or never) get both edges at once (common near the top of
    # the ROI, where lines converge) always fell back to the coarser
    # ROAD_CENTER estimate even with a confident single edge. This radius
    # lets that row borrow the width learned at the nearest row instead,
    # since lane width changes gradually with y (perspective), not
    # row-to-row. Does not affect BOTH_BOUNDARIES detection or both_ratio.
    width_profile_lookup_radius_px: float = 30.0
    # Lane-width sanity gate (sections 9/10): a per-row pair is only treated as a
    # trustworthy lane pair, and only allowed to update the learned width profile,
    # if it stays close to the recently observed width at that row. A pair that
    # suddenly opens up (intersection / lane opening) is kept as a lower-confidence
    # sample instead of being fed back into the width model or trusted at full weight.
    lane_width_relative_tolerance: float = 0.5
    lane_width_absolute_tolerance_px: float = 30.0
    lane_width_min_px: float = 20.0
    lane_width_max_px: float = 600.0
    lane_width_update_alpha: float = 0.2
    # Metric policy is converted through the observed/learned lane width; the
    # image planner still performs no BEV projection and never uses a fixed
    # pixel offset.  The configured values must be measured for the vehicle
    # and test road before deployment.
    nominal_lane_width_m: float = 3.0
    minimum_boundary_clearance_m: float = 1.5
    vehicle_width_m: float = 0.46
    vehicle_boundary_margin_m: float = 0.1
    # Seed lane width (px) used to estimate the opposite boundary from a
    # single confident edge BEFORE any BOTH_BOUNDARIES detection has been
    # learned for a nearby row. Without this, boot-time frames that only ever
    # see one lane edge cannot produce a center estimate. Set <= 0 to require
    # a learned width. The learned width_profile always takes precedence over
    # this seed once available.
    lane_width_seed_px: float = 0.0
    # Road-only planning remains image-space.  These thresholds describe
    # directly observable pixels/ratios and do not invent a metre-per-pixel
    # conversion.  Tune them from labelled D456 frames before deployment.
    road_edge_clip_margin_px: int = 3
    road_clipped_weight_scale: float = 0.55
    road_near_field_height_px: int = 80
    road_minimum_near_coverage_ratio: float = 0.55
    road_minimum_near_width_px: float = 60.0
    road_width_max_relative_deviation: float = 0.45
    road_width_outlier_window_rows: int = 2
    road_center_spike_px: float = 35.0
    road_single_center_disagreement_ratio: float = 0.10
    road_single_boundary_blend: float = 0.70
    road_transition_smoothing_alpha: float = 0.35
    road_transition_smoothing_max_px: float = 24.0
    road_min_width_stability_score: float = 0.55
    road_min_center_score: float = 0.45
    road_branch_min_rows: int = 3
    road_branch_expansion_ratio: float = 2.2
    # A real-world road-class mask routinely has a few noisy pixels notched
    # out of its edge (curb/shadow/cone/segmentation dropout), especially
    # near the perspective vanishing point -- particularly one that is not
    # fully enclosed (so fill_road_holes' border-flood-fill correctly leaves
    # it alone) but is still just noise, not a second road. Runs separated
    # by a gap up to this many px are merged into one run for branch-
    # evidence / run_count purposes only (not for the road mask itself).
    road_branch_gap_tolerance_px: int = 14
    # A branch signal must repeat for this many consecutive frames before it
    # is trusted (req 3.B: transient marking-induced splits should not read
    # as a branch). Does not delay road-center/candidate-path computation,
    # only the BRANCH_SUSPECTED/branch_critical gates.
    # Default 1 preserves single-frame branch reporting (existing behavior /
    # tests); raise it in deployment config for the debounced req 3.B
    # behavior once running against live video.
    branch_confirm_frames: int = 1

    # --- Ego-vehicle (bumper/bonnet) exclusion (req 1) ---------------------
    # The vehicle's own front bumper/bonnet/mounting hardware is visible at
    # the bottom of the frame and must not participate in road-center,
    # branch-evidence, candidate-path, virtual-center, or jump-baseline
    # computation. Stop-line detection/tracking is intentionally NOT routed
    # through this exclusion (see stop_line_memory.py); it reads the raw
    # masks directly.
    ego_exclusion_enabled: bool = True
    # Ratio (of image height, from the bottom) of a simple horizontal cut.
    # This is the primary, resolution-independent guard: it raises the
    # effective ROI/near-field bottom row so bumper rows are never sampled.
    ego_exclusion_bottom_ratio: float = 0.12
    # Polygon refinement, normalized to (x_ratio, y_ratio) in [0, 1] with
    # y=1 at the image bottom. Lets an asymmetric hood/mirror-stalk shape
    # reach slightly higher than the flat bottom_ratio cut without having to
    # raise that cut for the whole frame width. Empty tuple disables the
    # polygon refinement (ratio cut only). Default is a shallow trapezoid
    # roughly matching a centered bonnet, tuned for 640x480. NOTE: the
    # polygon's highest point sets the single effective_roi_bottom row used
    # everywhere (near-field/branch/sampling), so it is deliberately kept
    # shallow -- a real vehicle with a tall centered antenna/wiring bundle
    # that pokes up well above the bonnet line needs its own, per-vehicle
    # measured polygon (a tall narrow peak here would otherwise shrink the
    # near-field window for the ENTIRE frame width, not just that column;
    # see the video-regression report's "remaining risks" for this task).
    ego_exclusion_polygon: tuple = (
        (0.16, 1.00), (0.84, 1.00), (0.72, 0.86), (0.28, 0.86))
    # When True, exclusion only suppresses branch-evidence rows (near-field
    # bumper clutter reading as a fork) and leaves road-center/candidate-path
    # sampling at the original ROI bottom. Default False applies the full
    # exclusion described in req 1.
    ego_exclusion_branch_only: bool = False

    # --- Road-marking (diamond/text/arrow) suppression (req 3) -------------
    # A. Hole-filling / closing so an interior marking (segmented as a
    # non-road class) does not read as a break in the road corridor.
    road_hole_fill_enabled: bool = True
    road_hole_close_kernel_px: int = 9
    # A near-field crosswalk-approach diamond/arrow marking is large by
    # design (meant to be highly visible) and can easily exceed a few
    # thousand pixels once it is close to the camera -- far more than a
    # small text glyph. The cap only needs to stay below a REAL enclosed
    # opening's area (rare -- most real branches/intersections touch the
    # far/side image border and are never "enclosed" in the first place),
    # so it is set generously rather than tuned to one marking's size.
    road_hole_max_area_px: int = 8000
    # C. Interior white/yellow blobs (diamonds, letters, arrows) must not
    # become LEFT/RIGHT boundary candidates. A component is treated as a
    # real lane-boundary segment only if it is long and thin (elongation)
    # or sits close to an already-observed road edge; otherwise it is
    # dropped before boundary sampling.
    marking_suppression_enabled: bool = True
    # A real boundary line, even drawn diagonally by perspective, stays thin
    # per-row (area/height); a diamond/letter/arrow blob does not. Bounding
    # box aspect ratio is deliberately NOT used here -- a diagonal line has
    # a large bbox width even though it is visually thin.
    marking_max_row_width_px: float = 18.0
    marking_min_length_px: float = 40.0
    marking_edge_margin_px: float = 18.0

    # --- Local jump rejection (req 4) --------------------------------------
    # _reject_direction_outliers additionally screens each interior point
    # for a heading-angle spike and a curvature (2nd-derivative) spike, not
    # just a lateral residual. All three are local-outlier removals (one
    # sample), never a whole-horizon discard.
    heading_jump_threshold_deg: float = 35.0
    curvature_jump_threshold_px: float = 45.0
    # Upper bound, as a fraction of the sampled points, on how many local
    # outliers a single frame may drop. Prevents a noisy near-bumper band
    # from cascading into raw len < 3 -> INVALID; once the cap is hit the
    # remaining flagged points are kept (down-weighted) instead of removed.
    max_local_outlier_ratio: float = 0.35
    # A stale previous path (still being leaned on only because recent
    # frames failed validity) must not lock the temporal blend against a
    # newly-fit, otherwise-good path. Once fallback_age reaches this many
    # frames, the outlier-lock branch of _temporal_blend is bypassed and the
    # new fit is trusted at temporal_stale_recovery_alpha or higher.
    temporal_stale_lock_max_frames: int = 3
    temporal_stale_recovery_alpha: float = 0.85

    # --- Dynamic PATH ROI / corridor (req: ROI separation) -----------------
    # ROAD ROI stays roi_top/roi_bottom (+ ego exclusion) and is used for the
    # road mask / near-field / branch evidence -- unchanged by this section.
    # PATH ROI is this dynamic corridor: candidate lane/boundary points are
    # only accepted within predicted_center_x(y) +/- half_width(y). It never
    # clips the road mask itself, only which points may become path
    # candidates, so it cannot cut off a real wide/curved/intersection road.
    path_corridor_enabled: bool = True
    # Half-width as a ratio of image width, interpolated near/mid/far by row
    # position. near < mid < far by default (near-field perspective keeps
    # true lane edges close to the predicted center; far-field rows carry
    # more perspective/segmentation noise and need more slack).
    path_corridor_near_half_width_ratio: float = 0.14
    path_corridor_mid_half_width_ratio: float = 0.22
    path_corridor_far_half_width_ratio: float = 0.32
    path_corridor_min_half_width_px: float = 28.0
    # predicted_center_x(y) blends the previous final path (when available
    # at that row) with the current ego-connected road center at that row;
    # 1.0 = previous path only, 0.0 = road center only.
    path_corridor_previous_path_weight: float = 0.7
    path_corridor_expand_when_lost: bool = True
    path_corridor_max_expand_ratio: float = 2.5
    path_corridor_expand_step: float = 0.25
    path_corridor_shrink_step: float = 0.15

    # --- Boundary track filtering (marking suppression, stage 2) ----------
    # suppress_interior_markings() removes marking-shaped mask blobs before
    # sampling; this filters the assembled per-row left/right point
    # sequences themselves, catching a marking whose individual stroke is
    # locally thin enough to pass the mask filter but whose row-to-row
    # position zigzags (a diamond's two slanted edges, letter strokes).
    boundary_track_filter_enabled: bool = True
    boundary_min_vertical_span_ratio: float = 0.12
    # Max allowed change in local row-to-row slope (px of x per px of y)
    # between two adjacent segments of one boundary track. A real boundary
    # line's slope changes gradually with perspective/curvature; a marking
    # edge whips back and forth over a few rows.
    boundary_max_lateral_slope_change: float = 0.9
    boundary_temporal_tolerance_px: float = 40.0
    boundary_min_track_score: float = 0.5

    # --- Jump rejection reorder (raw-stage repair, final-stage warning) ---
    # Raw-stage local lateral/heading spikes are now REPAIRED (linear
    # interpolation from neighbors) instead of deleted, so one bad sample
    # never shrinks the horizon or cascades into rejecting later points.
    # Pixel curvature is checked on the final fitted+temporal path as a soft
    # repair warning. Controller-equivalent steering below is the hard gate.
    final_curvature_jump_px: float = 60.0

    # Horizon recovery and final vehicle-feasibility policy.  Candidate
    # geometry thresholds are warnings; the controller-equivalent steering
    # calculation below is the physical hard gate.
    target_path_points: int = 24
    minimum_drivable_horizon_ratio: float = 0.50
    max_gap_repair_rows: int = 4
    maximum_steering_deg: float = 22.0
    steering_lookahead_y_ratio: float = 0.50
    steering_proportional_gain_deg_per_norm: float = 22.0
    steering_derivative_gain_deg_per_norm_per_s: float = 2.0
    max_steering_rate_deg_per_sec: float = 180.0
    max_steering_delta_deg_per_frame: float = 12.0
    max_steering_delta_deg_per_segment: float = 12.0
    steering_repair_previous_weight: float = 0.65
    nominal_frame_period_sec: float = 0.05

    # --- ROAD_ONLY fallback (req 6) -----------------------------------
    # Looser point requirement when every accepted source is ROAD_CENTER:
    # lane detection can be completely absent while the road itself is a
    # perfectly safe, drivable corridor.
    valid_min_road_only_points: int = 4

    # --- Source-mode hysteresis (req 7) -------------------------------
    # BOTH_LANES/ONE_LANE/ROAD_ONLY/MIXED/TEMPORAL classification only
    # switches after this many consecutive frames agree, unless the
    # candidate path's near-field x barely moved (switching then cannot
    # itself cause a visible jump).
    source_confirm_frames: int = 2
    source_release_frames: int = 3
    source_switch_max_lateral_px: float = 45.0
    # Publish-boundary hard safety envelope.  Every final point is projected
    # into the exact ego-connected road run at its own image row, with a
    # perspective-aware horizontal clearance.  These are image-space safety
    # pixels, not fabricated metric calibration.
    final_path_safety_margin_near_px: float = 12.0
    final_path_safety_margin_mid_px: float = 8.0
    final_path_safety_margin_far_px: float = 4.0
    # req 8: a continuity_score dip below valid_min_continuity_score alone
    # (a couple of jump points, some sparse loss) only demotes VALID to
    # DEGRADED -- but a horizon this shredded (near-continuous rejection,
    # only a noise-level handful of points surviving) is the "sustained
    # severe geometry violation" case that IS a hard INVALID.
    continuity_hard_invalid_floor: float = 0.15

    # --- Steering-only validity (BEV standalone) ----------------------
    # When True, plan() marks a frame INVALID only if the path cannot be
    # expressed (no ego-connected road / fewer than the required points) or
    # the required steering exceeds maximum_steering_deg. Near-field loss,
    # branch_critical, clearance/physical geometry, final-road projection and
    # topology are downgraded from hard INVALID to DEGRADED so the BEV/metric
    # stack still receives a drivable path. Leave False for the strict
    # production gate.
    steering_only_validity: bool = False


@dataclass
class PathResult:
    points: np.ndarray
    sources: list
    confidence: float
    state: str
    valid: bool
    latency_ms: float
    road_component: np.ndarray
    left: np.ndarray
    right: np.ndarray
    raw: np.ndarray
    diagnostics: dict = None
    confidence_components: dict = None
    virtual: np.ndarray = None
    virtual_details: list = None


def _binary(mask):
    return (np.asarray(mask) > 0).astype(np.uint8)


def ego_connected_component(road, center_x, seed_half_width, seed_height,
                            minimum_pixels=20, seed_bottom_row=None):
    mask = _binary(road)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    height, width = mask.shape
    bottom = height if seed_bottom_row is None else max(1, min(height, int(seed_bottom_row)))
    x1, x2 = max(0, int(center_x-seed_half_width)), min(width, int(center_x+seed_half_width+1))
    y1 = max(0, bottom-int(seed_height))
    seed_labels = labels[y1:bottom, x1:x2]
    candidates = [(int(np.count_nonzero(seed_labels == label)), label)
                  for label in range(1, count)
                  if stats[label, cv2.CC_STAT_AREA] >= minimum_pixels]
    if not candidates or max(candidates)[0] == 0:
        return np.zeros_like(mask)
    label = max(candidates)[1]
    return (labels == label).astype(np.uint8)


def exclude_transverse_markings(lane_mask, words, stop, c_line, ratio=0.35):
    lane = _binary(lane_mask)
    exclusion = _binary(words) | _binary(stop) | _binary(c_line)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
    output = np.zeros_like(lane)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & exclusion.astype(bool)))
        if area and overlap/area < ratio:
            output[component] = 1
    return output


def exclude_one_semantic(lane_mask, exclusion_mask, ratio):
    lane = _binary(lane_mask); exclusion = _binary(exclusion_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
    output = np.zeros_like(lane); removed = 0
    for label in range(1, count):
        component = labels == label; area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero(component & exclusion.astype(bool)))
        if area and overlap/area >= ratio:
            removed += 1
        else:
            output[component] = 1
    return output, removed


def fill_road_holes(road_mask, close_kernel_px=9, max_hole_area_px=1600):
    """Close small interior gaps (diamond/text/arrow markings) in a road mask.

    A pixel-perfect segmentation of the road class leaves a hole wherever a
    road-surface marking was classified as a different (non-road) class.
    Without retraining, those holes must not read as a break in the road
    corridor (req 3.A). A hole is filled when it is fully enclosed by road
    (does not touch the image border -- an off-road gap or a genuine branch
    opening reaches the border/far edge) and its area is below the cap, so a
    real large opening (intersection) is left alone.
    """
    mask = _binary(road_mask)
    kernel_size = max(1, int(close_kernel_px))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    height, width = mask.shape
    flood = mask.copy()
    flood_flags = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(flood, flood_flags, (0, 0), 1)
    # Background pixels the border flood-fill never reached are enclosed by
    # road on all sides -- hole candidates.
    holes = (flood == 0).astype(np.uint8)
    if not np.any(holes):
        return mask
    if max_hole_area_px and max_hole_area_px > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
        kept = np.zeros_like(holes)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= max_hole_area_px:
                kept[labels == label] = 1
        holes = kept
    return np.clip(mask | holes, 0, 1).astype(np.uint8)


def suppress_interior_markings(lane_mask, road_mask, max_row_width_px=18.0,
                               min_length_px=40.0, edge_margin_px=18.0):
    """Drop lane-mask blobs that are road-surface markings, not boundaries.

    A crosswalk-approach diamond/letter/arrow is frequently painted white and
    can be picked up by the white-line class. It is kept as a lane-boundary
    candidate only if it stays thin on a per-row basis over a long vertical
    span (a real line segment -- even drawn diagonally by perspective, its
    bounding-box width can be large, so aspect ratio alone is not used) or
    sits close to an already-observed road edge; a compact, isolated,
    centered blob is removed (req 3.C).
    """
    lane = _binary(lane_mask)
    road = _binary(road_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
    output = np.zeros_like(lane)
    height, width = lane.shape
    removed = 0
    for label in range(1, count):
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        avg_row_width = area/max(1.0, float(comp_h))
        boundary_like = (comp_h >= min_length_px and
                         avg_row_width <= max_row_width_px)
        mid_y = y0+comp_h//2
        y1, y2 = max(0, mid_y-2), min(height, mid_y+3)
        row_x = np.flatnonzero(np.any(road[y1:y2], axis=0))
        near_edge = True
        if len(row_x):
            comp_center = x0+comp_w/2.0
            edge_distance = min(abs(comp_center-float(row_x.min())),
                                abs(comp_center-float(row_x.max())))
            near_edge = edge_distance <= edge_margin_px
        if boundary_like or near_edge:
            output[labels == label] = 1
        else:
            removed += 1
    return output, removed


def ego_exclusion_mask(shape, bottom_ratio=0.12, polygon=(), enabled=True):
    """Pixel mask covering the vehicle bumper/bonnet/mounts (req 1).

    ``bottom_ratio`` is a simple, resolution-independent horizontal cut from
    the image bottom; ``polygon`` (normalized (x_ratio, y_ratio) points, y=1
    at the bottom) refines that cut for an asymmetric shape. Both stay
    ratio-based so 640x480 tuning holds at other resolutions.
    """
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if not enabled:
        return mask
    ratio = float(np.clip(bottom_ratio, 0.0, 0.9))
    if ratio > 0.0:
        cutoff = height-int(round(ratio*height))
        mask[max(0, cutoff):height, :] = 1
    if polygon:
        points = np.asarray([[x*width, y*height] for x, y in polygon],
                            dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


def ego_exclusion_top_row(shape, bottom_ratio=0.12, polygon=(),
                          roi_top=0, enabled=True):
    """Highest row (smallest y) excluded by :func:`ego_exclusion_mask`."""
    height, width = shape
    if not enabled:
        return height
    ratio = float(np.clip(bottom_ratio, 0.0, 0.9))
    top = height-int(round(ratio*height)) if ratio > 0.0 else height
    if polygon:
        top_ratio = min(float(y) for _, y in polygon)
        top = min(top, height-int(round((1.0-top_ratio)*height)))
    return max(int(roi_top)+1, top)


def _runs(xs, gap_tolerance_px=1):
    """Contiguous x-runs, merging internal gaps up to ``gap_tolerance_px``.

    The default (1) is the original strict behavior (any single-pixel gap
    splits). A larger tolerance is used for branch-evidence run counting
    (see ``_road_geometry``): a real-world road-class segmentation mask
    routinely has a few noisy pixels notched out along its edge near the
    vanishing point/curb/shadow -- a 2-5px gap there is segmentation noise,
    not a second road. It must not read as a branch fork.
    """
    if not len(xs):
        return []
    split = np.where(np.diff(xs) > max(1, int(gap_tolerance_px)))[0] + 1
    return [part for part in np.split(xs, split) if len(part)]


class ImagePathPlanner:
    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.previous = None
        self.previous_confidence = 0.0
        self.fallback_age = 0
        self.single_boundary_age = 0
        self.width_profile = {}
        self.shift_direction = 0
        self.consistent_shift_frames = 0
        self._last_virtual_points = np.empty((0, 2), dtype=float)
        self._last_virtual_details = []
        self._last_ego_anchor_applied = False
        self.previous_boundary_mode = None
        self.previous_source_mode = None
        self._pending_width_updates = []
        self.last_plan_time = None
        self._branch_suspected_streak = 0
        self._branch_critical_streak = 0
        self._corridor_expand_level = 0.0
        self._corridor_state = "TRACKING"
        self._previous_left_track = {}
        self._previous_right_track = {}
        self._source_confirmed_mode = None
        self._source_pending_mode = None
        self._source_pending_streak = 0
        self._source_release_streak = 0
        self._last_confirmed_near_x = None
        self._previous_feasibility_offset_px = None
        self._previous_feasibility_steering_deg = None
        self._previous_feasibility_timestamp_sec = None

    def reset(self):
        self.previous = None
        self.previous_confidence = 0.0
        self.fallback_age = 0
        self.single_boundary_age = 0
        self.width_profile.clear()
        self.shift_direction = 0
        self.consistent_shift_frames = 0
        self._last_virtual_points = np.empty((0, 2), dtype=float)
        self._last_virtual_details = []
        self._last_ego_anchor_applied = False
        self.previous_boundary_mode = None
        self.previous_source_mode = None
        self._pending_width_updates = []
        self.last_plan_time = None
        self._branch_suspected_streak = 0
        self._branch_critical_streak = 0
        self._corridor_expand_level = 0.0
        self._corridor_state = "TRACKING"
        self._previous_left_track = {}
        self._previous_right_track = {}
        self._source_confirmed_mode = None
        self._source_pending_mode = None
        self._source_pending_streak = 0
        self._source_release_streak = 0
        self._last_confirmed_near_x = None
        self._previous_feasibility_offset_px = None
        self._previous_feasibility_steering_deg = None
        self._previous_feasibility_timestamp_sec = None

    def reset_temporal(self):
        """Drop stale frame-to-frame state while retaining learned lane width."""
        self.previous = None
        self.previous_confidence = 0.0
        self.fallback_age = 0
        self.single_boundary_age = 0
        self.shift_direction = 0
        self.consistent_shift_frames = 0
        self.previous_boundary_mode = None
        self.previous_source_mode = None
        self._branch_suspected_streak = 0
        self._branch_critical_streak = 0
        self._corridor_expand_level = 0.0
        self._corridor_state = "TRACKING"
        self._previous_left_track = {}
        self._previous_right_track = {}
        self._source_confirmed_mode = None
        self._source_pending_mode = None
        self._source_pending_streak = 0
        self._source_release_streak = 0
        self._last_confirmed_near_x = None
        self._previous_feasibility_offset_px = None
        self._previous_feasibility_steering_deg = None
        self._previous_feasibility_timestamp_sec = None

    def inactive(self, shape=(480, 640)):
        self.reset()
        empty = np.empty((0, 2), dtype=float)
        return PathResult(empty, [], 0.0, INACTIVE, False, 0.0,
                          np.zeros(shape, np.uint8), empty, empty, empty)

    @staticmethod
    def _row_list(height, roi_top, roi_bottom, interval):
        top, bottom = max(0, int(roi_top)), min(int(height)-1, int(roi_bottom))
        return list(range(bottom, top-1, -max(1, int(interval))))

    def _samples(self, road, lane, roi_bottom=None, corridor=None):
        cfg = self.config
        height, width = road.shape
        bottom_cfg = cfg.roi_bottom if roi_bottom is None else roi_bottom
        ys = self._row_list(height, cfg.roi_top, bottom_cfg, cfg.sample_interval_px)
        left, right, road_edges = [], [], []
        counters = {"NO_ROAD": 0, "NO_BOUNDARY": 0, "LATERAL_JUMP": 0,
                    "ROAD_CONTAINMENT_FAILURE": 0, "CORRIDOR_REJECT": 0}
        previous_left = previous_right = None
        left_missing = right_missing = False
        for y in ys:
            y1, y2 = max(0, y-cfg.sample_band_half_height_px), min(height, y+cfg.sample_band_half_height_px+1)
            road_x = np.flatnonzero(np.any(road[y1:y2], axis=0))
            if len(road_x):
                road_edges.append((y, float(road_x.min()), float(road_x.max())))
            else:
                # Road is supporting evidence, not a prerequisite for reading
                # the lane mask. A missing / holey road mask must not hide two
                # otherwise clear yellow boundaries.
                counters["NO_ROAD"] += 1
            lane_x = np.flatnonzero(np.any(lane[y1:y2], axis=0))
            candidates = [float(np.mean(run)) for run in _runs(lane_x)
                          if len(run) >= max(1, cfg.minimum_component_pixels//10)]
            lc = [x for x in candidates if x < cfg.vehicle_center_x_px]
            rc = [x for x in candidates if x > cfg.vehicle_center_x_px]
            if lc and rc:
                lx, rx = max(lc), min(rc)
            elif (len(candidates) >= 2 and previous_left is not None and
                  previous_right is not None):
                # Both perspective boundaries may lie on the same side of the
                # fixed image center after the vehicle moves laterally. Keep
                # their geometric order and, when tracks exist, select the
                # ordered pair closest to those tracks.
                ordered = sorted(candidates)
                lx, rx = min(
                    ((left_x, right_x) for index, left_x in enumerate(ordered[:-1])
                     for right_x in ordered[index+1:]),
                    key=lambda pair: (abs(pair[0]-previous_left) +
                                      abs(pair[1]-previous_right)))
            elif candidates:
                if previous_left is not None and previous_right is not None:
                    candidate = min(
                        candidates,
                        key=lambda value: min(abs(value-previous_left),
                                              abs(value-previous_right)))
                    if abs(candidate-previous_left) <= abs(candidate-previous_right):
                        lx, rx = candidate, None
                    else:
                        lx, rx = None, candidate
                elif previous_left is not None:
                    candidate = min(candidates, key=lambda value: abs(value-previous_left))
                    lx, rx = candidate, None
                elif previous_right is not None:
                    candidate = min(candidates, key=lambda value: abs(value-previous_right))
                    lx, rx = None, candidate
                elif candidates[-1] < cfg.vehicle_center_x_px:
                    lx, rx = max(candidates), None
                else:
                    lx, rx = None, min(candidates)
            else:
                lx = rx = None
            # Dynamic PATH ROI (req: ROI separation, section 3): a candidate
            # outside predicted_center_x(y) +/- half_width(y) is dropped
            # before it can ever become a lane-boundary sample. This never
            # touches the road mask itself (ROAD/BRANCH ROI), only which
            # points are eligible for path/candidate generation.
            if corridor is not None:
                bounds = corridor.get(int(y))
                if bounds is not None:
                    lo, hi = bounds
                    if lx is not None and not lo <= lx <= hi:
                        lx = None; counters["CORRIDOR_REJECT"] += 1
                    if rx is not None and not lo <= rx <= hi:
                        rx = None; counters["CORRIDOR_REJECT"] += 1
            observed_left = lx is not None
            observed_right = rx is not None
            if lx is None and rx is None: counters["NO_BOUNDARY"] += 1
            # Only compare adjacent observations. After one or more missing
            # bands, perspective can move a valid boundary far from its last
            # observed x. The reacquired boundary is still checked by lane
            # width, center continuity, road-corridor, direction-outlier, and
            # temporal gates in _raw_path / plan.
            if (lx is not None and previous_left is not None and not left_missing and
                    abs(lx-previous_left) > cfg.maximum_lateral_jump_px):
                lx = None; counters["LATERAL_JUMP"] += 1
            if (rx is not None and previous_right is not None and not right_missing and
                    abs(rx-previous_right) > cfg.maximum_lateral_jump_px):
                rx = None; counters["LATERAL_JUMP"] += 1
            if lx is not None:
                left.append((lx, float(y))); previous_left = lx
            if rx is not None:
                right.append((rx, float(y))); previous_right = rx
            left_missing = not observed_left
            right_missing = not observed_right
        return (ys, np.asarray(left, float).reshape(-1, 2),
                np.asarray(right, float).reshape(-1, 2), road_edges, counters)

    @staticmethod
    def _at_y(points, y, tolerance):
        if not len(points): return None
        index = int(np.argmin(np.abs(points[:, 1]-y)))
        return float(points[index, 0]) if abs(points[index, 1]-y) <= tolerance else None

    def _width_near(self, y, radius):
        if self.width_profile:
            rows = np.asarray(sorted(self.width_profile), dtype=float)
            if rows[0] <= y <= rows[-1]:
                values = np.asarray([self.width_profile[int(row)] for row in rows])
                return float(np.interp(float(y), rows, values))
            nearest = int(rows[np.argmin(np.abs(rows-y))])
            if abs(nearest-y) <= radius:
                return float(self.width_profile[nearest])
        # No learned width nearby: use the configured seed if enabled. This
        # only affects single-boundary rows (opposite-edge estimation); it
        # never overrides a learned width and never touches BOTH_BOUNDARIES.
        seed = getattr(self.config, "lane_width_seed_px", 0.0)
        return seed if seed and seed > 0.0 else None

    def _road_bounds_at_y(self, road, y):
        band = max(1, self.config.sample_band_half_height_px)
        y1, y2 = max(0, int(y)-band), min(road.shape[0], int(y)+band+1)
        xs = np.flatnonzero(np.any(road[y1:y2], axis=0))
        if not len(xs):
            return None
        return float(xs.min()), float(xs.max())

    def _predicted_center_map(self, ys, road_geometry):
        """predicted_center_x(y): previous final path blended with the
        current ego-connected road center, falling back to image center.

        Priority (per row, when available): recent VALID/DEGRADED final
        path (``self.previous``) blended with ego-connected road center via
        ``path_corridor_previous_path_weight``; road center alone; then the
        image center. ``self.previous`` also stands in for "recent temporal
        path" since it is the last accepted fitted path in every case
        (including the TEMPORAL_FALLBACK branch, which reuses it directly).
        """
        cfg = self.config
        weight = float(np.clip(cfg.path_corridor_previous_path_weight, 0.0, 1.0))
        mapping = {}
        for y in ys:
            prev_x = None
            if self.previous is not None and len(self.previous):
                prev_x = self._at_y(self.previous, y, cfg.sample_interval_px*3)
            item = road_geometry.get(int(y))
            road_x = item["center"] if item is not None else None
            if prev_x is not None and road_x is not None:
                center, source = weight*prev_x+(1.0-weight)*road_x, "previous+road"
            elif prev_x is not None:
                center, source = prev_x, "previous_path"
            elif road_x is not None:
                center, source = road_x, "road_center"
            else:
                center, source = float(cfg.vehicle_center_x_px), "image_center"
            mapping[int(y)] = (float(center), source)
        return mapping

    def _corridor_half_width_px(self, y, width, roi_top, roi_bottom):
        cfg = self.config
        pos = float(np.clip((float(y)-roi_top)/max(1.0, roi_bottom-roi_top), 0.0, 1.0))
        if pos <= 0.5:
            frac = pos/0.5
            ratio = ((1.0-frac)*cfg.path_corridor_far_half_width_ratio +
                     frac*cfg.path_corridor_mid_half_width_ratio)
        else:
            frac = (pos-0.5)/0.5
            ratio = ((1.0-frac)*cfg.path_corridor_mid_half_width_ratio +
                     frac*cfg.path_corridor_near_half_width_ratio)
        base = ratio*float(width)
        # No track has ever been established yet (cold start / just after a
        # reset): there is no basis to narrow the search, so search as wide
        # as EXPANDING/LOST would, independent of the expand-level state
        # machine (which only reacts to a previously-tracked path failing).
        expand_level = (1.0 if self.previous is None
                        else self._corridor_expand_level)
        expand = 1.0+expand_level*(cfg.path_corridor_max_expand_ratio-1.0)
        return max(cfg.path_corridor_min_half_width_px, base*expand)

    def _corridor_bounds_map(self, ys, predicted_map, width, roi_top, roi_bottom):
        bounds = {}
        for y in ys:
            item = predicted_map.get(int(y))
            if item is None:
                continue
            center, _ = item
            half = self._corridor_half_width_px(y, width, roi_top, roi_bottom)
            bounds[int(y)] = (center-half, center+half)
        return bounds

    def _update_corridor_state(self, tracking_ok):
        """Progressive corridor widen-on-loss / narrow-on-reacquire (section 2).

        A lost path widens the corridor gradually over several frames
        instead of snapping to the full ROAD ROI in one frame; a reacquired
        path narrows it back down just as gradually.
        """
        cfg = self.config
        if tracking_ok:
            self._corridor_expand_level = max(
                0.0, self._corridor_expand_level-cfg.path_corridor_shrink_step)
            self._corridor_state = ("TRACKING" if self._corridor_expand_level <= 1e-6
                                    else "REACQUIRING")
        else:
            if cfg.path_corridor_expand_when_lost:
                self._corridor_expand_level = min(
                    1.0, self._corridor_expand_level+cfg.path_corridor_expand_step)
            self._corridor_state = ("LOST" if self._corridor_expand_level >= 1.0-1e-6
                                    else "EXPANDING")

    def _boundary_track_filter(self, points, previous_track):
        """Reject a boundary track's zigzag rows (req: marking as track, not class).

        A diamond/letter's edge can be locally thin enough to survive
        suppress_interior_markings() yet whip the row-to-row x position back
        and forth. This screens the assembled left/right point sequence for
        that pattern (large local slope change), disagreement with the
        previous frame's track at the same rows, and overall vertical span,
        producing a 0..1 ``boundary_track_score``.
        """
        cfg = self.config
        if not len(points):
            return points, 1.0, 0
        order = np.argsort(points[:, 1])
        pts = points[order]
        ys_arr, xs_arr = pts[:, 1], pts[:, 0]
        n = len(pts)
        keep = np.ones(n, dtype=bool)
        if n >= 3:
            dy = np.diff(ys_arr)
            dy = np.where(np.abs(dy) < 1e-6, 1e-6, dy)
            slopes = np.diff(xs_arr)/dy
            slope_change = np.abs(np.diff(slopes))
            for i in range(len(slope_change)):
                if slope_change[i] > cfg.boundary_max_lateral_slope_change:
                    keep[i+1] = False
        if previous_track:
            rows_prev = np.asarray(sorted(previous_track), dtype=float)
            for i in range(n):
                if not keep[i]:
                    continue
                nearest = rows_prev[int(np.argmin(np.abs(rows_prev-ys_arr[i])))]
                if (abs(nearest-ys_arr[i]) <= cfg.sample_interval_px*2 and
                        abs(previous_track[int(nearest)]-xs_arr[i]) >
                        cfg.boundary_temporal_tolerance_px):
                    keep[i] = False
        span = float(ys_arr.max()-ys_arr.min()) if n else 0.0
        roi_span = max(1.0, cfg.roi_bottom-cfg.roi_top)
        span_ratio = span/roi_span
        retained_ratio = float(np.count_nonzero(keep))/max(1, n)
        score = retained_ratio*(1.0 if span_ratio >= cfg.boundary_min_vertical_span_ratio else 0.4)
        removed = int(n-np.count_nonzero(keep))
        if score < cfg.boundary_min_track_score:
            return np.empty((0, 2), float), score, n
        return pts[keep], score, removed

    def _update_source_hysteresis(self, raw_mode, near_x):
        """Debounce BOTH_LANES/ONE_LANE/ROAD_ONLY/MIXED switching (req 7)."""
        cfg = self.config
        if self._source_confirmed_mode is None:
            self._source_confirmed_mode = raw_mode
            self._source_pending_mode = raw_mode
            self._source_pending_streak = 0
            self._last_confirmed_near_x = near_x
            return self._source_confirmed_mode, True
        if raw_mode == self._source_confirmed_mode:
            self._source_pending_streak = 0
            self._source_release_streak = 0
            self._last_confirmed_near_x = near_x
            return self._source_confirmed_mode, False
        self._source_release_streak += 1
        if raw_mode == self._source_pending_mode:
            self._source_pending_streak += 1
        else:
            self._source_pending_mode = raw_mode
            self._source_pending_streak = 1
        transitioned = False
        # Confirmation proves that the replacement is persistent; release
        # proves that the current source really disappeared.  Both gates are
        # required, so a one/two-frame segmentation flicker cannot bounce the
        # source state even when its lateral position happens to be close.
        if (self._source_pending_streak >= max(1, cfg.source_confirm_frames) and
                self._source_release_streak >= max(1, cfg.source_release_frames)):
            self._source_confirmed_mode = raw_mode
            self._source_pending_streak = 0
            self._source_release_streak = 0
            transitioned = True
            self._last_confirmed_near_x = near_x
        return self._source_confirmed_mode, transitioned

    def _road_geometry(self, road, ys):
        """Track one ego-connected road run and score row continuity.

        Rows are visited near-to-far.  When a connected component forks, the
        run closest to the preceding center is retained; the ambiguity is
        exposed separately instead of silently bridging the two branches.
        """
        cfg = self.config
        mask = _binary(road)
        height, width = mask.shape
        observations = {}
        previous_center = float(cfg.vehicle_center_x_px)
        for y in ys:
            band = max(1, cfg.sample_band_half_height_px)
            y1, y2 = max(0, int(y)-band), min(height, int(y)+band+1)
            runs = _runs(np.flatnonzero(np.any(mask[y1:y2], axis=0)),
                        cfg.road_branch_gap_tolerance_px)
            if not runs:
                continue
            selected = min(
                runs,
                key=lambda run: (0.0 if float(run[0]) <= previous_center <=
                                  float(run[-1]) else
                                  min(abs(previous_center-float(run[0])),
                                      abs(previous_center-float(run[-1]))),
                                 -len(run)))
            left, right = float(selected[0]), float(selected[-1])
            raw_center = 0.5*(left+right)
            left_clipped = left <= cfg.road_edge_clip_margin_px
            right_clipped = right >= width-1-cfg.road_edge_clip_margin_px
            if left_clipped and right_clipped:
                center = previous_center
            elif left_clipped or right_clipped:
                # A clipped edge is the FOV limit, not a measured road edge.
                # Keep most of the near-to-far center trend and admit only a
                # small portion of the raw midpoint.
                center = 0.75*previous_center+0.25*raw_center
            else:
                center = raw_center
            observations[int(y)] = {
                "left": left, "right": right, "width": right-left+1.0,
                "center": center, "raw_center": raw_center,
                "run_count": len(runs),
                "left_clipped": left_clipped,
                "right_clipped": right_clipped,
                "width_outlier": False, "center_spike": False,
                "width_score": 1.0,
            }
            previous_center = center

        ordered = [int(y) for y in ys if int(y) in observations]
        radius = max(1, int(cfg.road_width_outlier_window_rows))
        for index, y in enumerate(ordered):
            neighbor_rows = ordered[max(0, index-radius):index] + \
                ordered[index+1:index+radius+1]
            if neighbor_rows:
                local_width = float(np.median([
                    observations[row]["width"] for row in neighbor_rows]))
                deviation = abs(observations[y]["width"]-local_width) / \
                    max(1.0, local_width)
                observations[y]["width_score"] = max(
                    0.0, 1.0-deviation/max(
                        1e-6, cfg.road_width_max_relative_deviation))
                observations[y]["width_outlier"] = (
                    deviation > cfg.road_width_max_relative_deviation)
            if 0 < index < len(ordered)-1:
                expected = 0.5*(observations[ordered[index-1]]["center"] +
                                observations[ordered[index+1]]["center"])
                observations[y]["center_spike"] = (
                    abs(observations[y]["center"]-expected) >
                    cfg.road_center_spike_px)
        return observations

    def _road_safety(self, observations, ys, ego_component, roi_bottom=None):
        cfg = self.config
        bottom_cfg = cfg.roi_bottom if roi_bottom is None else roi_bottom
        near_rows = [int(y) for y in ys
                     if y >= bottom_cfg-cfg.road_near_field_height_px]
        observed_near = [observations[y] for y in near_rows
                         if y in observations]
        coverage = len(observed_near)/max(1, len(near_rows))
        median_width = (float(np.median([item["width"]
                                        for item in observed_near]))
                        if observed_near else 0.0)
        width_factor = min(1.0, median_width/max(
            1.0, cfg.road_minimum_near_width_px))
        near_score = min(coverage, width_factor)
        near_ok = bool(
            np.any(ego_component) and
            coverage >= cfg.road_minimum_near_coverage_ratio and
            median_width >= cfg.road_minimum_near_width_px)

        values = list(observations.values())
        stable = sum(not item["width_outlier"] for item in values)
        width_stability = stable/max(1, len(values))
        spike_count = sum(item["center_spike"] for item in values)
        clipped_count = sum(item["left_clipped"] or item["right_clipped"]
                            for item in values)
        center_score = max(
            0.0, 1.0-spike_count/max(1, len(values)))
        if values:
            clip_ratio = clipped_count/len(values)
            center_score *= 1.0-0.35*clip_ratio

        branch_rows = sum(item["run_count"] > 1 for item in values)
        near_widths = [item["width"] for item in observed_near]
        baseline_width = float(np.median(near_widths)) if near_widths else 0.0
        expanded_rows = sum(
            baseline_width > 0.0 and
            item["width"] > cfg.road_branch_expansion_ratio*baseline_width
            for item in values)
        branch_suspected = max(branch_rows, expanded_rows) >= \
            cfg.road_branch_min_rows
        near_branch_rows = sum(item["run_count"] > 1
                               for item in observed_near)
        branch_critical = near_branch_rows >= cfg.road_branch_min_rows
        return {
            "near_field_score": float(near_score),
            "near_field_ok": near_ok,
            "near_field_coverage_ratio": float(coverage),
            "near_field_median_width_px": float(median_width),
            "road_width_stability_score": float(width_stability),
            "road_center_score": float(center_score),
            "road_width_outlier_rows": len(values)-stable,
            "road_center_spike_rows": spike_count,
            "road_clipped_rows": clipped_count,
            "branch_rows": branch_rows,
            "branch_expansion_rows": expanded_rows,
            "branch_suspected": branch_suspected,
            "branch_critical": branch_critical,
        }

    def _smooth_source_transitions(self, points, sources, weights):
        if len(points) < 2:
            return points, weights, 0
        cfg = self.config
        output = points.copy()
        smoothed = 0
        for index in range(1, len(output)):
            if sources[index] == sources[index-1]:
                continue
            # BOTH is the anchor.  Only move the lower-reliability side of the
            # boundary and only over one adjacent row.
            movable = index if sources[index] != BOTH_BOUNDARIES else index-1
            anchor = index-1 if movable == index else index
            delta = output[anchor, 0]-output[movable, 0]
            correction = np.clip(
                cfg.road_transition_smoothing_alpha*delta,
                -cfg.road_transition_smoothing_max_px,
                cfg.road_transition_smoothing_max_px)
            output[movable, 0] += correction
            weights[movable] *= 0.95
            smoothed += 1
        return output, weights, smoothed

    @staticmethod
    def _source_mode(sources):
        count = max(1, len(sources))
        both = sources.count(BOTH_BOUNDARIES)/count
        road = sources.count(ROAD_CENTER)/count
        single = sum(source in (LEFT_BOUNDARY, RIGHT_BOUNDARY)
                     for source in sources)/count
        if both >= 0.6:
            return LANE_DOMINANT
        if road >= 0.6:
            return ROAD_DOMINANT
        if both or single or road:
            return MIXED_SOURCES
        return None

    def _clearance_fraction(self):
        cfg = self.config
        values = (cfg.nominal_lane_width_m,
                  cfg.minimum_boundary_clearance_m,
                  cfg.vehicle_width_m, cfg.vehicle_boundary_margin_m)
        if not all(np.isfinite(value) for value in values):
            return None
        if (cfg.nominal_lane_width_m <= 0.0 or
                cfg.minimum_boundary_clearance_m <= 0.0 or
                cfg.vehicle_width_m <= 0.0 or
                cfg.vehicle_boundary_margin_m < 0.0):
            return None
        clearance_m = max(
            cfg.minimum_boundary_clearance_m,
            cfg.vehicle_width_m/2.0 + cfg.vehicle_boundary_margin_m)
        if clearance_m > cfg.nominal_lane_width_m/2.0:
            return None
        return float(clearance_m/cfg.nominal_lane_width_m)

    def _road_width_from_boundary(self, road, boundary_point, source):
        """Estimate a boot-time lane width from the observed road corridor."""
        bounds = self._road_bounds_at_y(road, boundary_point[1])
        if bounds is None:
            return None
        if source == LEFT_BOUNDARY:
            width = bounds[1]-float(boundary_point[0])
        else:
            width = float(boundary_point[0])-bounds[0]
        if not self.config.lane_width_min_px <= width <= self.config.lane_width_max_px:
            return None
        return float(width)

    def _virtual_clearance_ok(self, detail):
        cfg = self.config
        body_clearance_m = (cfg.vehicle_width_m/2.0 +
                            cfg.vehicle_boundary_margin_m)
        required_px = (detail["lane_width_px"]*body_clearance_m /
                       cfg.nominal_lane_width_m)
        return (detail["offset_px"] >= required_px and
                detail["lane_width_px"]-detail["offset_px"] >= required_px)

    def _normal_offset_curve(self, boundary, source, road, roi_bottom=None,
                             corridor=None):
        """Offset a sampled boundary along its local inward unit normal.

        Boundary samples are ordered from the image bottom towards the top.
        For that traversal direction ``(-t_y, t_x)`` points image-right for
        a vertical boundary.  It is therefore the inward normal for a LEFT
        boundary; RIGHT uses the opposite normal.  The offset magnitude is
        the interpolated lane-width profile at each sample row divided by two.
        """
        points = np.asarray(boundary, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            return {}
        tangents = np.gradient(points, axis=0)
        lengths = np.linalg.norm(tangents, axis=1)
        valid_tangent = lengths > 1e-6
        tangents[valid_tangent] /= lengths[valid_tangent, None]
        tangents[~valid_tangent] = (0.0, -1.0)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        desired_x = 1.0 if source == LEFT_BOUNDARY else -1.0
        flip = normals[:, 0]*desired_x < 0.0
        normals[flip] *= -1.0

        curve = {}
        cfg = self.config
        for boundary_point, tangent, normal in zip(points, tangents, normals):
            row = int(round(boundary_point[1]))
            width = self._width_near(row, cfg.width_profile_lookup_radius_px)
            width_source = "width_profile" if width is not None else "road_corridor"
            if width is None:
                width = self._road_width_from_boundary(
                    road, boundary_point, source)
            clearance_fraction = self._clearance_fraction()
            if (width is None or not np.isfinite(width) or
                    not cfg.lane_width_min_px <= width <= cfg.lane_width_max_px or
                    clearance_fraction is None):
                continue
            offset = float(width)*clearance_fraction
            virtual = boundary_point + normal*offset
            # The geometric normal may point below the image for near-field
            # perspective samples. Those points have no observable road pixel
            # for containment and must not enter fitting; the ego anchor fills
            # that near-field segment instead.
            bottom_cfg = cfg.roi_bottom if roi_bottom is None else roi_bottom
            if not cfg.roi_top <= virtual[1] <= bottom_cfg:
                continue
            # Dynamic PATH ROI also applies to the single-boundary virtual
            # center (section 3): a normal-offset point projected outside
            # the predicted corridor is not a trustworthy path candidate.
            if corridor is not None:
                bounds = corridor.get(row)
                if bounds is not None and not bounds[0] <= virtual[0] <= bounds[1]:
                    continue
            curve[row] = {
                "boundary": boundary_point.copy(),
                "tangent": tangent.copy(),
                "normal": normal.copy(),
                "lane_width_px": float(width),
                "lane_width_source": width_source,
                "offset_px": offset,
                "clearance_m": float(
                    clearance_fraction*cfg.nominal_lane_width_m),
                "virtual": virtual,
                "source": source,
            }
        if not curve:
            return curve

        # A true normal offset changes both image x and y.  Re-sample that
        # geometric curve on the planner's established y rows so downstream
        # fitting/controller code keeps the same row convention as bilateral
        # paths.  ``virtual`` remains the exact normal-offset point for debug.
        items = list(curve.values())
        virtual_y = np.asarray([item["virtual"][1] for item in items])
        virtual_x = np.asarray([item["virtual"][0] for item in items])
        order = np.argsort(virtual_y)
        virtual_y, virtual_x = virtual_y[order], virtual_x[order]
        y_min, y_max = float(virtual_y[0]), float(virtual_y[-1])
        for row in list(curve):
            if not y_min <= float(row) <= y_max:
                del curve[row]
                continue
            curve[row]["resampled"] = np.asarray([
                float(np.interp(float(row), virtual_y, virtual_x)), float(row)
            ])
        return curve

    def _apply_ego_anchor(self, curve, sample_y, roi_bottom=None):
        """Blend the nearest single-boundary segment out of the ego pixel."""
        item = curve[int(sample_y)]
        cfg = self.config
        bottom_cfg = cfg.roi_bottom if roi_bottom is None else roi_bottom
        span = float(max(cfg.seed_height_px, 3*cfg.sample_interval_px))
        progress = np.clip((float(bottom_cfg)-float(sample_y))/span, 0.0, 1.0)
        alpha = progress*progress*(3.0-2.0*progress)
        virtual = item["resampled"]
        anchored = np.asarray([
            (1.0-alpha)*float(cfg.vehicle_center_x_px)+alpha*float(virtual[0]),
            float(sample_y),
        ])
        item["anchor_alpha"] = float(alpha)
        item["anchored"] = anchored
        return anchored

    def _repair_local_spikes(self, points, sources, weights):
        """Local-only spike REPAIR (req 5 reorder): lateral + heading only.

        Each interior point is screened for a lateral residual or a heading
        (segment-angle) spike against its immediate neighbors. A flagged
        point is repaired by linear interpolation between its nearest
        unflagged neighbors -- never deleted -- so the horizon length is
        preserved and one bad raw sample cannot cascade into rejecting
        later points or shrinking below the fitting minimum. The number
        repaired per frame is capped. Curvature is deliberately NOT checked
        here; it is evaluated once on the final fitted+temporal-blended path
        (see plan()) as a physical-drivability gate instead of a per-point
        raw-stage rejection.
        """
        counters = {"LATERAL_JUMP": 0, "HEADING_JUMP": 0, "CURVATURE_JUMP": 0}
        if len(points) < 3:
            return points, sources, weights, 0, counters
        cfg = self.config
        lateral_residual = np.zeros(len(points), dtype=float)
        heading_residual = np.zeros(len(points), dtype=float)
        for index in range(1, len(points)-1):
            previous_point, current_point, next_point = (
                points[index-1], points[index], points[index+1])
            expected_x = (previous_point[0]+next_point[0])/2.0
            lateral_residual[index] = abs(current_point[0]-expected_x)
            vector_in = current_point-previous_point
            vector_out = next_point-current_point
            norm_in = np.linalg.norm(vector_in)
            norm_out = np.linalg.norm(vector_out)
            if norm_in > 1e-6 and norm_out > 1e-6:
                cosine = np.clip(
                    np.dot(vector_in, vector_out)/(norm_in*norm_out), -1.0, 1.0)
                heading_residual[index] = np.degrees(np.arccos(cosine))
        lateral_score = lateral_residual/max(1e-6, cfg.center_direction_outlier_px)
        heading_score = heading_residual/max(1e-6, cfg.heading_jump_threshold_deg)
        combined = np.maximum(lateral_score, heading_score)
        cap = max(1, int(np.ceil(cfg.max_local_outlier_ratio*len(points))))
        flagged = []
        for index in np.argsort(combined)[::-1]:
            if combined[index] < 1.0:
                break
            if len(flagged) >= cap:
                break
            if all(abs(int(index)-other) > 1 for other in flagged):
                flagged.append(int(index))
        flagged_set = set(flagged)
        output = points.copy()
        output_weights = weights.copy()
        repaired = 0
        for index in sorted(flagged):
            left_index = index-1
            while left_index in flagged_set and left_index > 0:
                left_index -= 1
            right_index = index+1
            while right_index in flagged_set and right_index < len(points)-1:
                right_index += 1
            if (left_index < 0 or right_index >= len(points) or
                    left_index in flagged_set or right_index in flagged_set or
                    left_index == right_index):
                continue
            y0, y1v = points[left_index, 1], points[right_index, 1]
            if abs(y1v-y0) < 1e-6:
                continue
            fraction = (points[index, 1]-y0)/(y1v-y0)
            x0, x1v = points[left_index, 0], points[right_index, 0]
            output[index, 0] = x0+fraction*(x1v-x0)
            output_weights[index] *= 0.5  # repaired point: reduced trust
            repaired += 1
            counters["LATERAL_JUMP" if lateral_score[index] >= heading_score[index]
                     else "HEADING_JUMP"] += 1
        return output, sources, output_weights, repaired, counters

    def _repair_path_gaps(self, points, sources, weights, sample_rows, road,
                          roi_bottom):
        """Recover missing sampled rows without extrapolating beyond road.

        Short interior gaps use their two surviving path anchors.  Longer or
        edge gaps may use the previous path, then the current ego-connected
        road centre.  Every inserted point must still fit the final
        vehicle-width/margin envelope on its exact row.
        """
        cfg = self.config
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        weights = np.asarray(weights, dtype=float)
        if not sample_rows or not np.any(road):
            return points, list(sources), weights, 0

        existing = {int(round(point[1])): index
                    for index, point in enumerate(points)}
        rows = [int(row) for row in sample_rows]
        missing = [index for index, row in enumerate(rows) if row not in existing]
        if not missing:
            return points, list(sources), weights, 0

        groups = []
        for index in missing:
            if not groups or index != groups[-1][-1]+1:
                groups.append([index])
            else:
                groups[-1].append(index)

        additions = []
        for group in groups:
            left_pos = group[0]-1
            right_pos = group[-1]+1
            left_row = rows[left_pos] if left_pos >= 0 else None
            right_row = rows[right_pos] if right_pos < len(rows) else None
            left_point = (points[existing[left_row]] if left_row in existing
                          else None)
            right_point = (points[existing[right_row]] if right_row in existing
                           else None)
            short_bracketed = (
                len(group) <= max(1, cfg.max_gap_repair_rows) and
                left_point is not None and right_point is not None)
            for position in group:
                row = rows[position]
                intervals = self._safe_row_intervals(
                    road, row, self._final_safety_margin_px(row, roi_bottom))
                if not intervals:
                    continue
                source = ROAD_CENTER
                weight = cfg.road_weight*0.75
                if short_bracketed:
                    fraction = ((row-left_point[1]) /
                                max(1e-6, right_point[1]-left_point[1]))
                    candidate_x = (left_point[0] +
                                   fraction*(right_point[0]-left_point[0]))
                else:
                    previous_x = None
                    if self.previous is not None and len(self.previous):
                        previous_x = self._at_y(
                            self.previous, row, cfg.sample_interval_px*2)
                    bounds = self._road_bounds_at_y(road, row)
                    road_center = (None if bounds is None else
                                   0.5*(bounds[0]+bounds[1]))
                    if previous_x is not None and road_center is not None:
                        candidate_x = 0.7*previous_x+0.3*road_center
                        source = TEMPORAL_FALLBACK
                        weight = cfg.temporal_weight
                    elif previous_x is not None:
                        candidate_x = previous_x
                        source = TEMPORAL_FALLBACK
                        weight = cfg.temporal_weight
                    elif road_center is not None:
                        candidate_x = road_center
                    else:
                        continue
                lo, hi = min(
                    intervals,
                    key=lambda bounds: (0.0 if bounds[0] <= candidate_x <= bounds[1]
                                        else min(abs(candidate_x-bounds[0]),
                                                 abs(candidate_x-bounds[1]))))
                additions.append((float(np.clip(candidate_x, lo, hi)),
                                  float(row), source, weight))

        if not additions:
            return points, list(sources), weights, 0
        combined = [(float(x), float(y), source, float(weight))
                    for (x, y), source, weight in zip(points, sources, weights)]
        combined.extend(additions)
        combined.sort(key=lambda item: item[1], reverse=True)
        repaired_points = np.asarray(
            [[item[0], item[1]] for item in combined], dtype=float)
        repaired_sources = [item[2] for item in combined]
        repaired_weights = np.asarray([item[3] for item in combined], dtype=float)
        return repaired_points, repaired_sources, repaired_weights, len(additions)

    def _raw_path(self, ys, left, right, road, allow_single_boundary,
                  road_geometry=None, roi_bottom=None, corridor=None):
        cfg = self.config
        self._last_virtual_points = np.empty((0, 2), dtype=float)
        self._last_virtual_details = []
        self._last_ego_anchor_applied = False
        points, sources, weights = [], [], []
        counters = {"CONTINUITY_FAILURE": 0, "BOUNDARY_ORDER": 0,
                    "INVALID_LANE_WIDTH": 0, "ROAD_GROSS_OUTLIER": 0,
                    "DIRECTION_OUTLIER": 0, "VEHICLE_CLEARANCE": 0,
                    "ROAD_WIDTH_OUTLIER": 0, "ROAD_CENTER_SPIKE": 0,
                    "SOURCE_TRANSITION_SMOOTHED": 0,
                    "SINGLE_ROAD_CONFLICT": 0, "LATERAL_JUMP_LOCAL": 0,
                    "HEADING_JUMP": 0, "CURVATURE_JUMP": 0}
        self._pending_width_updates = []
        width_errors = []
        road_geometry = (road_geometry if road_geometry is not None
                         else self._road_geometry(road, ys))
        left_virtual = (self._normal_offset_curve(left, LEFT_BOUNDARY, road, roi_bottom, corridor)
                        if allow_single_boundary and self.width_profile else {})
        right_virtual = (self._normal_offset_curve(right, RIGHT_BOUNDARY, road, roi_bottom, corridor)
                         if allow_single_boundary and self.width_profile else {})
        pair_rows = {
            int(y) for y in ys
            if (self._at_y(left, y, cfg.sample_interval_px/2) is not None and
                self._at_y(right, y, cfg.sample_interval_px/2) is not None)
        }
        single_rows = sorted({int(y) for y in ys if int(y) not in pair_rows and
                              ((self._at_y(left, y, cfg.sample_interval_px/2)
                                is not None) !=
                               (self._at_y(right, y, cfg.sample_interval_px/2)
                                is not None)) and int(y) in road_geometry},
                             reverse=True)
        first_single_y = single_rows[0] if single_rows else None
        anchor_single = (first_single_y is not None and
                         not any(row >= first_single_y for row in pair_rows))
        for y in ys:
            lx = self._at_y(left, y, cfg.sample_interval_px/2)
            rx = self._at_y(right, y, cfg.sample_interval_px/2)
            road_item = road_geometry.get(int(y))
            pair_width = None
            if lx is not None and rx is not None and rx <= lx:
                counters["BOUNDARY_ORDER"] += 1
                lx = rx = None
            if lx is not None and rx is not None:
                width = rx-lx
                expected_width = self._width_near(
                    int(y), cfg.width_profile_lookup_radius_px)
                tolerance = (cfg.lane_width_absolute_tolerance_px if expected_width is None
                             else max(cfg.lane_width_absolute_tolerance_px,
                                      cfg.lane_width_relative_tolerance*expected_width))
                width_is_sane = (cfg.lane_width_min_px <= width <= cfg.lane_width_max_px and
                                 (expected_width is None or
                                  abs(width-expected_width) <= tolerance))
                if not width_is_sane:
                    counters["INVALID_LANE_WIDTH"] += 1
                    lx = rx = None
                else:
                    x = (lx+rx)/2
                    body_fraction = ((cfg.vehicle_width_m/2.0 +
                                      cfg.vehicle_boundary_margin_m) /
                                     cfg.nominal_lane_width_m)
                    required_clearance = width*body_fraction
                    if (x-lx < required_clearance or
                            rx-x < required_clearance):
                        counters["VEHICLE_CLEARANCE"] += 1
                        lx = rx = None
                    else:
                        pair_width = width
                        if expected_width is not None:
                            width_errors.append(
                                abs(width-expected_width)/max(1.0, tolerance))
            if pair_width is not None:
                x, source, weight = (lx+rx)/2, BOTH_BOUNDARIES, cfg.both_weight
                center_y = float(y)
                self._pending_width_updates.append((int(y), float(pair_width)))
            elif (allow_single_boundary and (lx is not None) != (rx is not None)
                  and road_item is not None and
                  not road_item["width_outlier"] and
                  not road_item["center_spike"]):
                road_width = road_item["width"]
                road_center = road_item["center"]
                learned_detail = None
                if road_item["left_clipped"] or road_item["right_clipped"]:
                    learned_detail = (left_virtual.get(int(y)) if lx is not None
                                      else right_virtual.get(int(y)))
                if (learned_detail is not None and
                        learned_detail["lane_width_source"] == "width_profile"):
                    virtual = (self._apply_ego_anchor(
                        left_virtual if lx is not None else right_virtual, y,
                        roi_bottom)
                        if anchor_single else learned_detail["resampled"])
                    x, center_y = map(float, virtual)
                    source = LEFT_BOUNDARY if lx is not None else RIGHT_BOUNDARY
                    weight = cfg.single_weight*cfg.road_clipped_weight_scale
                    self._last_virtual_details.append(learned_detail)
                    road_width = None
                else:
                    center_y = float(y)
                    if lx is not None:
                        candidate = lx+0.5*road_width
                        source = LEFT_BOUNDARY
                        boundary = lx
                    else:
                        candidate = rx-0.5*road_width
                        source = RIGHT_BOUNDARY
                        boundary = rx
                    disagreement = abs(candidate-road_center)
                    if disagreement > (cfg.road_single_center_disagreement_ratio *
                                       max(1.0, road_width)):
                        # A contradictory isolated edge cannot pull the vehicle
                        # outside the observed corridor.  Fall through to C.
                        counters["SINGLE_ROAD_CONFLICT"] += 1
                        x = road_center
                        source = ROAD_CENTER
                        weight = cfg.road_weight
                    else:
                        blend = np.clip(cfg.road_single_boundary_blend, 0.0, 1.0)
                        x = blend*candidate+(1.0-blend)*road_center
                        weight = cfg.single_weight
                        detail = {
                            "boundary": np.asarray([boundary, float(y)]),
                            "virtual": np.asarray([candidate, float(y)]),
                            "resampled": np.asarray([x, float(y)]),
                            "lane_width_px": float(road_width),
                            "lane_width_source": "road_corridor",
                            "offset_px": float(abs(candidate-boundary)),
                            "clearance_m": None,
                            "method": "observed_road_width_midpoint",
                            "road_center": float(road_center),
                            "center_disagreement_px": float(disagreement),
                            "source": source,
                        }
                        self._last_virtual_details.append(detail)
                    if anchor_single:
                        bottom_cfg = (cfg.roi_bottom if roi_bottom is None
                                     else roi_bottom)
                        span = float(max(
                            cfg.seed_height_px, 3*cfg.sample_interval_px))
                        progress = np.clip(
                            (bottom_cfg-center_y)/span, 0.0, 1.0)
                        alpha = progress*progress*(3.0-2.0*progress)
                        x = ((1.0-alpha)*cfg.vehicle_center_x_px + alpha*x)
            elif road_item is not None:
                if road_item["width_outlier"]:
                    counters["ROAD_WIDTH_OUTLIER"] += 1
                    continue
                if road_item["center_spike"]:
                    counters["ROAD_CENTER_SPIKE"] += 1
                    continue
                x, center_y = road_item["center"], float(y)
                source = ROAD_CENTER
                near_progress = np.clip(
                    (center_y-cfg.roi_top) /
                    max(1.0, cfg.roi_bottom-cfg.roi_top), 0.0, 1.0)
                weight = cfg.road_weight*(0.55+0.45*near_progress)
            else:
                continue
            if road_item is not None:
                road_bounds = (road_item["left"], road_item["right"])
                margin = cfg.road_gross_outlier_margin_px
                if not road_bounds[0]-margin <= x <= road_bounds[1]+margin:
                    counters["ROAD_GROSS_OUTLIER"] += 1
                    continue
                body_fraction = ((cfg.vehicle_width_m/2.0 +
                                  cfg.vehicle_boundary_margin_m) /
                                 cfg.nominal_lane_width_m)
                clearance_width = road_item["width"]
                if (road_item["left_clipped"] or
                        road_item["right_clipped"]):
                    clearance_width = min(
                        clearance_width, cfg.road_minimum_near_width_px)
                required = min(0.5*clearance_width,
                               body_fraction*clearance_width)
                if (x-road_bounds[0] < required or
                        road_bounds[1]-x < required):
                    counters["VEHICLE_CLEARANCE"] += 1
                    continue
                if (road_item["left_clipped"] or
                        road_item["right_clipped"]):
                    weight *= cfg.road_clipped_weight_scale
            if points and abs(x-points[-1][0]) > cfg.maximum_lateral_jump_px:
                counters["CONTINUITY_FAILURE"] += 1
                continue
            points.append((x, center_y)); sources.append(source); weights.append(weight)
        self._last_ego_anchor_applied = bool(anchor_single and self._last_virtual_details)
        if self._last_virtual_details:
            self._last_virtual_points = np.asarray(
                [item["virtual"] for item in self._last_virtual_details],
                dtype=float).reshape(-1, 2)
        point_array = np.asarray(points, float).reshape(-1, 2)
        weight_array = np.asarray(weights, float)
        (point_array, sources, weight_array,
         gap_repaired) = self._repair_path_gaps(
             point_array, sources, weight_array, ys, road,
             cfg.roi_bottom if roi_bottom is None else roi_bottom)
        counters["GAP_REPAIRED"] = gap_repaired
        point_array, weight_array, smoothed = self._smooth_source_transitions(
            point_array, sources, weight_array)
        counters["SOURCE_TRANSITION_SMOOTHED"] = smoothed
        # req 5 reorder: raw-stage local spikes are repaired (interpolated),
        # not deleted -- point count is preserved. Curvature is checked once
        # on the final fitted+temporal-blended path in plan().
        (point_array, sources, weight_array, repaired,
         jump_counters) = self._repair_local_spikes(
            point_array, sources, weight_array)
        counters["LATERAL_JUMP_LOCAL"] = jump_counters["LATERAL_JUMP"]
        counters["HEADING_JUMP"] = jump_counters["HEADING_JUMP"]
        counters["CURVATURE_JUMP"] = 0
        counters["DIRECTION_OUTLIER"] = repaired
        width_score = (max(0.0, 1.0-float(np.mean(width_errors)))
                       if width_errors else 1.0)
        return point_array, sources, weight_array, counters, width_score

    def _fit(self, raw, weights):
        if len(raw) < 3: return raw.copy()
        degree = min(self.config.polynomial_degree, len(raw)-1)
        normalized_y = raw[:, 1] / max(1.0, float(np.max(raw[:, 1])))
        coefficients = np.polyfit(normalized_y, raw[:, 0], degree, w=weights)
        fitted_x = np.polyval(coefficients, normalized_y)
        margin = max(0.0, self.config.fit_overshoot_margin_px)
        fitted_x = np.clip(fitted_x, np.min(raw[:, 0])-margin,
                           np.max(raw[:, 0])+margin)
        return np.column_stack((fitted_x, raw[:, 1]))

    @staticmethod
    def _curvature_px(points):
        if len(points) < 3:
            return 0.0
        return float(np.percentile(np.abs(np.diff(points[:, 0], n=2)), 90))

    def _lookahead_ratio(self, curvature):
        cfg = self.config
        if curvature <= cfg.curve_moderate_threshold_px:
            return cfg.lookahead_straight_ratio
        if curvature >= cfg.curve_sharp_threshold_px:
            return cfg.lookahead_sharp_ratio
        fraction = ((curvature-cfg.curve_moderate_threshold_px) /
                    max(1e-6, cfg.curve_sharp_threshold_px-
                        cfg.curve_moderate_threshold_px))
        return ((1-fraction)*cfg.lookahead_curve_ratio +
                fraction*cfg.lookahead_sharp_ratio)

    def _curvature_alpha(self, curvature):
        cfg = self.config
        if curvature <= cfg.curve_moderate_threshold_px:
            fraction = curvature/max(1e-6, cfg.curve_moderate_threshold_px)
            return ((1-fraction)*cfg.temporal_straight_alpha +
                    fraction*cfg.temporal_alpha)
        fraction = min(1.0, (curvature-cfg.curve_moderate_threshold_px) /
                       max(1e-6, cfg.curve_sharp_threshold_px-
                           cfg.curve_moderate_threshold_px))
        return ((1-fraction)*cfg.temporal_alpha +
                fraction*cfg.temporal_curve_alpha)

    def _update_shift_hysteresis(self, signed_shift):
        threshold = self.config.temporal_small_shift_px
        direction = 0 if abs(signed_shift) < threshold else int(np.sign(signed_shift))
        if direction == 0:
            self.shift_direction = 0
            self.consistent_shift_frames = 0
        elif direction == self.shift_direction:
            self.consistent_shift_frames += 1
        else:
            self.shift_direction = direction
            self.consistent_shift_frames = 1
        return direction

    def _temporal_blend(self, fitted, curvature, reacquiring=False,
                        boundary_mode_transition=False, fallback_age=0):
        details = {"temporal_shift_px": 0.0, "temporal_signed_shift_px": 0.0,
                   "temporal_alpha_used": 1.0, "temporal_outlier_rejected": False,
                   "consistent_shift_frames": self.consistent_shift_frames,
                   "temporal_score": 1.0, "temporal_ok": True,
                   "temporal_stale_recovery": False}
        if self.previous is None or not len(self.previous):
            return fitted, details
        previous_x = [self._at_y(self.previous, y,
                                 self.config.sample_interval_px/2)
                      for y in fitted[:, 1]]
        matched = np.asarray([value is not None for value in previous_x])
        if not matched.any():
            return fitted, details
        previous_points = np.asarray(self.previous, dtype=float)
        order = np.argsort(previous_points[:, 1])
        previous_aligned = np.interp(
            fitted[:, 1], previous_points[order, 1], previous_points[order, 0])
        deltas = fitted[matched, 0]-previous_aligned[matched]
        shift = float(np.mean(np.abs(deltas)))
        signed_shift = float(np.median(deltas))
        direction = self._update_shift_hysteresis(signed_shift)
        details.update({"temporal_shift_px": shift,
                        "temporal_signed_shift_px": signed_shift,
                        "consistent_shift_frames": self.consistent_shift_frames})

        cfg = self.config
        sustained = (direction != 0 and self.consistent_shift_frames >=
                     cfg.temporal_hysteresis_frames)
        # A previous path kept alive only through repeated fallback frames is
        # stale: it must not lock the blend against a fresh, otherwise-valid
        # fit (req 4). Past the configured age, treat it like a sustained
        # direction change instead of segmentation noise.
        stale_previous = fallback_age >= cfg.temporal_stale_lock_max_frames
        if shift > cfg.maximum_temporal_shift_px and not sustained and not stale_previous:
            # A single large change is segmentation noise until the same
            # direction persists for the configured hysteresis window.
            fitted[:, 0] = previous_aligned
            details.update({"temporal_alpha_used": 0.0,
                            "temporal_outlier_rejected": True,
                            "temporal_score": 0.25})
            return fitted, details

        alpha = self._curvature_alpha(curvature)
        if sustained:
            alpha = min(0.9, alpha+cfg.temporal_consistent_boost)
        if reacquiring and not stale_previous:
            alpha = min(alpha, cfg.temporal_reacquire_alpha)
        if stale_previous:
            alpha = max(alpha, cfg.temporal_stale_recovery_alpha)
            details["temporal_stale_recovery"] = True
        elif boundary_mode_transition:
            # Ease across the bilateral/normal-offset representation change
            # for one frame. Existing hysteresis resumes on the next frame.
            alpha = min(alpha, 0.2*cfg.temporal_straight_alpha)
        if shift > 2.0*cfg.maximum_temporal_shift_px:
            details["temporal_ok"] = False
            details["temporal_score"] = 0.2
        else:
            details["temporal_score"] = max(
                0.0, 1.0-shift/max(1.0, 2.0*cfg.maximum_temporal_shift_px))
        fitted[:, 0] = alpha*fitted[:, 0] + (1-alpha)*previous_aligned
        details["temporal_alpha_used"] = alpha
        return fitted, details

    def _road_containment(self, points, road):
        """Return evaluated and contained point counts, tolerating small holes."""
        cfg = self.config
        mask = _binary(road)
        height, width = mask.shape
        checked = inside = 0
        band = max(1, cfg.sample_band_half_height_px)
        tolerance = max(0, cfg.road_containment_tolerance_px)
        for x, y in points:
            yi = int(round(y))
            y1, y2 = max(0, yi-band), min(height, yi+band+1)
            row_x = np.flatnonzero(np.any(mask[y1:y2], axis=0))
            if not len(row_x):
                continue
            checked += 1
            xi = int(round(x))
            x1, x2 = max(0, xi-tolerance), min(width, xi+tolerance+1)
            if np.any(mask[y1:y2, x1:x2]):
                inside += 1
        return checked, inside

    def _final_safety_margin_px(self, y, roi_bottom):
        """Perspective-aware horizontal road-edge clearance at image row y."""
        cfg = self.config
        position = float(np.clip(
            (float(y)-cfg.roi_top)/max(1.0, float(roi_bottom-cfg.roi_top)),
            0.0, 1.0))
        if position <= 0.5:
            fraction = position/0.5
            return float((1.0-fraction)*cfg.final_path_safety_margin_far_px +
                         fraction*cfg.final_path_safety_margin_mid_px)
        fraction = (position-0.5)/0.5
        return float((1.0-fraction)*cfg.final_path_safety_margin_mid_px +
                     fraction*cfg.final_path_safety_margin_near_px)

    def _safe_row_intervals(self, road, y, margin):
        """Return exact-row runs with vehicle-body and edge clearance.

        The vehicle half-width plus its configured metric body margin is
        converted to pixels as a fraction of the observed road-run width.
        The larger of that clearance and the perspective image margin is
        used, avoiding an invented global metre-per-pixel scale.
        """
        mask = _binary(road)
        yi = int(round(y))
        if not 0 <= yi < mask.shape[0]:
            return []
        intervals = []
        for run in _runs(np.flatnonzero(mask[yi])):
            run_width = float(run[-1]-run[0]+1)
            cfg = self.config
            body_fraction = ((cfg.vehicle_width_m/2.0+
                              cfg.vehicle_boundary_margin_m) /
                             max(1e-6, cfg.nominal_lane_width_m))
            clearance = max(float(margin), run_width*body_fraction)
            lo, hi = float(run[0])+clearance, float(run[-1])-clearance
            if lo <= hi:
                intervals.append((lo, hi))
        return intervals

    def _project_final_path_to_road(self, points, road, roi_bottom):
        """Project every final point into its exact current-road safe run.

        Projection is local in x and never fabricates a road row.  After a
        small shape-preserving smoothing pass, it is projected once more so
        smoothing itself cannot cross the mask edge. A missing road row or
        margin envelope makes the path unrecoverable; large corrections and
        pixel curvature request repair, then vehicle steering decides safety.
        """
        original = np.asarray(points, dtype=float).reshape(-1, 2)
        details = {
            "final_road_safety_evaluated": bool(len(original)),
            "final_road_offroad_points_before": 0,
            "final_road_margin_violations_before": 0,
            "final_road_projected_points": 0,
            "final_road_dropped_points": 0,
            "final_road_kept_indices": list(range(len(original))),
            "final_road_projection_max_px": 0.0,
            "final_road_recovered": False,
            "final_road_unrecoverable": False,
            "final_road_min_clearance_px": None,
        }
        if not len(original):
            return original.copy(), True, details
        road_mask = _binary(road)
        if not np.any(road_mask):
            details["final_road_offroad_points_before"] = len(original)
            details["final_road_unrecoverable"] = True
            return np.empty((0, 2), float), False, details

        projected = original.copy()
        corrections = []
        accepted_corrections = []
        minimum_clearance = []
        keep = np.ones(len(original), dtype=bool)
        for index, (x, y) in enumerate(original):
            yi, xi = int(round(y)), int(round(x))
            margin = self._final_safety_margin_px(y, roi_bottom)
            intervals = self._safe_row_intervals(road_mask, y, margin)
            exact_inside = bool(
                0 <= yi < road_mask.shape[0] and
                0 <= xi < road_mask.shape[1] and road_mask[yi, xi])
            if not exact_inside:
                details["final_road_offroad_points_before"] += 1
            inside_margin = any(lo <= x <= hi for lo, hi in intervals)
            if not inside_margin:
                details["final_road_margin_violations_before"] += 1
            if not intervals:
                keep[index] = False
                corrections.append(0.0)
                continue
            lo, hi = min(intervals, key=lambda bounds:
                         0.0 if bounds[0] <= x <= bounds[1] else
                         min(abs(x-bounds[0]), abs(x-bounds[1])))
            new_x = float(np.clip(x, lo, hi))
            correction = abs(new_x-x)
            # Do not turn a large but recoverable lateral correction into an
            # image-geometry hard stop.  The projected/smoothed result still
            # has to pass exact road, topology and vehicle-steering gates.
            projected[index, 0] = new_x
            corrections.append(correction)
            accepted_corrections.append(correction)
            minimum_clearance.append(min(new_x-lo, hi-new_x)+margin)

        details["final_road_dropped_points"] = int(np.count_nonzero(~keep))
        details["final_road_kept_indices"] = np.flatnonzero(keep).tolist()
        projected = projected[keep]
        original_kept = original[keep]
        if len(projected) < 3:
            details["final_road_projection_max_px"] = float(max(corrections or [0.0]))
            details["final_road_unrecoverable"] = True
            return np.empty((0, 2), float), False, details

        # Local three-point smoothing preserves endpoints and path shape.
        if (len(projected) >= 3 and
                any(value > 1e-6 for value in accepted_corrections)):
            smooth_x = projected[:, 0].copy()
            smooth_x[1:-1] = (0.25*projected[:-2, 0] +
                              0.50*projected[1:-1, 0] +
                              0.25*projected[2:, 0])
            projected[:, 0] = smooth_x

        # Re-project after smoothing, then require exact mask containment.
        for point in projected:
            margin = self._final_safety_margin_px(point[1], roi_bottom)
            intervals = self._safe_row_intervals(road_mask, point[1], margin)
            if not intervals:
                details["final_road_unrecoverable"] = True
                return np.empty((0, 2), float), False, details
            lo, hi = min(intervals, key=lambda bounds:
                         0.0 if bounds[0] <= point[0] <= bounds[1] else
                         min(abs(point[0]-bounds[0]), abs(point[0]-bounds[1])))
            point[0] = np.clip(point[0], lo, hi)
            xi, yi = int(round(point[0])), int(round(point[1]))
            if not (0 <= yi < road_mask.shape[0] and
                    0 <= xi < road_mask.shape[1] and road_mask[yi, xi]):
                details["final_road_unrecoverable"] = True
                return np.empty((0, 2), float), False, details

        maximum_correction = float(np.max(np.abs(
            projected[:, 0]-original_kept[:, 0])))
        # Pixel second-difference is a warning only.  Physical feasibility is
        # decided later using the production controller's steering law.
        curvature_warning = (
            self._curvature_px(projected) > self.config.final_curvature_jump_px)
        details.update({
            "final_road_projected_points": int(np.count_nonzero(
                np.abs(projected[:, 0]-original_kept[:, 0]) > 1e-6)),
            "final_road_projection_max_px": maximum_correction,
            # Dropping only the raster-boundary row at roi_bottom is routine
            # ego-exclusion sanitation, not a lateral path recovery.  A real
            # x projection or loss of any interior row remains DEGRADED.
            "final_road_recovered": bool(
                maximum_correction > 1e-6 or
                np.any(original[~keep, 1] < float(roi_bottom)-0.5)),
            "final_road_min_clearance_px": (
                float(min(minimum_clearance)) if minimum_clearance else None),
            "final_pixel_curvature_warning": bool(curvature_warning),
        })
        return projected, True, details

    def _final_topology_ok(self, points):
        """Require a finite, near-to-far, non-self-intersecting polyline."""
        array = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(array) < 3 or not np.all(np.isfinite(array)):
            return False
        # Planner samples are ordered from the ego-near bottom towards the
        # image horizon.  Equal/reversing rows imply a loop or rearward step.
        if np.any(np.diff(array[:, 1]) >= -1e-6):
            return False
        return not has_self_intersection(array)

    def _steering_feasibility(self, points, image_width, timestamp_sec):
        """Evaluate the final path with the production pixel-PD steering law."""
        cfg = self.config
        array = np.asarray(points, dtype=float).reshape(-1, 2)
        details = {
            "required_steering_deg": 0.0,
            "max_required_steering_deg": 0.0,
            "max_horizon_steering_delta_deg": 0.0,
            "steering_frame_delta_deg": 0.0,
            "steering_rate_deg_per_sec": 0.0,
            "steering_angle_ok": False,
            "steering_rate_ok": False,
            "steering_continuity_ok": False,
            "steering_offset_px": None,
            "steering_dt_sec": cfg.nominal_frame_period_sec,
        }
        if len(array) < 3:
            return details
        offset = lookahead_offset_px(
            array, image_width, cfg.steering_lookahead_y_ratio)
        if offset is None or not np.isfinite(offset):
            return details
        if (timestamp_sec is not None and
                self._previous_feasibility_timestamp_sec is not None):
            dt = timestamp_sec-self._previous_feasibility_timestamp_sec
            if not np.isfinite(dt) or dt <= 0.0:
                dt = cfg.nominal_frame_period_sec
        else:
            dt = cfg.nominal_frame_period_sec
        required = steering_from_offset_deg(
            offset, self._previous_feasibility_offset_px, dt, image_width,
            cfg.steering_proportional_gain_deg_per_norm,
            cfg.steering_derivative_gain_deg_per_norm_per_s,
            1.0e6)

        # Evaluate several controller lookahead positions with the same
        # proportional mapping.  This catches a +20/-20 zigzag whose single
        # configured lookahead could otherwise happen to look benign.
        horizon_angles = []
        for ratio in np.linspace(0.0, 1.0, min(9, len(array))):
            horizon_offset = lookahead_offset_px(array, image_width, ratio)
            if horizon_offset is None:
                continue
            horizon_angles.append(steering_from_offset_deg(
                horizon_offset, None, 0.0, image_width,
                cfg.steering_proportional_gain_deg_per_norm,
                cfg.steering_derivative_gain_deg_per_norm_per_s, 1.0e6))
        maximum = max([abs(required)]+[abs(value) for value in horizon_angles])
        segment_delta = (max(np.abs(np.diff(horizon_angles)))
                         if len(horizon_angles) >= 2 else 0.0)
        previous_steering = self._previous_feasibility_steering_deg
        frame_delta = (abs(required-previous_steering)
                       if previous_steering is not None else 0.0)
        rate = frame_delta/max(1e-6, dt)
        rate_ok = (
            previous_steering is None or
            (frame_delta <= cfg.max_steering_delta_deg_per_frame and
             rate <= cfg.max_steering_rate_deg_per_sec))
        continuity_ok = segment_delta <= cfg.max_steering_delta_deg_per_segment
        details.update({
            "required_steering_deg": float(required),
            "max_required_steering_deg": float(maximum),
            "max_horizon_steering_delta_deg": float(segment_delta),
            "steering_frame_delta_deg": float(frame_delta),
            "steering_rate_deg_per_sec": float(rate),
            "steering_angle_ok": bool(maximum <= cfg.maximum_steering_deg),
            "steering_rate_ok": bool(rate_ok),
            "steering_continuity_ok": bool(continuity_ok),
            "steering_offset_px": float(offset),
            "steering_dt_sec": float(dt),
        })
        return details

    def _repair_final_steering(self, points, road=None):
        """Smooth a final path towards recent safe geometry before stopping."""
        repaired = np.asarray(points, dtype=float).reshape(-1, 2).copy()
        if len(repaired) < 3:
            return repaired
        if self.previous is not None and len(self.previous):
            previous = np.asarray(self.previous, dtype=float)
            order = np.argsort(previous[:, 1])
            aligned = np.interp(
                repaired[:, 1], previous[order, 1], previous[order, 0])
            previous_weight = float(np.clip(
                self.config.steering_repair_previous_weight, 0.0, 0.95))
            repaired[:, 0] = (previous_weight*aligned+
                              (1.0-previous_weight)*repaired[:, 0])
        if road is not None and np.any(road):
            corridor_centers = []
            for x, y in repaired:
                intervals = self._safe_row_intervals(
                    road, y, self._final_safety_margin_px(
                        y, self.config.roi_bottom))
                if not intervals:
                    corridor_centers.append(x)
                    continue
                lo, hi = min(
                    intervals,
                    key=lambda bounds: (0.0 if bounds[0] <= x <= bounds[1]
                                        else min(abs(x-bounds[0]),
                                                 abs(x-bounds[1]))))
                corridor_centers.append(0.5*(lo+hi))
            # Use current-road centre only as a repair reference.  The
            # dynamic/previous path remains the majority contribution, so a
            # wide multi-lane road cannot cause an abrupt centre snap.
            repaired[:, 0] = (0.70*repaired[:, 0] +
                              0.30*np.asarray(corridor_centers))
        # A low-order refit removes alternating steering commands across the
        # horizon.  It is followed by repeated local smoothing and, at the
        # caller, an exact road/vehicle-envelope projection, so the fit can
        # never authorize an off-road extrapolation.
        repaired = self._fit(repaired, np.ones(len(repaired), dtype=float))
        for _ in range(3):
            smooth = repaired[:, 0].copy()
            smooth[1:-1] = (0.25*repaired[:-2, 0] +
                            0.50*repaired[1:-1, 0] +
                            0.25*repaired[2:, 0])
            repaired[:, 0] = smooth
        return repaired

    def _commit_steering_state(self, details, timestamp_sec):
        offset = details.get("steering_offset_px")
        steering = details.get("required_steering_deg")
        if offset is None or not np.isfinite(offset) or not np.isfinite(steering):
            return
        self._previous_feasibility_offset_px = float(offset)
        self._previous_feasibility_steering_deg = float(steering)
        if timestamp_sec is not None:
            self._previous_feasibility_timestamp_sec = float(timestamp_sec)

    def plan(self, road, white, yellow, words=None, stop=None, c_line=None,
             timestamp_sec=None):
        started = time.perf_counter()
        if timestamp_sec is not None:
            timestamp_sec = float(timestamp_sec)
            if not np.isfinite(timestamp_sec):
                raise ValueError("timestamp_sec must be finite")
            if (self.last_plan_time is not None and
                    timestamp_sec-self.last_plan_time >
                    self.config.temporal_state_timeout_sec):
                self.reset_temporal()
            self.last_plan_time = timestamp_sec
        shape = np.asarray(road).shape
        cfg = self.config
        zero = np.zeros(shape, np.uint8)
        words, stop, c_line = [zero if item is None else item for item in (words, stop, c_line)]
        # req 3.A: close/hole-fill the road mask so a diamond/text/arrow
        # marking (segmented as a non-road class) does not read as a break
        # in the road corridor.
        road_binary = fill_road_holes(
            road, cfg.road_hole_close_kernel_px, cfg.road_hole_max_area_px
        ) if cfg.road_hole_fill_enabled else _binary(road)

        # req 1: ego bumper/bonnet/mount exclusion. ``exclusion_mask`` is the
        # pixel-precise polygon+ratio region; ``roi_bottom`` is the
        # resolution-independent row it raises the sampling/near-field
        # window to, so the bumper band is never treated as evidence for
        # road-center, branch, candidate-path, virtual-center or jump-
        # baseline computation. Stop-line tracking never goes through this
        # planner path (see stop_line_memory.py), so it is unaffected.
        exclusion_mask = ego_exclusion_mask(
            shape, cfg.ego_exclusion_bottom_ratio, cfg.ego_exclusion_polygon,
            cfg.ego_exclusion_enabled)
        exclusion_top_row = ego_exclusion_top_row(
            shape, cfg.ego_exclusion_bottom_ratio, cfg.ego_exclusion_polygon,
            cfg.roi_top, cfg.ego_exclusion_enabled)
        apply_full_exclusion = cfg.ego_exclusion_enabled and not cfg.ego_exclusion_branch_only
        effective_roi_bottom = (min(cfg.roi_bottom, exclusion_top_row-1)
                                if apply_full_exclusion else cfg.roi_bottom)
        if apply_full_exclusion:
            road_for_path = road_binary & (1-exclusion_mask)
        else:
            road_for_path = road_binary
        seed_bottom_row = (min(shape[0], effective_roi_bottom+1)
                           if apply_full_exclusion else None)
        ego_component = ego_connected_component(
            road_for_path, cfg.vehicle_center_x_px,
            cfg.seed_half_width_px, cfg.seed_height_px,
            cfg.minimum_component_pixels, seed_bottom_row)
        component = ego_component
        disconnected_road_pixels = int(
            np.count_nonzero(road_for_path) - np.count_nonzero(component))
        # branch-only mode: keep the main path pipeline on the un-excluded
        # geometry but still compute an excluded variant purely to gate
        # branch evidence below.
        branch_component = (component & (1-exclusion_mask)
                            if cfg.ego_exclusion_enabled and cfg.ego_exclusion_branch_only
                            else component)
        lane_before = _binary(white) | _binary(yellow)
        if apply_full_exclusion:
            lane_before = lane_before & (1-exclusion_mask)
        lane, words_removed = exclude_one_semantic(
            lane_before, words, self.config.exclusion_overlap_ratio)
        lane, stop_removed = exclude_one_semantic(
            lane, stop, self.config.exclusion_overlap_ratio)
        lane, c_line_removed = exclude_one_semantic(
            lane, c_line, self.config.exclusion_overlap_ratio)
        # req 3.C: drop interior diamond/text/arrow blobs from the lane mask
        # before boundary sampling, so they cannot masquerade as a
        # LEFT/RIGHT boundary candidate or steer the candidate path (3.D).
        marking_removed = 0
        if cfg.marking_suppression_enabled:
            lane, marking_removed = suppress_interior_markings(
                lane, component, cfg.marking_max_row_width_px,
                cfg.marking_min_length_px, cfg.marking_edge_margin_px)
        # ROAD ROI / BRANCH ROI: road_geometry/road_safety are computed on
        # the full ego-connected road extent (unchanged by the dynamic PATH
        # ROI below), so a real wide/curved/intersection road is never cut.
        ys = self._row_list(shape[0], cfg.roi_top, effective_roi_bottom,
                            cfg.sample_interval_px)
        road_geometry = self._road_geometry(component, ys)
        road_safety = self._road_safety(
            road_geometry, ys, component, effective_roi_bottom)
        # Dynamic PATH ROI (section 2): predicted_center_x(y) +/-
        # half_width(y), built from the previous final path / current road
        # center, widened while LOST and narrowed back down while TRACKING.
        # Only used to filter lane/boundary CANDIDATES below -- never to cut
        # the road mask or the branch-evidence computation above.
        predicted_center = self._predicted_center_map(ys, road_geometry)
        corridor = (self._corridor_bounds_map(
            ys, predicted_center, shape[1], cfg.roi_top, effective_roi_bottom)
            if cfg.path_corridor_enabled else None)
        ys, left, right, edges, rejections = self._samples(
            component, lane, effective_roi_bottom, corridor)
        # Boundary continuity/track filtering (section 4): reject a
        # left/right boundary track whose row-to-row position zigzags
        # (a marking edge), lacks vertical span, or disagrees with the
        # previous frame's track -- independent of the mask-blob filter
        # suppress_interior_markings() already applied.
        left_track_score = right_track_score = 1.0
        if cfg.boundary_track_filter_enabled:
            left, left_track_score, _ = self._boundary_track_filter(
                left, self._previous_left_track)
            right, right_track_score, _ = self._boundary_track_filter(
                right, self._previous_right_track)
        if cfg.ego_exclusion_enabled and cfg.ego_exclusion_branch_only:
            # Re-derive branch evidence alone from the exclusion-masked
            # geometry so a bumper artifact cannot flag BRANCH_SUSPECTED
            # while the rest of the path still uses the full-bottom data.
            branch_ys = [y for y in ys if y < exclusion_top_row]
            branch_geometry = self._road_geometry(branch_component, branch_ys)
            branch_safety = self._road_safety(
                branch_geometry, branch_ys, branch_component, exclusion_top_row)
            road_safety["branch_rows"] = branch_safety["branch_rows"]
            road_safety["branch_expansion_rows"] = branch_safety["branch_expansion_rows"]
            road_safety["branch_suspected"] = branch_safety["branch_suspected"]
            road_safety["branch_critical"] = branch_safety["branch_critical"]
        # req 3.B: a branch signal must repeat for several consecutive
        # frames before being trusted; a transient marking-induced split
        # must not immediately flip BRANCH_SUSPECTED/branch_critical.
        raw_branch_suspected = bool(road_safety["branch_suspected"])
        raw_branch_critical = bool(road_safety["branch_critical"])
        self._branch_suspected_streak = (
            self._branch_suspected_streak+1 if raw_branch_suspected else 0)
        self._branch_critical_streak = (
            self._branch_critical_streak+1 if raw_branch_critical else 0)
        road_safety["branch_suspected_raw"] = raw_branch_suspected
        road_safety["branch_critical_raw"] = raw_branch_critical
        road_safety["branch_suspected"] = (
            self._branch_suspected_streak >= cfg.branch_confirm_frames)
        road_safety["branch_critical"] = (
            self._branch_critical_streak >= cfg.branch_confirm_frames)
        physical_geometry_ok = self._clearance_fraction() is not None
        rejections.update({"WORDS_OVERLAP": words_removed, "STOP_OVERLAP": stop_removed,
                           "C_LINE_OVERLAP": c_line_removed,
                           "MARKING_SUPPRESSED": marking_removed,
                           "TOO_SMALL_COMPONENT": 0, "TEMPORAL_REJECT": 0})
        pair_rows = sum(
            self._at_y(left, y, self.config.sample_interval_px/2) is not None and
            self._at_y(right, y, self.config.sample_interval_px/2) is not None
            for y in ys)
        if pair_rows:
            self.single_boundary_age = 0
        else:
            self.single_boundary_age += 1
        # The frame limit is only a boot-time grace period when no trustworthy
        # width has ever been learned. Once bilateral observations established
        # a plausible width profile, a sustained one-sided corner remains a
        # geometry-validated normal-offset path instead of expiring merely due
        # to its age. Complete loss still uses the separate 5-frame temporal
        # fallback below.
        allow_single = (pair_rows > 0 or bool(self.width_profile) or
                        bool(road_geometry) or
                        self.single_boundary_age <=
                        self.config.max_single_boundary_fallback_frames)
        raw, sources, weights, raw_rejections, width_score = self._raw_path(
            ys, left, right, component, allow_single, road_geometry,
            effective_roi_bottom, corridor)
        rejections.update(raw_rejections)
        both_ratio = sources.count(BOTH_BOUNDARIES)/max(1, len(sources))
        single_ratio = sum(source in (LEFT_BOUNDARY, RIGHT_BOUNDARY)
                           for source in sources)/max(1, len(sources))
        road_ratio = sources.count(ROAD_CENTER)/max(1, len(sources))
        point_score = min(1.0, len(raw)/max(
            1, self.config.valid_min_points, len(ys)//2))
        boundary_score = float(np.mean(weights)) if len(weights) else 0.0
        near_score = road_safety["near_field_score"]
        continuity_failures = (rejections["LATERAL_JUMP"] +
                               rejections["CONTINUITY_FAILURE"] +
                               rejections["DIRECTION_OUTLIER"])
        continuity_score = max(0.0, 1.0-continuity_failures/max(
            1, len(raw)+continuity_failures))
        components = {"road_score": 0.5, "boundary_score": boundary_score,
                      "coverage_score": point_score, "point_score": point_score,
                      "near_field_score": near_score,
                      "bilateral_score": both_ratio,
                      "both_ratio": both_ratio,
                      "single_ratio": single_ratio,
                      "road_ratio": road_ratio,
                      "road_center_score": road_safety["road_center_score"],
                      "road_width_stability_score":
                          road_safety["road_width_stability_score"],
                      "road_containment_score": 0.0,
                      "continuity_score": continuity_score,
                      "lane_width_score": width_score,
                      "polynomial_score": 0.0, "fallback_score": 0.0,
                      "temporal_score": 1.0, "freshness_score": 1.0}
        diagnostics = {"road_sample_rows": len(edges),
                       "physical_geometry_ok": physical_geometry_ok,
                       "ego_road_component_present": bool(np.any(component)),
                       "disconnected_road_pixels": disconnected_road_pixels,
                       "ego_exclusion_enabled": bool(cfg.ego_exclusion_enabled),
                       "ego_exclusion_branch_only": bool(cfg.ego_exclusion_branch_only),
                       "ego_exclusion_top_row_px": int(exclusion_top_row),
                       "effective_roi_bottom_px": int(effective_roi_bottom),
                       "marking_suppressed_components": int(marking_removed),
                       "branch_suspected_streak": self._branch_suspected_streak,
                       "branch_critical_streak": self._branch_critical_streak,
                       "lane_candidate_rows": len({int(y) for _, y in np.vstack((left, right))}) if len(left)+len(right) else 0,
                       "left_candidate_rows": len(left), "right_candidate_rows": len(right),
                       "lane_pixels_before_exclusion": int(np.count_nonzero(lane_before)),
                       "lane_pixels_after_exclusion": int(np.count_nonzero(lane)),
                       "raw_center_points": len(raw), "fitting_input_points": len(raw),
                       "both_boundary_rows": pair_rows,
                       "both_boundary_ratio": both_ratio,
                       "both_ratio": both_ratio,
                       "single_ratio": single_ratio,
                       "road_ratio": road_ratio,
                       "single_boundary_fallback_age": self.single_boundary_age,
                       "single_boundary_fallback_allowed": allow_single,
                       "lane_width_profile_rows": len(self.width_profile),
                       "expected_lane_width_near_px": self._width_near(
                           self.config.roi_bottom, self.config.width_profile_lookup_radius_px),
                       "expected_lane_width_middle_px": self._width_near(
                           (self.config.roi_top+self.config.roi_bottom)//2,
                           self.config.width_profile_lookup_radius_px),
                       "rejections": rejections}
        diagnostics.update(road_safety)
        diagnostics.update({
            "single_boundary_path": False,
            "road_dependent_path": False,
            "vehicle_containment_ok": False,
            "continuity_ok": False,
            "temporal_ok": False,
            "source_mode": self._source_mode(sources),
            "source_mode_transition": False,
            "path_corridor_enabled": bool(cfg.path_corridor_enabled),
            "path_corridor_state": self._corridor_state,
            "path_corridor_expand_level": float(self._corridor_expand_level),
            "path_corridor_near_half_width_px": self._corridor_half_width_px(
                effective_roi_bottom, shape[1], cfg.roi_top, effective_roi_bottom),
            "path_corridor_bounds": ([{"y": y, "lo": bounds[0], "hi": bounds[1]}
                                      for y, bounds in sorted(corridor.items())]
                                     if corridor else []),
            "left_boundary_track_score": float(left_track_score),
            "right_boundary_track_score": float(right_track_score),
        })
        if len(raw) >= 3:
            boundary_mode = (BOTH_BOUNDARIES if BOTH_BOUNDARIES in sources
                             else (sources[0] if sources else None))
            source_mode = self._source_mode(sources)
            near_x = float(raw[0, 0]) if len(raw) else None
            confirmed_source_mode, _ = self._update_source_hysteresis(source_mode, near_x)
            boundary_mode_transition = (
                self.previous_boundary_mode is not None and
                boundary_mode != self.previous_boundary_mode)
            # req 7: ease the temporal blend only on a CONFIRMED source-mode
            # change that survived both the confirmation and release gates,
            # not on every single-frame segmentation flicker.
            source_mode_transition = (
                self.previous_source_mode is not None and
                confirmed_source_mode != self.previous_source_mode)
            fitted = self._fit(raw, weights)
            residual = float(np.sqrt(np.mean(np.square(raw[:, 0]-fitted[:, 0]))))
            polynomial_score = max(
                0.0, 1.0-residual/max(1.0, self.config.polynomial_residual_scale_px))
            # The hysteresis state controls actual path adoption, not only a
            # diagnostic label.  Until the old source is released and the new
            # one confirmed, retain the previous geometry for at most the
            # debounce window; the hard current-road projection below still
            # decides whether that held geometry is safe this frame.
            source_pending_hold = bool(
                confirmed_source_mode != source_mode and
                self.previous is not None and len(self.previous))
            if source_pending_hold:
                fitted = self.previous.copy()
                sources = [TEMPORAL_FALLBACK]*len(fitted)
            curvature_before_temporal = self._curvature_px(fitted)
            reacquiring = self.fallback_age > 0
            fitted, temporal = self._temporal_blend(
                fitted, curvature_before_temporal, reacquiring,
                boundary_mode_transition or source_mode_transition,
                fallback_age=self.fallback_age)
            if temporal["temporal_outlier_rejected"]:
                rejections["TEMPORAL_REJECT"] += 1
            # Publish-boundary safety: polynomial fitting and temporal
            # blending are both upstream of this exact-mask projection.
            # Consequently neither fit overshoot nor an old previous path can
            # escape the current ego-connected road corridor.
            fitted, final_road_safe, final_road_safety = (
                self._project_final_path_to_road(
                    fitted, component, effective_roi_bottom))
            diagnostics.update(final_road_safety)
            sources = ([sources[index] for index in
                        final_road_safety["final_road_kept_indices"]]
                       if final_road_safe else [])
            topology_ok = self._final_topology_ok(fitted)
            steering = self._steering_feasibility(
                fitted, shape[1], timestamp_sec)
            steering_repaired = False
            steering_repair_requested = bool(
                final_road_safe and len(fitted) >= 3 and
                (final_road_safety.get("final_pixel_curvature_warning", False) or
                 not steering["steering_angle_ok"] or
                 not steering["steering_rate_ok"] or
                 not steering["steering_continuity_ok"]))
            steering_repair_attempts = 0
            if steering_repair_requested:
                for _ in range(3):
                    steering_repair_attempts += 1
                    repaired_candidate = self._repair_final_steering(
                        fitted, component)
                    (repaired_candidate, repaired_safe,
                     repaired_safety) = self._project_final_path_to_road(
                         repaired_candidate, component, effective_roi_bottom)
                    if not repaired_safe:
                        fitted = np.empty((0, 2), dtype=float)
                        sources = []
                        final_road_safe = False
                        final_road_safety = repaired_safety
                        diagnostics.update(final_road_safety)
                        topology_ok = False
                        steering = self._steering_feasibility(
                            fitted, shape[1], timestamp_sec)
                        break
                    sources = [
                        sources[index]
                        for index in repaired_safety["final_road_kept_indices"]]
                    fitted = repaired_candidate
                    steering_repaired = True
                    repaired_safety["final_road_projected_points"] += int(
                        final_road_safety["final_road_projected_points"])
                    repaired_safety["final_road_dropped_points"] += int(
                        final_road_safety["final_road_dropped_points"])
                    repaired_safety["final_road_recovered"] = bool(
                        repaired_safety["final_road_recovered"] or
                        final_road_safety["final_road_recovered"])
                    final_road_safety = repaired_safety
                    diagnostics.update(final_road_safety)
                    topology_ok = self._final_topology_ok(fitted)
                    steering = self._steering_feasibility(
                        fitted, shape[1], timestamp_sec)
                    if (topology_ok and steering["steering_angle_ok"] and
                            steering["steering_rate_ok"] and
                            steering["steering_continuity_ok"]):
                        break
            road_checked, road_inside = self._road_containment(fitted, component)
            road_available = road_checked >= self.config.road_validation_min_rows
            road_score = (road_inside/road_checked if road_checked else 0.65)
            road_ok = (not road_available or road_score >=
                       self.config.valid_min_road_containment_ratio)
            fallback_ratio = (sum(source != BOTH_BOUNDARIES for source in sources) /
                              max(1, len(sources)))
            fallback_score = max(0.0, 1.0-fallback_ratio)
            components.update({"road_score": road_score,
                               "road_containment_score": road_score,
                               "polynomial_score": polynomial_score,
                               "fallback_score": fallback_score,
                               "temporal_score": temporal["temporal_score"]})
            lane_width_relevance = max(both_ratio, single_ratio)
            lane_width_component = (
                lane_width_relevance*width_score +
                (1.0-lane_width_relevance)*
                road_safety["road_width_stability_score"])
            branch_factor = 0.82 if road_safety["branch_suspected"] else 1.0
            confidence = float(np.clip(branch_factor*(
                .16*point_score + .16*boundary_score +
                .10*road_safety["road_center_score"] +
                .10*road_safety["road_width_stability_score"] +
                .12*near_score + .10*road_score +
                .10*continuity_score + .08*temporal["temporal_score"] +
                .04*polynomial_score + .04*lane_width_component),
                0.0, 1.0))
            single_boundary_path = (
                any(source in (LEFT_BOUNDARY, RIGHT_BOUNDARY)
                    for source in sources) and
                not any(source == BOTH_BOUNDARIES for source in sources))
            # req 6 (ROAD_ONLY): lane detection can be entirely absent while
            # the road itself is a perfectly safe, drivable corridor -- that
            # case gets its own (lower) point requirement rather than being
            # held to the lane-based thresholds it structurally cannot meet.
            road_only_path = bool(sources) and all(
                source == ROAD_CENTER for source in sources)
            if road_only_path:
                quality_target_points = self.config.valid_min_road_only_points
            elif single_boundary_path:
                quality_target_points = self.config.valid_min_single_boundary_points
            else:
                quality_target_points = self.config.valid_min_points
            # Three finite points are enough for the production controller.
            # A shorter-than-target horizon is a DEGRADED quality condition,
            # not a candidate-geometry hard stop.
            required_points = 3
            enough_points = len(fitted) >= required_points
            horizon_ratio = min(
                1.0, len(fitted)/max(1, self.config.target_path_points))
            horizon_sufficient = (
                horizon_ratio >= self.config.minimum_drivable_horizon_ratio)
            continuity_ok = continuity_score >= self.config.valid_min_continuity_score
            # Road is deliberately soft: gross off-road samples were already
            # removed before fitting, while holes only lower the quality score.
            # Sparse one-sided recovery needs positive road containment; this
            # prevents a geometrically offset boundary from authorizing drive
            # when its inferred center falls outside the observed corridor.
            road_dependent_path = bool(sources) and any(
                source != BOTH_BOUNDARIES for source in sources)
            containment_ok = bool(final_road_safe)
            width_stability_ok = (
                road_safety["road_width_stability_score"] >=
                self.config.road_min_width_stability_score)
            center_ok = (road_safety["road_center_score"] >=
                         self.config.road_min_center_score)
            # Pixel curvature is a final-path soft warning only, never a raw
            # point deletion or hard validity gate (see _repair_local_spikes).
            final_curvature_px = self._curvature_px(fitted)
            final_curvature_ok = final_curvature_px <= self.config.final_curvature_jump_px
            # req 8: INVALID means "cannot actually drive this", not "lane
            # detail is imperfect". Only conditions that make the corridor
            # itself unusable/unsafe gate validity; lane-quality softness
            # (continuity_ok, width_stability_ok, center_ok, sparse points
            # above the mode-appropriate minimum) only demotes VALID to
            # DEGRADED through the confidence/branch_suspected checks below.
            if self.config.steering_only_validity:
                # Steering-angle-only gate: a path is emitted as long as it is
                # geometrically expressible (>=3 points, ego road present) and
                # the required steering stays within the mechanical limit.
                # Every other condition (near-field loss, branch, clearance,
                # topology, road projection) only demotes VALID->DEGRADED via
                # the state test below, so the metric/BEV stack can run on the
                # emitted points without inheriting the strict corridor gate.
                hard_invalid = (
                    not np.any(component) or               # no ego-connected road
                    not steering["steering_angle_ok"])      # exceeds +/-max steering
            else:
                hard_invalid = (
                    not road_safety["near_field_ok"] or       # near-field road lost
                    not np.any(component) or                   # no ego-connected road
                    road_safety["branch_critical"] or           # real branch, no safe pick
                    not physical_geometry_ok or                 # vehicle/lane geometry invalid
                    not final_road_safe or                       # exact final path/margin unsafe
                    not topology_ok or                          # rearward/looping path
                    not steering["steering_angle_ok"])          # exceeds +/-22 degrees
            valid = (not hard_invalid) and enough_points
            # A suspected far/mid-field branch is still usable as a cautious
            # observation when the near corridor is safe, but it must not be
            # advertised as full-quality lane tracking. ImagePath.path_state
            # carries this distinction to the production controller without
            # expanding the message contract.
            state = (VALID if (valid and
                               confidence >= self.config.valid_min_confidence and
                               not road_safety["branch_suspected"] and
                               not final_road_safety["final_road_recovered"] and
                               not steering_repaired and
                               steering["steering_continuity_ok"] and
                               steering["steering_rate_ok"] and
                               final_curvature_ok and
                               temporal["temporal_ok"] and
                               horizon_sufficient and
                               continuity_ok and width_stability_ok and center_ok)
                     else DEGRADED if valid else INVALID)
            if valid:
                alpha = np.clip(self.config.lane_width_update_alpha, 0.0, 1.0)
                # Sparse lane fragments may coexist with a perfectly usable
                # ROAD_CENTER path, but they must not poison the learned lane
                # width model.  Update only from a complete-enough A frame.
                if pair_rows >= self.config.valid_min_points:
                    for row, width in self._pending_width_updates:
                        old = self.width_profile.get(row, width)
                        self.width_profile[row] = (1-alpha)*old + alpha*width
                self.previous = fitted.copy()
                self.previous_confidence = confidence
                self.fallback_age = 0
                self.previous_boundary_mode = boundary_mode
                self.previous_source_mode = confirmed_source_mode
                self._commit_steering_state(steering, timestamp_sec)
                self._previous_left_track = ({int(y): float(x) for x, y in left}
                                             if len(left) else {})
                self._previous_right_track = ({int(y): float(x) for x, y in right}
                                              if len(right) else {})
                diagnostics.update({
                    "lane_width_profile_rows": len(self.width_profile),
                    "expected_lane_width_near_px": self._width_near(
                        self.config.roi_bottom,
                        self.config.width_profile_lookup_radius_px),
                    "expected_lane_width_middle_px": self._width_near(
                        (self.config.roi_top+self.config.roi_bottom)//2,
                        self.config.width_profile_lookup_radius_px),
                })
            else:
                self.fallback_age += 1
            rejections["ROAD_CONTAINMENT_FAILURE"] = road_checked-road_inside
            curvature = final_curvature_px
            lookahead = self._lookahead_ratio(curvature)
            diagnostics.update({"road_evaluated_points": road_checked,
                                "road_contained_points": road_inside,
                                "road_containment_ratio": road_score,
                                "road_validation_available": road_available,
                                "road_soft_validation_passed": road_ok,
                                "enough_points": enough_points,
                                "required_points": required_points,
                                "quality_target_points": quality_target_points,
                                "target_path_points": self.config.target_path_points,
                                "path_horizon_ratio": horizon_ratio,
                                "horizon_sufficient": horizon_sufficient,
                                "single_boundary_path": single_boundary_path,
                                "road_only_path": road_only_path,
                                "road_dependent_path": road_dependent_path,
                                "vehicle_containment_ok": containment_ok,
                                "near_field_ok": road_safety["near_field_ok"],
                                "road_width_stability_ok": width_stability_ok,
                                "road_center_ok": center_ok,
                                "continuity_ok": continuity_ok,
                                "temporal_ok": temporal["temporal_ok"],
                                "temporal_shift_px": temporal["temporal_shift_px"],
                                "temporal_signed_shift_px": temporal["temporal_signed_shift_px"],
                                "temporal_alpha_used": temporal["temporal_alpha_used"],
                                "temporal_outlier_rejected": temporal["temporal_outlier_rejected"],
                                "temporal_stale_recovery": temporal["temporal_stale_recovery"],
                                "consistent_shift_frames": temporal["consistent_shift_frames"],
                                "boundary_mode": boundary_mode,
                                "boundary_mode_transition": boundary_mode_transition,
                                "source_mode": source_mode,
                               "source_mode_confirmed": confirmed_source_mode,
                               "source_mode_transition": source_mode_transition,
                                "source_pending_hold": source_pending_hold,
                                "source_release_streak": self._source_release_streak,
                                "final_curvature_ok": final_curvature_ok,
                                "final_curvature_warning": not final_curvature_ok,
                                "steering_repair_requested": steering_repair_requested,
                                "steering_repaired": steering_repaired,
                                "steering_repair_attempts": steering_repair_attempts,
                                "topology_ok": topology_ok,
                                "hard_invalid": hard_invalid,
                                "polynomial_residual_px": residual,
                                "fallback_ratio": fallback_ratio,
                                "path_curvature_px": curvature,
                                "recommended_lookahead_y_ratio": lookahead,
                                "reacquired_after_loss": reacquiring})
            diagnostics.update(steering)
            diagnostics["final_path_points"] = len(fitted)
            result = PathResult(fitted, sources, confidence, state, valid, 0.0,
                                component, left, right, raw, diagnostics, components)
        elif (self.previous is not None and
              self.fallback_age < self.config.max_temporal_fallback_frames):
            self.fallback_age += 1
            confidence = self.previous_confidence * max(
                0.0, 1-self.config.temporal_fallback_confidence_decay*self.fallback_age)
            freshness = max(0.0, 1-self.fallback_age /
                            (self.config.max_temporal_fallback_frames+1))
            components.update({"temporal_score": freshness,
                               "freshness_score": freshness,
                               "fallback_score": 0.0})
            fallback_points, fallback_safe, fallback_safety = (
                self._project_final_path_to_road(
                    self.previous, component, effective_roi_bottom))
            fallback_topology_ok = self._final_topology_ok(fallback_points)
            fallback_steering = self._steering_feasibility(
                fallback_points, shape[1], timestamp_sec)
            fallback_feasible = bool(
                fallback_safe and fallback_topology_ok and
                fallback_steering["steering_angle_ok"])
            curvature = self._curvature_px(fallback_points)
            diagnostics.update({"fallback_ratio": 1.0,
                                "complete_lane_loss_fallback_age": self.fallback_age,
                                "path_curvature_px": curvature,
                                "recommended_lookahead_y_ratio": self._lookahead_ratio(curvature),
                                "temporal_alpha_used": 0.0,
                                "temporal_outlier_rejected": False})
            diagnostics.update(fallback_safety)
            diagnostics.update(fallback_steering)
            diagnostics["topology_ok"] = fallback_topology_ok
            diagnostics["vehicle_containment_ok"] = bool(fallback_safe)
            if fallback_feasible and len(fallback_points):
                result = PathResult(
                    fallback_points,
                    [TEMPORAL_FALLBACK]*len(fallback_points), confidence,
                    DEGRADED, True, 0.0, component, left, right, raw,
                    diagnostics, components)
                self._commit_steering_state(fallback_steering, timestamp_sec)
            else:
                empty = np.empty((0, 2), float)
                result = PathResult(empty, [], 0.0, INVALID, False, 0.0,
                                    component, left, right, raw,
                                    diagnostics, components)
        else:
            self.fallback_age += 1
            empty = np.empty((0, 2), float)
            result = PathResult(empty, [], 0.0, INVALID, False, 0.0,
                                component, left, right, raw, diagnostics, components)
        # Empty output is vacuously off-road-free.  Keep evaluation separate
        # so INVALID/no-path frames are not mislabeled as an off-road path.
        if not len(result.points):
            result.diagnostics["vehicle_containment_ok"] = True
            result.diagnostics.setdefault("final_road_safety_evaluated", False)
        result.diagnostics["final_path_points"] = len(result.points)
        result.virtual = self._last_virtual_points.copy()
        result.virtual_details = self._last_virtual_details.copy()
        details = self._last_virtual_details
        result.diagnostics.update({
            "virtual_center_points": len(self._last_virtual_points),
            "ego_anchor_applied": self._last_ego_anchor_applied,
            "ego_anchor_x_px": float(self.config.vehicle_center_x_px),
            "ego_anchor_y_px": float(self.config.roi_bottom),
            "virtual_offset_mean_px": (float(np.mean(
                [item["offset_px"] for item in details])) if details else None),
            "virtual_offset_min_px": (float(np.min(
                [item["offset_px"] for item in details])) if details else None),
            "virtual_offset_max_px": (float(np.max(
                [item["offset_px"] for item in details])) if details else None),
        })
        result.latency_ms = (time.perf_counter()-started)*1000.0
        # Progressive corridor widen-on-loss / narrow-on-reacquire, driven by
        # this frame's own outcome so the NEXT frame's dynamic PATH ROI
        # reflects it (section 2). Never snaps straight to full width/back.
        self._update_corridor_state(bool(result.valid))
        result.diagnostics["path_corridor_state"] = self._corridor_state
        result.diagnostics["path_corridor_expand_level"] = float(self._corridor_expand_level)
        return result
