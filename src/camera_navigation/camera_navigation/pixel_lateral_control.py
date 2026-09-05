"""Pixel-space lateral controller (no BEV, no camera extrinsics needed).

This is the geometry-free alternative to the metric Pure Pursuit path. It
takes the pixel-space path (race_interfaces/ImagePath: points with x_px/y_px
plus image_width/height) and produces a steering angle from the lateral
offset of a look-ahead row, using a PD law. No ground-plane calibration,
no IMU attitude lock, no camera mount config is required -- the trade-off is
that steering is a tuned image-plane response rather than a metric-accurate
Pure Pursuit solution.

All functions are pure and ROS-independent so they can be unit tested
off-vehicle.
"""
import math


def lookahead_offset_px(points_xy, image_width, lookahead_y_ratio):
    """Signed lateral pixel offset of the path from image center.

    points_xy: iterable of (x_px, y_px), image origin top-left, y grows down.
    Picks the path point whose y is closest to the look-ahead row
    (lookahead_y_ratio * image_height, but expressed via the point that best
    matches it), then returns (x_px - center_x). Positive = path is to the
    RIGHT of center. Returns None if no usable point.

    The look-ahead row is chosen as a fraction of the vertical span of the
    path: 0.0 = nearest (bottom of image, closest to vehicle), 1.0 = farthest
    (top). A mid value (~0.5) trades reaction distance against stability.
    """
    pts = [(float(x), float(y)) for x, y in points_xy
           if math.isfinite(x) and math.isfinite(y)]
    if not pts or not math.isfinite(image_width) or image_width <= 0:
        return None
    if not math.isfinite(lookahead_y_ratio):
        return None
    ratio = min(1.0, max(0.0, lookahead_y_ratio))
    y_values = [y for _, y in pts]
    y_near, y_far = max(y_values), min(y_values)  # near=bottom(large y)
    # target row: interpolate between near(ratio=0) and far(ratio=1)
    target_y = y_near + (y_far - y_near) * ratio
    best = min(pts, key=lambda p: abs(p[1] - target_y))
    center_x = image_width / 2.0
    return best[0] - center_x


def steering_from_offset_deg(
        offset_px, previous_offset_px, dt_s,
        image_width,
        proportional_gain_deg_per_norm,
        derivative_gain_deg_per_norm_per_s,
        max_steering_deg):
    """PD steering (degrees) from a normalized lateral pixel offset.

    The offset is normalized by half the image width so gains are resolution
    independent: norm = offset_px / (image_width/2), in roughly [-1, 1].
    Positive offset (path to the right) yields positive steering before the
    caller applies the vehicle steering_sign.

    Fail-closed: any non-finite input yields 0.0 steering (straight), which
    the caller's state machine treats as a benign command, never propulsion.
    """
    try:
        half = image_width / 2.0
        if not math.isfinite(half) or half <= 0.0:
            return 0.0
        if not all(math.isfinite(v) for v in (
                offset_px, proportional_gain_deg_per_norm,
                derivative_gain_deg_per_norm_per_s, max_steering_deg)):
            return 0.0
        norm = offset_px / half
        p_term = proportional_gain_deg_per_norm * norm
        d_term = 0.0
        if (previous_offset_px is not None and math.isfinite(previous_offset_px)
                and math.isfinite(dt_s) and dt_s > 0.0):
            prev_norm = previous_offset_px / half
            d_term = derivative_gain_deg_per_norm_per_s * (norm - prev_norm) / dt_s
        steering = p_term + d_term
        if not math.isfinite(steering):
            return 0.0
        return float(max(-max_steering_deg, min(max_steering_deg, steering)))
    except Exception:
        return 0.0
