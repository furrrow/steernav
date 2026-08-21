"""
https://github.com/gershom96/VLA_DataGeneration/blob/main/utils/pointcloud_utils.py
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def camera_to_base_transform(camera_height=1.5):
    """
    Transform from camera frame (x right, y down, z forward)
    to base frame (x forward, y left, z up).

    The returned translation places the camera `camera_height`
    meters above the ground plane in the base frame.
    """
    rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    translation = np.asarray([0.0, 0.0, camera_height], dtype=np.float32)
    return rotation, translation


def transform_points(points_xyz, R=None, t=None):
    """
    Transform points from camera frame to robot/base frame.

    Args:
        points_xyz: (H, W, 3) or (N, 3) array
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        transformed points with same leading shape
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    orig_shape = pts.shape
    pts_flat = pts.reshape(-1, 3)

    if R is None:
        R = np.eye(3, dtype=np.float32)
    if t is None:
        t = np.zeros(3, dtype=np.float32)

    pts_out = (pts_flat @ R.T) + t[None, :]
    return pts_out.reshape(orig_shape)


def _normalize_vector(vector, eps=1e-8):
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return vec / norm


def rotation_matrix_from_axis_angle(axis, angle_rad):
    axis_unit = _normalize_vector(axis)
    if axis_unit is None or abs(float(angle_rad)) <= 1e-8:
        return np.eye(3, dtype=np.float32)

    x, y, z = axis_unit
    skew = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    sin_theta = float(np.sin(angle_rad))
    cos_theta = float(np.cos(angle_rad))
    identity = np.eye(3, dtype=np.float32)
    return identity + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)


def _fit_plane_from_points(points_xyz):
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = _normalize_vector(normal)
    if normal is None:
        return None
    if normal[2] < 0.0:
        normal = -normal
    offset = float(-np.dot(normal, centroid))
    distances = pts @ normal + offset
    rmse = float(np.sqrt(np.mean(distances**2)))
    return {
        "normal": normal,
        "offset": offset,
        "rmse": rmse,
    }


def select_ground_plane_candidates(
    points_xyz,
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    min_forward=0.5,
    z_min=-2.0,
    z_max=2.0,
    cell_size=0.75,
    max_points=4000,
):
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return np.empty((0, 3), dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    roi = (
        (x >= max(float(x_min), float(min_forward)))
        & (x <= float(x_max))
        & (y >= float(y_min))
        & (y <= float(y_max))
        & (z >= float(z_min))
        & (z <= float(z_max))
    )
    pts = pts[finite & roi]
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    coarse_x_min = max(float(x_min), float(min_forward))
    coarse_y_min = float(y_min)
    cols = np.floor((pts[:, 0] - coarse_x_min) / float(cell_size)).astype(np.int32)
    rows = np.floor((pts[:, 1] - coarse_y_min) / float(cell_size)).astype(np.int32)
    num_cols = max(1, int(np.ceil((float(x_max) - coarse_x_min) / float(cell_size))))
    keys = rows.astype(np.int64) * int(num_cols) + cols.astype(np.int64)

    order = np.lexsort((pts[:, 2], keys))
    pts_sorted = pts[order]
    keys_sorted = keys[order]
    keep = np.ones(len(keys_sorted), dtype=bool)
    keep[1:] = keys_sorted[1:] != keys_sorted[:-1]
    candidates = pts_sorted[keep]

    if len(candidates) > int(max_points):
        sample_idx = np.linspace(0, len(candidates) - 1, int(max_points)).astype(np.int32)
        candidates = candidates[sample_idx]

    return candidates.astype(np.float32, copy=False)


def fit_ground_plane_ransac(
    candidate_points_xyz,
    max_iterations=120,
    distance_threshold=0.08,
    max_tilt_deg=35.0,
    min_inlier_ratio=0.12,
    min_inlier_count=40,
    rng=None,
):
    pts = np.asarray(candidate_points_xyz, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None

    rng = np.random.default_rng() if rng is None else rng
    min_cos_tilt = float(np.cos(np.deg2rad(max_tilt_deg)))
    min_support = max(int(min_inlier_count), int(np.ceil(float(min_inlier_ratio) * len(pts))))

    best_inliers = None
    best_count = 0
    best_error = np.inf

    for _ in range(int(max_iterations)):
        sample_idx = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        normal = _normalize_vector(normal)
        if normal is None:
            continue
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < min_cos_tilt:
            continue

        offset = float(-np.dot(normal, p0))
        distances = np.abs(pts @ normal + offset)
        inliers = distances <= float(distance_threshold)
        inlier_count = int(inliers.sum())
        if inlier_count < min_support:
            continue

        mean_error = float(distances[inliers].mean())
        if inlier_count > best_count or (inlier_count == best_count and mean_error < best_error):
            best_inliers = inliers
            best_count = inlier_count
            best_error = mean_error

    if best_inliers is None:
        return None

    fit = _fit_plane_from_points(pts[best_inliers])
    if fit is None:
        return None

    if fit["normal"][2] < min_cos_tilt:
        return None

    refined_distances = np.abs(pts @ fit["normal"] + fit["offset"])
    refined_inliers = refined_distances <= float(distance_threshold)
    fit["inlier_mask"] = refined_inliers
    fit["inlier_count"] = int(refined_inliers.sum())
    fit["candidate_count"] = int(len(pts))
    fit["inlier_ratio"] = float(fit["inlier_count"] / max(1, len(pts)))
    fit["inlier_rmse"] = float(np.sqrt(np.mean(refined_distances[refined_inliers] ** 2))) if np.any(refined_inliers) else None
    fit["tilt_deg"] = float(np.degrees(np.arccos(np.clip(fit["normal"][2], -1.0, 1.0))))
    return fit


def estimate_ground_plane_correction(
    points_xyz,
    sensor_position,
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    min_forward=0.5,
    candidate_z_min=-2.0,
    candidate_z_max=2.0,
    candidate_cell_size=0.75,
    ransac_iterations=120,
    ransac_distance_threshold=0.08,
    max_ground_tilt_deg=35.0,
    max_correction_deg=8.0,
    prior_ground_normal=None,
    smoothing=0.65,
):
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    diagnostics = {
        "enabled": True,
        "fit_success": False,
        "normal_source": "failed",
        "candidate_count": 0,
        "inlier_count": 0,
        "raw_ground_normal_xyz": None,
        "smoothed_ground_normal_xyz": None,
        "raw_tilt_deg": None,
        "applied_tilt_deg": 0.0,
        "inlier_rmse_m": None,
        "sensor_origin_xyz": [float(v) for v in np.asarray(sensor_position, dtype=np.float32)],
        "correction_rotation_matrix": np.eye(3, dtype=np.float32).tolist(),
    }

    candidates = select_ground_plane_candidates(
        points_xyz,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_forward=min_forward,
        z_min=candidate_z_min,
        z_max=candidate_z_max,
        cell_size=candidate_cell_size,
    )
    diagnostics["candidate_count"] = int(len(candidates))

    fit = fit_ground_plane_ransac(
        candidates,
        max_iterations=ransac_iterations,
        distance_threshold=ransac_distance_threshold,
        max_tilt_deg=max_ground_tilt_deg,
    )

    smoothed_normal = None
    if fit is not None:
        diagnostics["fit_success"] = True
        diagnostics["raw_ground_normal_xyz"] = [float(v) for v in fit["normal"]]
        diagnostics["raw_tilt_deg"] = float(fit["tilt_deg"])
        diagnostics["inlier_count"] = int(fit["inlier_count"])
        diagnostics["inlier_rmse_m"] = float(fit["inlier_rmse"]) if fit["inlier_rmse"] is not None else None
        smoothed_normal = np.asarray(fit["normal"], dtype=np.float32)
        prior_normal = _normalize_vector(prior_ground_normal) if prior_ground_normal is not None else None
        blend = float(np.clip(smoothing, 0.0, 0.999))
        if prior_normal is not None and blend > 0.0:
            blended = blend * prior_normal + (1.0 - blend) * smoothed_normal
            blended = _normalize_vector(blended)
            if blended is not None and blended[2] >= 0.0:
                smoothed_normal = blended
                diagnostics["normal_source"] = "fit_smoothed"
            else:
                diagnostics["normal_source"] = "fit"
        else:
            diagnostics["normal_source"] = "fit"
    elif prior_ground_normal is not None:
        prior_normal = _normalize_vector(prior_ground_normal)
        if prior_normal is not None and prior_normal[2] >= 0.0:
            smoothed_normal = prior_normal
            diagnostics["normal_source"] = "prior"

    if smoothed_normal is None:
        return np.eye(3, dtype=np.float32), diagnostics

    smoothed_normal = _normalize_vector(smoothed_normal)
    if smoothed_normal is None or smoothed_normal[2] < 0.0:
        return np.eye(3, dtype=np.float32), diagnostics

    diagnostics["smoothed_ground_normal_xyz"] = [float(v) for v in smoothed_normal]
    raw_angle = float(np.arccos(np.clip(np.dot(smoothed_normal, up), -1.0, 1.0)))
    apply_angle = min(raw_angle, float(np.deg2rad(max_correction_deg)))
    diagnostics["applied_tilt_deg"] = float(np.degrees(apply_angle))

    axis = np.cross(smoothed_normal, up)
    axis = _normalize_vector(axis)
    if axis is None or apply_angle <= 1e-8:
        return np.eye(3, dtype=np.float32), diagnostics

    correction = rotation_matrix_from_axis_angle(axis, apply_angle)
    diagnostics["correction_rotation_matrix"] = correction.astype(np.float32).tolist()
    return correction.astype(np.float32), diagnostics


def bbox_to_pointmap_bounds(bbox, full_image_size, map_shape, bbox_center_fraction=1.0):
    """
    Project an image-space bounding box into point-map pixel bounds.

    Returns:
        (sx1, sy1, sx2, sy2) with end-exclusive bounds, or None if invalid.
    """
    image_w, image_h = full_image_size
    map_h, map_w = map_shape[:2]
    if image_w <= 0 or image_h <= 0 or map_w <= 0 or map_h <= 0:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    fraction = float(np.clip(bbox_center_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return None

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = max(0.5, 0.5 * (x2 - x1) * fraction)
    half_h = max(0.5, 0.5 * (y2 - y1) * fraction)

    sx1 = int(np.clip(np.floor((cx - half_w) / image_w * map_w), 0, map_w - 1))
    sy1 = int(np.clip(np.floor((cy - half_h) / image_h * map_h), 0, map_h - 1))
    sx2 = int(np.clip(np.ceil((cx + half_w) / image_w * map_w), sx1 + 1, map_w))
    sy2 = int(np.clip(np.ceil((cy + half_h) / image_h * map_h), sy1 + 1, map_h))
    return sx1, sy1, sx2, sy2


def _estimate_camera_goal_from_valid_mask(roi_points, valid_mask):
    """
    Reduce a bbox ROI to one camera-frame point using per-column nearest returns.

    Returns:
        (xyz_camera, valid_column_count) or (None, 0) if no valid columns remain.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return None, 0

    planar_ranges = np.linalg.norm(roi_points[..., (0, 2)], axis=-1)
    planar_ranges = np.where(valid, planar_ranges, np.inf)

    best_rows = np.argmin(planar_ranges, axis=0)
    col_indices = np.arange(planar_ranges.shape[1])
    best_ranges = planar_ranges[best_rows, col_indices]
    valid_cols = np.isfinite(best_ranges)
    if not np.any(valid_cols):
        return None, 0

    selected_points = roi_points[best_rows[valid_cols], col_indices[valid_cols]]
    xyz_camera = np.median(selected_points, axis=0).astype(np.float32)
    return xyz_camera, int(valid_cols.sum())


def _build_overlap_exclusion_mask(
    target_bounds,
    other_bboxes,
    full_image_size,
    map_shape,
):
    """
    Build a ROI-local mask for pixels overlapped by other image-space detections.
    """
    sx1, sy1, sx2, sy2 = target_bounds
    overlap_mask = np.zeros((sy2 - sy1, sx2 - sx1), dtype=bool)

    for other_bbox in other_bboxes or ():
        other_bounds = bbox_to_pointmap_bounds(
            bbox=other_bbox,
            full_image_size=full_image_size,
            map_shape=map_shape,
            bbox_center_fraction=1.0,
        )
        if other_bounds is None:
            continue

        ox1, oy1, ox2, oy2 = other_bounds
        ix1 = max(sx1, ox1)
        iy1 = max(sy1, oy1)
        ix2 = min(sx2, ox2)
        iy2 = min(sy2, oy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        overlap_mask[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1] = True

    return overlap_mask


def estimate_goal_from_bbox_columns(
    points_map,
    mask,
    bbox,
    full_image_size,
    bbox_center_fraction=1.0,
    camera_height=0.0,
    other_bboxes=None,
    base_frame_correction_R=None,
    base_frame_correction_origin=None,
):
    """
    Estimate a target point by collapsing each bbox column to its nearest return.

    The selected 3D point is converted from camera coordinates
    (x right, y down, z forward) into base coordinates
    (x forward, y left, z up).
    """
    bounds = bbox_to_pointmap_bounds(
        bbox=bbox,
        full_image_size=full_image_size,
        map_shape=points_map.shape,
        bbox_center_fraction=bbox_center_fraction,
    )
    if bounds is None:
        return None

    sx1, sy1, sx2, sy2 = bounds
    roi_points = np.asarray(points_map[sy1:sy2, sx1:sx2], dtype=np.float32)
    roi_mask = np.asarray(mask[sy1:sy2, sx1:sx2], dtype=bool)
    if roi_points.size == 0 or roi_mask.size == 0:
        return None

    base_valid = roi_mask & np.isfinite(roi_points).all(axis=-1)
    if not np.any(base_valid):
        return None

    xyz_camera, base_valid_cols = _estimate_camera_goal_from_valid_mask(roi_points, base_valid)
    if xyz_camera is None:
        return None

    overlap_mask = _build_overlap_exclusion_mask(
        target_bounds=bounds,
        other_bboxes=other_bboxes,
        full_image_size=full_image_size,
        map_shape=points_map.shape,
    )
    if np.any(overlap_mask):
        visible_valid = base_valid & ~overlap_mask
        visible_xyz_camera, visible_valid_cols = _estimate_camera_goal_from_valid_mask(roi_points, visible_valid)

        # If overlap removal leaves only a tiny sliver of the detection,
        # fall back to the full bbox estimate to avoid overfitting to noise.
        min_required_cols = max(1, int(np.ceil(0.25 * base_valid_cols)))
        if visible_xyz_camera is not None and visible_valid_cols >= min_required_cols:
            xyz_camera = visible_xyz_camera

    rotation, translation = camera_to_base_transform(camera_height=camera_height)
    xyz_base = transform_points(xyz_camera[None, :], R=rotation, t=translation)[0].astype(np.float32)
    if base_frame_correction_R is not None:
        correction_origin = translation if base_frame_correction_origin is None else np.asarray(base_frame_correction_origin, dtype=np.float32)
        xyz_base = transform_points(
            xyz_base[None, :] - correction_origin[None, :],
            R=np.asarray(base_frame_correction_R, dtype=np.float32),
            t=correction_origin,
        )[0].astype(np.float32)

    return {
        "goal_xy_m": [float(xyz_base[0]), float(xyz_base[1])],
        "goal_xyz_m": [float(x) for x in xyz_base],
        "goal_xyz_camera_m": [float(x) for x in xyz_camera],
    }


def filter_points_by_height_and_roi(
    points_xyz,
    h_min,
    h_max,
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    z_axis=2,
):
    """
    Filter 3D points by:
      - valid finite values
      - forward/lateral ROI
      - height range [h_min, h_max]

    Assumes robot-frame axes:
      x = forward
      y = lateral
      z = height

    Args:
        points_xyz: (H, W, 3) or (N, 3)
        h_min, h_max: keep points with h_min <= z <= h_max
        x_min, x_max, y_min, y_max: BEV region of interest
        z_axis: axis index corresponding to height (usually 2)

    Returns:
        filtered_points: (M, 3)
        valid_mask: boolean mask over flattened input
    """
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)

    finite = np.isfinite(pts).all(axis=1)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, z_axis]

    roi = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    height_ok = (z >= h_min) & (z <= h_max)

    valid_mask = finite & roi & height_ok
    return pts[valid_mask], valid_mask


def bev_grid_params(x_min, x_max, y_min, y_max, resolution):
    """
    Compute BEV grid dimensions.
    """
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    return height, width


def points_to_bev_indices(points_xyz, x_min, y_min, resolution, H, W):
    """
    Convert robot-frame XY points into BEV grid indices.

    Returns:
        rows, cols, in_bounds
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]

    cols = np.floor((x - x_min) / resolution).astype(np.int32)
    rows = np.floor((y - y_min) / resolution).astype(np.int32)

    in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    return rows, cols, in_bounds

def points_to_pseudo_laserscan(
    points_xyz,
    angle_min=-np.pi / 2,
    angle_max=np.pi / 2,
    angle_increment=np.deg2rad(0.5),
    range_min=0.05,
    range_max=20.0,
):
    """
    Collapse 3D points to a 2D pseudo-LaserScan in the robot frame.

    Uses XY plane only:
      range = sqrt(x^2 + y^2)
      angle = atan2(y, x)

    For each angular bin, keeps the nearest point.

    Returns:
        scan dict with fields similar to ROS LaserScan:
        {
            'angle_min', 'angle_max', 'angle_increment',
            'range_min', 'range_max', 'ranges'
        }
    """
    num_bins = int(np.floor((angle_max - angle_min) / angle_increment)) + 1
    ranges = np.full((num_bins,), np.inf, dtype=np.float32)

    if len(points_xyz) > 0:
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        r = np.sqrt(x**2 + y**2)
        a = np.arctan2(y, x)

        valid = (
            np.isfinite(r)
            & np.isfinite(a)
            & (r >= range_min)
            & (r <= range_max)
            & (a >= angle_min)
            & (a <= angle_max)
        )

        r = r[valid]
        a = a[valid]

        bin_idx = np.floor((a - angle_min) / angle_increment).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

        for i, rr in zip(bin_idx, r):
            if rr < ranges[i]:
                ranges[i] = rr

    return {
        "angle_min": float(angle_min),
        "angle_max": float(angle_max),
        "angle_increment": float(angle_increment),
        "range_min": float(range_min),
        "range_max": float(range_max),
        "ranges": ranges,
    }


def world_to_grid(x, y, x_min, y_min, resolution):
    """
    Continuous XY -> integer grid indices
    """
    col = int(np.floor((x - x_min) / resolution))
    row = int(np.floor((y - y_min) / resolution))
    return row, col


def bresenham_line(r0, c0, r1, c1):
    """
    Integer grid cells along a line using Bresenham.
    """
    cells = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    err = dc - dr
    r, c = r0, c0

    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr

    return cells


def _grid_centers_from_indices(rows, cols, x_min, y_min, resolution):
    x = x_min + (cols.astype(np.float32) + 0.5) * resolution
    y = y_min + (rows.astype(np.float32) + 0.5) * resolution
    z = np.zeros_like(x, dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def _continuous_neighbor_mask(points_a, points_b, max_neighbor_distance_m):
    finite = np.isfinite(points_a).all(axis=-1) & np.isfinite(points_b).all(axis=-1)
    if not np.any(finite):
        return finite
    distance = np.linalg.norm(points_a - points_b, axis=-1)
    return finite & (distance <= float(max_neighbor_distance_m))


def rasterize_organized_occluders(
    points_xyz,
    x_min,
    x_max,
    y_min,
    y_max,
    resolution,
    H,
    W,
    max_neighbor_distance_m=0.5,
):
    """
    Rasterize an organized point map into BEV occluder cells.

    Each valid depth pixel is a surface return. Adjacent image pixels on a
    continuous surface represent the surface between their projected BEV
    samples, even when the ESDF resolution is finer than the pixel footprint
    at range. This fills those sub-pixel BEV gaps for visibility ray stopping
    without treating every return as an obstacle.
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        flat = pts.reshape(-1, 3)
        rows, cols, in_bounds = points_to_bev_indices(flat, x_min, y_min, resolution, H, W)
        occluder = np.zeros((H, W), dtype=bool)
        occluder[rows[in_bounds], cols[in_bounds]] = True
        return occluder

    finite = np.isfinite(pts).all(axis=-1)
    roi = (
        finite
        & (pts[..., 0] >= x_min)
        & (pts[..., 0] <= x_max)
        & (pts[..., 1] >= y_min)
        & (pts[..., 1] <= y_max)
    )
    occluder = np.zeros((H, W), dtype=bool)
    if not np.any(roi):
        return occluder

    rows = np.zeros(pts.shape[:2], dtype=np.int32)
    cols = np.zeros(pts.shape[:2], dtype=np.int32)
    rows[roi] = np.floor((pts[..., 1][roi] - y_min) / resolution).astype(np.int32)
    cols[roi] = np.floor((pts[..., 0][roi] - x_min) / resolution).astype(np.int32)
    rows[roi] = np.clip(rows[roi], 0, H - 1)
    cols[roi] = np.clip(cols[roi], 0, W - 1)
    occluder[rows[roi], cols[roi]] = True

    neighbor_pairs = (
        (pts[:, :-1], pts[:, 1:], roi[:, :-1], roi[:, 1:], rows[:, :-1], rows[:, 1:], cols[:, :-1], cols[:, 1:]),
        (pts[:-1, :], pts[1:, :], roi[:-1, :], roi[1:, :], rows[:-1, :], rows[1:, :], cols[:-1, :], cols[1:, :]),
    )
    for points_a, points_b, roi_a, roi_b, rows_a, rows_b, cols_a, cols_b in neighbor_pairs:
        continuous = roi_a & roi_b & _continuous_neighbor_mask(points_a, points_b, max_neighbor_distance_m)
        for r0, r1, c0, c1 in zip(rows_a[continuous], rows_b[continuous], cols_a[continuous], cols_b[continuous]):
            for rr, cc in bresenham_line(int(r0), int(c0), int(r1), int(c1)):
                if 0 <= rr < H and 0 <= cc < W:
                    occluder[rr, cc] = True

    return occluder


def raytrace_visibility_from_points(
    points_xyz,
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    resolution=0.05,
    sensor_xy=(0.0, 0.0),
    fov_points_xyz=None,
    occluder_points_xyz=None,
    angle_min=None,
    angle_max=None,
    angle_increment=None,
    fill_between_beams=True,
    bridge_occluder_neighbors=True,
    max_occluder_neighbor_distance_m=0.5,
):
    """
    Build:
      - occupied cells from all retained point endpoints
      - visible free cells by raytracing one beam per angle bin

    Occupancy keeps every in-bounds point so vertically separated structures
    remain represented in the obstacle map. Visibility ray stops are built
    from the broader set of ROI-valid depth returns, not only obstacle-height
    points, so visible surfaces occlude the cells behind them. When an
    organized point map is provided, neighboring image pixels on continuous
    surfaces are bridged in BEV to cover the growing meter-per-pixel footprint
    at range. Beams with no observed return inside the angular span are marked
    visible free out to the ROI boundary.

    Returns:
        occupied_mask: (H, W) bool
        visible_free_mask: (H, W) bool
        known_mask: (H, W) bool
    """
    H, W = bev_grid_params(x_min, x_max, y_min, y_max, resolution)

    occupied = np.zeros((H, W), dtype=bool)
    visible_free = np.zeros((H, W), dtype=bool)

    sensor_x, sensor_y = float(sensor_xy[0]), float(sensor_xy[1])
    sensor_r, sensor_c = world_to_grid(sensor_x, sensor_y, x_min, y_min, resolution)
    sensor_r = int(np.clip(sensor_r, 0, H - 1))
    sensor_c = int(np.clip(sensor_c, 0, W - 1))

    occ_pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    occ_pts = occ_pts[np.isfinite(occ_pts).all(axis=1)]
    if len(occ_pts) > 0:
        rows, cols, in_bounds = points_to_bev_indices(occ_pts, x_min, y_min, resolution, H, W)
        rows = rows[in_bounds]
        cols = cols[in_bounds]
        occupied[rows, cols] = True
        occ_pts = occ_pts[in_bounds]

    fov_source = occ_pts if fov_points_xyz is None else np.asarray(fov_points_xyz, dtype=np.float32)
    occluder_source = (
        fov_source
        if occluder_points_xyz is None
        else np.asarray(occluder_points_xyz, dtype=np.float32)
    )

    if fov_points_xyz is None:
        fov_pts = occ_pts
    else:
        fov_pts = fov_source.reshape(-1, 3)
        fov_pts = fov_pts[np.isfinite(fov_pts).all(axis=1)]
        if len(fov_pts) > 0:
            _, _, fov_in_bounds = points_to_bev_indices(fov_pts, x_min, y_min, resolution, H, W)
            fov_pts = fov_pts[fov_in_bounds]

    if len(occ_pts) == 0 and len(fov_pts) == 0:
        known = occupied | visible_free
        return occupied, visible_free, known

    if len(fov_pts) == 0:
        fov_pts = occ_pts

    observed_angles = np.arctan2(fov_pts[:, 1] - sensor_y, fov_pts[:, 0] - sensor_x)
    if angle_min is None:
        angle_min = float(np.min(observed_angles))
    else:
        angle_min = float(angle_min)
    if angle_max is None:
        angle_max = float(np.max(observed_angles))
    else:
        angle_max = float(angle_max)

    if angle_max < angle_min:
        angle_min, angle_max = angle_max, angle_min

    if angle_increment is None:
        corners = np.asarray(
            [
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_min],
                [x_max, y_max],
            ],
            dtype=np.float32,
        )
        max_range = float(np.max(np.linalg.norm(corners - np.asarray([sensor_x, sensor_y], dtype=np.float32), axis=1)))
        angle_increment = float(np.arctan2(resolution, max(max_range, resolution)))
    else:
        angle_increment = float(angle_increment)

    angle_increment = max(angle_increment, 1e-4)

    num_bins = int(np.floor((angle_max - angle_min) / angle_increment)) + 1
    hit_ranges = np.full((num_bins,), np.inf, dtype=np.float32)
    hit_points = np.full((num_bins, 2), np.nan, dtype=np.float32)

    if bridge_occluder_neighbors and occluder_source.ndim == 3:
        occluder_mask = rasterize_organized_occluders(
            occluder_source,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            H=H,
            W=W,
            max_neighbor_distance_m=max_occluder_neighbor_distance_m,
        )
        occluder_rows, occluder_cols = np.nonzero(occluder_mask)
        hit_pts = _grid_centers_from_indices(occluder_rows, occluder_cols, x_min, y_min, resolution)
    else:
        hit_pts = occluder_source.reshape(-1, 3)
        hit_pts = hit_pts[np.isfinite(hit_pts).all(axis=1)]
        if len(hit_pts) > 0:
            _, _, hit_in_bounds = points_to_bev_indices(hit_pts, x_min, y_min, resolution, H, W)
            hit_pts = hit_pts[hit_in_bounds]

    if len(hit_pts) > 0:
        x = hit_pts[:, 0]
        y = hit_pts[:, 1]
        r = np.sqrt((x - sensor_x) ** 2 + (y - sensor_y) ** 2)
        a = np.arctan2(y - sensor_y, x - sensor_x)

        valid = (
            np.isfinite(r)
            & np.isfinite(a)
            & (r >= 0.05)
            & (a >= angle_min)
            & (a <= angle_max)
        )

        r = r[valid]
        a = a[valid]
        x = x[valid]
        y = y[valid]

        bin_idx = np.floor((a - angle_min) / angle_increment).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

        for idx, rr, xx, yy in zip(bin_idx, r, x, y):
            if rr < hit_ranges[idx]:
                hit_ranges[idx] = rr
                hit_points[idx] = [xx, yy]

    def boundary_endpoint(angle):
        dx = np.cos(angle)
        dy = np.sin(angle)
        candidates = []

        if abs(dx) > 1e-8:
            tx = (x_min - sensor_x) / dx
            y_at_tx = sensor_y + tx * dy
            if tx > 0.0 and y_min - 1e-6 <= y_at_tx <= y_max + 1e-6:
                candidates.append(tx)

            tx = (x_max - sensor_x) / dx
            y_at_tx = sensor_y + tx * dy
            if tx > 0.0 and y_min - 1e-6 <= y_at_tx <= y_max + 1e-6:
                candidates.append(tx)

        if abs(dy) > 1e-8:
            ty = (y_min - sensor_y) / dy
            x_at_ty = sensor_x + ty * dx
            if ty > 0.0 and x_min - 1e-6 <= x_at_ty <= x_max + 1e-6:
                candidates.append(ty)

            ty = (y_max - sensor_y) / dy
            x_at_ty = sensor_x + ty * dx
            if ty > 0.0 and x_min - 1e-6 <= x_at_ty <= x_max + 1e-6:
                candidates.append(ty)

        if not candidates:
            return sensor_x, sensor_y

        t = min(candidates)
        return sensor_x + t * dx, sensor_y + t * dy

    def clamp_grid_endpoint(x, y):
        row, col = world_to_grid(x, y, x_min, y_min, resolution)
        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        return row, col

    angles = angle_min + np.arange(num_bins, dtype=np.float32) * angle_increment
    beam_ranges = np.zeros((num_bins,), dtype=np.float32)
    beam_hits = np.isfinite(hit_ranges)
    for idx, angle in enumerate(angles):
        if beam_hits[idx]:
            end_x = float(hit_points[idx, 0])
            end_y = float(hit_points[idx, 1])
            beam_ranges[idx] = float(hit_ranges[idx])
            include_endpoint = False
        else:
            end_x, end_y = boundary_endpoint(float(angle))
            beam_ranges[idx] = float(np.hypot(end_x - sensor_x, end_y - sensor_y))
            include_endpoint = True
        end_r, end_c = clamp_grid_endpoint(end_x, end_y)
        line_cells = bresenham_line(sensor_r, sensor_c, end_r, end_c)
        ray_cells = line_cells if include_endpoint else line_cells[:-1]

        for rr, cc in ray_cells:
            if 0 <= rr < H and 0 <= cc < W:
                visible_free[rr, cc] = True

    if fill_between_beams and num_bins > 1:
        wedge_mask = np.zeros((H, W), dtype=np.uint8)
        sensor_poly = np.asarray([sensor_c, sensor_r], dtype=np.int32)
        for idx in range(num_bins - 1):
            fill_range = float(min(beam_ranges[idx], beam_ranges[idx + 1]))
            if not np.isfinite(fill_range) or fill_range <= 0.0:
                continue

            end0_x = sensor_x + fill_range * np.cos(float(angles[idx]))
            end0_y = sensor_y + fill_range * np.sin(float(angles[idx]))
            end1_x = sensor_x + fill_range * np.cos(float(angles[idx + 1]))
            end1_y = sensor_y + fill_range * np.sin(float(angles[idx + 1]))
            end0_r, end0_c = clamp_grid_endpoint(end0_x, end0_y)
            end1_r, end1_c = clamp_grid_endpoint(end1_x, end1_y)

            poly = np.asarray(
                [
                    sensor_poly,
                    [end0_c, end0_r],
                    [end1_c, end1_r],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(wedge_mask, poly, 1)

        visible_free |= wedge_mask.astype(bool)

    visible_free[occupied] = False
    known = occupied | visible_free
    return occupied, visible_free, known


def compute_esdf_from_occupancy(
    occupied_mask,
    known_mask,
    resolution=0.05,
    signed=True,
):
    """
    Compute ESDF/SDF on a 2D grid.

    Positive values are computed only from observed occupied cells, so
    unknown/FOV-boundary cells do not create artificial positive-distance
    cliffs inside visible free space. Unknown cells are still marked negative
    in the signed output, preserving the invariant that ESDF < 0 is invalid
    or colliding.

    Args:
        occupied_mask: bool array, True = occupied
        resolution: meters per grid cell
        signed: if False, returns unsigned distance-to-obstacle in free space

    Returns:
        esdf: float32 array
            if signed:
                positive in known free space
                negative in occupied or unknown cells
            else:
                zero in occupied or unknown cells, positive in known free space
    """
    occupied = occupied_mask.astype(bool)
    known = known_mask.astype(bool)
    unknown = ~known
    invalid = occupied | unknown
    known_free = known & ~occupied

    max_distance = float(np.hypot(occupied.shape[0], occupied.shape[1]) * resolution)
    if np.any(occupied):
        # distance to nearest observed occupied cell for every cell
        dist_out = distance_transform_edt(~occupied) * resolution
    else:
        # If no obstacle was observed in the FOV, known-free cells have no
        # obstacle-limited distance inside this map.
        dist_out = np.full(occupied.shape, max_distance, dtype=np.float32)

    dist_out = dist_out.astype(np.float32)
    dist_out[invalid] = 0.0

    if not signed:
        return dist_out.astype(np.float32)

    if np.any(known_free):
        # distance from invalid cells to nearest known-free cell
        dist_in = distance_transform_edt(invalid) * resolution
    else:
        dist_in = np.full(occupied.shape, max_distance, dtype=np.float32)

    esdf = dist_out.astype(np.float32)
    esdf[invalid] = -dist_in[invalid].astype(np.float32)
    return esdf


def inflate_obstacles_via_esdf(esdf, robot_radius):
    """
    Convert ESDF to clearance wrt robot footprint by subtracting radius.

    clearance > 0   safe
    clearance = 0   touching
    clearance < 0   collision
    """
    return esdf - float(robot_radius)


def build_unknown_mask(known_mask):
    """
    Unknown cells are those not observed as occupied or visible-free.
    """
    return ~known_mask.astype(bool)


def obstacle_and_unknown_cost(
    clearance_map,
    unknown_mask,
    obstacle_margin=0.2,
    unknown_cost=1.0,
):
    """
    Produce a scalar field that can be sampled along a trajectory.

    Obstacle part:
        penalize low clearance with a quadratic hinge
    Unknown part:
        additive constant penalty where visibility is unknown

    Returns:
        cost_map
    """
    obs_cost = np.maximum(0.0, obstacle_margin - clearance_map) ** 2
    unk_cost = unknown_cost * unknown_mask.astype(np.float32)
    return obs_cost + unk_cost


def pointcloud_to_esdf_pipeline(
    points_xyz,
    h_min,
    h_max,
    R=None,
    t=None,
    x_min=0.0,
    x_max=25.0,
    y_min=-10.0,
    y_max=10.0,
    resolution=0.05,
    sensor_xy=(0.0, 0.0),
    robot_radius=0.25,
    ground_alignment=True,
    ground_alignment_prior_normal=None,
    ground_alignment_smoothing=0.65,
    ground_candidate_min_forward=0.5,
    ground_candidate_z_min=-2.0,
    ground_candidate_z_max=2.0,
    ground_candidate_cell_size=0.75,
    ground_ransac_iterations=120,
    ground_ransac_distance_threshold=0.08,
    ground_max_tilt_deg=35.0,
    ground_max_correction_deg=8.0,
    visibility_angle_min=None,
    visibility_angle_max=None,
    visibility_angle_increment=None,
):
    """
    End-to-end pipeline:
      1) transform to robot frame
      2) filter obstacle points by height and XY ROI
      3) infer observed FOV from all ROI-valid points
      4) raytrace occupancy + visible free
      5) compute ESDF
      6) subtract robot radius to get clearance

    Returns:
        dict containing all intermediate maps
    """
    pts_robot_nominal = transform_points(points_xyz, R=R, t=t)
    sensor_position = np.zeros(3, dtype=np.float32) if t is None else np.asarray(t, dtype=np.float32)
    ground_info = {
        "enabled": bool(ground_alignment),
        "fit_success": False,
        "normal_source": "disabled" if not ground_alignment else "failed",
        "candidate_count": 0,
        "inlier_count": 0,
        "raw_ground_normal_xyz": None,
        "smoothed_ground_normal_xyz": None,
        "raw_tilt_deg": None,
        "applied_tilt_deg": 0.0,
        "inlier_rmse_m": None,
    }
    pts_robot = pts_robot_nominal
    if ground_alignment:
        correction, ground_info = estimate_ground_plane_correction(
            pts_robot_nominal,
            sensor_position=sensor_position,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            min_forward=ground_candidate_min_forward,
            candidate_z_min=ground_candidate_z_min,
            candidate_z_max=ground_candidate_z_max,
            candidate_cell_size=ground_candidate_cell_size,
            ransac_iterations=ground_ransac_iterations,
            ransac_distance_threshold=ground_ransac_distance_threshold,
            max_ground_tilt_deg=ground_max_tilt_deg,
            max_correction_deg=ground_max_correction_deg,
            prior_ground_normal=ground_alignment_prior_normal,
            smoothing=ground_alignment_smoothing,
        )
        pts_robot = transform_points(pts_robot_nominal - sensor_position, R=correction, t=sensor_position)

    pts_filt, valid_mask = filter_points_by_height_and_roi(
        pts_robot,
        h_min=h_min,
        h_max=h_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_axis=2,
    )
    pts_occluder = pts_robot.astype(np.float32, copy=True)
    pts_occluder.reshape(-1, 3)[~valid_mask] = np.nan
    occupied, visible_free, known = raytrace_visibility_from_points(
        pts_filt,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=resolution,
        sensor_xy=sensor_xy,
        fov_points_xyz=pts_robot,
        occluder_points_xyz=pts_occluder,
        angle_min=visibility_angle_min,
        angle_max=visibility_angle_max,
        angle_increment=visibility_angle_increment,
    )

    esdf = compute_esdf_from_occupancy(occupied, known, resolution=resolution, signed=True)
    clearance = inflate_obstacles_via_esdf(esdf, robot_radius=robot_radius)
    unknown = build_unknown_mask(known)

    return {
        "points_robot_nominal": pts_robot_nominal,
        "points_robot": pts_robot,
        "points_filtered": pts_filt,
        "valid_mask_flat": valid_mask,
        "occupied_mask": occupied,
        "visible_free_mask": visible_free,
        "known_mask": known,
        "unknown_mask": unknown,
        "esdf": esdf,
        "clearance": clearance,
        "ground_alignment": ground_info,
        "ground_alignment_correction_R": correction if ground_alignment else np.eye(3, dtype=np.float32),
        "ground_alignment_sensor_origin_xyz": sensor_position,
    }