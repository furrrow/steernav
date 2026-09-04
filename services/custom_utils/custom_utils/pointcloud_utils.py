"""
https://github.com/gershom96/VLA_DataGeneration/blob/main/utils/pointcloud_utils.py
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from scipy.ndimage import distance_transform_edt
from dataclasses import dataclass
import time
import numba

@dataclass
class BEVGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    height: int
    width: int

    @classmethod
    def create(cls, x_min, x_max, y_min, y_max, resolution,):
        # formerly def bev_grid_params(x_min, x_max, y_min, y_max, resolution)
        """
        Compute BEV grid dimensions.
        """
        width = int(np.ceil((x_max - x_min) / resolution))
        height = int(np.ceil((y_max - y_min) / resolution))
        return cls(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=resolution,
            height=height,
            width=width,
        )

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
    pts = np.ascontiguousarray(points_xyz, dtype=np.float32).reshape(-1, 3)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, z_axis]

    roi = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    height_ok = (z >= h_min) & (z <= h_max)

    valid_mask = roi & height_ok
    return pts[valid_mask], valid_mask

def points_to_bev_indices(points_xyz, grid: BEVGrid):
    """
    Convert robot-frame XY points into BEV grid indices.

    Returns:
        rows, cols, in_bounds
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]

    cols = np.floor((x - grid.x_min) / grid.resolution).astype(np.int32)
    rows = np.floor((y - grid.y_min) / grid.resolution).astype(np.int32)

    in_bounds = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
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


def rasterize_lines_vectorized(r0, r1, c0, c1, occluder_grid):
    """
    Vectorized batch line rasterization for multiple line segments.
    r0, r1, c0, c1 are 1D arrays of endpoints.
    """
    dr = np.abs(r1 - r0)
    dc = np.abs(c1 - c0)
    num_steps = np.maximum(dr, dc)
    max_len = np.max(num_steps)

    if max_len == 0:
        return

    # Normalize step progress from 0.0 to 1.0
    t = np.linspace(0, 1, num=max_len + 1, dtype=np.float32)  # Shape: (S,)

    # Interpolate for all line segments simultaneously via broadcasting
    # Shape: (N_lines, S)
    r_interp = np.round(r0[:, None] + (r1 - r0)[:, None] * t).astype(np.int32)
    c_interp = np.round(c0[:, None] + (c1 - c0)[:, None] * t).astype(np.int32)

    # Valid step mask per line length
    step_indices = np.arange(max_len + 1)
    valid_mask = step_indices <= num_steps[:, None]

    # Boundary filtering & grid setting
    H, W = occluder_grid.shape
    valid_mask &= (r_interp >= 0) & (r_interp < H) & (c_interp >= 0) & (c_interp < W)

    occluder_grid[r_interp[valid_mask], c_interp[valid_mask]] = True

@numba.njit(fastmath=True)
def draw_lines_numba(r0_arr, r1_arr, c0_arr, c1_arr, occluder):
    # faster version of bresenham_line
    H, W = occluder.shape
    for i in numba.prange(len(r0_arr)):
        r0, r1 = r0_arr[i], r1_arr[i]
        c0, c1 = c0_arr[i], c1_arr[i]

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr

        r, c = r0, c0
        while True:
            if 0 <= r < H and 0 <= c < W:
                occluder[r, c] = True
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

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
        grid: BEVGrid,
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
    t0 = time.perf_counter()
    pts = np.asarray(points_xyz, dtype=np.float32)
    H, W = grid.height, grid.width

    if pts.ndim != 3 or pts.shape[-1] != 3:
        flat_pts = pts.reshape(-1, 3)
        rows, cols, in_bounds = points_to_bev_indices(flat_pts, grid)
        occluder = np.zeros((H, W), dtype=bool)
        occluder[rows[in_bounds], cols[in_bounds]] = True
        return occluder

    # t1 = time.perf_counter()
    # print(f"initialize: {(t1 - t0) * 1000:.2f} ms")

    finite = np.isfinite(pts).all(axis=-1)
    x_pts, y_pts = pts[..., 0], pts[..., 1]

    roi = (
            finite
            & (x_pts >= grid.x_min) & (x_pts <= grid.x_max)
            & (y_pts >= grid.y_min) & (y_pts <= grid.y_max)
    )
    occluder = np.zeros((H, W), dtype=np.uint8)
    if not np.any(roi):
        return occluder

    # Precompute reciprocal resolution for speed
    inv_res = 1.0 / grid.resolution
    rows = np.zeros(pts.shape[:2], dtype=np.int32)
    cols = np.zeros(pts.shape[:2], dtype=np.int32)

    # t2 = time.perf_counter()
    # print(f"finite/roi: {(t2 - t1) * 1000:.2f} ms")

    rows[roi] = np.clip(np.floor((y_pts[roi] - grid.y_min) * inv_res), 0, H - 1).astype(np.int32)
    cols[roi] = np.clip(np.floor((x_pts[roi] - grid.x_min) * inv_res), 0, W - 1).astype(np.int32)
    occluder[rows[roi], cols[roi]] = True

    # shortcut portion: ======================================================
    # Raw projected surface

    occluder[rows[roi], cols[roi],] = 1
    # Bridge small gaps
    kernel_size = int(max_neighbor_distance_m // grid.resolution) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    occluder = cv2.morphologyEx(occluder, cv2.MORPH_CLOSE, kernel,)
    # t4 = time.perf_counter()
    # print(f"rasterize: new occluder using cv2.morphologyEx: {(t4 - t2) * 1000:.2f} ms")
    return occluder.astype(bool)
    # shortcut portion: ======================================================

    # Check horizontal and vertical neighbor pairs
    neighbor_slices = (
        (slice(None), slice(None, -1), slice(None), slice(1, None)),  # Horizontal
        (slice(None, -1), slice(None), slice(1, None), slice(None)),  # Vertical
    )

    t3 = time.perf_counter()
    # print(f"projection, neighbor slices: {(t3 - t2) * 1000:.2f} ms")

    for r_a, c_a, r_b, c_b in neighbor_slices:
        roi_a, roi_b = roi[r_a, c_a], roi[r_b, c_b]
        pts_a, pts_b = pts[r_a, c_a], pts[r_b, c_b]
        rows_a, rows_b = rows[r_a, c_a], rows[r_b, c_b]
        cols_a, cols_b = cols[r_a, c_a], cols[r_b, c_b]

        continuous = roi_a & roi_b & _continuous_neighbor_mask(pts_a, pts_b, max_neighbor_distance_m)
        needs_line = continuous & ((np.abs(rows_a - rows_b) > 1) | (np.abs(cols_a - cols_b) > 1))

        if np.any(needs_line):
            draw_lines_numba(
                rows_a[needs_line],
                rows_b[needs_line],
                cols_a[needs_line],
                cols_b[needs_line],
                occluder
            )
    t4 = time.perf_counter()
    print(f"rasterize_find neighbors: {(t4 - t3) * 1000:.2f} ms")
    return occluder

def get_inbound_points(points_xyz, grid:BEVGrid):
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)

    if len(points) == 0:
        return points

    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        return points

    _, _, in_bounds = points_to_bev_indices(points, grid)

    return points[in_bounds]

@numba.njit
def _find_min_bin_hits(bin_idx, ranges, x_pts, y_pts, hit_ranges, hit_points):
    for i in numba.prange(len(bin_idx)):
        b = bin_idx[i]
        r = ranges[i]
        if r < hit_ranges[b]:
            hit_ranges[b] = r
            hit_points[b, 0] = x_pts[i]
            hit_points[b, 1] = y_pts[i]

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
    bridge_occluder_neighbors=False,
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
    # 1. Grid
    grid = BEVGrid.create(x_min, x_max, y_min, y_max, resolution,)
    H, W = grid.height, grid.width
    # 2. check for sources, prepare sources occupancy, pov, and occluder points
    occupancy_points = get_inbound_points(points_xyz, grid)
    fov_source = occupancy_points if fov_points_xyz is None else np.asarray(fov_points_xyz, dtype=np.float32)
    occluder_source = fov_source if occluder_points_xyz is None else np.asarray(occluder_points_xyz, dtype=np.float32)

    # 3. handle fov_points based on fov_points_xyz, bounds
    occupied = np.zeros((H, W), dtype=bool)
    visible_free = np.zeros((H, W), dtype=bool)
    if fov_points_xyz is None:
        fov_points = occupancy_points
    else:
        fov_points = fov_source.reshape(-1, 3)
        fov_points = fov_points[np.isfinite(fov_points).all(axis=1)]
        if len(fov_points) > 0:
            _, _, fov_in_bounds = points_to_bev_indices(fov_points, grid)
            fov_points = fov_points[fov_in_bounds]

    if len(fov_points) == 0:
        if len(occupancy_points) == 0:
            return occupied, visible_free, occupied | visible_free
        print("no fov_points left, defaulting back to occupancy_points")
        fov_points = occupancy_points

    # sensor x y r c
    sensor_x, sensor_y = float(sensor_xy[0]), float(sensor_xy[1])
    sensor_r, sensor_c = world_to_grid(sensor_x, sensor_y, x_min, y_min, resolution)
    sensor_r = int(np.clip(sensor_r, 0, H - 1))
    sensor_c = int(np.clip(sensor_c, 0, W - 1))

    # 4. Handle angle ranges and increments
    observed_angles = np.arctan2(fov_points[:, 1] - sensor_y, fov_points[:, 0] - sensor_x)

    angle_min = float(np.min(observed_angles)) if angle_min is None else float(angle_min)
    angle_max = float(np.max(observed_angles)) if angle_max is None else float(angle_max)

    if angle_max < angle_min:
        angle_min, angle_max = angle_max, angle_min

    if angle_increment is None:
        corners = np.asarray([[x_min, y_min],
                              [x_min, y_max],
                              [x_max, y_min],
                              [x_max, y_max],],dtype=np.float32,)
        max_range = float(np.max(np.linalg.norm(corners - np.asarray([sensor_x, sensor_y], dtype=np.float32), axis=1)))
        angle_increment = float(np.arctan2(resolution, max(max_range, resolution)))
    else:
        angle_increment = float(angle_increment)

    angle_increment = max(angle_increment, 1e-4)

    num_bins = int(np.floor((angle_max - angle_min) / angle_increment)) + 1
    hit_ranges = np.full((num_bins,), np.inf, dtype=np.float32)
    hit_points = np.full((num_bins, 2), np.nan, dtype=np.float32)

    # 5. bridge_occluder_neighbors buggy, results in false rays of  "free", default False.
    if bridge_occluder_neighbors and occluder_source.ndim == 3:
        occluder_mask = rasterize_organized_occluders(
            occluder_source,
            grid=grid,
            max_neighbor_distance_m=max_occluder_neighbor_distance_m * 3,
        )
        occluder_rows, occluder_cols = np.nonzero(occluder_mask)
        hit_pts = _grid_centers_from_indices(occluder_rows, occluder_cols, x_min, y_min, resolution)
    else:
        # Fallback: create occluder_mask from points in grid
        occluder_mask = np.zeros((H, W), dtype=bool)
        hit_pts = occluder_source.reshape(-1, 3)
        hit_pts = hit_pts[np.isfinite(hit_pts).all(axis=1)]
        if len(hit_pts) > 0:
            hit_r, hit_c, hit_in_bounds = points_to_bev_indices(hit_pts, grid)
            hit_pts = hit_pts[hit_in_bounds]
            occluder_mask[hit_r[hit_in_bounds], hit_c[hit_in_bounds]] = True
    if len(hit_pts) > 0:
        x, y = hit_pts[:, 0], hit_pts[:, 1]
        dx, dy = x - sensor_x, y - sensor_y
        r = np.hypot(dx, dy)
        a = np.arctan2(dy, dx)

        valid = (np.isfinite(r)) & (r >= 0.05) & (a >= angle_min) & (a <= angle_max)
        r, a, x, y = r[valid], a[valid], x[valid], y[valid]
        if len(r) > 0:
            bin_idx = np.clip(((a - angle_min) / angle_increment).astype(np.int32), 0, num_bins - 1)
            _find_min_bin_hits(bin_idx, r, x, y, hit_ranges, hit_points)

    angles = angle_min + np.arange(num_bins, dtype=np.float32) * angle_increment
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # t2 = time.perf_counter()
    # print(f"raytrace: _find_min_bin_hits: {(t2 - t1) * 1000:.2f} ms")

    # Vectorized boundary intersections
    with np.errstate(divide='ignore', invalid='ignore'):
        tx1 = np.where(np.abs(cos_a) > 1e-8, (x_min - sensor_x) / cos_a, np.nan)
        tx2 = np.where(np.abs(cos_a) > 1e-8, (x_max - sensor_x) / cos_a, np.nan)
        ty1 = np.where(np.abs(sin_a) > 1e-8, (y_min - sensor_y) / sin_a, np.nan)
        ty2 = np.where(np.abs(sin_a) > 1e-8, (y_max - sensor_y) / sin_a, np.nan)

    def valid_t(t, val, bound_min, bound_max):
        return (t > 0.0) & (val >= bound_min - 1e-6) & (val <= bound_max + 1e-6)

    t_candidates = np.stack([
        np.where(valid_t(tx1, sensor_y + tx1 * sin_a, y_min, y_max), tx1, np.inf),
        np.where(valid_t(tx2, sensor_y + tx2 * sin_a, y_min, y_max), tx2, np.inf),
        np.where(valid_t(ty1, sensor_x + ty1 * cos_a, x_min, x_max), ty1, np.inf),
        np.where(valid_t(ty2, sensor_x + ty2 * cos_a, x_min, x_max), ty2, np.inf),
    ], axis=1)

    t_boundary = np.min(t_candidates, axis=1)
    boundary_x = sensor_x + t_boundary * cos_a
    boundary_y = sensor_y + t_boundary * sin_a

    has_hit = np.isfinite(hit_ranges)
    final_x = np.where(has_hit, hit_points[:, 0], boundary_x)
    final_y = np.where(has_hit, hit_points[:, 1], boundary_y)
    final_ranges = np.where(has_hit, hit_ranges, np.hypot(final_x - sensor_x, final_y - sensor_y))

    # t3 = time.perf_counter()
    # print(f"raytrace: boundary intersections: {(t3 - t2) * 1000:.2f} ms")

    # Single-pass OpenCV batch rasterization
    wedge_mask = np.zeros((H, W), dtype=np.uint8)

    if fill_between_beams and num_bins > 1:
        fill_ranges = np.minimum(final_ranges[:-1], final_ranges[1:])
        valid_wedges = np.isfinite(fill_ranges) & (fill_ranges > 0.0)

        e0_x = sensor_x + fill_ranges[valid_wedges] * cos_a[:-1][valid_wedges]
        e0_y = sensor_y + fill_ranges[valid_wedges] * sin_a[:-1][valid_wedges]
        e1_x = sensor_x + fill_ranges[valid_wedges] * cos_a[1:][valid_wedges]
        e1_y = sensor_y + fill_ranges[valid_wedges] * sin_a[1:][valid_wedges]

        inv_res = 1.0 / resolution
        # FIX: Ensure column (X) maps to W-1 and row (Y) maps to H-1
        e0_c = np.clip(((e0_x - x_min) * inv_res).astype(np.int32), 0, W - 1)
        e0_r = np.clip(((e0_y - y_min) * inv_res).astype(np.int32), 0, H - 1)
        e1_c = np.clip(((e1_x - x_min) * inv_res).astype(np.int32), 0, W - 1)
        e1_r = np.clip(((e1_y - y_min) * inv_res).astype(np.int32), 0, H - 1)

        num_valid = len(e0_c)
        if num_valid > 0:
            # Interleave endpoints into one single perimeter contour
            contour = np.empty((2 * num_valid + 1, 2), dtype=np.int32)
            contour[0] = [sensor_c, sensor_r]
            contour[1::2, 0] = e0_c
            contour[1::2, 1] = e0_r
            contour[2::2, 0] = e1_c
            contour[2::2, 1] = e1_r

            cv2.fillPoly(wedge_mask, [contour], 1)

    visible_free = wedge_mask.astype(bool)
    visible_free[occluder_mask] = False
    known = occluder_mask | visible_free
    # t4 = time.perf_counter()
    # print(f"raytrace: fill_between_beams: {(t4 - t3) * 1000:.2f} ms")
    return occluder_mask, visible_free, known

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
    ground_alignment=False,
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
    # t0 = time.perf_counter()
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
    # t1 = time.perf_counter()
    # print(f"1. transform_points: {(t1 - t0) * 1000:.2f} ms")

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
    # t2 = time.perf_counter()
    # print(f"2. ground_alignment: {(t2 - t1) * 1000:.2f} ms")

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
    # t3 = time.perf_counter()
    # print(f"3. filter_points_by_height_and_roi: {(t3 - t2) * 1000:.2f} ms")

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
    # t4 = time.perf_counter()
    # print(f"4. raytrace_visibility_from_points: {(t4 - t3) * 1000:.2f} ms")

    esdf = compute_esdf_from_occupancy(occupied, known, resolution=resolution, signed=True)
    # t5 = time.perf_counter()
    # print(f"5. compute_esdf_from_occupancy: {(t5 - t4) * 1000:.2f} ms")

    clearance = inflate_obstacles_via_esdf(esdf, robot_radius=robot_radius)
    unknown = build_unknown_mask(known)
    # t6 = time.perf_counter()
    # print(f"6. inflate obstacle, build unknown mask: {(t6 - t5) * 1000:.2f} ms")

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


def merge_points(points_input:np.ndarray, projected_patch: np.ndarray,
                 x1:int, y1:int, x2:int, y2:int):
    """
    merge the original point cloud with forward projection of moving obstacles
    retain whichever point is closer to the camera.

    :param points_input:
    :param projected_patch:
    :param x1, y1, x2, y2: bounding box coordinates from moge
    :return:
    """
    merged = points_input.copy()
    original_patch = merged[y1:y2, x1:x2] # (y, x, 3)
    result = original_patch.copy()

    original_z = original_patch[..., 2] # (y, x)
    projected_z = projected_patch[..., 2] # (y, x)

    use_projected = projected_z < original_z
    result[use_projected] = projected_patch[use_projected]

    merged[y1:y2, x1:x2] = result

    return merged


def update_points(points_input, detection_queue: list[sv.Detections],
                  min_record_num=6, robot_velocity_camera=np.array([0, 0, 0.1]), time_incr=0.1, time_look_ahead=1.0):
    """
    inflate detected objects' pointcloudds in the direction of their travel
    :param points_input:
    :param detection_queue:
    :param robot_velocity_camera: in the CAMERA FRAME: x-right, y-down, z-forward
    :param time_incr: how much time passes between frames. 19fps ~ 0.05
    :param time_look_ahead:
    :return:
    """
    last_detection = detection_queue[-1]
    if len(detection_queue) < min_record_num:
        return points_input
    if len(last_detection.tracker_id) == 0:
        return points_input

    for id in last_detection.tracker_id:
        median_depth_list = []
        # extract depth of the bounding boxes
        for i in range(len(detection_queue)):
            # print(i, detection_queue[i])
            pos_dict = detection_queue[i].data
            if id in pos_dict:
                median_depth_list.append(pos_dict[id])
        # in case not enough detection on a specific id:
        if len(median_depth_list) < min_record_num:
            continue
        # calculate position change over time
        median_depth_list = np.array(median_depth_list)
        changes = np.diff(median_depth_list, axis=0)
        median_position_shift =np.median(changes, axis=0)

        # extract points from bounding box, project forward to future position
        last_idx = np.argwhere(last_detection.tracker_id == id)[0][0]
        x1, y1, x2, y2 = last_detection.xyxy[last_idx].astype(int)
        box_3d_pts = points_input[y1:y2, x1:x2]
        observed_velocity = median_position_shift / time_incr
        relative_velocity = observed_velocity + robot_velocity_camera
        print(f"\n ----------------- > DEBUG: relative_velocity: {relative_velocity[2]:.2f}\n")
        if True in np.isnan(relative_velocity):
            print("nan velocity for position shift:", median_position_shift)
            continue
        if relative_velocity[2] > 0: # Object is moving away from camera
            continue
        if abs(relative_velocity[2]) < 0.01: # ignore slow moving objects from noise
            continue
        future_shift = relative_velocity * time_look_ahead
        box_project_forward = box_3d_pts + future_shift

        print(f"moving 3d loc by: {np.median(box_3d_pts, axis=(0, 1)) - np.median(box_project_forward, axis=(0, 1))}")
        points_input = merge_points(points_input, box_project_forward, x1, y1, x2, y2,)
    return points_input
