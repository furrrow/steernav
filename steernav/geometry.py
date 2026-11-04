"""Pure path, pose, and coordinate-frame geometry for SteerNav."""

from __future__ import annotations

import numpy as np


def resample_path(path, num_waypoints: int = 8, eps: float = 1e-8) -> np.ndarray:
    """Resample a path to equally spaced XY waypoints along its arc length."""
    if not isinstance(num_waypoints, (int, np.integer)) or num_waypoints <= 0:
        raise ValueError("num_waypoints must be a positive integer")

    path_xy = np.asarray(path, dtype=np.float64)
    if path_xy.ndim == 1:
        if path_xy.shape[0] < 2:
            return np.zeros((num_waypoints, 2), dtype=np.float64)
        path_xy = path_xy[:2][None, :]
    elif path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError("path must have shape (N, 2) or (N, >=2)")
    else:
        path_xy = path_xy[:, :2]

    path_xy = path_xy[np.isfinite(path_xy).all(axis=1)]
    if len(path_xy) == 0:
        return np.zeros((num_waypoints, 2), dtype=np.float64)

    if len(path_xy) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(path_xy, axis=0), axis=1) > eps]
        path_xy = path_xy[keep]
    if len(path_xy) == 1:
        return np.repeat(path_xy, num_waypoints, axis=0)

    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative_distance = np.r_[0.0, np.cumsum(segment_lengths)]
    sample_distances = np.linspace(0.0, cumulative_distance[-1], num_waypoints)
    return np.column_stack([
        np.interp(sample_distances, cumulative_distance, path_xy[:, axis])
        for axis in range(2)
    ])


def pick_waypoint(
    opt_path_xy,
    target_distance: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Pick a waypoint at a fixed lookahead distance along a base-frame path."""
    path_xy = np.asarray(opt_path_xy, dtype=np.float64)
    if path_xy.size == 0:
        return np.zeros(2, dtype=np.float64)

    if path_xy.ndim == 1:
        if path_xy.shape[0] < 2:
            return np.zeros(2, dtype=np.float64)
        path_xy = path_xy[:2][None, :]
    elif path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise ValueError("opt_path_xy must have shape (N, 2) or (N, >=2)")
    else:
        path_xy = path_xy[:, :2]

    path_xy = path_xy[np.isfinite(path_xy).all(axis=1)]
    if len(path_xy) == 0:
        return np.zeros(2, dtype=np.float64)

    points = np.vstack((np.zeros(2, dtype=np.float64), path_xy))
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    remaining = float(target_distance)
    if remaining <= eps:
        return np.zeros(2, dtype=np.float64)

    last_reachable = points[0]
    for start, vector, length in zip(points[:-1], segment_vectors, segment_lengths):
        if length <= eps:
            continue
        last_reachable = start + vector
        if remaining <= length:
            return start + (remaining / length) * vector
        remaining -= length

    return last_reachable


def transform_robot_twist_to_camera(
    linear_velocity_base,
    angular_velocity_base,
    T_base_from_cam,
):
    """Return camera-origin linear and angular velocity in camera axes."""
    linear_velocity_base = np.asarray(linear_velocity_base, dtype=np.float64)
    angular_velocity_base = np.asarray(angular_velocity_base, dtype=np.float64)
    T_base_from_cam = np.asarray(T_base_from_cam, dtype=np.float64)
    if linear_velocity_base.shape != (3,) or angular_velocity_base.shape != (3,):
        raise ValueError("linear and angular velocity must each have shape (3,)")
    if T_base_from_cam.shape != (4, 4):
        raise ValueError("T_base_from_cam must have shape (4, 4)")

    rotation_cam_from_base = T_base_from_cam[:3, :3].T
    camera_offset_base = T_base_from_cam[:3, 3]
    camera_linear_velocity_base = (
        linear_velocity_base
        + np.cross(angular_velocity_base, camera_offset_base)
    )
    return (
        rotation_cam_from_base @ camera_linear_velocity_base,
        rotation_cam_from_base @ angular_velocity_base,
    )


def quaternion_to_rotation_matrix(quaternion_xyzw):
    """Convert an XYZW quaternion to a 3-by-3 rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be a finite, nonzero XYZW vector")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def interpolate_pose_samples(pose_samples, timestamp, max_endpoint_gap=0.1):
    """Interpolate a pose, allowing a nearby endpoint when one side is absent."""
    if not pose_samples:
        return None
    times = np.asarray([sample[0] for sample in pose_samples], dtype=np.float64)
    timestamp = float(timestamp)
    right = int(np.searchsorted(times, timestamp, side="left"))
    if right < len(times) and abs(times[right] - timestamp) <= 1e-9:
        position, quaternion = pose_samples[right][1:]
    else:
        if right == 0 or right == len(times):
            endpoint = 0 if right == 0 else len(times) - 1
            if abs(timestamp - times[endpoint]) > float(max_endpoint_gap):
                return None
            position, quaternion = pose_samples[endpoint][1:]
            q0 = np.asarray(quaternion, dtype=np.float64).copy()
            q1 = q0.copy()
            fraction = 0.0
        else:
            left = right - 1
            interval = times[right] - times[left]
            if interval <= 1e-9:
                return None
            fraction = (timestamp - times[left]) / interval
            position = (
                (1.0 - fraction) * np.asarray(pose_samples[left][1], dtype=np.float64)
                + fraction * np.asarray(pose_samples[right][1], dtype=np.float64)
            )
            q0 = np.asarray(pose_samples[left][2], dtype=np.float64).copy()
            q1 = np.asarray(pose_samples[right][2], dtype=np.float64).copy()
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            quaternion = q0 + fraction * (q1 - q0)
            quaternion /= np.linalg.norm(quaternion)
        else:
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            quaternion = (
                np.sin((1.0 - fraction) * angle) * q0
                + np.sin(fraction * angle) * q1
            ) / np.sin(angle)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    transform[:3, 3] = position
    return transform


def rotate_path_deformation_between_base_frames(
    deformation_xy,
    previous_world_from_base,
    current_world_from_base,
):
    """Express previous-frame XY displacement vectors in the current base frame."""
    deformation_xy = np.asarray(deformation_xy, dtype=np.float64)
    previous_world_from_base = np.asarray(previous_world_from_base, dtype=np.float64)
    current_world_from_base = np.asarray(current_world_from_base, dtype=np.float64)
    if deformation_xy.ndim != 2 or deformation_xy.shape[1] != 2:
        raise ValueError("deformation_xy must have shape (N, 2)")
    if previous_world_from_base.shape != (4, 4) or current_world_from_base.shape != (4, 4):
        raise ValueError("base poses must have shape (4, 4)")

    current_from_previous_rotation = (
        current_world_from_base[:3, :3].T
        @ previous_world_from_base[:3, :3]
    )
    deformation_xyz = np.column_stack(
        (deformation_xy, np.zeros(len(deformation_xy), dtype=np.float64))
    )
    return (deformation_xyz @ current_from_previous_rotation.T)[:, :2]
