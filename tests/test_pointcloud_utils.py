import numpy as np
from types import SimpleNamespace

from custom_utils.pointcloud_utils import (
    _find_min_bin_hits,
    compensate_camera_motion,
    raytrace_visibility_from_points,
    update_points,
)


def test_find_min_bin_hits_keeps_nearest_point_per_bin():
    bin_idx = np.array([1, 0, 1, 2, 0], dtype=np.int32)
    ranges = np.array([3.0, 2.0, 1.0, 4.0, 0.5], dtype=np.float32)
    x_pts = np.array([30.0, 20.0, 10.0, 40.0, 5.0], dtype=np.float32)
    y_pts = -x_pts
    hit_ranges = np.full(4, np.inf, dtype=np.float32)
    hit_points = np.full((4, 2), np.nan, dtype=np.float32)

    _find_min_bin_hits(bin_idx, ranges, x_pts, y_pts, hit_ranges, hit_points)

    np.testing.assert_allclose(hit_ranges[:3], [0.5, 1.0, 4.0])
    np.testing.assert_allclose(hit_points[:3], [[5.0, -5.0], [10.0, -10.0], [40.0, -40.0]])
    assert np.isinf(hit_ranges[3])
    assert np.isnan(hit_points[3]).all()


def test_raytrace_visibility_uses_nearest_angular_hit():
    points = np.array(
        [
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    occupied, visible_free, known = raytrace_visibility_from_points(
        points,
        x_min=0.0,
        x_max=3.0,
        y_min=-1.0,
        y_max=2.0,
        resolution=0.25,
        angle_increment=0.1,
    )

    assert occupied.shape == visible_free.shape == known.shape == (12, 12)
    assert occupied.any()
    assert visible_free.any()
    np.testing.assert_array_equal(known, occupied | visible_free)


def test_camera_translation_is_removed_from_relative_velocity():
    camera_velocity = np.array([0.0, 0.0, 1.2])

    estimated = compensate_camera_motion(
        observed_relative_velocity=-camera_velocity,
        object_position_camera=np.array([2.0, 0.0, 5.0]),
        robot_velocity_camera=camera_velocity,
        robot_angular_velocity_camera=np.zeros(3),
    )

    np.testing.assert_allclose(estimated, np.zeros(3), atol=1e-12)


def test_camera_rotation_is_removed_from_relative_velocity():
    omega_camera = np.array([0.0, -0.4, 0.0])
    position_camera = np.array([2.0, 0.0, 5.0])
    apparent_velocity = -np.cross(omega_camera, position_camera)

    estimated = compensate_camera_motion(
        observed_relative_velocity=apparent_velocity,
        object_position_camera=position_camera,
        robot_velocity_camera=np.zeros(3),
        robot_angular_velocity_camera=omega_camera,
    )

    np.testing.assert_allclose(estimated, np.zeros(3), atol=1e-12)


def test_update_points_uses_observation_timestamps():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(tracker_id=np.array([7]), data={7: np.array([0.0, 0.0, 5.0])}),
        SimpleNamespace(
            tracker_id=np.array([7]),
            data={7: np.array([0.0, 0.0, 4.0])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
    ]

    updated, _ = update_points(
        points,
        detections,
        last_n_records=2,
        detection_timestamps=[10.0, 10.5],
        time_incr=0.01,
        time_look_ahead=1.0,
    )

    np.testing.assert_allclose(updated[..., 2], 2.0)
