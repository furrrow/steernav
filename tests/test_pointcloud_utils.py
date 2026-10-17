import numpy as np

from custom_utils.pointcloud_utils import (
    _find_min_bin_hits,
    raytrace_visibility_from_points,
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
