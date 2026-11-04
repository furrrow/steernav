import numpy as np
from types import SimpleNamespace

from custom_utils.pointcloud_utils import (
    _find_min_bin_hits,
    compensate_camera_motion,
    pointcloud_to_esdf_pipeline,
    raytrace_visibility_from_points,
    transform_point_between_camera_poses,
    update_points,
)


def test_esdf_pipeline_preserves_dynamic_projection_provenance():
    points = np.array(
        [
            [[1.0, 0.0, 0.75], [2.0, 0.0, 0.75]],
            [[1.0, 0.5, 0.75], [2.0, 0.5, 0.75]],
        ],
        dtype=np.float32,
    )
    dynamic_projection_mask = np.array(
        [[False, True], [False, False]], dtype=bool
    )

    result = pointcloud_to_esdf_pipeline(
        points,
        h_min=0.5,
        h_max=1.0,
        x_min=0.0,
        x_max=3.0,
        y_min=-1.0,
        y_max=1.0,
        resolution=0.25,
        robot_radius=0.25,
        dynamic_projection_mask=dynamic_projection_mask,
    )

    dynamic_occupied = result["dynamic_occupied_mask"]
    assert dynamic_occupied.sum() == 1
    assert np.all(result["occupied_mask"][dynamic_occupied])


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


def test_pose_delta_maps_stationary_point_into_later_camera_frame():
    world_from_early_camera = np.eye(4)
    world_from_late_camera = np.eye(4)
    world_from_late_camera[2, 3] = 1.0

    aligned = transform_point_between_camera_poses(
        np.array([0.0, 0.0, 5.0]),
        world_from_early_camera,
        world_from_late_camera,
    )

    np.testing.assert_allclose(aligned, [0.0, 0.0, 4.0])


def test_update_points_pose_delta_removes_camera_motion():
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
    world_from_early_camera = np.eye(4)
    world_from_late_camera = np.eye(4)
    world_from_late_camera[2, 3] = 1.0

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[10.0, 11.0],
        camera_poses_world=[world_from_early_camera, world_from_late_camera],
    )

    np.testing.assert_array_equal(updated, points)
    assert movement == []


def test_update_points_pose_delta_preserves_world_relative_motion():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 3.5
    detections = [
        SimpleNamespace(tracker_id=np.array([8]), data={8: np.array([0.0, 0.0, 5.0])}),
        SimpleNamespace(
            tracker_id=np.array([8]),
            data={8: np.array([0.0, 0.0, 3.5])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
    ]
    world_from_early_camera = np.eye(4)
    world_from_late_camera = np.eye(4)
    world_from_late_camera[2, 3] = 1.0

    _, movement = update_points(
        points,
        detections,
        detection_timestamps=[10.0, 11.0],
        camera_poses_world=[world_from_early_camera, world_from_late_camera],
        time_look_ahead=1.0,
        min_dynamic_observations=2,
        sweep_steps=2,
    )

    assert len(movement) == 1
    np.testing.assert_allclose(movement[0][1] - movement[0][0], [0.0, 0.0, -0.5])


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
        min_dynamic_observations=2,
        sweep_steps=2,
    )

    np.testing.assert_allclose(np.unique(updated[..., 2]), [2.0, 4.0])


def test_update_points_smooths_velocity_after_initial_detection():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
    # Constant -0.5 m/s depth motion with a noisy final observation. A two-point
    # estimate would report -2.5 m/s for the final interval.
    depths = [5.0, 4.95, 4.90, 4.85, 4.60]
    detections = [
        SimpleNamespace(
            tracker_id=np.array([12]),
            data={12: np.array([0.0, 0.0, depth])},
            xyxy=np.array([[0, 0, 2, 2]]),
        )
        for depth in depths
    ]
    identity_poses = [np.eye(4) for _ in timestamps]

    _, movement = update_points(
        points,
        detections,
        detection_timestamps=timestamps,
        camera_poses_world=identity_poses,
        last_n_records=2,
        velocity_window=5,
        time_look_ahead=1.0,
        sweep_steps=2,
    )

    assert len(movement) == 1
    # The median pairwise slope recovers -0.5 m/s despite the noisy last pair.
    np.testing.assert_allclose(movement[0][1] - movement[0][0], [0.0, 0.0, -0.5])


def test_update_points_does_not_override_smoothed_direction_for_one_noisy_frame():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    timestamps = [0.0, 1.0, 2.0, 3.0]
    depths = [6.0, 5.0, 4.0, 4.6]
    detections = [
        SimpleNamespace(
            tracker_id=np.array([13]),
            data={13: np.array([0.0, 0.0, depth])},
            xyxy=np.array([[0, 0, 2, 2]]),
        )
        for depth in depths
    ]

    _, movement = update_points(
        points,
        detections,
        detection_timestamps=timestamps,
        camera_poses_world=[np.eye(4) for _ in timestamps],
        velocity_window=5,
        time_look_ahead=1.0,
        sweep_steps=2,
    )

    assert len(movement) == 1
    # The robust history still indicates approach; the final noisy sample must
    # not replace it with the latest two-point +0.6 m/s estimate.
    assert movement[0][1][2] - movement[0][0][2] < 0.0


def test_update_points_ignores_too_short_velocity_baseline():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(
            tracker_id=np.array([15]),
            data={15: np.array([0.0, 0.0, depth])},
            xyxy=np.array([[0, 0, 2, 2]]),
        )
        for depth in [5.0, 4.9, 4.95]
    ]

    _, movement = update_points(
        points,
        detections,
        detection_timestamps=[0.0, 0.1, 0.116666667],
        camera_poses_world=[np.eye(4) for _ in detections],
        min_dynamic_observations=2,
        velocity_window=3,
        min_velocity_interval=0.075,
        time_look_ahead=1.0,
        sweep_steps=2,
    )

    assert len(movement) == 1
    # The 16.7 ms final pair alone says +3 m/s. Longer baselines retain the
    # underlying modest approaching motion.
    assert movement[0][1][2] - movement[0][0][2] < 0.0


def test_update_points_drops_history_across_long_tracking_gap():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    present = lambda depth: SimpleNamespace(
        tracker_id=np.array([14]),
        data={14: np.array([0.0, 0.0, depth])},
        xyxy=np.array([[0, 0, 2, 2]]),
    )
    missing = SimpleNamespace(tracker_id=np.array([0]), data={})
    detections = [present(6.0), present(5.0), missing, missing, present(4.0)]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
        camera_poses_world=[np.eye(4) for _ in detections],
        min_dynamic_observations=2,
    )

    np.testing.assert_array_equal(updated, points)
    assert movement == []


def test_update_points_detects_lateral_motion_and_preserves_swept_path():
    points = np.zeros((2, 3, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(
            tracker_id=np.array([4]),
            data={4: np.array([0.0, 0.0, 4.0])},
            xyxy=np.array([[0, 0, 3, 2]]),
        ),
        SimpleNamespace(
            tracker_id=np.array([4]),
            data={4: np.array([0.5, 0.0, 4.0])},
            xyxy=np.array([[0, 0, 3, 2]]),
        ),
    ]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[1.0, 2.0],
        time_look_ahead=1.0,
        min_dynamic_speed=0.15,
        min_dynamic_observations=2,
        sweep_steps=3,
    )

    np.testing.assert_allclose(np.unique(updated[..., 0]), [0.0, 0.25, 0.5])
    assert len(movement) == 1
    np.testing.assert_allclose(movement[0][1] - movement[0][0], [0.5, 0.0, 0.0])


def test_update_points_ignores_motion_below_dynamic_speed():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(tracker_id=np.array([3]), data={3: np.array([0.0, 0.0, 4.0])}),
        SimpleNamespace(
            tracker_id=np.array([3]),
            data={3: np.array([0.01, 0.0, 4.0])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
    ]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[1.0, 2.0],
        min_dynamic_speed=0.15,
        min_dynamic_observations=2,
    )

    np.testing.assert_array_equal(updated, points)
    assert movement == []


def test_update_points_retains_track_through_one_missed_detection():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(tracker_id=np.array([9]), data={9: np.array([0.0, 0.0, 5.0])}),
        SimpleNamespace(
            tracker_id=np.array([9]),
            data={9: np.array([0.0, 0.0, 4.75])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
        SimpleNamespace(
            tracker_id=np.array([9]),
            data={9: np.array([0.0, 0.0, 4.5])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
        SimpleNamespace(tracker_id=np.array([0]), data={}),
    ]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[1.0, 1.5, 2.0, 2.5],
        time_look_ahead=1.0,
        sweep_steps=2,
    )

    assert len(movement) == 1
    np.testing.assert_allclose(movement[0][1][2], 3.25)
    np.testing.assert_allclose(np.unique(updated[..., 2]), [3.25, 4.0])


def test_update_points_does_not_project_provisional_two_observation_track():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(tracker_id=np.array([5]), data={5: np.array([0.0, 0.0, 5.0])}),
        SimpleNamespace(
            tracker_id=np.array([5]),
            data={5: np.array([0.0, 0.0, 4.0])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
    ]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[1.0, 1.2],
        time_look_ahead=1.0,
    )

    np.testing.assert_array_equal(updated, points)
    assert movement == []


def test_update_points_rejects_implausibly_fast_tracking_jump():
    points = np.zeros((2, 2, 3), dtype=np.float64)
    points[..., 2] = 4.0
    detections = [
        SimpleNamespace(tracker_id=np.array([6]), data={6: np.array([0.0, 0.0, 5.0])}),
        SimpleNamespace(tracker_id=np.array([6]), data={6: np.array([0.0, 0.0, 4.9])}),
        SimpleNamespace(
            tracker_id=np.array([6]),
            data={6: np.array([0.0, 0.0, 2.0])},
            xyxy=np.array([[0, 0, 2, 2]]),
        ),
    ]

    updated, movement = update_points(
        points,
        detections,
        detection_timestamps=[1.0, 1.1, 1.2],
        max_dynamic_speed=3.0,
    )

    np.testing.assert_array_equal(updated, points)
    assert movement == []
