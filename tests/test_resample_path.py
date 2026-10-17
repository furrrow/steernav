import ast
from pathlib import Path

import numpy as np


def load_path_function(name):
    source_path = Path(__file__).resolve().parents[1] / "steernav" / "ros_inference.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


resample_path = load_path_function("resample_path")
pick_waypoint = load_path_function("pick_waypoint")


def test_interpolates_when_first_waypoint_is_too_far():
    waypoint = pick_waypoint(
        np.array([[2.0, 0.0]]),
        target_distance=1.0,
    )

    np.testing.assert_allclose(waypoint, [1.0, 0.0])
    assert np.isclose(np.linalg.norm(waypoint), 1.0)


def test_accumulates_across_short_inconsistent_segments():
    path = np.array([
        [0.05, 0.0],
        [0.10, 0.0],
        [0.10, 1.0],
    ])

    waypoint = pick_waypoint(path, target_distance=0.3)

    np.testing.assert_allclose(waypoint, [0.10, 0.20])


def test_returns_last_point_when_path_is_shorter_than_lookahead():
    path = np.array([
        [0.05, 0.0],
        [0.10, 0.0],
    ])

    waypoint = pick_waypoint(path, target_distance=0.3)

    np.testing.assert_allclose(waypoint, [0.10, 0.0])


def test_skips_duplicate_points():
    path = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])

    waypoint = pick_waypoint(path, target_distance=0.3)

    np.testing.assert_allclose(waypoint, [0.0, 0.3])


def test_accepts_extra_columns_but_returns_xy():
    path = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])

    waypoint = pick_waypoint(path, target_distance=0.3)

    np.testing.assert_allclose(waypoint, [0.3, 0.0])
    assert waypoint.shape == (2,)


def test_resamples_vla_path_to_eight_arc_length_waypoints():
    path = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.2, 0.0, 1.0, 0.0],
        [1.4, 0.0, 1.0, 0.0],
    ])

    result = resample_path(path)

    assert result.shape == (8, 2)
    np.testing.assert_allclose(result[:, 0], np.linspace(0.0, 1.4, 8))
    np.testing.assert_allclose(result[:, 1], 0.0)


def test_resampling_is_independent_of_input_sampling_density():
    sparse = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    dense = np.array([
        [0.0, 0.0],
        [0.25, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ])

    np.testing.assert_allclose(resample_path(sparse), resample_path(dense))


def test_resampling_repeats_a_single_valid_waypoint():
    result = resample_path(np.array([[0.4, -0.2]]), num_waypoints=8)

    assert result.shape == (8, 2)
    np.testing.assert_allclose(result, np.repeat([[0.4, -0.2]], 8, axis=0))
