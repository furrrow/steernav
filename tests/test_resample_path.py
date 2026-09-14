import ast
from pathlib import Path

import numpy as np


def load_resample_path_and_pick_waypoint():
    source_path = Path(__file__).resolve().parents[1] / "steernav" / "ros_inference.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resample_path_and_pick_waypoint"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["resample_path_and_pick_waypoint"]


resample_path_and_pick_waypoint = load_resample_path_and_pick_waypoint()


def test_interpolates_when_first_waypoint_is_too_far():
    waypoint = resample_path_and_pick_waypoint(np.array([[1.0, 0.0]]))

    np.testing.assert_allclose(waypoint, [0.3, 0.0])
    assert np.isclose(np.linalg.norm(waypoint), 0.3)


def test_accumulates_across_short_inconsistent_segments():
    path = np.array([
        [0.05, 0.0],
        [0.10, 0.0],
        [0.10, 1.0],
    ])

    waypoint = resample_path_and_pick_waypoint(path)

    np.testing.assert_allclose(waypoint, [0.10, 0.20])


def test_returns_last_point_when_path_is_shorter_than_lookahead():
    path = np.array([
        [0.05, 0.0],
        [0.10, 0.0],
    ])

    waypoint = resample_path_and_pick_waypoint(path)

    np.testing.assert_allclose(waypoint, [0.10, 0.0])


def test_skips_duplicate_points():
    path = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ])

    waypoint = resample_path_and_pick_waypoint(path)

    np.testing.assert_allclose(waypoint, [0.0, 0.3])


def test_accepts_extra_columns_but_returns_xy():
    path = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])

    waypoint = resample_path_and_pick_waypoint(path)

    np.testing.assert_allclose(waypoint, [0.3, 0.0])
    assert waypoint.shape == (2,)
