import ast
from pathlib import Path

import numpy as np
import pytest


def load_path_function(name):
    source_path = Path(__file__).resolve().parents[1] / "steernav" / "steer_dummy_path.py"
    tree = ast.parse(source_path.read_text())
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


constrain_path_to_feasible_prefix = load_path_function(
    "constrain_path_to_feasible_prefix"
)


def test_fully_blocked_path_becomes_stop_path():
    path = np.array([[0.2, 0.0], [0.4, 0.1], [0.6, 0.2]])

    result = constrain_path_to_feasible_prefix(path, np.array([0.1, 0.2, 0.3]))

    np.testing.assert_array_equal(result, np.zeros_like(path))


def test_path_stops_at_first_blocked_waypoint():
    path = np.array([[0.2, 0.0], [0.4, 0.1], [0.6, 0.2], [0.8, 0.3]])
    z_scaled = np.array([-0.3, -0.1, 0.2, -0.2])

    result = constrain_path_to_feasible_prefix(path, z_scaled)

    np.testing.assert_allclose(
        result,
        [[0.2, 0.0], [0.4, 0.1], [0.4, 0.1], [0.4, 0.1]],
    )


def test_fully_feasible_path_is_unchanged():
    path = np.array([[0.2, 0.0], [0.4, 0.1]])

    result = constrain_path_to_feasible_prefix(path, np.array([-0.2, -0.1]))

    np.testing.assert_array_equal(result, path)


def test_small_positive_values_within_tolerance_are_feasible():
    path = np.array([[0.2, 0.0], [0.4, 0.1], [0.6, 0.2]])

    result = constrain_path_to_feasible_prefix(
        path,
        np.array([0.02, 0.03, 0.031]),
    )

    np.testing.assert_allclose(result, [[0.2, 0.0], [0.4, 0.1], [0.4, 0.1]])


def test_path_and_feasibility_lengths_must_match():
    with pytest.raises(ValueError, match="same number of points"):
        constrain_path_to_feasible_prefix(
            np.array([[0.2, 0.0], [0.4, 0.1]]),
            np.array([-0.2]),
        )
